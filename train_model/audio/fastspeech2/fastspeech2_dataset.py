"""
FastSpeech2 训练 - 使用 HuggingFace LJSpeech 数据集
需要安装: pip install datasets librosa phonemizer pyworld
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import librosa
from datasets import load_dataset
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import json
import pickle
import subprocess


class PhonemeTokenizer:
    """音素分词器"""
    def __init__(self):
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.next_id = len(self.special_tokens)
        
    def add_token(self, token):
        """添加新的音素token"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
    
    def encode(self, phonemes):
        """将音素序列转换为ID"""
        ids = [self.token_to_id.get(p, self.token_to_id[self.unk_token]) for p in phonemes]
        return ids
    
    def decode(self, ids):
        """将ID转换回音素序列"""
        return [self.id_to_token.get(i, self.unk_token) for i in ids]
    
    def __len__(self):
        return len(self.token_to_id)
    
    def save(self, path):
        """保存tokenizer"""
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': {int(k): v for k, v in self.id_to_token.items()}
            }, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """加载tokenizer"""
        tokenizer = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            tokenizer.token_to_id = data['token_to_id']
            tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
            tokenizer.next_id = len(tokenizer.token_to_id)
        return tokenizer


class AudioProcessor:
    """音频处理器 - 提取梅尔频谱、pitch、energy"""
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mels=80,
                 fmin=0,
                 fmax=8000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
    def get_mel_spectrogram(self, audio):
        """提取梅尔频谱"""
        # 确保音频是float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 提取梅尔频谱
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # 转换到对数域
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return mel_db.T  # [T, n_mels]
    
    def get_pitch(self, audio):
        """提取pitch (使用更快的算法)"""
        try:
            # 方法1: 使用 librosa.yin (比 pyin 快很多)
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )
            
            # 处理未发声部分（用0填充）
            f0 = np.where(np.isnan(f0), 0.0, f0)
            
            return f0
        except Exception as e:
            print(f"YIN pitch 提取失败: {e}")
            # 备选方案：使用简单的自相关方法
            try:
                # 简化的 pitch 估计
                frame_length = self.win_length
                hop_length = self.hop_length
                n_frames = 1 + (len(audio) - frame_length) // hop_length
                f0 = np.zeros(n_frames)
                
                for i in range(n_frames):
                    start = i * hop_length
                    end = start + frame_length
                    if end <= len(audio):
                        frame = audio[start:end]
                        # 简单的 pitch 估计（基于零交叉率）
                        zcr = librosa.feature.zero_crossing_rate(frame.reshape(1, -1))[0, 0]
                        # 粗略的 pitch 估计
                        if zcr < 0.1:  # 低频信号
                            f0[i] = 200.0  # 默认 pitch
                        else:
                            f0[i] = 0.0  # 无声
                
                return f0
            except Exception as e2:
                print(f"备选 pitch 提取也失败: {e2}")
                # 最后的备选：返回固定值
                n_frames = 1 + (len(audio) - self.win_length) // self.hop_length
                return np.full(n_frames, 200.0)  # 默认 pitch
    
    def get_energy(self, audio):
        """提取能量"""
        # 使用RMS能量
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
        
        return energy


class LJSpeechDataset(Dataset):
    """LJSpeech 数据集（从 HuggingFace 加载）"""
    def __init__(self, 
                 split='train',
                 cache_dir='./cache',
                 processed_dir='./processed_data',
                 max_samples=None,
                 force_preprocess=False):
        """
        Args:
            split: 'train' 或 'validation'
            cache_dir: HuggingFace数据集缓存目录
            processed_dir: 预处理后的数据保存目录
            max_samples: 最大样本数（用于快速测试）
            force_preprocess: 是否强制重新预处理
        """
        self.split = split
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor()
        
        # 检查是否已有预处理数据
        tokenizer_path = os.path.join(processed_dir, 'tokenizer.json')
        stats_path = os.path.join(processed_dir, 'stats.json')
        samples_path = os.path.join(processed_dir, f'{split}_samples.pkl')
        
        if not force_preprocess and os.path.exists(samples_path) and os.path.exists(tokenizer_path):
            print(f"加载预处理的 {split} 数据...")
            self.tokenizer = PhonemeTokenizer.load(tokenizer_path)
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            with open(samples_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"✓ 加载了 {len(self.samples)} 个样本")
        else:
            print(f"预处理 {split} 数据...")
            # 加载 LJSpeech 数据集
            print("从 HuggingFace 下载 LJSpeech 数据集...")
            dataset = load_dataset("MikhailT/lj-speech", cache_dir=cache_dir)
            
            # LJSpeech 没有官方的 train/val split，我们手动分割
            # 使用90%作为训练集，10%作为验证集
            total_size = len(dataset['full'])
            train_size = int(total_size * 0.9)
            
            if split == 'train':
                raw_data = dataset['full'].select(range(train_size))
            else:  # validation
                raw_data = dataset['full'].select(range(train_size, total_size))
            
            if max_samples:
                raw_data = raw_data.select(range(min(max_samples, len(raw_data))))
            
            print(f"数据集大小: {len(raw_data)}")
            print(f"数据样本[0]: {raw_data[0]}")
            
            # 初始化或加载 tokenizer
            if os.path.exists(tokenizer_path) and split == 'validation':
                self.tokenizer = PhonemeTokenizer.load(tokenizer_path)
            else:
                self.tokenizer = PhonemeTokenizer()
            
            # 预处理所有样本
            self.samples = []
            pitch_values = []
            energy_values = []
            
            print("开始预处理样本...")
            for idx in tqdm(range(len(raw_data)), desc=f"Processing {split}"):
                try:
                    sample = self._preprocess_sample(raw_data[idx])
                    if sample is not None:
                        self.samples.append(sample)
                        pitch_values.extend(sample['pitch'].tolist())
                        energy_values.extend(sample['energy'].tolist())
                except Exception as e:
                    print(f"\n警告: 样本 {idx} 处理失败: {e}")
                    continue
            
            print(f"\n✓ 成功处理 {len(self.samples)} 个样本")
            
            # 保存 tokenizer 和统计信息（仅在训练集上）
            if split == 'train':
                self.tokenizer.save(tokenizer_path)
                print(f"✓ Tokenizer 已保存: {tokenizer_path}")
                print(f"  词表大小: {len(self.tokenizer)}")
                
                # 计算统计信息
                pitch_values = [p for p in pitch_values if p > 0]  # 只考虑有声部分
                self.stats = {
                    'pitch_min': float(np.min(pitch_values)) if pitch_values else 0.0,
                    'pitch_max': float(np.max(pitch_values)) if pitch_values else 800.0,
                    'pitch_mean': float(np.mean(pitch_values)) if pitch_values else 200.0,
                    'pitch_std': float(np.std(pitch_values)) if pitch_values else 50.0,
                    'energy_min': float(np.min(energy_values)) if energy_values else 0.0,
                    'energy_max': float(np.max(energy_values)) if energy_values else 1.0,
                    'energy_mean': float(np.mean(energy_values)) if energy_values else 0.5,
                    'energy_std': float(np.std(energy_values)) if energy_values else 0.1
                }
                
                with open(stats_path, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                print(f"✓ 统计信息已保存: {stats_path}")
                print(f"  Pitch range: [{self.stats['pitch_min']:.1f}, {self.stats['pitch_max']:.1f}]")
                print(f"  Energy range: [{self.stats['energy_min']:.4f}, {self.stats['energy_max']:.4f}]")
            else:
                # 验证集使用训练集的统计信息
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            
            # 保存预处理后的样本
            with open(samples_path, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"✓ 预处理数据已保存: {samples_path}")
    
    def _text_to_phonemes(self, text):
        """将文本转换为音素"""
        # 使用 espeak 后端进行 G2P
        try:
            # 清理文本
            text = text.lower().strip()
            
            '''
            # 方法1: 使用 phonemizer (指定 espeak 路径)
            try:
                phonemes = phonemize(
                    text,
                    language='en-us',
                    backend='espeak',
                    strip=True,
                    preserve_punctuation=False,
                    with_stress=False,
                    separator=' ',
                    njobs=1
                )
                phoneme_list = phonemes.split()
                if phoneme_list:
                    return phoneme_list
            except Exception as e1:
                print(f"phonemizer 失败: {e1}")
            
            '''
            # 方法2: 直接调用 espeak
            try:
                result = subprocess.run(
                    ['/opt/homebrew/bin/espeak', '-q', '--ipa', '-v', 'en-us', text],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    phonemes = result.stdout.strip()
                    print(f"phonemes: {phonemes}")
                    # 简单的音素分割（移除 IPA 符号）
                    phoneme_list = [p for p in phonemes if p.isalpha() or p in 'ɪəʊʌɑːæeɪaɪɔɪəʊaʊɪəeəʊə']
                    if phoneme_list:
                        return phoneme_list
            except Exception as e2:
                print(f"espeak 直接调用失败: {e2}")
            
            # 方法3: 字符级分割作为备选
            print("使用字符级分割作为备选...")
            return list(text.replace(' ', '_'))
            
        except Exception as e:
            print(f"音素转换完全失败: {e}")
            return list(text.replace(' ', '_'))
    
    def _estimate_duration(self, phonemes, mel_len):
        """
        估计每个音素的 duration
        
        注意: 这是一个简化的估计方法
        实际项目中应该使用 Montreal Forced Aligner 进行精确对齐
        """
        n_phonemes = len(phonemes)
        if n_phonemes == 0:
            return np.array([])
        
        # 简单平均分配
        avg_duration = mel_len / n_phonemes
        durations = np.full(n_phonemes, avg_duration)
        
        # 调整以确保总和等于 mel_len
        diff = mel_len - durations.sum()
        durations[0] += diff
        
        # 确保所有 duration >= 1
        durations = np.maximum(durations, 1.0)
        
        return durations
    
    def _phoneme_level_pitch_energy(self, pitch, energy, durations):
        """
        将帧级的 pitch 和 energy 转换为音素级（取平均）
        
        Args:
            pitch: [T] 帧级 pitch
            energy: [T] 帧级 energy
            durations: [N] 每个音素的帧数
        
        Returns:
            phoneme_pitch: [N] 音素级 pitch
            phoneme_energy: [N] 音素级 energy
        """
        phoneme_pitch = []
        phoneme_energy = []
        
        start = 0
        for dur in durations:
            end = start + int(dur)
            end = min(end, len(pitch))
            
            if start < end:
                # 计算该音素对应帧的平均值
                p_segment = pitch[start:end]
                e_segment = energy[start:end]
                
                # Pitch: 只对有声部分求平均
                voiced_pitch = p_segment[p_segment > 0]
                if len(voiced_pitch) > 0:
                    phoneme_pitch.append(np.mean(voiced_pitch))
                else:
                    phoneme_pitch.append(0.0)
                
                # Energy: 直接求平均
                phoneme_energy.append(np.mean(e_segment))
            else:
                phoneme_pitch.append(0.0)
                phoneme_energy.append(0.0)
            
            start = end
        
        return np.array(phoneme_pitch), np.array(phoneme_energy)
    
    def _preprocess_sample(self, raw_sample):
        """预处理单个样本"""
        # 获取音频和文本
        audio = raw_sample['audio']['array']
        sr = raw_sample['audio']['sampling_rate']
        text = raw_sample['normalized_text']

        # 重采样到22050Hz
        if sr != self.audio_processor.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_processor.sample_rate)
        
        # 归一化音频
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # 文本转音素
        phonemes = self._text_to_phonemes(text)
        if len(phonemes) == 0:
            return None
        
        # 添加音素到 tokenizer
        for p in phonemes:
            self.tokenizer.add_token(p)
        
        # 转换为 ID
        phoneme_ids = self.tokenizer.encode(phonemes)
        
        # 提取声学特征
        mel = self.audio_processor.get_mel_spectrogram(audio)
        pitch = self.audio_processor.get_pitch(audio)
        energy = self.audio_processor.get_energy(audio)
        
        # 确保长度一致
        min_len = min(mel.shape[0], len(pitch), len(energy))
        mel = mel[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]
        
        # 估计 duration
        durations = self._estimate_duration(phonemes, mel.shape[0])
        
        # 转换为音素级的 pitch 和 energy
        phoneme_pitch, phoneme_energy = self._phoneme_level_pitch_energy(
            pitch, energy, durations
        )
        
        return {
            'phoneme_ids': np.array(phoneme_ids, dtype=np.int64),
            'text_length': len(phoneme_ids),
            'mel': mel.astype(np.float32),
            'mel_length': mel.shape[0],
            'duration': durations.astype(np.float32),
            'pitch': phoneme_pitch.astype(np.float32),
            'energy': phoneme_energy.astype(np.float32),
            'text': text,
            'phonemes': phonemes
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'text': torch.from_numpy(sample['phoneme_ids']),
            'text_length': sample['text_length'],
            'mel': torch.from_numpy(sample['mel']),
            'mel_length': sample['mel_length'],
            'duration': torch.from_numpy(sample['duration']),
            'pitch': torch.from_numpy(sample['pitch']),
            'energy': torch.from_numpy(sample['energy'])
        }


def collate_fn(batch):
    """批处理函数"""
    # 获取最大长度
    max_text_len = max([item['text_length'] for item in batch])
    max_mel_len = max([item['mel_length'] for item in batch])
    batch_size = len(batch)
    n_mels = batch[0]['mel'].shape[1]
    
    # 初始化 padded 张量
    text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, max_mel_len, n_mels)
    duration_padded = torch.zeros(batch_size, max_text_len)
    pitch_padded = torch.zeros(batch_size, max_text_len)
    energy_padded = torch.zeros(batch_size, max_text_len)
    
    text_lengths = torch.LongTensor([item['text_length'] for item in batch])
    mel_lengths = torch.LongTensor([item['mel_length'] for item in batch])
    
    # 填充数据
    for i, item in enumerate(batch):
        text_len = item['text_length']
        mel_len = item['mel_length']
        
        text_padded[i, :text_len] = item['text']
        mel_padded[i, :mel_len] = item['mel']
        duration_padded[i, :text_len] = item['duration']
        pitch_padded[i, :text_len] = item['pitch']
        energy_padded[i, :text_len] = item['energy']
    
    return {
        'text': text_padded,
        'text_lengths': text_lengths,
        'mel': mel_padded,
        'mel_lengths': mel_lengths,
        'duration': duration_padded,
        'pitch': pitch_padded,
        'energy': energy_padded
    }


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("FastSpeech2 - 使用 HuggingFace LJSpeech 数据集")
    print("=" * 70)
    
    # 检查依赖
    print("\n检查依赖...")
    try:
        import datasets
        import phonemizer
        import librosa
        print("✓ 所有依赖已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("\n请安装:")
        print("pip install datasets phonemizer librosa")
        exit(1)
    
    # 创建数据集（首次运行会下载和预处理数据）
    print("\n" + "=" * 70)
    print("加载训练集...")
    print("=" * 70)
    
    train_dataset = LJSpeechDataset(
        split='train',
        cache_dir='./cache',
        processed_dir='./processed_data',
        #max_samples=10,  # 快速测试，使用100个样本；实际训练时设为None
        force_preprocess=False
    )
    
    print("\n" + "=" * 70)
    print("加载验证集...")
    print("=" * 70)
    
    val_dataset = LJSpeechDataset(
        split='validation',
        cache_dir='./cache',
        processed_dir='./processed_data',
        #max_samples=2,  # 快速测试；实际训练时设为None
        force_preprocess=False
    )
    
    print("\n" + "=" * 70)
    print("数据集信息")
    print("=" * 70)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"词表大小: {len(train_dataset.tokenizer)}")
    
    # 查看样本
    print("\n" + "=" * 70)
    print("样本示例")
    print("=" * 70)
    sample = train_dataset[0]
    print(f"Text shape: {sample['text'].shape}")
    print(f"Text length: {sample['text_length']}")
    print(f"Mel shape: {sample['mel'].shape}")
    print(f"Mel length: {sample['mel_length']}")
    print(f"Duration shape: {sample['duration'].shape}")
    print(f"Duration sum: {sample['duration'].sum().item():.1f}")
    print(f"Pitch shape: {sample['pitch'].shape}")
    print(f"Energy shape: {sample['energy'].shape}")
    print(f"\n验证对齐: sum(duration)={sample['duration'].sum().item():.1f}, mel_length={sample['mel_length']}")
    
    # 测试 DataLoader
    print("\n" + "=" * 70)
    print("测试 DataLoader")
    print("=" * 70)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"Batch text shape: {batch['text'].shape}")
    print(f"Batch text lengths: {batch['text_lengths'].tolist()}")
    print(f"Batch mel shape: {batch['mel'].shape}")
    print(f"Batch mel lengths: {batch['mel_lengths'].tolist()}")
    
    print("\n" + "=" * 70)
    print("✓ 数据加载测试完成！")
    print("=" * 70)
    
    print("\n下一步:")
    print("1. 使用 fastspeech2.py 中的模型")
    print("2. 使用 fastspeech2_training_.py 中的训练代码")
    print("3. 开始训练!")