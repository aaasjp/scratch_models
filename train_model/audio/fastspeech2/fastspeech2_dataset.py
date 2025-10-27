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
from typing import Optional, List
import re
import tempfile
import shutil
from pathlib import Path

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
                 fmax=8000,
                 eps=1e-6):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

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
        # mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        
        return mel_db.T  # [T, n_mels]
    
    

    def get_pitch(self, audio: np.ndarray) -> np.ndarray:
        """提取基频，过滤静音帧无效值"""
        if audio.size == 0:
            raise Exception("Input audio is empty.")

        fmin = librosa.note_to_hz('C2')  # ~65.4 Hz
        fmax = librosa.note_to_hz('C7')  # ~2093 Hz

        f0 = librosa.yin(
            y=audio,
            fmin=fmin,
            fmax=fmax,
            sr=self.sample_rate,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )
        
        # 过滤无效F0（超出范围或接近0的值设为0，表示静音）
        f0[(f0 < fmin) | (f0 > fmax) | (f0 < 1e-3)] = 0.0
        
        # 仅对有声段进行log转换（避免静音帧的0值影响）
        f0[f0 > 0] = np.log(f0[f0 > 0] + 1)
        return f0



    def get_energy(self, audio):
        """提取RMS能量，过滤静音段噪声"""
        # 使用RMS计算能量，更贴近人耳对响度的感知
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]  # [T]
        
        # 计算能量阈值（例如：小于全局最大值的1%视为静音，能量设为0）
        energy_threshold = np.max(energy) * 0.01 if np.max(energy) > 0 else 0
        energy[energy < energy_threshold] = 0.0
        
        # 仅对有声段进行log转换（避免静音帧的0值影响）
        energy[energy > 0] = np.log(energy[energy > 0] + 1)
        return energy


class LJSpeechDataset(Dataset):
    """LJSpeech 数据集（从本地已对齐数据加载）"""
    def __init__(self, 
                 split='train',
                 corpus_dir='./corpus',
                 aligned_dir='./corpus_aligned',
                 processed_dir='./processed_data',
                 max_samples=None,
                 force_preprocess=False):
        """
        Args:
            split: 'train' 或 'validation'
            corpus_dir: 原始音频和文本文件目录
            aligned_dir: MFA对齐后的TextGrid文件目录
            processed_dir: 预处理后的数据保存目录
            max_samples: 最大样本数（用于快速测试）
            force_preprocess: 是否强制重新预处理
        """
        self.split = split
        self.corpus_dir = corpus_dir
        self.aligned_dir = aligned_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        print(f"corpus_dir: {corpus_dir}")
        print(f"aligned_dir: {aligned_dir}")
        print(f"processed_dir: {processed_dir}")
        
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
            # 获取所有可用的文件
            self.file_list = self._get_file_list()
            
            # 分割训练集和验证集
            total_size = len(self.file_list)
            train_size = int(total_size * 0.9)
            
            if split == 'train':
                self.file_list = self.file_list[:train_size]
            else:  # validation
                self.file_list = self.file_list[train_size:]
            
            if max_samples:
                self.file_list = self.file_list[:min(max_samples, len(self.file_list))]
            
            print(f"数据集大小: {len(self.file_list)}")
            
            # 初始化或加载 tokenizer
            if os.path.exists(tokenizer_path) and split == 'validation':
                self.tokenizer = PhonemeTokenizer.load(tokenizer_path)
            else:
                self.tokenizer = PhonemeTokenizer()
                
                # 如果是训练集，从所有TextGrid文件构建音素词表
                if split == 'train':
                    print("从所有TextGrid文件构建音素词表...")
                    all_phonemes = set()
                    
                    # 获取所有可用的文件（不仅仅是当前分割）
                    all_files = self._get_file_list()
                    for file_info in tqdm(all_files, desc="Building phoneme vocabulary"):
                        phonemes = self._extract_phonemes_from_textgrid(file_info['textgrid_path'])
                        all_phonemes.update(phonemes)
                    
                    # 添加所有音素到tokenizer
                    for phoneme in sorted(all_phonemes):
                        self.tokenizer.add_token(phoneme)
                    
                    print(f"✓ 构建了包含 {len(all_phonemes)} 个音素的词表")
            
            # 预处理所有样本
            self.samples = []
            pitch_values = []
            energy_values = []
            duration_values = []
            mel_values = []
            
            print("开始预处理样本...")
            for idx in tqdm(range(len(self.file_list)), desc=f"Processing {split}"):
                try:
                    sample = self._preprocess_sample(idx)
                    if sample is not None:
                        self.samples.append(sample)
                        pitch_values.extend(sample['pitch'].tolist())
                        energy_values.extend(sample['energy'].tolist())
                        duration_values.extend(sample['duration'].tolist())
                        mel_values.extend(sample['mel'].flatten().tolist())
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
                energy_values = [e for e in energy_values if e > 0]  # 只考虑有声部分
                self.stats = {
                    'pitch_min': float(np.min(pitch_values)),
                    'pitch_max': float(np.max(pitch_values)),
                    'pitch_mean': float(np.mean(pitch_values)),
                    'pitch_std': float(np.std(pitch_values)),
                    'energy_min': float(np.min(energy_values)),
                    'energy_max': float(np.max(energy_values)),
                    'energy_mean': float(np.mean(energy_values)),
                    'energy_std': float(np.std(energy_values)),
                    'duration_min': float(np.min(duration_values)),
                    'duration_max': float(np.max(duration_values)),
                    'duration_mean': float(np.mean(duration_values)),
                    'duration_std': float(np.std(duration_values)),
                    'mel_min': float(np.min(mel_values)),
                    'mel_max': float(np.max(mel_values)),
                    'mel_mean': float(np.mean(mel_values)),
                    'mel_std': float(np.std(mel_values))
                }
                
                with open(stats_path, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                print(f"✓ 统计信息已保存: {stats_path}")
                print(f"  Pitch range: [{self.stats['pitch_min']:.1f}, {self.stats['pitch_max']:.1f}]")
                print(f"  Energy range: [{self.stats['energy_min']:.4f}, {self.stats['energy_max']:.4f}]")
                print(f"  Duration range: [{self.stats['duration_min']:.4f}, {self.stats['duration_max']:.4f}]")
                print(f"  Mel range: [{self.stats['mel_min']:.4f}, {self.stats['mel_max']:.4f}]")
            else:
                # 验证集使用训练集的统计信息
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            
            # 保存预处理后的样本
            with open(samples_path, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"✓ 预处理数据已保存: {samples_path}")
    
    def _get_file_list(self):
        """获取所有可用的文件列表"""
        wav_files = []
        for file in os.listdir(self.corpus_dir):
            if file.endswith('.wav'):
                base_name = file[:-4]  # 去掉.wav扩展名
                wav_path = os.path.join(self.corpus_dir, file)
                lab_path = os.path.join(self.corpus_dir, base_name + '.lab')
                textgrid_path = os.path.join(self.aligned_dir, base_name + '.TextGrid')
                
                # 检查所有必需的文件是否存在
                if os.path.exists(lab_path) and os.path.exists(textgrid_path):
                    wav_files.append({
                        'wav_path': wav_path,
                        'lab_path': lab_path,
                        'textgrid_path': textgrid_path,
                        'base_name': base_name
                    })
        
        # 按文件名排序
        wav_files.sort(key=lambda x: x['base_name'])
        return wav_files
    
    def _read_textgrid_durations(self, textgrid_path: str) -> Optional[np.ndarray]:
        """
        从TextGrid文件读取MFA对齐结果，返回每个音素的duration（帧数）
        包含空音素（静音段）
        
        Args:
            textgrid_path: TextGrid文件路径
            
        Returns:
            durations: 每个音素的duration（帧数），如果读取失败返回None
        """
        try:
            import textgrid
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # 查找phones层
            phoneme_tier = None
            for tier in tg.tiers:
                if tier.name == 'phones':
                    phoneme_tier = tier
                    break
            
            if phoneme_tier is None:
                print(f"警告: 未找到phones层 in {textgrid_path}")
                return None
            
            # 计算每个音素的duration（帧数），包含空音素
            durations = []
            for interval in phoneme_tier:
                # 将时间转换为帧数
                duration_frames = int(interval.maxTime * self.audio_processor.sample_rate / self.audio_processor.hop_length) - \
                               int(interval.minTime * self.audio_processor.sample_rate / self.audio_processor.hop_length)
                durations.append(max(1, duration_frames))  # 确保至少1帧
            
            return np.array(durations, dtype=np.float32)
            
        except Exception as e:
            print(f"读取TextGrid文件失败 {textgrid_path}: {e}")
            return None
    
    def _read_lab_text(self, lab_path: str) -> str:
        """从.lab文件读取文本"""
        try:
            with open(lab_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"读取lab文件失败 {lab_path}: {e}")
            return ""
    
    def _extract_phonemes_from_textgrid(self, textgrid_path: str) -> List[str]:
        """
        从TextGrid文件中提取音素序列，包含空音素（静音段）
        
        Args:
            textgrid_path: TextGrid文件路径
            
        Returns:
            phonemes: 音素列表，空音素用特殊标记表示
        """
        try:
            import textgrid
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # 查找phones层
            phoneme_tier = None
            for tier in tg.tiers:
                if tier.name == 'phones':
                    phoneme_tier = tier
                    break
            
            if phoneme_tier is None:
                print(f"警告: 未找到phones层 in {textgrid_path}")
                return []
            
            # 提取音素，包含空音素
            phonemes = []
            for interval in phoneme_tier:
                if interval.mark.strip():  # 有内容的音素
                    phonemes.append(interval.mark.strip())
                else:  # 空音素（静音段）
                    phonemes.append('sil')  # 用特殊标记表示静音
            
            return phonemes
            
        except Exception as e:
            print(f"从TextGrid提取音素失败 {textgrid_path}: {e}")
            return []

    
    def _phoneme_level_pitch_energy(self, pitch, energy, durations, phonemes):
        """
        将帧级的 pitch 和 energy 转换为音素级
        使用phoneme标记来判断静音段
        
        Args:
            pitch: [T] 帧级 pitch
            energy: [T] 帧级 energy
            durations: [N] 每个音素的帧数
            phonemes: [N] 音素列表
        
        Returns:
            phoneme_pitch: [N] 音素级 pitch
            phoneme_energy: [N] 音素级 energy
        """
        # 定义静音音素标记
        SILENCE_PHONES = {'sil', 'sp', 'spn', '', 'SIL', 'SP'}
        
        phoneme_pitch = []
        phoneme_energy = []
        
        # 计算全局最小能量作为静音参考
        min_energy = np.percentile(energy[energy > 0], 5) if np.any(energy > 0) else 0.0
        
        start = 0
        for phoneme, dur in zip(phonemes, durations):
            end = start + int(dur)
            end = min(end, len(pitch))
            
            if start < end:
                p_segment = pitch[start:end]
                e_segment = energy[start:end]
                
                # 判断是否为静音
                is_silence = phoneme in SILENCE_PHONES
                
                if is_silence:
                    # 静音段：使用固定的低值
                    phoneme_pitch.append(0.0)  # 或者 np.log(1.0)
                    phoneme_energy.append(min_energy)
                else:
                    # 有声段
                    # Pitch: 只对voiced部分求平均（pitch > 阈值的部分）
                    pitch_threshold = np.log(50)  # 约50Hz的阈值
                    voiced_mask = p_segment > pitch_threshold
                    
                    if np.any(voiced_mask):
                        phoneme_pitch.append(np.mean(p_segment[voiced_mask]))
                    else:
                        # 清音（如 /s/, /f/）：使用0或很小的值
                        phoneme_pitch.append(0.0)
                    
                    # Energy: 直接求平均
                    phoneme_energy.append(np.mean(e_segment))
            else:
                # 异常情况
                phoneme_pitch.append(0.0)
                phoneme_energy.append(min_energy)
            
            start = end
        
        return np.array(phoneme_pitch), np.array(phoneme_energy)

    def _preprocess_sample(self, idx):
        """预处理单个样本"""
        file_info = self.file_list[idx]
        wav_path = file_info['wav_path']
        lab_path = file_info['lab_path']
        textgrid_path = file_info['textgrid_path']
        
        # 读取音频文件
        audio, sr = librosa.load(wav_path, sr=None)
        
        # 重采样到22050Hz
        if sr != self.audio_processor.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_processor.sample_rate)
        
        # 不需要归一化音频，因为librosa.load已经归一化了
        # audio = audio / (np.max(np.abs(audio)) + 1e-6) ## 峰值归一化
        
        # 读取文本
        text = self._read_lab_text(lab_path)
        if not text:
             return None
        
        # 从TextGrid文件直接获取音素
        phonemes = self._extract_phonemes_from_textgrid(textgrid_path)
        if len(phonemes) == 0:
            return None
        
        # 转换为 ID
        phoneme_ids = self.tokenizer.encode(phonemes)
        
        # 提取声学特征
        mel = self.audio_processor.get_mel_spectrogram(audio)
        pitch = self.audio_processor.get_pitch(audio)
        energy = self.audio_processor.get_energy(audio)

        print("--------------------------------")
        print(f"idx: {idx}")
        print(f"mel.shape: {mel.shape}")
        print(f"pitch.shape: {pitch.shape}")
        print(f"energy.shape: {energy.shape}")
        # 确保长度一致
        assert mel.shape[0] == len(pitch) == len(energy), "失败：mel, pitch, energy长度不一致"
        #min_len = min(mel.shape[0], len(pitch), len(energy))
        #mel = mel[:min_len]
        #pitch = pitch[:min_len]
        #energy = energy[:min_len]
        
        # 从TextGrid文件读取duration
        #durations = self._estimate_duration(phonemes, mel.shape[0], textgrid_path)
        durations=self._read_textgrid_durations(textgrid_path)
        print(f"durations: {durations}")
        print(f"durations.shape: {durations.shape}")
        print(f"durations.sum(): {durations.sum()}")
        print("--------------------------------")

        mel=mel[:int(durations.sum())]
        pitch=pitch[:int(durations.sum())]
        energy=energy[:int(durations.sum())]
        print("--------------------------------")
        
        # 转换为音素级的 pitch 和 energy
        phoneme_pitch, phoneme_energy = self._phoneme_level_pitch_energy(
            pitch, energy, durations, phonemes
        )
        
        return {
            'text': text,
            'phonemes': phonemes,
            'phoneme_ids': np.array(phoneme_ids, dtype=np.int64),
            'text_length': len(phoneme_ids),
            'phoneme_length': len(phonemes),
            'mel': mel.astype(np.float32),
            'mel_length': mel.shape[0],
            'duration': durations.astype(np.float32),
            'pitch': phoneme_pitch.astype(np.float32),
            'energy': phoneme_energy.astype(np.float32),
            'audio_path': wav_path
        }
    
    def __len__(self):
        return len(self.samples)
    
    ## 此处作变量名称转换，为了适配训练程序的变量名称!!
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'text': torch.from_numpy(sample['phoneme_ids']),
            'text_length': len(sample['phoneme_ids']),
            'raw_text': sample['text'],
            'raw_text_length': len(sample['text']),
            'phonemes': sample['phonemes'],
            'mel': torch.from_numpy(sample['mel']),
            'mel_length': sample['mel_length'],
            'duration': torch.from_numpy(sample['duration']),
            'pitch': torch.from_numpy(sample['pitch']),
            'energy': torch.from_numpy(sample['energy']),
            'audio_path': sample['audio_path']
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
        corpus_dir='./corpus',
        aligned_dir='./corpus_aligned',
        processed_dir='./processed_data',
        max_samples=None,  # 快速测试，使用150个样本；实际训练时设为None
        force_preprocess=False
    )
    
    print("\n" + "=" * 70)
    print("加载验证集...")
    print("=" * 70)
    
    val_dataset = LJSpeechDataset(
        split='validation',
        corpus_dir='./corpus',
        aligned_dir='./corpus_aligned',
        processed_dir='./processed_data',
        max_samples=None,  # 快速测试；实际训练时设为None
        force_preprocess=False
    )
    
    print("\n" + "=" * 70)
    print("数据集信息")
    print("=" * 70)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"词表大小: {len(train_dataset.tokenizer)}")
    
    # 查看所有样本
    print("\n" + "=" * 70)
    print(f"训练集样本数量: {len(train_dataset)}")
    if len(train_dataset) > 0:
        print("\n训练集样本详情:")
        for i in range(len(train_dataset)):
            print("=" * 70)
            sample = train_dataset[i]
            print(f"\n样本 {i+1}:")
            print(f"  文本: {sample['raw_text']}")
            print(f"  文本长度: {sample['raw_text_length']}")
            print(f"  音素: {sample['phonemes']}")
            print(f"  音素ID: {sample['text']}")
            print(f"  音素长度: {sample['text_length']}")
            print(f"  Mel频谱形状: {sample['mel'].shape}")
            print(f"  Mel长度: {sample['mel_length']}")
            print(f"  Duration长度: {len(sample['duration'])}")
            print(f"  Durations: {sample['duration']}")
            print(f"  Duration总和: {sample['duration'].sum().item():.1f}")
            print(f"  Pitch形状: {sample['pitch'].shape}")
            print(f"  Pitch: {sample['pitch']}")
            print(f"  Energy形状: {sample['energy'].shape}")
            print(f"  Energy: {sample['energy']}")
            print(f"  音频路径: {sample['audio_path']}")
            print(f"  验证对齐: sum(duration)={sample['duration'].sum().item():.1f}, mel_length={sample['mel_length']}")
            print("=" * 70)
    
    # 查看验证集所有样本
    print("\n" + "=" * 70)
    print(f"验证集样本数量: {len(val_dataset)}")
    if len(val_dataset) > 0:
        print("\n验证集样本详情:")
        for i in range(len(val_dataset)):
            print("=" * 70)
            sample = val_dataset[i]
            print(f"\n样本 {i+1}:")
            print(f"  文本: {sample['raw_text']}")
            print(f"  文本长度: {sample['raw_text_length']}")
            print(f"  音素: {sample['phonemes']}")
            print(f"  音素ID: {sample['text']}")
            print(f"  音素长度: {sample['text_length']}")
            print(f"  Mel频谱形状: {sample['mel'].shape}")
            print(f"  Mel长度: {sample['mel_length']}")
            print(f"  Duration长度: {len(sample['duration'])}")
            print(f"  Durations: {sample['duration']}")
            print(f"  Duration总和: {sample['duration'].sum().item():.1f}")
            print(f"  Pitch形状: {sample['pitch'].shape}")
            print(f"  Pitch: {sample['pitch']}")
            print(f"  Energy形状: {sample['energy'].shape}")
            print(f"  Energy: {sample['energy']}")
            print(f"  音频路径: {sample['audio_path']}")
            print(f"  验证对齐: sum(duration)={sample['duration'].sum().item():.1f}, mel_length={sample['mel_length']}")
            print("=" * 70)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print("\n" + "=" * 70)
    print("✓ 数据加载完成！")
    print("=" * 70)
    
    print("\n下一步:")
    print("1. 使用 fastspeech2.py 中的模型")
    print("2. 使用 fastspeech2_training_.py 中的训练代码")
    print("3. 开始训练!")