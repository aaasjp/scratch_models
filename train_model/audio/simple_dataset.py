#!/usr/bin/env python3
"""
简化的TTS数据集类 - 使用更可靠的数据集
"""

import torch
import numpy as np
import librosa
import re
from datasets import load_dataset
from torch.utils.data import Dataset

class SimpleTTSDataset(Dataset):
    """简化的TTS数据集类 - 使用LibriSpeech数据集"""
    
    def __init__(self, dataset_name="librispeech", split="train", max_samples=None):
        """
        使用更可靠的数据集
        
        Args:
            dataset_name: 数据集名称，支持 'librispeech', 'common_voice'
            split: 数据集分割 ('train', 'validation', 'test')
            max_samples: 最大样本数
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        
        # 加载数据集
        self.dataset = self._load_dataset()
        
        # 音频参数
        self.sample_rate = 22050
        self.n_mels = 80
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
    def _load_dataset(self):
        """加载数据集"""
        try:
            if self.dataset_name == "librispeech":
                # LibriSpeech数据集 - 更稳定
                dataset = load_dataset("librispeech_asr", "clean", split=self.split, trust_remote_code=True)
            elif self.dataset_name == "common_voice":
                # Common Voice数据集
                dataset = load_dataset("mozilla-foundation/common_voice_11_0", 
                                     language="en", split=self.split, trust_remote_code=True)
            else:
                raise ValueError(f"不支持的数据集: {self.dataset_name}")
            
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            
            return dataset
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("使用模拟数据...")
            return self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """创建模拟数据集"""
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 创建模拟音频数据
                duration = np.random.uniform(1.0, 5.0)  # 1-5秒
                sample_rate = 22050
                audio_length = int(duration * sample_rate)
                
                # 生成简单的音频信号
                t = np.linspace(0, duration, audio_length)
                frequency = np.random.uniform(100, 500)  # 随机频率
                audio = np.sin(2 * np.pi * frequency * t) * 0.1
                
                # 添加一些噪声
                noise = np.random.normal(0, 0.01, audio_length)
                audio = audio + noise
                
                # 生成随机文本
                text = f"Sample text {idx} with random content for testing"
                
                return {
                    'audio': {'array': audio, 'sampling_rate': sample_rate},
                    'text': text
                }
        
        return DummyDataset(self.max_samples or 100)
    
    def _extract_mel_spectrogram(self, audio):
        """提取梅尔频谱图"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        # 转换为对数域
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec.T)  # [time, n_mels]
    
    def _extract_pitch(self, audio):
        """提取音高序列"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 使用librosa提取音高
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            threshold=0.1
        )
        
        # 取每帧的最大音高
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
            else:
                pitch_values.append(0)
        
        return torch.FloatTensor(pitch_values)
    
    def _extract_energy(self, audio):
        """提取能量序列"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 计算RMS能量
        frame_length = self.hop_length
        energy = []
        
        for i in range(0, len(audio) - frame_length + 1, self.hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)
        
        return torch.FloatTensor(energy)
    
    def _text_to_sequence(self, text):
        """将文本转换为序列"""
        # 简单的字符级编码
        text = re.sub(r'[^\w\s]', '', text.lower())
        char_to_id = {char: idx for idx, char in enumerate(set(text))}
        sequence = [char_to_id.get(char, 0) for char in text]
        return torch.LongTensor(sequence)
    
    def _estimate_duration(self, mel_spec, text_length):
        """估计持续时间"""
        mel_length = mel_spec.size(0)
        duration_per_char = mel_length / text_length
        durations = [duration_per_char] * text_length
        return torch.FloatTensor(durations)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.dataset[idx]
        
        # 获取音频和文本
        if 'audio' in item:
            audio = item['audio']['array']
            text = item['text']
        else:
            # 处理不同的数据格式
            audio = item.get('audio', {}).get('array', np.random.randn(22050))
            text = item.get('text', item.get('sentence', f"Sample text {idx}"))
        
        # 文本预处理
        text_sequence = self._text_to_sequence(text)
        
        # 音频预处理
        mel_spec = self._extract_mel_spectrogram(audio)
        pitch = self._extract_pitch(audio)
        energy = self._extract_energy(audio)
        
        # 估计持续时间
        duration = self._estimate_duration(mel_spec, len(text_sequence))
        
        # 对齐序列长度
        min_len = min(len(text_sequence), len(pitch), len(energy), len(duration))
        text_sequence = text_sequence[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]
        duration = duration[:min_len]
        
        return {
            'text': text_sequence,
            'mel': mel_spec,
            'duration': duration,
            'pitch': pitch,
            'energy': energy
        }

def test_simple_dataset():
    """测试简化数据集"""
    print("测试简化数据集...")
    
    try:
        # 测试LibriSpeech数据集
        dataset = SimpleTTSDataset(
            dataset_name="librispeech",
            split="train",
            max_samples=5
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试第一个样本
        sample = dataset[0]
        print(f"样本信息:")
        print(f"  文本长度: {len(sample['text'])}")
        print(f"  梅尔频谱形状: {sample['mel'].shape}")
        print(f"  音高序列长度: {len(sample['pitch'])}")
        print(f"  能量序列长度: {len(sample['energy'])}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    test_simple_dataset()
