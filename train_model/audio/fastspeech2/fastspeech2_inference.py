"""
FastSpeech2 改进版推理脚本
使用 HiFi-GAN 声码器替代 Griffin-Lim

安装依赖:
pip install torch torchaudio phonemizer
"""

import torch
import numpy as np
import argparse
import os
import json
from phonemizer import phonemize
import librosa
from scipy.io import wavfile
import subprocess

from fastspeech2 import FastSpeech2
from fastspeech2_dataset import PhonemeTokenizer


class HiFiGANVocoder:
    """HiFi-GAN 声码器(使用预训练模型)"""
    def __init__(self, device='cuda'):
        self.device = device
        
        try:
            # 尝试使用torch.hub加载预训练HiFi-GAN
            # 这是一个LJSpeech预训练的HiFi-GAN
            print("正在加载 HiFi-GAN 预训练模型...")
            self.model = torch.hub.load(
                'descriptinc/melgan-neurips',
                'load_melgan',
                'multi_speaker'
            )
            self.model = self.model.to(device)
            self.model.eval()
            print("✓ HiFi-GAN 加载成功")
            self.available = True
        except Exception as e:
            print(f"警告: 无法加载HiFi-GAN: {e}")
            print("将使用改进的Griffin-Lim作为备选")
            self.available = False
            self.setup_griffin_lim()
    
    def setup_griffin_lim(self):
        """设置改进的Griffin-Lim参数"""
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mels = 80
        
        # 创建梅尔滤波器
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=8000
        )
    
    def mel_to_audio(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频
        
        Args:
            mel_spectrogram: [T, n_mels] numpy array (dB scale)
        
        Returns:
            audio: [T * hop_length] numpy array
        """
        if self.available:
            # 使用HiFi-GAN
            return self._hifigan_inference(mel_spectrogram)
        else:
            # 使用改进的Griffin-Lim
            return self._improved_griffin_lim(mel_spectrogram)
    
    def _hifigan_inference(self, mel_spectrogram):
        """使用HiFi-GAN生成音频"""
        # 转换为tensor并调整形状
        mel = torch.FloatTensor(mel_spectrogram.T).unsqueeze(0).to(self.device)
        
        # 归一化到 [-1, 1] 范围
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        
        with torch.no_grad():
            audio = self.model.inverse(mel).squeeze()
        
        audio = audio.cpu().numpy()
        return audio
    
    def _improved_griffin_lim(self, mel_spectrogram):
        """改进的Griffin-Lim算法"""
        # 转置为 [n_mels, T]
        mel = mel_spectrogram.T
        
        # 从dB转回功率
        mel_power = librosa.db_to_power(mel)
        
        # 确保所有值为正
        mel_power = np.maximum(mel_power, 1e-10)
        
        # 逆梅尔滤波器(使用伪逆)
        linear_spec = np.dot(np.linalg.pinv(self.mel_basis), mel_power)
        linear_spec = np.maximum(linear_spec, 1e-10)
        
        # 改进的Griffin-Lim: 更多迭代次数,更好的初始化
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=64,  # 增加迭代次数(从32到64)
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            window='hann',
            center=True,
            dtype=np.float32,
            length=None,
            pad_mode='reflect',
            momentum=0.99,  # 添加动量加速收敛
            init='random',  # 随机初始化相位
            random_state=0
        )
        
        # 后处理: 轻度滤波减少噪音
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        return audio


class FastSpeech2Inference:
    """FastSpeech2 推理器(改进版)"""
    def __init__(self, checkpoint_path, device='cuda', use_hifigan=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        self.config = checkpoint.get('config', {})
        vocab_size = self.config.get('vocab_size', 100)
        
        # 加载tokenizer
        tokenizer_path = './processed_data/tokenizer.json'
        if os.path.exists(tokenizer_path):
            self.tokenizer = PhonemeTokenizer.load(tokenizer_path)
            vocab_size = len(self.tokenizer)
            print(f"✓ 加载tokenizer,词表大小: {vocab_size}")
        else:
            raise FileNotFoundError(f"找不到tokenizer: {tokenizer_path}")
        
        # 加载统计信息
        stats_path = './processed_data/stats.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"✓ 加载统计信息")
            print(f"  Pitch range: [{self.stats['pitch_min']:.1f}, {self.stats['pitch_max']:.1f}]")
            print(f"  Energy range: [{self.stats['energy_min']:.4f}, {self.stats['energy_max']:.4f}]")
        else:
            print("警告: 找不到统计信息,使用默认值")
            self.stats = {
                'pitch_min': 0.0, 'pitch_max': 800.0,
                'pitch_mean': 200.0, 'pitch_std': 50.0,
                'energy_min': 0.0, 'energy_max': 1.0,
                'energy_mean': 0.5, 'energy_std': 0.1
            }
        
        # 创建模型
        self.model = FastSpeech2(
            vocab_size=vocab_size,
            d_model=self.config.get('d_model', 256),
            n_layers=self.config.get('n_layers', 4),
            n_heads=self.config.get('n_heads', 2),
            d_ff=self.config.get('d_ff', 1024),
            dropout=self.config.get('dropout', 0.1),
            n_mel_channels=80
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型加载成功")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
        
        # 初始化声码器
        self.vocoder = HiFiGANVocoder(device=self.device)
        
        # 更新模型的统计信息
        self.model.variance_adaptor.pitch_min = torch.tensor(self.stats['pitch_min'])
        self.model.variance_adaptor.pitch_max = torch.tensor(self.stats['pitch_max'])
        self.model.variance_adaptor.energy_min = torch.tensor(self.stats['energy_min'])
        self.model.variance_adaptor.energy_max = torch.tensor(self.stats['energy_max'])
    
    def text_to_phonemes(self, text):
        """将文本转换为音素"""
        try:
            text = text.lower().strip()
            
            # 使用subprocess直接调用espeak
            result = subprocess.run(
                ['espeak', '-q', '--ipa', '-v', 'en-us', text],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                phonemes = result.stdout.strip()
                # 简单的音素分割
                phoneme_list = [p for p in phonemes if p.isalpha() or p in 'ɪəʊʌɒËæeɪaɪɔɪəʊaʊɪəeəʊə']
                
                if phoneme_list:
                    return phoneme_list
                else:
                    return list(text.replace(' ', '_'))
            else:
                return list(text.replace(' ', '_'))
                
        except Exception as e:
            print(f"音素转换失败: {e}")
            return list(text.replace(' ', '_'))
    
    @torch.no_grad()
    def synthesize(self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
        """
        从文本合成语音(改进版)
        """
        print(f"\n{'='*60}")
        print(f"合成文本: {text}")
        print(f"{'='*60}")
        
        # 文本转音素
        phonemes = self.text_to_phonemes(text)
        if len(phonemes) == 0:
            raise ValueError("无法转换文本为音素")
        
        print(f"音素: {' '.join(phonemes)}")
        print(f"音素数量: {len(phonemes)}")
        
        # 转换为ID
        phoneme_ids = self.tokenizer.encode(phonemes)
        text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)
        
        print(f"\n控制参数:")
        print(f"  语速: {duration_control}x")
        print(f"  音高: {pitch_control}x")
        print(f"  能量: {energy_control}x")
        
        # 模型推理
        print(f"\n正在生成梅尔频谱...")
        mel_output, mel_mask, duration_pred, pitch_pred, energy_pred, mel_lengths = self.model(
            text=text_tensor,
            text_lengths=text_lengths,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control
        )
        
        # 获取有效长度的梅尔频谱
        mel_length = mel_lengths[0].item()
        mel_spectrogram = mel_output[0, :mel_length].cpu().numpy()
        
        print(f"✓ 梅尔频谱shape: {mel_spectrogram.shape}")
        print(f"  帧数: {mel_length}")
        print(f"  时长: {mel_length * 256 / 22050:.2f} 秒")
        
        # 梅尔频谱后处理(关键!)
        # 1. 去归一化(如果训练时做了归一化)
        mel_mean = mel_spectrogram.mean()
        mel_std = mel_spectrogram.std()
        print(f"  Mel统计: mean={mel_mean:.2f}, std={mel_std:.2f}")
        
        # 2. 确保范围合理
        # mel_spectrogram = np.clip(mel_spectrogram, -80, 0)
        
        # 使用声码器生成音频
        print(f"\n正在生成音频波形...")
        audio = self.vocoder.mel_to_audio(mel_spectrogram)
        
        print(f"✓ 音频生成完成")
        print(f"  长度: {len(audio)} 采样点")
        print(f"  时长: {len(audio) / 22050:.2f} 秒")
        
        info = {
            'text': text,
            'phonemes': phonemes,
            'mel_length': mel_length,
            'audio_length': len(audio),
            'duration_control': duration_control,
            'pitch_control': pitch_control,
            'energy_control': energy_control,
            'mel_mean': float(mel_mean),
            'mel_std': float(mel_std)
        }
        
        return audio, mel_spectrogram, info
    
    def save_audio(self, audio, output_path, sample_rate=22050):
        """保存音频文件"""
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        # 轻微限幅防止削波
        audio = np.clip(audio, -0.95, 0.95)
        audio = (audio * 32767).astype(np.int16)
        
        wavfile.write(output_path, sample_rate, audio)
        print(f"\n✓ 音频已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FastSpeech2 Inference (Improved)')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化推理器
    print("="*70)
    print("FastSpeech2 改进版推理系统")
    print("="*70)
    
    inferencer = FastSpeech2Inference(args.checkpoint, device=args.device)
    
    # 测试文本
    texts = [args.text] if args.text else [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for idx, text in enumerate(texts):
        try:
            audio, mel, info = inferencer.synthesize(
                text,
                duration_control=args.duration_control,
                pitch_control=args.pitch_control,
                energy_control=args.energy_control
            )
            
            audio_path = os.path.join(args.output_dir, f'output_{idx:03d}.wav')
            inferencer.save_audio(audio, audio_path)
            
            # 保存信息
            info_path = os.path.join(args.output_dir, f'info_{idx:03d}.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ 合成失败: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*70)
    print(f"✓ 全部完成!输出目录: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()