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
from typing import List
from fastspeech2 import FastSpeech2
from fastspeech2_dataset import PhonemeTokenizer
from fastspeech2_dataset import LJSpeechDataset

class HiFiGANVocoder:
    """HiFi-GAN 声码器(使用预训练模型)"""
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.h = None
        self._load_model()
    
    def _load_model(self):
        """加载HiFi-GAN模型"""
        try:
            # 导入必要的模块
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'hifi_gan'))
            
            from hifi_gan.models import Generator
            from hifi_gan.env import AttrDict
            from hifi_gan.meldataset import MAX_WAV_VALUE
            import json
            
            # 加载配置
            config_file = './hifi_gan/pretrained_models/LJ_V1/config.json'
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"找不到HiFi-GAN配置文件: {config_file}")
                
            with open(config_file) as f:
                data = f.read()
            json_config = json.loads(data)
            self.h = AttrDict(json_config)
            
            # 创建生成器
            self.generator = Generator(self.h).to(self.device)
            
            # 加载预训练权重
            checkpoint_path = './hifi_gan/pretrained_models/LJ_V1/generator_v1'
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"找不到HiFi-GAN预训练模型: {checkpoint_path}")
                
            state_dict_g = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(state_dict_g['generator'])
            
            self.generator.eval()
            self.generator.remove_weight_norm()
            
            print(f"✓ HiFi-GAN声码器加载成功")
            print(f"  采样率: {self.h.sampling_rate}")
            print(f"  设备: {self.device}")
            
        except Exception as e:
            print(f"✗ HiFi-GAN声码器加载失败: {e}")
            self.generator = None
    
    def mel_to_audio(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频
        
        Args:
            mel_spectrogram: [n_mels, T] 梅尔频谱
        
        Returns:
            audio: 音频波形
        """
        if self.generator is None:
            print("✗ HiFi-GAN声码器未加载，返回空音频")
            return np.array([])
        
        try:
            # 转换梅尔频谱格式 (参考test_convert_mel_to_wav.py)
            mel_magnitude = np.sqrt(mel_spectrogram + 1e-9)  # 对应torch中的sqrt操作
            
            # 使用torch风格的对数压缩
            C = 1
            clip_val = 1e-5
            mel_compressed = np.log(np.clip(mel_magnitude, a_min=clip_val, a_max=None) * C)
            
            # 转换为tensor并添加batch维度
            x = torch.FloatTensor(mel_compressed).to(self.device)
            x = x.unsqueeze(0)  # [1, n_mels, T]
            
            with torch.no_grad():
                # 生成音频
                y_g_hat = self.generator(x)
                audio = y_g_hat.squeeze()  # 移除batch维度
                
                # 转换为numpy数组并归一化到[-1, 1]范围
                audio = audio.cpu().numpy()
                
            return audio
            
        except Exception as e:
            print(f"✗ 音频生成失败: {e}")
            return np.array([]) 
    



class FastSpeech2Inference:
    """FastSpeech2 推理器(改进版)"""
    def __init__(self, checkpoint_path, device='cuda', use_hifigan=True, phonemes_file=None, textgrid_file=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.phonemes_file = phonemes_file
        self.textgrid_file = textgrid_file
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        self.config = checkpoint.get('config', {})
        
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
            print(f"  Duration range: [{self.stats['duration_min']:.4f}, {self.stats['duration_max']:.4f}]")
            print(f"  Mel range: [{self.stats['mel_min']:.4f}, {self.stats['mel_max']:.4f}]")
            print(f"  Pitch mean: {self.stats['pitch_mean']:.1f}")
            print(f"  Energy mean: {self.stats['energy_mean']:.4f}")
            print(f"  Duration mean: {self.stats['duration_mean']:.4f}")
            print(f"  Mel mean: {self.stats['mel_mean']:.4f}")
            print(f"  Pitch std: {self.stats['pitch_std']:.1f}")
            print(f"  Energy std: {self.stats['energy_std']:.4f}")
            print(f"  Duration std: {self.stats['duration_std']:.4f}")
            print(f"  Mel std: {self.stats['mel_std']:.4f}")
        else:
            raise FileNotFoundError(f"找不到统计信息: {stats_path}")
        
        # 创建模型
        self.model = FastSpeech2(
            vocab_size=vocab_size,
            d_model=self.config.get('d_model', 256),
            n_layers=self.config.get('n_layers', 8),
            n_heads=self.config.get('n_heads', 2),
            d_ff=self.config.get('d_ff', 1024),
            dropout=self.config.get('dropout', 0.1),
            n_mel_channels=80,
            max_seq_len=1000,
            stats_path=stats_path
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
        '''
        # 更新模型的统计信息
        self.model.variance_adaptor.pitch_min = torch.tensor(self.stats['pitch_min'])
        self.model.variance_adaptor.pitch_max = torch.tensor(self.stats['pitch_max'])
        self.model.variance_adaptor.energy_min = torch.tensor(self.stats['energy_min'])
        self.model.variance_adaptor.energy_max = torch.tensor(self.stats['energy_max'])
        '''
    
    def denormalize_mel(self, mel_spectrogram, eps=1e-6):
        """
        对梅尔频谱进行逆归一化处理
        
        Args:
            mel_spectrogram: 归一化后的梅尔频谱 [T, n_mels]
            
        Returns:
            denormalized_mel: 逆归一化后的梅尔频谱 [n_mels, T]
        """
        # 逆标准化: x = (x_norm * std) + mean
        mel_denorm = mel_spectrogram * (self.stats['mel_std'] + eps) + self.stats['mel_mean']
        
        # 逆裁剪: 确保在原始范围内
        mel_denorm = np.clip(mel_denorm, self.stats['mel_min'], self.stats['mel_max'])
        
        print("--------------------------------")
        print(f"mel_denorm.shape: {mel_denorm.shape}")
        print(f"mel_denorm: {mel_denorm}")
        print("--------------------------------")
        # 先转换shape: [T, n_mels] -> [n_mels, T]
        mel_denorm = mel_denorm.T
        
        # 逆对数处理: 从对数域转换回线性域
        # 原始处理: mel_db = librosa.power_to_db(mel, ref=1.0)
        # 逆处理: mel_linear = librosa.db_to_power(mel_db, ref=1.0)
        mel_linear = librosa.db_to_power(mel_denorm, ref=1.0)
        print("--------------------------------")
        print(f"mel_linear.shape: {mel_linear.shape}")
        print(f"mel_linear: {mel_linear}")
        print("--------------------------------")
        return mel_linear
    
    
    def load_phonemes_from_file(self, phonemes_file):
        """
        从音素文件中读取音素数据
        
        Args:
            phonemes_file: 音素文件路径
            
        Returns:
            phonemes: 音素列表
        """
        try:
            if not os.path.exists(phonemes_file):
                raise FileNotFoundError(f"音素文件不存在: {phonemes_file}")
            
            phonemes = []
            with open(phonemes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    # 解析格式: word\tphoneme1 phoneme2 phoneme3
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        phoneme_sequence = parts[1].strip()
                        
                        # 将音素序列分割成单个音素
                        phoneme_list = phoneme_sequence.split()
                        phonemes.extend(phoneme_list)
                    else:
                        # 如果格式不正确，尝试直接分割
                        phoneme_list = line.split()
                        phonemes.extend(phoneme_list)
            
            print(f"✓ 从文件加载音素: {len(phonemes)} 个音素")
            print(f"  前10个音素: {phonemes[:10]}")
            
            return phonemes
            
        except Exception as e:
            print(f"加载音素文件失败: {e}")
            return []
    
    def extract_phonemes_from_textgrid(self, textgrid_path: str) -> List[str]:
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
    
    @torch.no_grad()
    def synthesize(self, text=None, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
        """
        从文本或音素文件合成语音(改进版)
        
        Args:
            text: 输入文本(如果提供了phonemes_file则忽略)
            duration_control: 语速控制
            pitch_control: 音高控制  
            energy_control: 能量控制
        """
        print(f"\n{'='*60}")
        
        # 获取音素
        if self.phonemes_file and os.path.exists(self.phonemes_file):
            #print(f"从音素文件合成: {self.phonemes_file}")
            #phonemes = self.load_phonemes_from_file(self.phonemes_file)
            print(f"从TextGrid文件合成: {self.textgrid_file}")
            phonemes = self.extract_phonemes_from_textgrid(self.textgrid_file)
            if len(phonemes) == 0:
                raise ValueError("无法从音素文件加载音素")
        else:
            raise ValueError("必须提供phonemes_file参数")
            
        print(f"{'='*60}")
        
        print(f"音素: {' '.join(phonemes)}")
        print(f"音素数量: {len(phonemes)}")
        
        # 转换为ID
        phoneme_ids = self.tokenizer.encode(phonemes)
        print(f"phoneme_ids: {phoneme_ids}")
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
        # 逆归一化处理
        print(f"正在对梅尔频谱进行逆归一化...")
        mel_spectrogram = self.denormalize_mel(mel_spectrogram)
        
        print(f"✓ 逆归一化完成")
        print(f"  梅尔频谱shape: {mel_spectrogram.shape} (已转换为[n_mels, T])")
        print(f"  梅尔频谱范围: [{mel_spectrogram.min():.4f}, {mel_spectrogram.max():.4f}]")
        print(f"  梅尔频谱均值: {mel_spectrogram.mean():.4f}")
        print(f"  梅尔频谱标准差: {mel_spectrogram.std():.4f}")
        
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
            'mel_mean': float(mel_spectrogram.mean()),
            'mel_std': float(mel_spectrogram.std())
        }
        
        return audio, mel_spectrogram, info
    
    def save_audio(self, audio, output_path):
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            output_path: 输出路径
        """
        # 按照HiFi-GAN的方式保存音频
        # 音频乘以MAX_WAV_VALUE并转换为int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 保存为WAV文件
        wavfile.write(output_path, 22050, audio_int16)
        print(f"✓ 音频已保存到: {output_path}")
    


def main():
    parser = argparse.ArgumentParser(description='FastSpeech2 Inference (Improved)')
    
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_epoch_70.pth')
    parser.add_argument('--phonemes_file', type=str, default='./output_phonemes.txt')
    parser.add_argument('--textgrid_file', type=str, default='./corpus_aligned/audio_000017.TextGrid')
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
    
    inferencer = FastSpeech2Inference(args.checkpoint, device=args.device, phonemes_file=args.phonemes_file, textgrid_file=args.textgrid_file)
    
    # 处理输入
    if args.phonemes_file and os.path.exists(args.phonemes_file):
        # 使用音素文件
        #print(f"使用音素文件: {args.phonemes_file}")
        print(f"使用TextGrid文件: {args.textgrid_file}")
        try:
            audio, mel, info = inferencer.synthesize(
                duration_control=args.duration_control,
                pitch_control=args.pitch_control,
                energy_control=args.energy_control
            )
            
            audio_path = os.path.join(args.output_dir, 'output_from_phonemes.wav')
            inferencer.save_audio(audio, audio_path)
            
            # 保存信息
            info_path = os.path.join(args.output_dir, 'info_phonemes.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ 合成失败: {e}\n")
            import traceback
            traceback.print_exc()
    else:
        raise ValueError("必须提供phonemes_file参数")
    
    print("="*70)
    print(f"✓ 全部完成!输出目录: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()