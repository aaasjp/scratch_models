"""
FastSpeech2 推理脚本
使用训练好的模型进行语音合成

运行方式:
python inference_fastspeech2_real.py --text "Hello world" --checkpoint checkpoints/best_model.pth
"""

import torch
import numpy as np
import argparse
import os
import json
from phonemizer import phonemize
import librosa
from scipy.io import wavfile

from fastspeech2_improved import FastSpeech2
from fastspeech2_real_data import PhonemeTokenizer


class GriffinLimVocoder:
    """Griffin-Lim 声码器（简单版本）"""
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mels=80,
                 n_iter=32):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        
        # 创建梅尔滤波器
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
    def mel_to_audio(self, mel_spectrogram):
        """
        将梅尔频谱转换为音频
        
        Args:
            mel_spectrogram: [T, n_mels] numpy array
        
        Returns:
            audio: [T * hop_length] numpy array
        """
        # 转置为 [n_mels, T]
        mel = mel_spectrogram.T
        
        # 反归一化（从 dB 转回功率）
        mel_power = librosa.db_to_power(mel)
        
        # 逆梅尔滤波器
        linear_spec = np.dot(np.linalg.pinv(self.mel_basis), mel_power)
        
        # Griffin-Lim 算法
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft
        )
        
        return audio


class FastSpeech2Inference:
    """FastSpeech2 推理器"""
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        self.config = checkpoint.get('config', {})
        vocab_size = self.config.get('vocab_size', 100)
        
        # 加载 tokenizer
        tokenizer_path = './processed_data/tokenizer.json'
        if os.path.exists(tokenizer_path):
            self.tokenizer = PhonemeTokenizer.load(tokenizer_path)
            vocab_size = len(self.tokenizer)
            print(f"✓ 加载 tokenizer，词表大小: {vocab_size}")
        else:
            raise FileNotFoundError(f"找不到 tokenizer: {tokenizer_path}")
        
        # 加载统计信息
        stats_path = './processed_data/stats.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"✓ 加载统计信息")
        else:
            print("警告: 找不到统计信息，使用默认值")
            self.stats = {
                'pitch_min': 0.0, 'pitch_max': 800.0,
                'energy_min': 0.0, 'energy_max': 100.0
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
        self.vocoder = GriffinLimVocoder()
        print("✓ 声码器初始化完成")
        
        # 更新模型的统计信息
        self.model.variance_adaptor.pitch_min = torch.tensor(self.stats['pitch_min'])
        self.model.variance_adaptor.pitch_max = torch.tensor(self.stats['pitch_max'])
        self.model.variance_adaptor.energy_min = torch.tensor(self.stats['energy_min'])
        self.model.variance_adaptor.energy_max = torch.tensor(self.stats['energy_max'])
    
    def text_to_phonemes(self, text):
        """将文本转换为音素"""
        try:
            # 清理文本
            text = text.lower().strip()
            
            # G2P 转换
            phonemes = phonemize(
                text,
                language='en-us',
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
                with_stress=False
            )
            
            # 分割音素
            phoneme_list = phonemes.split()
            
            return phoneme_list
        except Exception as e:
            print(f"音素转换失败: {e}")
            return []
    
    @torch.no_grad()
    def synthesize(self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
        """
        从文本合成语音
        
        Args:
            text: 输入文本
            duration_control: 语速控制（1.0=正常，>1更慢，<1更快）
            pitch_control: 音高控制（1.0=正常，>1更高，<1更低）
            energy_control: 能量控制（1.0=正常，>1更响，<1更轻）
        
        Returns:
            audio: 音频波形
            mel_spectrogram: 梅尔频谱
            info: 合成信息
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
        
        # 转换为 ID
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
            'energy_control': energy_control
        }
        
        return audio, mel_spectrogram, info
    
    def save_audio(self, audio, output_path, sample_rate=22050):
        """保存音频文件"""
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        audio = (audio * 32767).astype(np.int16)
        
        wavfile.write(output_path, sample_rate, audio)
        print(f"\n✓ 音频已保存: {output_path}")
    
    def save_mel_spectrogram(self, mel, output_path):
        """保存梅尔频谱图"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.imshow(mel.T, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 频谱图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FastSpeech2 Inference')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--text', type=str, default=None,
                        help='要合成的文本')
    parser.add_argument('--text_file', type=str, default=None,
                        help='包含多个文本的文件（每行一个）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--save_mel', action='store_true',
                        help='保存梅尔频谱图')
    
    # 控制参数
    parser.add_argument('--duration_control', type=float, default=1.0,
                        help='语速控制（1.0=正常）')
    parser.add_argument('--pitch_control', type=float, default=1.0,
                        help='音高控制（1.0=正常）')
    parser.add_argument('--energy_control', type=float, default=1.0,
                        help='能量控制（1.0=正常）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='运行设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化推理器
    print("="*70)
    print("FastSpeech2 推理系统")
    print("="*70)
    
    inferencer = FastSpeech2Inference(args.checkpoint, device=args.device)
    
    # 准备文本列表
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # 默认示例文本
        texts = [
            "Hello world, this is a test.",
            "FastSpeech2 is a text to speech model.",
            "It can generate natural sounding speech."
        ]
    
    # 合成所有文本
    print(f"\n将合成 {len(texts)} 个文本")
    print("="*70)
    
    for idx, text in enumerate(texts):
        try:
            # 合成
            audio, mel, info = inferencer.synthesize(
                text,
                duration_control=args.duration_control,
                pitch_control=args.pitch_control,
                energy_control=args.energy_control
            )
            
            # 保存音频
            audio_path = os.path.join(args.output_dir, f'output_{idx:03d}.wav')
            inferencer.save_audio(audio, audio_path)
            
            # 保存梅尔频谱（可选）
            if args.save_mel:
                mel_path = os.path.join(args.output_dir, f'mel_{idx:03d}.png')
                inferencer.save_mel_spectrogram(mel, mel_path)
            
            # 保存信息
            info_path = os.path.join(args.output_dir, f'info_{idx:03d}.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ 合成失败: {e}\n")
            continue
    
    print("="*70)
    print(f"✓ 全部完成！输出目录: {args.output_dir}")
    print("="*70)


def interactive_mode():
    """交互式模式"""
    import sys
    
    print("="*70)
    print("FastSpeech2 交互式合成系统")
    print("="*70)
    
    # 获取检查点路径
    checkpoint_path = input("\n请输入模型检查点路径 [checkpoints/best_model.pth]: ").strip()
    if not checkpoint_path:
        checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ 检查点不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 初始化推理器
    inferencer = FastSpeech2Inference(checkpoint_path, device='cuda')
    
    # 控制参数
    duration_control = 1.0
    pitch_control = 1.0
    energy_control = 1.0
    
    output_counter = 0
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("命令:")
    print("  输入文本 - 合成语音")
    print("  speed <value> - 设置语速 (0.5-2.0)")
    print("  pitch <value> - 设置音高 (0.5-2.0)")
    print("  energy <value> - 设置能量 (0.5-2.0)")
    print("  reset - 重置所有控制参数")
    print("  quit - 退出")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("退出系统")
                break
            
            # 处理控制命令
            if user_input.startswith('speed '):
                try:
                    value = float(user_input.split()[1])
                    if 0.5 <= value <= 2.0:
                        duration_control = value
                        print(f"✓ 语速设置为: {value}x")
                    else:
                        print("✗ 语速值应在 0.5-2.0 之间")
                except:
                    print("✗ 无效的语速值")
                continue
            
            if user_input.startswith('pitch '):
                try:
                    value = float(user_input.split()[1])
                    if 0.5 <= value <= 2.0:
                        pitch_control = value
                        print(f"✓ 音高设置为: {value}x")
                    else:
                        print("✗ 音高值应在 0.5-2.0 之间")
                except:
                    print("✗ 无效的音高值")
                continue
            
            if user_input.startswith('energy '):
                try:
                    value = float(user_input.split()[1])
                    if 0.5 <= value <= 2.0:
                        energy_control = value
                        print(f"✓ 能量设置为: {value}x")
                    else:
                        print("✗ 能量值应在 0.5-2.0 之间")
                except:
                    print("✗ 无效的能量值")
                continue
            
            if user_input.lower() == 'reset':
                duration_control = 1.0
                pitch_control = 1.0
                energy_control = 1.0
                print("✓ 控制参数已重置")
                continue
            
            # 合成语音
            try:
                audio, mel, info = inferencer.synthesize(
                    user_input,
                    duration_control=duration_control,
                    pitch_control=pitch_control,
                    energy_control=energy_control
                )
                
                # 保存
                output_path = os.path.join(output_dir, f'interactive_{output_counter:03d}.wav')
                inferencer.save_audio(audio, output_path)
                output_counter += 1
                
            except Exception as e:
                print(f"\n✗ 合成失败: {e}")
        
        except KeyboardInterrupt:
            print("\n\n退出系统")
            break
        except Exception as e:
            print(f"\n✗ 错误: {e}")


if __name__ == "__main__":
    import sys
    
    # 如果没有命令行参数，进入交互模式
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()