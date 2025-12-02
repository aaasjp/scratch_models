"""
音频文件转mel频谱图工具
输入：音频文件路径
输出：mel频谱图（保存为图片和numpy数组）
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    elif system == "Windows":
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
    
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

class AudioToMelSpectrogram:
    """音频转mel频谱图工具"""
    
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mels=80,
                 fmin=0,
                 fmax=8000):
        """
        初始化参数
        
        Args:
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            win_length: 窗口长度
            n_mels: mel滤波器数量
            fmin: 最小频率
            fmax: 最大频率
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
    def load_audio(self, audio_path):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            audio: 音频数据
            sr: 采样率
        """
        print(f"正在加载音频文件: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=None)
        
        print(f"原始采样率: {sr} Hz")
        print(f"音频长度: {len(audio)} 采样点")
        print(f"音频时长: {len(audio) / sr:.2f} 秒")
        
        # 重采样到目标采样率
        if sr != self.sample_rate:
            print(f"重采样到 {self.sample_rate} Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            print(f"重采样后长度: {len(audio)} 采样点")
            print(f"重采样后时长: {len(audio) / self.sample_rate:.2f} 秒")
        
        return audio, self.sample_rate
    
    def extract_mel_spectrogram(self, audio):
        """
        提取mel频谱图
        
        Args:
            audio: 音频数据
            
        Returns:
            mel_spectrogram: mel频谱图 [n_mels, T]
            mel_db: 对数域mel频谱图 [n_mels, T]
        """
        print("正在提取mel频谱图...")
        
        # 确保音频是float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 提取mel频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
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
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        print(f"Mel频谱图形状: {mel_spectrogram.shape}")
        print(f"Mel频谱图范围: [{mel_spectrogram.min():.4f}, {mel_spectrogram.max():.4f}]")
        print(f"对数域Mel频谱图范围: [{mel_db.min():.4f}, {mel_db.max():.4f}]")
        
        return mel_spectrogram, mel_db
    
    def visualize_mel_spectrogram(self, mel_db, output_path=None, title="Mel频谱图"):
        """
        可视化mel频谱图
        
        Args:
            mel_db: 对数域mel频谱图 [n_mels, T]
            output_path: 输出图片路径
            title: 图片标题
        """
        print("正在生成mel频谱图可视化...")
        
        plt.figure(figsize=(12, 8))
        
        # 显示mel频谱图
        librosa.display.specshow(
            mel_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title, fontsize=16)
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('Mel频率', fontsize=12)
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Mel频谱图已保存到: {output_path}")
        
        plt.show()
    
    def save_mel_spectrogram(self, mel_spectrogram, output_path):
        """
        保存mel频谱图为numpy数组
        
        Args:
            mel_spectrogram: mel频谱图
            output_path: 输出路径
        """
        print(f"正在保存mel频谱图到: {output_path}")
        np.save(output_path, mel_spectrogram)
        print("保存完成!")
    
    def process_audio(self, audio_path, output_dir=None, save_visualization=True, save_numpy=True):
        """
        处理音频文件，生成mel频谱图
        
        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            save_visualization: 是否保存可视化图片
            save_numpy: 是否保存numpy数组
            
        Returns:
            mel_spectrogram: mel频谱图
            mel_db: 对数域mel频谱图
        """
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名（不含扩展名）
        audio_name = Path(audio_path).stem
        
        print("=" * 60)
        print("音频转Mel频谱图工具")
        print("=" * 60)
        
        # 1. 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 2. 提取mel频谱图
        mel_spectrogram, mel_db = self.extract_mel_spectrogram(audio)
        
        # 3. 可视化
        if save_visualization:
            viz_path = os.path.join(output_dir, f"{audio_name}_mel_spectrogram.png")
            self.visualize_mel_spectrogram(mel_db, viz_path, f"{audio_name} - Mel频谱图")
        
        # 4. 保存numpy数组
        if save_numpy:
            mel_path = os.path.join(output_dir, f"{audio_name}_mel_spectrogram.npy")
            self.save_mel_spectrogram(mel_spectrogram, mel_path)
        
        print("=" * 60)
        print("处理完成!")
        print("=" * 60)
        
        return mel_spectrogram, mel_db

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频文件转mel频谱图工具')
    parser.add_argument('audio_path', nargs='?', default='test_files/audio_000115.wav', help='输入音频文件路径（默认：test_files/audio_000115.wav）')
    parser.add_argument('--output_dir', '-o', default='./mel_output', help='输出目录（默认：./mel_output）')
    parser.add_argument('--sample_rate', type=int, default=22050, help='采样率（默认22050）')
    parser.add_argument('--n_mels', type=int, default=80, help='Mel滤波器数量（默认80）')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT窗口大小（默认1024）')
    parser.add_argument('--hop_length', type=int, default=256, help='跳跃长度（默认256）')
    parser.add_argument('--fmax', type=int, default=8000, help='最大频率（默认8000）')
    parser.add_argument('--no_viz', action='store_true', help='不生成可视化图片')
    parser.add_argument('--no_save', action='store_true', help='不保存numpy数组')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = AudioToMelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.n_fft,
        n_mels=args.n_mels,
        fmin=0,
        fmax=args.fmax
    )
    
    # 处理音频
    try:
        processor.process_audio(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            save_visualization=not args.no_viz,
            save_numpy=not args.no_save
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 如果直接运行，可以在这里测试
    if len(os.sys.argv) == 1:
        # 示例用法
        print("音频转Mel频谱图工具")
        print("使用方法:")
        print("python audio_to_mel_spectrogram.py <音频文件路径> [选项]")
        print("\n示例:")
        print("python audio_to_mel_spectrogram.py test.wav")
        print("python audio_to_mel_spectrogram.py test.wav --output_dir ./mel_output --n_mels 80")
    else:
        exit(main())
