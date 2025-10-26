#!/usr/bin/env python3
"""
测试音频质量改进效果
比较修复前后的音频质量
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import argparse

def analyze_audio_quality(audio_path, title="音频分析"):
    """
    分析音频质量
    
    Args:
        audio_path: 音频文件路径
        title: 分析标题
    """
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=22050)
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"文件: {audio_path}")
    print(f"采样率: {sr} Hz")
    print(f"长度: {len(audio)} 采样点")
    print(f"时长: {len(audio) / sr:.2f} 秒")
    print(f"音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"音频均值: {audio.mean():.6f}")
    print(f"音频标准差: {audio.std():.6f}")
    
    # 计算信噪比 (简单估计)
    signal_power = np.mean(audio ** 2)
    noise_power = np.var(audio - np.mean(audio))
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"信噪比估计: {snr:.2f} dB")
    
    # 分析频谱
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft)
    
    # 找到主要频率成分
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    
    # 找到峰值频率
    peak_indices = np.argsort(positive_magnitude)[-5:]
    peak_freqs = positive_freqs[peak_indices]
    peak_mags = positive_magnitude[peak_indices]
    
    print(f"\n主要频率成分:")
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
        if freq > 0:
            print(f"  {i+1}. {freq:.1f} Hz (幅度: {mag:.2f})")
    
    # 检查高频噪声
    high_freq_mask = positive_freqs > 8000
    high_freq_energy = np.sum(positive_magnitude[high_freq_mask])
    total_energy = np.sum(positive_magnitude)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    print(f"\n高频噪声分析:")
    print(f"高频能量比例: {high_freq_ratio:.4f}")
    if high_freq_ratio > 0.1:
        print("⚠️  检测到较多高频噪声，可能是电流声")
    else:
        print("✓ 高频噪声较少")
    
    return {
        'duration': len(audio) / sr,
        'range': [audio.min(), audio.max()],
        'std': audio.std(),
        'snr': snr if noise_power > 0 else float('inf'),
        'high_freq_ratio': high_freq_ratio
    }

def compare_audio_files(file1, file2, title1="文件1", title2="文件2"):
    """
    比较两个音频文件的质量
    """
    print(f"\n{'='*80}")
    print("音频质量对比分析")
    print(f"{'='*80}")
    
    results1 = analyze_audio_quality(file1, title1)
    results2 = analyze_audio_quality(file2, title2)
    
    if results1 and results2:
        print(f"\n{'='*60}")
        print("对比结果:")
        print(f"{'='*60}")
        
        # 比较信噪比
        if results1['snr'] != float('inf') and results2['snr'] != float('inf'):
            snr_diff = results2['snr'] - results1['snr']
            print(f"信噪比变化: {snr_diff:+.2f} dB")
            if snr_diff > 0:
                print("✓ 信噪比有所改善")
            else:
                print("⚠️  信噪比有所下降")
        
        # 比较高频噪声
        hf_diff = results2['high_freq_ratio'] - results1['high_freq_ratio']
        print(f"高频噪声变化: {hf_diff:+.4f}")
        if hf_diff < 0:
            print("✓ 高频噪声减少")
        else:
            print("⚠️  高频噪声增加")
        
        # 比较音频范围
        range1 = results1['range'][1] - results1['range'][0]
        range2 = results2['range'][1] - results2['range'][0]
        print(f"音频动态范围: {range1:.4f} -> {range2:.4f}")

def main():
    parser = argparse.ArgumentParser(description='音频质量分析工具')
    parser.add_argument('--audio1', help='第一个音频文件路径')
    parser.add_argument('--audio2', help='第二个音频文件路径（可选，用于对比）')
    parser.add_argument('--title1', default='音频1', help='第一个音频的标题')
    parser.add_argument('--title2', default='音频2', help='第二个音频的标题')
    
    args = parser.parse_args()
    
    if args.audio1:
        if args.audio2:
            compare_audio_files(args.audio1, args.audio2, args.title1, args.title2)
        else:
            analyze_audio_quality(args.audio1, args.title1)
    else:
        print("请提供至少一个音频文件路径")
        print("使用方法:")
        print("python test_audio_quality.py --audio1 output.wav")
        print("python test_audio_quality.py --audio1 old.wav --audio2 new.wav")

if __name__ == "__main__":
    main()
