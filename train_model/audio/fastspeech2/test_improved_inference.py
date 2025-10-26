#!/usr/bin/env python3
"""
测试改进后的音频合成质量
"""

import os
import sys
import argparse
import json
from fastspeech2_inference import FastSpeech2Inference

def test_audio_quality():
    """测试音频质量改进效果"""
    
    print("="*80)
    print("FastSpeech2 音频质量改进测试")
    print("="*80)
    
    # 检查必要文件
    checkpoint_path = './checkpoints/checkpoint_epoch_70.pth'
    textgrid_file = './corpus_aligned/audio_000017.TextGrid'
    output_dir = './outputs'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return False
    
    if not os.path.exists(textgrid_file):
        print(f"❌ TextGrid文件不存在: {textgrid_file}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化推理器
        print("正在初始化FastSpeech2推理器...")
        inferencer = FastSpeech2Inference(
            checkpoint_path=checkpoint_path,
            device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
            textgrid_file=textgrid_file
        )
        
        print("✓ 推理器初始化成功")
        
        # 测试不同的控制参数
        test_configs = [
            {
                'name': '标准参数',
                'duration_control': 1.0,
                'pitch_control': 1.0,
                'energy_control': 1.0
            },
            {
                'name': '慢速低音',
                'duration_control': 0.8,
                'pitch_control': 0.9,
                'energy_control': 1.0
            },
            {
                'name': '快速高音',
                'duration_control': 1.2,
                'pitch_control': 1.1,
                'energy_control': 1.0
            }
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\n{'='*60}")
            print(f"测试配置: {config['name']}")
            print(f"{'='*60}")
            
            try:
                # 合成音频
                audio, mel, info = inferencer.synthesize(
                    duration_control=config['duration_control'],
                    pitch_control=config['pitch_control'],
                    energy_control=config['energy_control']
                )
                
                # 保存音频
                output_path = os.path.join(output_dir, f"test_{config['name'].replace(' ', '_')}.wav")
                inferencer.save_audio(audio, output_path)
                
                # 保存信息
                info_path = os.path.join(output_dir, f"test_{config['name'].replace(' ', '_')}_info.json")
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                
                # 分析音频质量
                audio_analysis = analyze_audio_quality(audio, config['name'])
                results.append({
                    'config': config,
                    'output_path': output_path,
                    'analysis': audio_analysis
                })
                
                print(f"✓ {config['name']} 合成完成")
                print(f"  输出文件: {output_path}")
                print(f"  音频长度: {len(audio)} 采样点")
                print(f"  音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
                
            except Exception as e:
                print(f"❌ {config['name']} 合成失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 总结结果
        print(f"\n{'='*80}")
        print("测试结果总结")
        print(f"{'='*80}")
        
        for result in results:
            config = result['config']
            analysis = result['analysis']
            print(f"\n{config['name']}:")
            print(f"  音频范围: [{analysis['range'][0]:.4f}, {analysis['range'][1]:.4f}]")
            print(f"  标准差: {analysis['std']:.6f}")
            print(f"  高频噪声比例: {analysis['high_freq_ratio']:.4f}")
            
            if analysis['high_freq_ratio'] < 0.05:
                print("  ✓ 高频噪声较少，质量良好")
            elif analysis['high_freq_ratio'] < 0.1:
                print("  ⚠️  高频噪声适中")
            else:
                print("  ❌ 高频噪声较多，可能存在电流声")
        
        print(f"\n✓ 所有测试完成！输出目录: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_audio_quality(audio, name):
    """分析音频质量"""
    import numpy as np
    
    # 基本统计
    duration = len(audio)
    audio_range = [audio.min(), audio.max()]
    std = audio.std()
    
    # 计算高频噪声比例
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/22050)
    magnitude = np.abs(fft)
    
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    
    high_freq_mask = positive_freqs > 8000
    high_freq_energy = np.sum(positive_magnitude[high_freq_mask])
    total_energy = np.sum(positive_magnitude)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    return {
        'duration': duration,
        'range': audio_range,
        'std': std,
        'high_freq_ratio': high_freq_ratio
    }

def main():
    parser = argparse.ArgumentParser(description='测试改进后的音频合成质量')
    parser.add_argument('--checkpoint', default='./checkpoints/checkpoint_epoch_70.pth', help='模型检查点路径')
    parser.add_argument('--textgrid', default='./corpus_aligned/audio_000017.TextGrid', help='TextGrid文件路径')
    parser.add_argument('--output_dir', default='./outputs', help='输出目录')
    
    args = parser.parse_args()
    
    # 更新全局变量
    global checkpoint_path, textgrid_file, output_dir
    checkpoint_path = args.checkpoint
    textgrid_file = args.textgrid
    output_dir = args.output_dir
    
    success = test_audio_quality()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
