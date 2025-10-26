#!/usr/bin/env python3
"""
测试数据集功能，包括train和valid数据集
"""

import os
import sys
sys.path.append('.')

from fastspeech2_dataset import LJSpeechDataset

def test_dataset_samples():
    """测试数据集样本"""
    print("=" * 70)
    print("测试数据集样本")
    print("=" * 70)
    
    try:
        # 测试训练集
        print("\n📚 测试训练集 (train)")
        print("-" * 50)
        train_dataset = LJSpeechDataset(
            split='train',
            corpus_dir='./corpus',
            aligned_dir='./corpus_aligned',
            processed_dir='./processed_data',
            max_samples=2,  # 测试15000个样本
            force_preprocess=False
        )
        print("====================训练集样本详情============================")
        print(f"训练集样本数量: {len(train_dataset)}")
        if len(train_dataset) > 0:
            print("\n训练集样本详情:")
            for i in range(min(2, len(train_dataset))):
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
                print("=" * 70)
        
        # 测试验证集
        print("\n\n📚 测试验证集 (validation)")
        print("-" * 50)
        valid_dataset = LJSpeechDataset(
            split='validation',
            corpus_dir='./corpus',
            aligned_dir='./corpus_aligned',
            processed_dir='./processed_data',
            max_samples=2,  # 测试15000个样本
            force_preprocess=False
        )
        print("====================验证集样本详情============================")
        print(f"验证集样本数量: {len(valid_dataset)}")
        if len(valid_dataset) > 0:
            print("\n验证集样本详情:")
            for i in range(min(2, len(valid_dataset))):
                print("=" * 70)
                sample = valid_dataset[i]
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
                print("=" * 70)
        print("\n✓ 数据集测试完成！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_samples()
