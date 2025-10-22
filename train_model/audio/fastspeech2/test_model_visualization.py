#!/usr/bin/env python3
"""
FastSpeech2 模型可视化测试脚本
生成模型结构图和架构图
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastspeech2 import FastSpeech2, print_model_info, visualize_model_structure

def main():
    print("=" * 80)
    print("FastSpeech2 模型可视化测试")
    print("=" * 80)
    
    # 创建模型
    print("正在创建 FastSpeech2 模型...")
    model = FastSpeech2(
        vocab_size=100,
        d_model=256,
        n_layers=8,
        n_heads=2,
        d_ff=1024,
        n_mel_channels=80,
        max_seq_len=1000,
        stats_path='./processed_data/stats.json'
    )
    
    print("模型创建完成！")
    
    # 打印模型信息
    print_model_info(model)
    
    # 生成模型结构图
    print("\n开始生成模型结构图...")
    success = visualize_model_structure(model, "./model_structure")
    
    if success:
        print("\n✅ 模型结构图生成成功！")
        print("生成的文件:")
        print("  - ./model_structure/fastspeech2_architecture.png (架构图)")
        print("  - ./model_structure/fastspeech2_hierarchy.png (层次结构图)")
    else:
        print("\n❌ 模型结构图生成失败")
        print("请检查 matplotlib 是否正确安装")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
