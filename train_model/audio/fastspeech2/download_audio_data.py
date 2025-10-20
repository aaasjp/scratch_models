"""
MFA数据准备脚本
从LJ-Speech数据集下载音频文件，并转换为MFA所需格式
严格遵循MFA数据格式要求：
- 音频文件：WAV格式，单声道
- 文本文件：.lab格式，与音频文件同名
- 数据结构：所有文件放在同一文件夹下
"""

import datasets
import os
import librosa
import soundfile as sf
from pathlib import Path
import shutil
from tqdm import tqdm


def prepare_mfa_data(output_dir="./corpus", max_samples=None):
    """
    准备MFA格式的数据
    
    Args:
        output_dir: 输出目录，默认为./corpus
        max_samples: 最大样本数，None表示使用全部数据
    """
    print("正在加载LJ-Speech数据集...")
    # 使用本地缓存，避免重复下载
    dataset = datasets.load_dataset("MikhailT/lj-speech", cache_dir="./cache", download_mode="reuse_dataset_if_exists")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"开始处理数据，保存到: {output_path.absolute()}")
    
    # 获取数据集
    train_data = dataset['full']
    total_samples = len(train_data)
    if max_samples:
        total_samples = min(total_samples, max_samples)
    
    print(f"总共处理 {total_samples} 个样本")
    
    processed_count = 0
    failed_count = 0
    
    for i, sample in enumerate(tqdm(train_data, desc="处理音频文件")):
        if max_samples and i >= max_samples:
            break
            
        try:
            # 获取音频数据和文本
            audio_data = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            text = sample['normalized_text']
            print(f"audio_data.shape: {audio_data.shape}")
            print(f"audio_data.dtype: {audio_data.dtype}")
            print(f"audio_data: {audio_data}")
            print(f"sample_rate: {sample_rate}")
            print(f"text: {text}")

            # 生成文件名（使用索引确保唯一性）
            audio_filename = f"audio_{i:06d}.wav"
            text_filename = f"audio_{i:06d}.lab"
            
            # 音频文件路径
            audio_path = output_path / audio_filename
            
            # 确保音频为单声道
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # 重采样到22050Hz（MFA推荐采样率）
            if sample_rate != 22050:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
            
            # 保存音频文件为WAV格式
            sf.write(audio_path, audio_data, 22050, format='WAV', subtype='PCM_16')
            
            # 创建对应的.lab文本文件
            text_path = output_path / text_filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text.strip())
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理第{i}个样本时出错: {e}")
            failed_count += 1
            continue
    
    print(f"\n数据处理完成!")
    print(f"成功处理: {processed_count} 个样本")
    print(f"失败: {failed_count} 个样本")
    print(f"数据保存在: {output_path.absolute()}")
    
    return output_path


if __name__ == "__main__":
    # 处理数据（可以设置max_samples来限制样本数量，用于测试）
    prepare_mfa_data(output_dir="./corpus", max_samples=100000)  # 先处理10个样本进行测试
