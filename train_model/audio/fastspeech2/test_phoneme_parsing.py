#!/usr/bin/env python3
"""
测试音素文件解析功能
"""

import os

def test_phoneme_parsing():
    """测试音素文件解析"""
    
    phonemes_file = './output_phonemes.txt'
    if not os.path.exists(phonemes_file):
        print(f"音素文件不存在: {phonemes_file}")
        return False
    
    try:
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
                    print(f"单词: {word} -> 音素: {phoneme_list}")
                else:
                    # 如果格式不正确，尝试直接分割
                    phoneme_list = line.split()
                    phonemes.extend(phoneme_list)
                    print(f"直接解析: {phoneme_list}")
        
        print(f"\n总共解析出 {len(phonemes)} 个音素")
        print(f"前10个音素: {phonemes[:10]}")
        print(f"后10个音素: {phonemes[-10:]}")
        
        return True
        
    except Exception as e:
        print(f"解析失败: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("测试音素文件解析")
    print("="*60)
    
    success = test_phoneme_parsing()
    
    if success:
        print("\n✓ 解析测试通过!")
    else:
        print("\n✗ 解析测试失败!")
    
    print("="*60)
