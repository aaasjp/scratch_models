import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
LengthRegulator 对齐机制详解
解答两个关键问题：
1. 如何保证复制后的长度与梅尔频谱一致？
2. Mask 的作用和实现
"""

# ============================================
# 问题1: LengthRegulator 的对齐机制
# ============================================

print("=" * 70)
print("问题1: LengthRegulator 如何保证与梅尔频谱对齐？")
print("=" * 70)

print("\n【答案】: 通过真实的 duration 标注来保证！\n")

# 数据预处理阶段
print("【数据准备阶段】（训练前完成）:")
print("-" * 70)

print("""
1. 原始数据:
   - 文本: "hello"
   - 音频波形: [wav data]

2. 强制对齐 (Montreal Forced Aligner 等工具):
   - 文本 → 音素: ['h', 'ə', 'l', 'oʊ']
   - 对每个音素标注在音频中的起止时间
   
3. 计算 Duration:
   音素     起始时间    结束时间    持续时长    对应帧数(hop=256)
   'h'      0.00s      0.05s      0.05s       11 帧
   'ə'      0.05s      0.12s      0.07s       15 帧  
   'l'      0.12s      0.18s      0.06s       13 帧
   'oʊ'     0.18s      0.30s      0.12s       26 帧
   
   Duration = [11, 15, 13, 26]  总计: 65 帧

4. 提取梅尔频谱:
   - 从音频提取 mel spectrogram
   - 形状: [65, 80]  (65帧 × 80个mel bins)
   
5. 保存训练数据:
   {
     'phoneme_ids': [ph_id_h, ph_id_ə, ph_id_l, ph_id_oʊ],
     'duration': [11, 15, 13, 26],
     'mel': [65, 80],  # 这就是对齐的关键！
     'pitch': [11, 15, 13, 26],  # 每个音素的平均pitch
     'energy': [11, 15, 13, 26]  # 每个音素的平均energy
   }
""")

print("\n【关键点】:")
print("✓ duration 是从音频中通过强制对齐提取的")
print("✓ sum(duration) 天然等于梅尔频谱的帧数")
print("✓ 训练时用真实 duration，所以长度必然对齐")
print("✓ 推理时用预测 duration，可能不完全对齐但接近")

# 代码演示
print("\n" + "=" * 70)
print("代码演示: 对齐机制")
print("=" * 70)

class LengthRegulatorWithAlignment(nn.Module):
    """带详细说明的 Length Regulator"""
    
    def forward(self, x, duration, max_len=None):
        """
        Args:
            x: [batch, phoneme_len, d_model] - 音素级别的表征
            duration: [batch, phoneme_len] - 每个音素的持续帧数
            max_len: 最大长度（用于 padding）
        
        Returns:
            output: [batch, mel_len, d_model] - 帧级别的表征
        """
        output = []
        mel_lens = []
        
        for batch_idx in range(x.size(0)):
            expanded = []
            for phoneme_idx in range(x.size(1)):
                # 获取当前音素的表征和持续时长
                phoneme_repr = x[batch_idx, phoneme_idx]  # [d_model]
                dur = int(duration[batch_idx, phoneme_idx].item())
                
                # 复制 dur 次 - 这就是对齐的核心！
                # 如果 dur=15，就把这个音素的表征复制15帧
                if dur > 0:
                    expanded.append(phoneme_repr.unsqueeze(0).expand(dur, -1))
            
            if expanded:
                # 拼接所有音素的扩展表征
                expanded_seq = torch.cat(expanded, dim=0)  # [sum(duration), d_model]
                output.append(expanded_seq)
                mel_lens.append(expanded_seq.size(0))
            else:
                output.append(torch.zeros(1, x.size(2)).to(x.device))
                mel_lens.append(1)
        
        # Padding 到相同长度
        if max_len is None:
            max_len = max(mel_lens)
        
        output_padded = []
        for seq in output:
            if seq.size(0) < max_len:
                padding = torch.zeros(max_len - seq.size(0), seq.size(1)).to(x.device)
                seq = torch.cat([seq, padding], dim=0)
            output_padded.append(seq)
        
        output_tensor = torch.stack(output_padded, dim=0)
        
        return output_tensor, mel_lens

# 测试对齐
print("\n【测试】:")
phoneme_repr = torch.randn(1, 4, 256)  # 4个音素
duration = torch.tensor([[11, 15, 13, 26]])  # 对应的duration

regulator = LengthRegulatorWithAlignment()
mel_repr, mel_lens = regulator(phoneme_repr, duration)

print(f"输入: {phoneme_repr.shape} (音素级)")
print(f"Duration: {duration[0].tolist()}")
print(f"Duration 总和: {duration.sum().item()}")
print(f"输出: {mel_repr.shape} (帧级)")
print(f"实际长度: {mel_lens[0]}")
print(f"\n✓ 验证: Duration总和({duration.sum().item()}) == 输出长度({mel_lens[0]})")

# 可视化对齐过程
print("\n可视化对齐过程...")
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# 音素级表征
phoneme_vis = phoneme_repr[0, :, :50].detach().numpy()
axes[0].imshow(phoneme_vis.T, aspect='auto', cmap='viridis')
axes[0].set_title('Phoneme-level Representation (4 phonemes)')
axes[0].set_xlabel('Phoneme Index')
axes[0].set_ylabel('Feature Dimension')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(['h(11)', 'ə(15)', 'l(13)', 'oʊ(26)'])

# 帧级表征（扩展后）
mel_vis = mel_repr[0, :mel_lens[0], :50].detach().numpy()
axes[1].imshow(mel_vis.T, aspect='auto', cmap='viridis')
axes[1].set_title('Frame-level Representation (65 frames)')
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Feature Dimension')

# 标注边界
boundaries = [0]
for d in duration[0].tolist():
    boundaries.append(boundaries[-1] + d)
for b in boundaries[1:-1]:
    axes[1].axvline(x=b, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('length_regulator_alignment.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存: length_regulator_alignment.png")
plt.close()

# ============================================
# 问题2: Mask 的必要性和实现
# ============================================

print("\n" + "=" * 70)
print("问题2: Mask 是否需要？如何实现？")
print("=" * 70)

print("\n【答案】: 需要！有两种主要的 Mask:\n")

print("1. Padding Mask - 处理变长序列")
print("2. (可选) Attention Mask - 控制注意力模式")

print("\n" + "-" * 70)
print("Mask 类型详解")
print("-" * 70)

class MaskGenerator:
    """Mask 生成器"""
    
    @staticmethod
    def get_padding_mask(lengths, max_len=None):
        """
        生成 Padding Mask
        
        Args:
            lengths: [batch] - 每个序列的实际长度
            max_len: 最大长度
        
        Returns:
            mask: [batch, max_len] - 1表示有效位置，0表示padding
        """
        batch_size = lengths.size(0)
        if max_len is None:
            max_len = lengths.max().item()
        
        # 创建位置索引
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
        # 比较得到mask
        mask = (ids < lengths.unsqueeze(1)).float()
        
        return mask
    
    @staticmethod
    def get_attention_mask(mask, n_heads=None):
        """
        将 padding mask 转换为 attention mask
        
        Args:
            mask: [batch, seq_len]
            n_heads: 注意力头数
        
        Returns:
            attn_mask: [batch, n_heads, seq_len, seq_len] 或 [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len = mask.size()
        
        # [batch, 1, seq_len] @ [batch, seq_len, 1] -> [batch, seq_len, seq_len]
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        
        # 添加 head 维度
        if n_heads:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
        else:
            attn_mask = attn_mask.unsqueeze(1)
        
        return attn_mask

# 演示
print("\n【演示】:")
batch_size = 3
lengths = torch.tensor([3, 5, 2])  # 3个序列的实际长度
max_len = 6

mask_gen = MaskGenerator()

# 生成 padding mask
padding_mask = mask_gen.get_padding_mask(lengths, max_len)
print(f"\nLengths: {lengths.tolist()}")
print(f"Padding Mask shape: {padding_mask.shape}")
print(f"Padding Mask:\n{padding_mask}")

# 生成 attention mask
attn_mask = mask_gen.get_attention_mask(padding_mask, n_heads=2)
print(f"\nAttention Mask shape: {attn_mask.shape}")
print(f"Attention Mask (Head 0, Batch 0):\n{attn_mask[0, 0]}")

# ============================================
# 完整的 FastSpeech2 with Mask
# ============================================

print("\n" + "=" * 70)
print("完整实现: FastSpeech2 with Mask")
print("=" * 70)

class MultiHeadAttentionWithMask(nn.Module):
    """带 Mask 的多头注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        # 应用 mask（关键！）
        if mask is not None:
            # mask shape: [batch, 1, seq_len, seq_len]
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc(x)
        
        return x, attention


class FastSpeech2WithMask(nn.Module):
    """完整的 FastSpeech2 with Mask"""
    
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadAttentionWithMask(d_model, n_heads=2)
        self.length_regulator = LengthRegulatorWithAlignment()
        
    def forward(self, text, duration, text_lengths=None, mel_lengths=None):
        """
        Args:
            text: [batch, text_len]
            duration: [batch, text_len]
            text_lengths: [batch] - 文本的实际长度
            mel_lengths: [batch] - 梅尔频谱的实际长度（训练时提供）
        """
        batch_size = text.size(0)
        
        # 1. Embedding
        x = self.embedding(text)  # [batch, text_len, d_model]
        
        # 2. 生成 text padding mask（编码器阶段）
        if text_lengths is not None:
            text_mask = MaskGenerator.get_padding_mask(text_lengths, text.size(1))
            text_attn_mask = MaskGenerator.get_attention_mask(text_mask, n_heads=2)
        else:
            text_attn_mask = None
        
        # 3. Self-attention (编码器)
        x, _ = self.attention(x, x, x, mask=text_attn_mask)
        
        # 4. Length Regulation
        mel_repr, actual_mel_lens = self.length_regulator(x, duration)
        
        # 5. 生成 mel padding mask（解码器阶段）
        if mel_lengths is not None:
            mel_mask = MaskGenerator.get_padding_mask(mel_lengths, mel_repr.size(1))
            mel_attn_mask = MaskGenerator.get_attention_mask(mel_mask, n_heads=2)
        else:
            mel_attn_mask = None
        
        # 6. Self-attention (解码器)
        output, _ = self.attention(mel_repr, mel_repr, mel_repr, mask=mel_attn_mask)
        
        return output, mel_mask

# 测试
print("\n【测试完整模型】:")
model = FastSpeech2WithMask(vocab_size=100, d_model=64)

batch_size = 2
text = torch.randint(0, 100, (batch_size, 10))
duration = torch.randint(1, 5, (batch_size, 10)).float()
text_lengths = torch.tensor([7, 10])  # 实际长度

# 计算梅尔频谱的实际长度
mel_lengths = torch.tensor([
    duration[0, :text_lengths[0]].sum().long(),
    duration[1, :text_lengths[1]].sum().long()
])

print(f"\n输入:")
print(f"  Text shape: {text.shape}")
print(f"  Text lengths: {text_lengths.tolist()}")
print(f"  Duration: {duration.shape}")
print(f"  Mel lengths (computed): {mel_lengths.tolist()}")

output, mel_mask = model(text, duration, text_lengths, mel_lengths)

print(f"\n输出:")
print(f"  Output shape: {output.shape}")
print(f"  Mel mask shape: {mel_mask.shape}")
print(f"  Mel mask:\n{mel_mask}")

# ============================================
# 总结
# ============================================

print("\n" + "=" * 70)
print("总结")
print("=" * 70)

summary = """
【问题1: 如何保证对齐？】

✓ 训练阶段:
  - Duration 是从音频强制对齐得到的真实标注
  - sum(duration) 天然等于 mel_len
  - 所以 LengthRegulator 输出长度必然正确

✓ 推理阶段:
  - Duration 是模型预测的
  - 可能与真实长度有偏差，但通常很接近
  - 即使有偏差也不影响，因为 Vocoder 可以处理不同长度

【问题2: Mask 是否需要？】

✓ 绝对需要！原因:
  1. Batch 中序列长度不同，需要 padding
  2. Padding 位置不应参与注意力计算
  3. 不用 mask 会导致模型学习到 padding 的虚假模式

✓ 两种 Mask:
  1. Text Mask: 用于编码器（音素级别）
  2. Mel Mask: 用于解码器（帧级别）

✓ Mask 的作用:
  - 在注意力计算中将 padding 位置的分数设为 -inf
  - Softmax 后这些位置的权重接近 0
  - 避免 padding 影响模型学习
"""

print(summary)

print("\n关键代码模式:")
print("""
# 生成 mask
mask = (ids < lengths.unsqueeze(1)).float()

# 在注意力中应用
energy = energy.masked_fill(mask == 0, -1e10)

# 在损失计算中应用
loss = loss * mask  # 只计算有效位置的损失
""")

print("\n" + "=" * 70)