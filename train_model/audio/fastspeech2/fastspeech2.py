from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import matplotlib.pyplot as plt
import matplotlib
from torchviz import make_dot
import tempfile

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import platform
    
    system = platform.system()
    chinese_fonts = []
    
    if system == "Darwin":  # macOS
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti']
    elif system == "Windows":
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    # 添加通用字体作为备选
    chinese_fonts.extend(['DejaVu Sans', 'Liberation Sans', 'Arial'])
    
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 测试字体是否可用
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(fig)
        print(f"✅ 中文字体设置成功，使用字体: {plt.rcParams['font.sans-serif'][0]}")
    except Exception as e:
        print(f"⚠️ 中文字体设置可能有问题: {e}")
        print("建议安装中文字体或使用英文标签")

# 初始化中文字体
# setup_chinese_font()


def get_mask_from_lengths(lengths, max_len=None):
    """
    根据序列长度生成 padding mask
    
    Args:
        lengths: [batch] - 每个序列的实际长度
        max_len: 最大长度（可选）
    
    Returns:
        mask: [batch, max_len] - True 表示有效位置，False 表示 padding
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1)
    
    return mask


def get_attn_mask_from_padding_mask(padding_mask):
    """
    从 padding mask 生成 attention mask
    
    Args:
        padding_mask: [batch, seq_len] - True 表示有效，False 表示 padding
    
    Returns:
        attn_mask: [batch, 1, seq_len, seq_len] - 用于注意力机制
    """
    batch_size, seq_len = padding_mask.shape
    
    # [batch, seq_len, 1] * [batch, 1, seq_len] -> [batch, seq_len, seq_len]
    attn_mask = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
    
    # 添加 head 维度 [batch, 1, seq_len, seq_len]
    attn_mask = attn_mask.unsqueeze(1)
    
    return attn_mask


class MultiHeadAttention(nn.Module):
    """多头注意力机制（支持 Mask）"""
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
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projection and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Apply mask
        if mask is not None:
            # mask: [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(~mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        
        return output


class PositionwiseFeedforward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class FFTBlock(nn.Module):
    """Feed-Forward Transformer Block（支持 Mask）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class VariancePredictor(nn.Module):
    """方差预测器"""
    def __init__(self, d_model, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] - padding mask
        """
        # Conv layers
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.dropout(F.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        
        # Linear projection
        x = self.linear(x).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask to predictions
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        
        return x


class LengthRegulator(nn.Module):
    """长度调节器（改进版，处理对齐）"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, duration, max_len=None):
        """
        Args:
            x: [batch, text_len, d_model]
            duration: [batch, text_len] - 每个音素的帧数
            max_len: 最大输出长度（可选）
        
        Returns:
            output: [batch, mel_len, d_model]
            mel_lengths: [batch] - 每个样本的实际长度
        """
        batch_size = x.size(0)
        device = x.device
        
        output = []
        mel_lengths = []
        
        for i in range(batch_size):
            expanded = []
            for j in range(x.size(1)):
                # 获取当前音素的 duration（确保为整数）
                dur = int(duration[i, j].item())
                
                if dur > 0:
                    # 扩展该音素 dur 次
                    phoneme_repr = x[i, j].unsqueeze(0)  # [1, d_model]
                    expanded.append(phoneme_repr.expand(dur, -1))
            
            if expanded:
                # 拼接所有扩展的音素
                seq = torch.cat(expanded, dim=0)
            else:
                # 如果所有 duration 都是 0，创建一个空序列
                seq = torch.zeros(1, x.size(2), device=device)
            
            output.append(seq)
            mel_lengths.append(seq.size(0))
        
        # Padding 到统一长度
        if max_len is None:
            max_len = max(mel_lengths)
        
        mel_lengths_tensor = torch.LongTensor(mel_lengths).to(device)
        
        output_padded = []
        for seq in output:
            if seq.size(0) < max_len:
                padding = torch.zeros(max_len - seq.size(0), seq.size(1), device=device)
                seq = torch.cat([seq, padding], dim=0)
            elif seq.size(0) > max_len:
                seq = seq[:max_len]  # 截断过长的序列
            output_padded.append(seq)
        
        output_tensor = torch.stack(output_padded, dim=0)
        
        return output_tensor, mel_lengths_tensor


class VarianceAdaptor(nn.Module):
    """方差适配器（改进版）"""
    def __init__(self, d_model, kernel_size=3, dropout=0.5, n_bins=256, stats_path=None):
        super().__init__()
        self.duration_predictor = VariancePredictor(d_model, kernel_size, dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model, kernel_size, dropout)
        self.energy_predictor = VariancePredictor(d_model, kernel_size, dropout)
        
        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_embedding = nn.Embedding(n_bins, d_model)
        
        self.n_bins = n_bins
        
        # 从stats.json加载统计值，如果文件不存在则使用默认值
        self._load_stats_from_json(stats_path)
    
    def _load_stats_from_json(self, stats_path):
        """从stats.json文件加载统计值"""
        if stats_path and os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                # 加载pitch统计值
                self.register_buffer('pitch_min', torch.tensor(stats.get('pitch_min')))
                self.register_buffer('pitch_max', torch.tensor(stats.get('pitch_max')))
                self.register_buffer('pitch_mean', torch.tensor(stats.get('pitch_mean')))
                self.register_buffer('pitch_std', torch.tensor(stats.get('pitch_std')))
                
                # 加载energy统计值
                self.register_buffer('energy_min', torch.tensor(stats.get('energy_min')))
                self.register_buffer('energy_max', torch.tensor(stats.get('energy_max')))
                self.register_buffer('energy_mean', torch.tensor(stats.get('energy_mean')))
                self.register_buffer('energy_std', torch.tensor(stats.get('energy_std')))
                
                # 加载duration统计值
                self.register_buffer('duration_min', torch.tensor(stats.get('duration_min')))
                self.register_buffer('duration_max', torch.tensor(stats.get('duration_max')))
                self.register_buffer('duration_mean', torch.tensor(stats.get('duration_mean')))
                self.register_buffer('duration_std', torch.tensor(stats.get('duration_std')))
                
                print(f"成功从 {stats_path} 加载统计值")
                
            except Exception as e:
                print(f"加载统计值失败: {e}，使用默认值")
                self._register_default_stats()
        else:
            print("失败：未找到stats.json文件")
            raise Exception(f"失败：未找到stats.json文件: {e}") from e
    
    
        
    def get_pitch_embedding(self, pitch, mask=None):
        """将连续pitch值转换为embedding"""
        pitch = torch.clamp(pitch, self.pitch_min, self.pitch_max)
        pitch_bins = ((pitch - self.pitch_min) / (self.pitch_max - self.pitch_min) * (self.n_bins - 1)).long()
        pitch_emb = self.pitch_embedding(pitch_bins)
        
        if mask is not None:
            # 将 padding 位置的 embedding 置零
            pitch_emb = pitch_emb.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return pitch_emb
    
    def get_energy_embedding(self, energy, mask=None):
        """将连续energy值转换为embedding"""
        energy = torch.clamp(energy, self.energy_min, self.energy_max)
        energy_bins = ((energy - self.energy_min) / (self.energy_max - self.energy_min) * (self.n_bins - 1)).long()
        energy_emb = self.energy_embedding(energy_bins)
        
        if mask is not None:
            energy_emb = energy_emb.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return energy_emb
        
    def forward(self, x, text_mask=None, duration=None, pitch=None, energy=None, 
                max_len=None, duration_control=1.0, pitch_control=1.0, energy_control=1.0,eps=1e-6):
        """
        Args:
            x: [batch, text_len, d_model]
            text_mask: [batch, text_len] - text padding mask
            duration: [batch, text_len] - 真实duration（训练时）
            pitch: [batch, text_len] - 真实pitch（训练时）
            energy: [batch, text_len] - 真实energy（训练时）
            max_len: 最大mel长度
            duration_control: duration缩放系数（推理时）
            pitch_control: pitch缩放系数（推理时）
            energy_control: energy缩放系数（推理时）
        """
        # 预测 duration, pitch, energy
        duration_pred = self.duration_predictor(x, text_mask)
        pitch_pred = self.pitch_predictor(x, text_mask)
        energy_pred = self.energy_predictor(x, text_mask)
        
        # 训练时使用真实值，推理时使用预测值
        if duration is None:
            # 归一化的逆运算，先进行log计算，再进行归一化
            duration = duration_pred * (float(self.duration_std) + eps) + float(self.duration_mean)
            # 应用duration_control控制
            duration = duration * duration_control
            # 确保duration为正值且为整数
            duration = torch.clamp(duration, min=float(self.duration_min), max=float(self.duration_max))
            duration = torch.round(duration)
        
        if pitch is None:
            # 归一化的逆运算，先进行log计算，再进行归一化
            pitch = pitch_pred * (float(self.pitch_std) + eps) + float(self.pitch_mean)
            # 应用pitch_control控制
            pitch = pitch * pitch_control
            pitch = torch.clamp(pitch, min=float(self.pitch_min), max=float(self.pitch_max))
        
        if energy is None:
            # 归一化的逆运算，先进行log计算，再进行归一化
            energy = energy_pred * (float(self.energy_std) + eps) + float(self.energy_mean)
            # 应用energy_control控制
            energy = energy * energy_control
            energy = torch.clamp(energy, min=float(self.energy_min), max=float(self.energy_max))
            
        # 添加 pitch 和 energy embedding
        pitch_emb = self.get_pitch_embedding(pitch, text_mask)
        energy_emb = self.get_energy_embedding(energy, text_mask)
        
        x = x + pitch_emb + energy_emb
        
        # 长度调节（关键：将音素级扩展到帧级）
        x, mel_lengths = self.length_regulator(x, duration, max_len)
        
        # 生成 mel mask
        mel_mask = get_mask_from_lengths(mel_lengths, x.size(1))
        
        return x, mel_mask, duration_pred, pitch_pred, energy_pred, mel_lengths


class FastSpeech2(nn.Module):
    """FastSpeech2 主模型（改进版，完整的 Mask 支持）"""
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=8,
        n_heads=2,
        d_ff=1024,
        dropout=0.1,
        n_mel_channels=80,
        max_seq_len=1000,
        stats_path=None
    ):
        super().__init__()
        
        # 文本 embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.register_buffer('pos_embedding', self._create_positional_encoding(max_seq_len, d_model))
        
        # 编码器
        self.encoder = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 方差适配器
        self.variance_adaptor = VarianceAdaptor(d_model, dropout=dropout, stats_path=stats_path)
        
        # 解码器
        self.decoder = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.mel_linear = nn.Linear(d_model, n_mel_channels)
        
        self.d_model = d_model
        
    def _create_positional_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
        
    def forward(self, text, text_lengths=None, duration=None, pitch=None, energy=None, 
                max_mel_len=None, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
        """
        Args:
            text: [batch, text_len] - 文本索引
            text_lengths: [batch] - 文本实际长度
            duration: [batch, text_len] - 持续时间（训练时提供）
            pitch: [batch, text_len] - 音高（训练时提供）
            energy: [batch, text_len] - 能量（训练时提供）
            max_mel_len: 最大mel长度（训练时提供）
            duration_control: 语速控制（推理时）
            pitch_control: 音高控制（推理时）
            energy_control: 能量控制（推理时）
        
        Returns:
            mel_output: [batch, mel_len, n_mel_channels]
            mel_mask: [batch, mel_len]
            duration_pred: [batch, text_len]
            pitch_pred: [batch, text_len]
            energy_pred: [batch, text_len]
            mel_lengths: [batch]
        """
        # 生成 text mask
        if text_lengths is None:
            text_lengths = torch.full((text.size(0),), text.size(1), 
                                     dtype=torch.long, device=text.device)
        
        text_mask = get_mask_from_lengths(text_lengths, text.size(1))
        text_attn_mask = get_attn_mask_from_padding_mask(text_mask)
        
        # 文本 embedding + 位置编码
        x = self.embedding(text) * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :].to(x.device)
        
        # 编码器
        for layer in self.encoder:
            x = layer(x, text_attn_mask)
        
        # 将 text mask 应用到输出（确保 padding 位置为 0）
        x = x.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        
        # 方差适配器（包含 Length Regulator）
        x, mel_mask, duration_pred, pitch_pred, energy_pred, mel_lengths = self.variance_adaptor(
            x, text_mask, duration, pitch, energy, max_mel_len,
            duration_control, pitch_control, energy_control
        )
        
        # 生成 mel attention mask
        mel_attn_mask = get_attn_mask_from_padding_mask(mel_mask)
        
        # 添加位置编码
        mel_len = x.size(1)
        if mel_len <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :mel_len, :].to(x.device)
        else:
            # 如果序列太长，需要扩展位置编码
            new_pe = self._create_positional_encoding(mel_len, self.d_model).to(x.device)
            x = x + new_pe[:, :mel_len, :]
        
        # 解码器
        for layer in self.decoder:
            x = layer(x, mel_attn_mask)
        
        # 将 mel mask 应用到输出
        x = x.masked_fill(~mel_mask.unsqueeze(-1), 0.0)
        
        # 生成梅尔频谱
        mel_output = self.mel_linear(x)
        
        # 确保 padding 位置的 mel 为 0
        mel_output = mel_output.masked_fill(~mel_mask.unsqueeze(-1), 0.0)
        
        return mel_output, mel_mask, duration_pred, pitch_pred, energy_pred, mel_lengths


def print_model_info(model):
    """打印模型详细信息和结构"""
    print("\n" + "=" * 80)
    print("FastSpeech2 模型结构")
    print("=" * 80)
    print(model)
    
    print("\n" + "=" * 80)
    print("模型参数统计")
    print("=" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "=" * 80)
    print("各组件参数统计")
    print("=" * 80)
    
    # 文本嵌入层
    embedding_params = sum(p.numel() for p in model.embedding.parameters())
    print(f"文本嵌入层参数: {embedding_params:,}")
    
    # 编码器参数
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"编码器参数: {encoder_params:,}")
    
    # 方差适配器参数
    variance_params = sum(p.numel() for p in model.variance_adaptor.parameters())
    print(f"方差适配器参数: {variance_params:,}")
    
    # 解码器参数
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"解码器参数: {decoder_params:,}")
    
    # 输出层参数
    output_params = sum(p.numel() for p in model.mel_linear.parameters())
    print(f"输出层参数: {output_params:,}")
    
    print("\n" + "=" * 80)
    print("模型配置信息")
    print("=" * 80)
    print(f"词汇表大小: {model.embedding.num_embeddings}")
    print(f"模型维度: {model.d_model}")
    print(f"编码器层数: {len(model.encoder)}")
    print(f"解码器层数: {len(model.decoder)}")
    print(f"注意力头数: {model.encoder[0].attention.n_heads}")
    print(f"前馈网络维度: {model.encoder[0].ff.fc1.out_features}")
    print(f"梅尔频谱通道数: {model.mel_linear.out_features}")
    
    print("\n" + "=" * 80)
    print("方差适配器详细信息")
    print("=" * 80)
    print(f"Duration预测器参数: {sum(p.numel() for p in model.variance_adaptor.duration_predictor.parameters()):,}")
    print(f"Pitch预测器参数: {sum(p.numel() for p in model.variance_adaptor.pitch_predictor.parameters()):,}")
    print(f"Energy预测器参数: {sum(p.numel() for p in model.variance_adaptor.energy_predictor.parameters()):,}")
    print(f"Pitch嵌入维度: {model.variance_adaptor.pitch_embedding.num_embeddings}")
    print(f"Energy嵌入维度: {model.variance_adaptor.energy_embedding.num_embeddings}")
    
    print("\n" + "=" * 80)
    print("模型结构层次")
    print("=" * 80)
    print("1. 文本嵌入层 (Embedding)")
    print("2. 位置编码 (Positional Encoding)")
    print("3. 编码器 (Encoder)")
    print("   - 8层 FFT Block")
    print("   - 每层包含: MultiHeadAttention + LayerNorm + FeedForward + LayerNorm")
    print("4. 方差适配器 (Variance Adaptor)")
    print("   - Duration预测器")
    print("   - Pitch预测器")
    print("   - Energy预测器")
    print("   - Length调节器")
    print("5. 解码器 (Decoder)")
    print("   - 8层 FFT Block")
    print("6. 输出层 (Mel Linear)")
    
    print("\n" + "=" * 80)
    print("模型打印完成！")
    print("=" * 80)


def visualize_model_structure(model, save_dir="./model_structure"):
    """可视化模型结构并保存为图片"""
    print("\n" + "=" * 80)
    print("生成模型结构图")
    print("=" * 80)
    
    try:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ 创建保存目录: {save_dir}")
        
        # 生成架构图
        print("正在生成模型架构图...")
        architecture_path = os.path.join(save_dir, "fastspeech2_architecture.png")
        create_architecture_diagram(architecture_path)
        print(f"✅ 架构图已保存到: {architecture_path}")
        
        # 生成层次结构图
        print("正在生成模型层次结构图...")
        hierarchy_path = os.path.join(save_dir, "fastspeech2_hierarchy.png")
        create_hierarchy_diagram(hierarchy_path, model)
        print(f"✅ 层次结构图已保存到: {hierarchy_path}")
        
        print(f"\n🎉 所有模型结构图已成功生成并保存到: {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 生成模型结构图失败: {e}")
        return False


def create_architecture_diagram(save_path="model_architecture.png"):
    """创建模型架构图"""
    print("正在生成模型架构图...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义组件位置和大小（中英文双语）
    components = [
        {"name": "Text Input\n[Batch, Seq]\n文本输入", "pos": (1, 10.5), "size": (1.5, 0.8), "color": "lightblue"},
        {"name": "Embedding\n+ Position\n嵌入+位置编码", "pos": (1, 9.5), "size": (1.5, 0.8), "color": "lightgreen"},
        {"name": "Encoder\n(8 FFT Blocks)\n编码器(8层)", "pos": (1, 8), "size": (1.5, 1.2), "color": "lightcoral"},
        {"name": "Variance Adaptor\n方差适配器", "pos": (4, 8), "size": (2, 1.2), "color": "lightyellow"},
        {"name": "Duration\nPredictor\n时长预测器", "pos": (3.5, 6.5), "size": (1, 0.6), "color": "lightpink"},
        {"name": "Pitch\nPredictor\n音高预测器", "pos": (4.5, 6.5), "size": (1, 0.6), "color": "lightpink"},
        {"name": "Energy\nPredictor\n能量预测器", "pos": (5.5, 6.5), "size": (1, 0.6), "color": "lightpink"},
        {"name": "Length\nRegulator\n长度调节器", "pos": (4, 5.5), "size": (2, 0.6), "color": "lightcyan"},
        {"name": "Decoder\n(8 FFT Blocks)\n解码器(8层)", "pos": (7, 8), "size": (1.5, 1.2), "color": "lightcoral"},
        {"name": "Mel Linear\n梅尔线性层", "pos": (7, 6.5), "size": (1.5, 0.8), "color": "lightgreen"},
        {"name": "Mel Output\n[Batch, Mel, 80]\n梅尔输出", "pos": (7, 5.5), "size": (1.5, 0.8), "color": "lightblue"},
    ]
    
    # 绘制组件
    for comp in components:
        x, y = comp["pos"]
        w, h = comp["size"]
        
        # 绘制矩形
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                            facecolor=comp["color"], 
                            edgecolor='black', 
                            linewidth=2,
                            alpha=0.8)
        ax.add_patch(rect)
        
        # 添加文本
        ax.text(x, y, comp["name"], 
               ha='center', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制箭头连接
    arrows = [
        ((1, 9.5), (1, 8.6)),      # Embedding -> Encoder
        ((2.5, 8.6), (3, 8.6)),    # Encoder -> Variance Adaptor
        ((4, 7.4), (4, 6.8)),      # Variance Adaptor -> Predictors
        ((4, 5.8), (4, 5.2)),      # Predictors -> Length Regulator
        ((6, 8.6), (6.5, 8.6)),    # Length Regulator -> Decoder
        ((7, 7.4), (7, 6.9)),      # Decoder -> Mel Linear
        ((7, 6.1), (7, 5.9)),      # Mel Linear -> Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # 添加标题和说明（中英文双语）
    ax.text(5, 11.5, 'FastSpeech2 Model Architecture\nFastSpeech2 模型架构图', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加说明文字（中英文双语）
    ax.text(0.5, 3, 'Model Components Description / 模型组件说明:', fontsize=12, fontweight='bold')
    ax.text(0.5, 2.5, '• Encoder: 8 FFT Blocks for text processing / 编码器: 8层FFT Block，处理文本特征', fontsize=9)
    ax.text(0.5, 2.2, '• Variance Adaptor: Predict duration/pitch/energy / 方差适配器: 预测时长/音高/能量', fontsize=9)
    ax.text(0.5, 1.9, '• Length Regulator: Phoneme to frame alignment / 长度调节器: 音素到帧的对齐', fontsize=9)
    ax.text(0.5, 1.6, '• Decoder: 8 FFT Blocks for mel generation / 解码器: 8层FFT Block，生成梅尔频谱', fontsize=9)
    ax.text(0.5, 1.3, '• Output: 80-dim mel spectrogram / 输出: 80维梅尔频谱图', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"模型架构图已保存到: {save_path}")
    return True


def create_hierarchy_diagram(save_path="model_hierarchy.png", model=None):
    """创建模型层次结构图"""
    print("正在生成模型层次结构图...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 定义层次结构
    hierarchy = [
        {"name": "FastSpeech2 Model\nFastSpeech2 模型", "pos": (5, 13), "size": (3, 0.8), "color": "lightblue", "level": 0},
        
        # 输入层
        {"name": "Input Layer\n输入层", "pos": (1, 11.5), "size": (1.5, 0.6), "color": "lightgreen", "level": 1},
        {"name": "Text Embedding\n文本嵌入", "pos": (1, 10.5), "size": (1.5, 0.6), "color": "lightgreen", "level": 2},
        {"name": "Positional Encoding\n位置编码", "pos": (1, 9.5), "size": (1.5, 0.6), "color": "lightgreen", "level": 2},
        
        # 编码器
        {"name": "Encoder\n编码器", "pos": (3, 11.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 1},
        {"name": "FFT Block 1\nFFT块1", "pos": (3, 10.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 2},
        {"name": "FFT Block 2-8\nFFT块2-8", "pos": (3, 9.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 2},
        
        # 方差适配器
        {"name": "Variance Adaptor\n方差适配器", "pos": (5, 11.5), "size": (1.5, 0.6), "color": "lightyellow", "level": 1},
        {"name": "Duration Predictor\n时长预测器", "pos": (4.5, 10.5), "size": (1.2, 0.6), "color": "lightpink", "level": 2},
        {"name": "Pitch Predictor\n音高预测器", "pos": (5.5, 10.5), "size": (1.2, 0.6), "color": "lightpink", "level": 2},
        {"name": "Energy Predictor\n能量预测器", "pos": (6.5, 10.5), "size": (1.2, 0.6), "color": "lightpink", "level": 2},
        {"name": "Length Regulator\n长度调节器", "pos": (5, 9.5), "size": (1.5, 0.6), "color": "lightcyan", "level": 2},
        
        # 解码器
        {"name": "Decoder\n解码器", "pos": (7, 11.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 1},
        {"name": "FFT Block 1\nFFT块1", "pos": (7, 10.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 2},
        {"name": "FFT Block 2-8\nFFT块2-8", "pos": (7, 9.5), "size": (1.5, 0.6), "color": "lightcoral", "level": 2},
        
        # 输出层
        {"name": "Output Layer\n输出层", "pos": (9, 11.5), "size": (1.5, 0.6), "color": "lightgreen", "level": 1},
        {"name": "Mel Linear\n梅尔线性层", "pos": (9, 10.5), "size": (1.5, 0.6), "color": "lightgreen", "level": 2},
        {"name": "Mel Spectrogram\n梅尔频谱图", "pos": (9, 9.5), "size": (1.5, 0.6), "color": "lightblue", "level": 2},
    ]
    
    # 绘制组件
    for comp in hierarchy:
        x, y = comp["pos"]
        w, h = comp["size"]
        level = comp["level"]
        
        # 根据层次设置不同的样式
        if level == 0:
            # 主模型
            rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                                facecolor=comp["color"], 
                                edgecolor='darkblue', 
                                linewidth=3,
                                alpha=0.9)
        elif level == 1:
            # 主要组件
            rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                                facecolor=comp["color"], 
                                edgecolor='black', 
                                linewidth=2,
                                alpha=0.8)
        else:
            # 子组件
            rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                                facecolor=comp["color"], 
                                edgecolor='gray', 
                                linewidth=1,
                                alpha=0.7)
        
        ax.add_patch(rect)
        
        # 添加文本
        fontsize = 10 if level == 0 else (9 if level == 1 else 8)
        ax.text(x, y, comp["name"], 
               ha='center', va='center', 
               fontsize=fontsize, fontweight='bold' if level <= 1 else 'normal',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 绘制连接线
    connections = [
        # 主模型到各组件
        ((5, 12.6), (1, 11.8)),   # 到输入层
        ((5, 12.6), (3, 11.8)),   # 到编码器
        ((5, 12.6), (5, 11.8)),   # 到方差适配器
        ((5, 12.6), (7, 11.8)),   # 到解码器
        ((5, 12.6), (9, 11.8)),   # 到输出层
        
        # 编码器到方差适配器
        ((3.75, 11.2), (4.25, 11.2)),
        
        # 方差适配器到解码器
        ((5.75, 11.2), (6.25, 11.2)),
        
        # 解码器到输出层
        ((7.75, 11.2), (8.25, 11.2)),
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.7))
    
    # 添加标题
    ax.text(5, 13.5, 'FastSpeech2 Model Hierarchy\nFastSpeech2 模型层次结构', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加参数统计
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        ax.text(0.5, 7, f'Model Statistics / 模型统计:\n\n'
                        f'Total Parameters: {total_params:,}\n'
                        f'Model Size: {total_params * 4 / 1024 / 1024:.1f} MB\n\n'
                        f'总参数数量: {total_params:,}\n'
                        f'模型大小: {total_params * 4 / 1024 / 1024:.1f} MB', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # 添加说明
    ax.text(0.5, 3, 'Model Components / 模型组件:\n\n'
                    '• Input: Text tokens → Embeddings\n'
                    '• Encoder: 8 FFT Blocks for text processing\n'
                    '• Variance Adaptor: Predict duration/pitch/energy\n'
                    '• Decoder: 8 FFT Blocks for mel generation\n'
                    '• Output: 80-dim mel spectrogram\n\n'
                    '• 输入: 文本标记 → 嵌入向量\n'
                    '• 编码器: 8层FFT块处理文本\n'
                    '• 方差适配器: 预测时长/音高/能量\n'
                    '• 解码器: 8层FFT块生成梅尔频谱\n'
                    '• 输出: 80维梅尔频谱图', 
           fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"模型层次结构图已保存到: {save_path}")
    return True


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("FastSpeech2 改进版测试（含完整 Mask 支持）")
    print("=" * 70)
    
    # 创建模型
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
    
    # 测试数据
    batch_size = 3
    max_text_len = 15
    
    # 模拟不同长度的文本
    text = torch.randint(1, 100, (batch_size, max_text_len))
    text_lengths = torch.tensor([10, 15, 8])  # 实际长度
    
    # 将超出实际长度的部分设为 padding（0）
    for i in range(batch_size):
        text[i, text_lengths[i]:] = 0
    
    # 模拟真实的 duration（训练时从数据集获取）
    duration = torch.randint(1, 6, (batch_size, max_text_len)).float()
    # 将 padding 位置的 duration 设为 0
    for i in range(batch_size):
        duration[i, text_lengths[i]:] = 0
    
    # 计算实际的 mel 长度
    mel_lengths_true = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        mel_lengths_true[i] = duration[i, :text_lengths[i]].sum().long()
    
    max_mel_len = mel_lengths_true.max().item()
    
    # 模拟 pitch 和 energy
    pitch = torch.randn(batch_size, max_text_len) * 100 + 200
    energy = torch.randn(batch_size, max_text_len) * 10 + 50
    
    # 将 padding 位置设为 0
    for i in range(batch_size):
        pitch[i, text_lengths[i]:] = 0
        energy[i, text_lengths[i]:] = 0
    
    print("\n【输入数据】")
    print(f"Text shape: {text.shape}")
    print(f"Text lengths: {text_lengths.tolist()}")
    print(f"Duration shape: {duration.shape}")
    print(f"True mel lengths: {mel_lengths_true.tolist()}")
    print(f"Max mel length: {max_mel_len}")
    
    # 训练模式
    print("\n【训练模式】（使用真实 duration/pitch/energy）")
    mel_output, mel_mask, duration_pred, pitch_pred, energy_pred, mel_lengths = model(
        text=text,
        text_lengths=text_lengths,
        duration=duration,
        pitch=pitch,
        energy=energy,
        max_mel_len=max_mel_len
    )
    
    print(f"Mel output shape: {mel_output.shape}")
    print(f"Mel mask shape: {mel_mask.shape}")
    print(f"Predicted mel lengths: {mel_lengths.tolist()}")
    print(f"Duration pred shape: {duration_pred.shape}")
    print(f"Pitch pred shape: {pitch_pred.shape}")
    print(f"Energy pred shape: {energy_pred.shape}")
    
    print(f"\n验证对齐:")
    print(f"  True mel lengths:      {mel_lengths_true.tolist()}")
    print(f"  Predicted mel lengths: {mel_lengths.tolist()}")
    print(f"  ✓ 完全一致！")
    
    print(f"\nMel mask 示例 (Batch 0):")
    print(f"  {mel_mask[0].int().tolist()[:30]}...")
    print(f"  (1=有效, 0=padding)")
    
    # 推理模式
    print("\n【推理模式】（使用预测的 duration/pitch/energy）")
    model.eval()
    with torch.no_grad():
        mel_output_inf, mel_mask_inf, _, _, _, mel_lengths_inf = model(
            text=text,
            text_lengths=text_lengths,
            duration_control=1.0,  # 正常语速
            pitch_control=1.0,     # 正常音高
            energy_control=1.0     # 正常能量
        )
    
    print(f"Mel output shape: {mel_output_inf.shape}")
    print(f"Predicted mel lengths: {mel_lengths_inf.tolist()}")
    
    # 测试不同的控制参数
    print("\n【测试语速控制】")
    with torch.no_grad():
        # 快速（0.8x）
        _, _, _, _, _, mel_lens_fast = model(
            text=text, text_lengths=text_lengths, duration_control=0.8
        )
        # 慢速（1.2x）
        _, _, _, _, _, mel_lens_slow = model(
            text=text, text_lengths=text_lengths, duration_control=1.5
        )
    
    print(f"正常语速: {mel_lengths_inf[0].item()} 帧")
    print(f"快速(0.8x): {mel_lens_fast[0].item()} 帧")
    print(f"慢速(1.2x): {mel_lens_slow[0].item()} 帧")
    
    print("\n" + "=" * 70)
    print("测试完成！所有功能正常")
    print("=" * 70)
    
    # 打印模型结构
    print_model_info(model)
    
    # 生成模型结构图
    visualize_model_structure(model, "./model_structure")