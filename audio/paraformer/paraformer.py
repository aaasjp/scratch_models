"""
Paraformer模型完整实现
包含Encoder、Predictor、CIF Sampler、Decoder等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """Paraformer编码器（类似Conformer）"""
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Predictor(nn.Module):
    """
    预测器：预测CIF的alpha权重
    alpha决定了每个encoder输出位置对应多少个decoder token
    """
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, encoder_output):
        # encoder_output: [batch_size, seq_len, d_model]
        x = encoder_output.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # 预测alpha权重（使用sigmoid保证在0-1之间）
        alphas = torch.sigmoid(self.fc(x)).squeeze(-1)  # [batch_size, seq_len]
        
        return alphas


class CIFSampler(nn.Module):
    """
    CIF (Continuous Integrate-and-Fire) 采样器
    根据alpha权重累积encoder特征，达到阈值时"发射"一个token
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, encoder_output, alphas, target_length=None):
        """
        Args:
            encoder_output: [batch_size, enc_len, d_model]
            alphas: [batch_size, enc_len]
            target_length: 目标序列长度（训练时使用）
        Returns:
            sampled_features: [batch_size, dec_len, d_model]
            num_fires: 实际发射的token数量
        """
        batch_size, enc_len, d_model = encoder_output.shape
        device = encoder_output.device
        
        # 如果指定了目标长度，调整alphas使其总和接近目标长度
        if target_length is not None:
            alpha_sum = alphas.sum(dim=1, keepdim=True)
            scale = target_length / (alpha_sum + 1e-8)
            alphas = alphas * scale
        
        # 初始化
        integrated = torch.zeros(batch_size, device=device)
        sampled_features = []
        fire_positions = []
        
        for t in range(enc_len):
            # 累积alpha权重
            integrated = integrated + alphas[:, t]
            
            # 当累积值超过阈值时，发射一个token
            fire_mask = integrated >= self.threshold
            
            if fire_mask.any():
                # 收集当前时刻的特征
                features = encoder_output[:, t, :]  # [batch_size, d_model]
                sampled_features.append(features)
                fire_positions.append(t)
                
                # 重置已发射的累积值
                integrated = integrated - fire_mask.float() * self.threshold
        
        if len(sampled_features) == 0:
            # 如果没有发射任何token，至少返回一个
            sampled_features = [encoder_output[:, 0, :]]
        
        # 堆叠所有发射的特征
        sampled_features = torch.stack(sampled_features, dim=1)  # [batch_size, num_fires, d_model]
        
        return sampled_features, len(fire_positions)


class DecoderLayer(nn.Module):
    """解码器层（非自回归）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 非自回归，所以不需要mask
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    """Paraformer解码器（非自回归）"""
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # 输出logits
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        
        return logits


class Paraformer(nn.Module):
    """
    完整的Paraformer模型
    包含: Encoder -> Predictor -> CIF Sampler -> Decoder
    """
    def __init__(
        self,
        input_dim=80,           # 输入特征维度（如FBANK特征）
        d_model=256,            # 模型维度
        enc_layers=12,          # 编码器层数
        dec_layers=6,           # 解码器层数
        num_heads=4,            # 注意力头数
        d_ff=2048,              # 前馈网络维度
        vocab_size=4233,        # 词表大小
        dropout=0.1,
        cif_threshold=1.0
    ):
        super().__init__()
        
        # 编码器
        self.encoder = Encoder(input_dim, d_model, enc_layers, num_heads, d_ff, dropout)
        
        # 预测器
        self.predictor = Predictor(d_model)
        
        # CIF采样器
        self.cif_sampler = CIFSampler(threshold=cif_threshold)
        
        # 解码器
        self.decoder = Decoder(d_model, dec_layers, num_heads, d_ff, vocab_size, dropout)
        
        self.d_model = d_model
        
    def forward(self, audio_features, audio_lengths=None, target_lengths=None):
        """
        Args:
            audio_features: [batch_size, seq_len, input_dim]
            audio_lengths: [batch_size] 音频实际长度
            target_lengths: [batch_size] 目标文本长度（训练时使用）
        Returns:
            logits: [batch_size, dec_len, vocab_size]
            alphas: [batch_size, enc_len]
            num_fires: 发射的token数量
        """
        # 1. 编码音频特征
        encoder_output = self.encoder(audio_features)  # [B, T, D]
        
        # 2. 预测alpha权重
        alphas = self.predictor(encoder_output)  # [B, T]
        
        # 3. CIF采样
        sampled_features, num_fires = self.cif_sampler(
            encoder_output, 
            alphas, 
            target_lengths
        )  # [B, L, D]
        
        # 4. 解码生成文本
        logits = self.decoder(sampled_features)  # [B, L, V]
        
        return logits, alphas, num_fires
    
    def recognize(self, audio_features):
        """推理模式：直接识别音频"""
        self.eval()
        with torch.no_grad():
            logits, alphas, num_fires = self.forward(audio_features)
            # 取最大概率的token
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        return predictions, alphas


# 使用示例
if __name__ == "__main__":
    # 模型参数
    batch_size = 4
    seq_len = 200
    input_dim = 80  # FBANK特征维度
    vocab_size = 4233
    
    # 创建模型
    model = Paraformer(
        input_dim=input_dim,
        d_model=256,
        enc_layers=6,
        dec_layers=3,
        num_heads=4,
        d_ff=1024,
        vocab_size=vocab_size,
        dropout=0.1
    )
    
    print("=" * 60)
    print("Paraformer模型结构:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print("=" * 60)
    
    # 模拟输入数据
    audio_features = torch.randn(batch_size, seq_len, input_dim)
    target_lengths = torch.tensor([50, 45, 55, 48])
    
    # 训练模式
    print("\n训练模式测试:")
    model.train()
    logits, alphas, num_fires = model(audio_features, target_lengths=target_lengths)
    print(f"输入形状: {audio_features.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"Alpha权重形状: {alphas.shape}")
    print(f"发射token数量: {num_fires}")
    
    # 推理模式
    print("\n推理模式测试:")
    predictions, alphas = model.recognize(audio_features[:1])
    print(f"预测token序列: {predictions[0][:20].tolist()}...")
    print(f"序列长度: {predictions.shape[1]}")
    
    print("\n" + "=" * 60)
    print("Paraformer模型特点:")
    print("1. 非自回归解码：并行生成所有token，推理速度快")
    print("2. CIF机制：自动对齐音频帧和文本token")
    print("3. Predictor-Sampler-Decoder：三阶段处理流程")
    print("4. 端到端训练：直接优化音频到文本的映射")
    print("=" * 60)