"""
Paraformer模型完整实现 - 增强版（支持热词功能）
包含: Encoder -> Predictor -> CIF Sampler -> Decoder + Hotword Biasing
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
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attn_weights


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
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """Paraformer编码器"""
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
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Predictor(nn.Module):
    """预测器：预测CIF的alpha权重"""
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, encoder_output):
        x = encoder_output.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        
        alphas = torch.sigmoid(self.fc(x)).squeeze(-1)
        
        return alphas


class CIFSampler(nn.Module):
    """CIF采样器"""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, encoder_output, alphas, target_length=None):
        batch_size, enc_len, d_model = encoder_output.shape
        device = encoder_output.device
        
        if target_length is not None:
            alpha_sum = alphas.sum(dim=1, keepdim=True)
            scale = target_length / (alpha_sum + 1e-8)
            alphas = alphas * scale
        
        integrated = torch.zeros(batch_size, device=device)
        sampled_features = []
        
        for t in range(enc_len):
            integrated = integrated + alphas[:, t]
            fire_mask = integrated >= self.threshold
            
            if fire_mask.any():
                features = encoder_output[:, t, :]
                sampled_features.append(features)
                integrated = integrated - fire_mask.float() * self.threshold
        
        if len(sampled_features) == 0:
            sampled_features = [encoder_output[:, 0, :]]
        
        sampled_features = torch.stack(sampled_features, dim=1)
        
        return sampled_features, len(sampled_features)


# ===================== 热词相关模块 =====================

class HotwordEncoder(nn.Module):
    """
    热词编码器
    将热词文本编码为向量表示
    """
    def __init__(self, vocab_size, d_model, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 使用轻量级的Transformer编码热词
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hotword_ids, hotword_mask=None):
        """
        Args:
            hotword_ids: [batch_size, num_hotwords, max_hotword_len]
            hotword_mask: [batch_size, num_hotwords, max_hotword_len]
        Returns:
            hotword_embeds: [batch_size, num_hotwords, d_model]
        """
        batch_size, num_hotwords, max_len = hotword_ids.shape
        
        # 展平进行编码
        hotword_ids_flat = hotword_ids.view(batch_size * num_hotwords, max_len)
        
        # 嵌入 + 位置编码
        x = self.embedding(hotword_ids_flat)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x)
        
        # 平均池化得到每个热词的表示
        if hotword_mask is not None:
            hotword_mask_flat = hotword_mask.view(batch_size * num_hotwords, max_len).unsqueeze(-1)
            x = (x * hotword_mask_flat).sum(dim=1) / (hotword_mask_flat.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)  # [batch_size * num_hotwords, d_model]
        
        # 恢复形状
        hotword_embeds = x.view(batch_size, num_hotwords, -1)
        
        return hotword_embeds


class HotwordAttention(nn.Module):
    """
    热词注意力模块
    计算解码器状态与热词的相似度，用于偏置输出概率
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.gate = nn.Linear(d_model * 2, 1)  # 控制热词影响程度的门控
        
    def forward(self, decoder_output, hotword_embeds):
        """
        Args:
            decoder_output: [batch_size, dec_len, d_model]
            hotword_embeds: [batch_size, num_hotwords, d_model]
        Returns:
            hotword_context: [batch_size, dec_len, d_model]
            attention_weights: [batch_size, dec_len, num_hotwords]
        """
        # 计算解码器输出与热词的注意力
        hotword_context, attn_weights = self.attention(
            decoder_output, 
            hotword_embeds, 
            hotword_embeds
        )
        
        # 门控机制：决定使用多少热词信息
        gate_input = torch.cat([decoder_output, hotword_context], dim=-1)
        gate_value = torch.sigmoid(self.gate(gate_input))  # [batch_size, dec_len, 1]
        
        # 加权的热词上下文
        hotword_context = gate_value * hotword_context
        
        return hotword_context, attn_weights.mean(dim=1)  # 平均多个头的注意力


class ContextualBiasing(nn.Module):
    """
    上下文偏置模块
    基于热词调整输出概率分布（Shallow Fusion方式）
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.bias_proj = nn.Linear(d_model, vocab_size)
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, decoder_output, hotword_context, base_logits, hotword_ids=None):
        """
        Args:
            decoder_output: [batch_size, dec_len, d_model]
            hotword_context: [batch_size, dec_len, d_model]
            base_logits: [batch_size, dec_len, vocab_size]
            hotword_ids: [batch_size, num_hotwords, max_hotword_len] 用于计算偏置权重
        Returns:
            biased_logits: [batch_size, dec_len, vocab_size]
        """
        # 计算热词偏置
        bias_logits = self.bias_proj(hotword_context)
        
        # 自适应权重：根据上下文决定偏置强度
        bias_weight = self.weight_net(hotword_context)  # [batch_size, dec_len, 1]
        
        # 组合基础logits和偏置logits
        biased_logits = base_logits + bias_weight * bias_logits
        
        return biased_logits


# ===================== 增强的解码器 =====================

class DecoderLayer(nn.Module):
    """解码器层（支持热词）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    """Paraformer解码器（支持热词）"""
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
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.output_proj(x)
        
        return logits, x  # 返回logits和隐藏状态


# ===================== 完整的Paraformer模型 =====================

class ParaformerWithHotword(nn.Module):
    """
    带热词功能的Paraformer模型
    """
    def __init__(
        self,
        input_dim=80,
        d_model=256,
        enc_layers=12,
        dec_layers=6,
        num_heads=4,
        d_ff=2048,
        vocab_size=4233,
        dropout=0.1,
        cif_threshold=1.0,
        use_hotword=True
    ):
        super().__init__()
        
        # 基础组件
        self.encoder = Encoder(input_dim, d_model, enc_layers, num_heads, d_ff, dropout)
        self.predictor = Predictor(d_model)
        self.cif_sampler = CIFSampler(threshold=cif_threshold)
        self.decoder = Decoder(d_model, dec_layers, num_heads, d_ff, vocab_size, dropout)
        
        # 热词相关组件
        self.use_hotword = use_hotword
        if use_hotword:
            self.hotword_encoder = HotwordEncoder(vocab_size, d_model, num_layers=2, num_heads=4)
            self.hotword_attention = HotwordAttention(d_model, num_heads=4)
            self.contextual_biasing = ContextualBiasing(vocab_size, d_model)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
    def forward(self, audio_features, audio_lengths=None, target_lengths=None, 
                hotword_ids=None, hotword_mask=None):
        """
        Args:
            audio_features: [batch_size, seq_len, input_dim]
            audio_lengths: [batch_size]
            target_lengths: [batch_size]
            hotword_ids: [batch_size, num_hotwords, max_hotword_len] 热词的token ids
            hotword_mask: [batch_size, num_hotwords, max_hotword_len] 热词的mask
        Returns:
            logits: [batch_size, dec_len, vocab_size]
            alphas: [batch_size, enc_len]
            num_fires: int
            hotword_attention: [batch_size, dec_len, num_hotwords] (如果使用热词)
        """
        # 1. 编码音频
        encoder_output = self.encoder(audio_features)
        
        # 2. 预测alpha
        alphas = self.predictor(encoder_output)
        
        # 3. CIF采样
        sampled_features, num_fires = self.cif_sampler(
            encoder_output, alphas, target_lengths
        )
        
        # 4. 解码
        base_logits, decoder_hidden = self.decoder(sampled_features)
        
        # 5. 热词处理
        hotword_attn_weights = None
        if self.use_hotword and hotword_ids is not None:
            # 编码热词
            hotword_embeds = self.hotword_encoder(hotword_ids, hotword_mask)
            
            # 计算热词注意力
            hotword_context, hotword_attn_weights = self.hotword_attention(
                decoder_hidden, hotword_embeds
            )
            
            # 应用上下文偏置
            logits = self.contextual_biasing(
                decoder_hidden, hotword_context, base_logits, hotword_ids
            )
        else:
            logits = base_logits
        
        return logits, alphas, num_fires, hotword_attn_weights
    
    def recognize(self, audio_features, hotwords=None, hotword_boost=1.5):
        """
        推理模式：识别音频（支持热词）
        
        Args:
            audio_features: [batch_size, seq_len, input_dim]
            hotwords: List[List[int]] - 热词的token ids列表
            hotword_boost: float - 热词增强权重
        """
        self.eval()
        with torch.no_grad():
            # 准备热词
            hotword_ids = None
            hotword_mask = None
            
            if hotwords is not None and self.use_hotword:
                hotword_ids, hotword_mask = self._prepare_hotwords(hotwords, audio_features.device)
            
            # 前向传播
            logits, alphas, num_fires, hotword_attn = self.forward(
                audio_features, 
                hotword_ids=hotword_ids, 
                hotword_mask=hotword_mask
            )
            
            # 应用额外的热词增强（可选）
            if hotwords is not None and self.use_hotword:
                logits = self._apply_hotword_boost(logits, hotwords, hotword_boost, audio_features.device)
            
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions, alphas, hotword_attn
    
    def _prepare_hotwords(self, hotwords, device):
        """准备热词数据"""
        batch_size = len(hotwords)
        max_num_hotwords = max(len(hw_list) for hw_list in hotwords)
        max_hotword_len = max(max(len(hw) for hw in hw_list) if hw_list else 0 
                             for hw_list in hotwords)
        
        # 初始化
        hotword_ids = torch.zeros(batch_size, max_num_hotwords, max_hotword_len, 
                                  dtype=torch.long, device=device)
        hotword_mask = torch.zeros(batch_size, max_num_hotwords, max_hotword_len, 
                                   dtype=torch.float, device=device)
        
        # 填充
        for b, hw_list in enumerate(hotwords):
            for i, hw in enumerate(hw_list):
                length = len(hw)
                hotword_ids[b, i, :length] = torch.tensor(hw, device=device)
                hotword_mask[b, i, :length] = 1.0
        
        return hotword_ids, hotword_mask
    
    def _apply_hotword_boost(self, logits, hotwords, boost_weight, device):
        """对热词token应用额外的boost"""
        boosted_logits = logits.clone()
        
        for b, hw_list in enumerate(hotwords):
            for hw in hw_list:
                for token_id in hw:
                    boosted_logits[b, :, token_id] += boost_weight
        
        return boosted_logits


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("=" * 70)
    print("Paraformer with Hotword - 带热词功能的语音识别模型")
    print("=" * 70)
    
    # 模型参数
    batch_size = 2
    seq_len = 200
    input_dim = 80
    vocab_size = 4233
    
    # 创建模型
    model = ParaformerWithHotword(
        input_dim=input_dim,
        d_model=256,
        enc_layers=6,
        dec_layers=3,
        num_heads=4,
        d_ff=1024,
        vocab_size=vocab_size,
        use_hotword=True
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 模拟输入
    audio_features = torch.randn(batch_size, seq_len, input_dim)
    
    # 模拟热词（例如：专有名词、产品名称等）
    hotwords = [
        [[100, 101, 102], [200, 201]],  # 第1个样本的热词
        [[150, 151], [250, 251, 252]]   # 第2个样本的热词
    ]
    
    print("\n" + "=" * 70)
    print("测试1: 不使用热词")
    print("=" * 70)
    predictions, alphas, _ = model.recognize(audio_features)
    print(f"输入形状: {audio_features.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"预测结果示例: {predictions[0, :10].tolist()}")
    
    print("\n" + "=" * 70)
    print("测试2: 使用热词")
    print("=" * 70)
    predictions_hw, alphas_hw, hotword_attn = model.recognize(
        audio_features, 
        hotwords=hotwords,
        hotword_boost=2.0
    )
    print(f"预测形状: {predictions_hw.shape}")
    print(f"热词注意力形状: {hotword_attn.shape if hotword_attn is not None else 'None'}")
    print(f"预测结果示例: {predictions_hw[0, :10].tolist()}")
    
    print("\n" + "=" * 70)
    print("热词功能说明")
    print("=" * 70)
    print("""
🎯 热词功能的3种实现方式:

1. **Hotword Encoder（热词编码器）**
   - 将热词文本编码为语义向量
   - 使用轻量级Transformer提取热词特征
   
2. **Hotword Attention（热词注意力）**
   - 计算解码器状态与热词的相似度
   - 门控机制动态调整热词影响
   
3. **Contextual Biasing（上下文偏置）**
   - 在输出层调整热词token的概率
   - Shallow Fusion方式融合

📊 热词效果对比:
   不使用热词: "今天天气很好,我要去阿里吧吧" (错误识别)
   使用热词:   "今天天气很好,我要去阿里巴巴" (正确识别)

💡 适用场景:
   ✓ 人名、地名等专有名词
   ✓ 产品名称、品牌名
   ✓ 行业术语、专业词汇
   ✓ 低频词、新词
    """)
    print("=" * 70)