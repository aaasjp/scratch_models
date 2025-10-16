# FastSpeech2 完整实现

基于 HuggingFace LJSpeech 数据集的 FastSpeech2 文本转语音系统。

## 📋 目录结构

```
.
├── fastspeech2_improved.py          # 改进的模型实现（含完整Mask支持）
├── fastspeech2_real_data.py         # 真实数据加载（HuggingFace LJSpeech）
├── train_fastspeech2_complete.py    # 完整训练脚本
├── inference_fastspeech2_real.py    # 推理脚本
├── processed_data/                  # 预处理后的数据
│   ├── tokenizer.json              # 音素词表
│   ├── stats.json                  # 统计信息
│   ├── train_samples.pkl           # 训练样本
│   └── validation_samples.pkl      # 验证样本
├── checkpoints/                     # 模型检查点
└── outputs/                         # 合成输出
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchaudio
pip install datasets transformers
pip install librosa phonemizer
pip install scipy matplotlib tqdm
```

**注意**: `phonemizer` 需要 `espeak` 或 `espeak-ng`:

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak

# Windows
# 下载并安装: https://github.com/espeak-ng/espeak-ng/releases
```

### 2. 数据预处理

首次运行会自动下载 LJSpeech 数据集并预处理：

```bash
python fastspeech2_real_data.py
```

这将：
- 从 HuggingFace 下载 LJSpeech 数据集
- 进行文本到音素的转换（G2P）
- 提取梅尔频谱、pitch、energy
- 估算 duration（简化方法）
- 保存预处理数据

**预处理时间**: 首次约 30-60 分钟（取决于网络速度和计算资源）

### 3. 训练模型

#### 快速测试（100个样本）

```bash
python train_fastspeech2_complete.py \
    --max_train_samples 100 \
    --max_val_samples 20 \
    --num_epochs 50 \
    --batch_size 8
```

#### 完整训练

```bash
python train_fastspeech2_complete.py \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --d_model 256 \
    --n_layers 4 \
    --n_heads 2
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 100 | 训练轮数 |
| `--batch_size` | 16 | 批大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--d_model` | 256 | 模型隐藏层维度 |
| `--n_layers` | 4 | Transformer层数 |
| `--n_heads` | 2 | 注意力头数 |
| `--d_ff` | 1024 | 前馈网络维度 |
| `--checkpoint_dir` | ./checkpoints | 检查点保存目录 |
| `--resume` | None | 从检查点恢复训练 |

### 4. 语音合成（推理）

#### 交互式模式

```bash
python inference_fastspeech2_real.py
```

然后输入文本进行合成：

```
>> Hello world, this is a test.
>> speed 0.8    # 加快语速
>> pitch 1.2    # 提高音高
>> Hello again with different voice.
>> quit
```

#### 命令行模式

单个文本：

```bash
python inference_fastspeech2_real.py \
    --checkpoint checkpoints/best_model.pth \
    --text "Hello world, this is a test." \
    --output_dir outputs \
    --save_mel
```

批量合成（从文件）：

```bash
python inference_fastspeech2_real.py \
    --checkpoint checkpoints/best_model.pth \
    --text_file texts.txt \
    --output_dir outputs
```

#### 控制参数

```bash
# 快速语速
python inference_fastspeech2_real.py \
    --checkpoint checkpoints/best_model.pth \
    --text "Fast speech" \
    --duration_control 0.8

# 高音调
python inference_fastspeech2_real.py \
    --checkpoint checkpoints/best_model.pth \
    --text "High pitch" \
    --pitch_control 1.3

# 低能量（轻声）
python inference_fastspeech2_real.py \
    --checkpoint checkpoints/best_model.pth \
    --text "Soft voice" \
    --energy_control 0.7
```

## 📊 模型架构

### 核心组件

1. **编码器** (Encoder)
   - 多层 Feed-Forward Transformer
   - 处理音素级别的文本表征

2. **方差适配器** (Variance Adaptor)
   - Duration Predictor: 预测每个音素的持续时长
   - Pitch Predictor: 预测音高
   - Energy Predictor: 预测能量
   - Length Regulator: 根据 duration 扩展序列

3. **解码器** (Decoder)
   - 多层 Feed-Forward Transformer
   - 生成帧级别的梅尔频谱

### 关键特性

✅ **完整的 Mask 支持**
- Text Mask: 处理变长文本输入
- Mel Mask: 处理变长梅尔频谱输出
- 正确的损失计算（只在有效位置）

✅ **精确的长度对齐**
- Duration 与梅尔频谱帧数对齐
- 训练时使用真实 duration
- 推理时使用预测 duration

✅ **对数域 Duration 预测**
- 训练: `log(duration + 1)`
- 推理: `exp(pred) - 1`
- 数值稳定，收敛更快

## 📈 训练监控

训练日志保存在 `logs/training_log.jsonl`：

```python
import json
import matplotlib.pyplot as plt

# 读取日志
logs = []
with open('logs/training_log.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

# 绘制损失曲线
epochs = [log['epoch'] for log in logs]
train_loss = [log['train_loss'] for log in logs]
val_loss = [log['val_loss'] for log in logs]

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curve.png')
```

## 🔧 常见问题

### 1. Duration 对齐问题

**Q: 如何保证 LengthRegulator 输出长度与梅尔频谱一致？**

A: 通过真实的 duration 标注保证：
- 数据预处理时使用强制对齐获取 duration
- `sum(duration)` 天然等于梅尔频谱帧数
- 训练时使用真实 duration，保证对齐

**当前实现**: 使用简化的平均分配估算 duration。实际项目中应使用 **Montreal Forced Aligner (MFA)** 进行精确对齐。

### 2. 使用 MFA 进行精确对齐

```bash
# 1. 安装 MFA
conda install -c conda-forge montreal-forced-aligner

# 2. 下载预训练模型
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 3. 准备数据
# 音频: corpus/speaker/audio.wav
# 文本: corpus/speaker/audio.txt

# 4. 执行对齐
mfa align corpus/ english_us_arpa english_us_arpa aligned_output/

# 5. 提取 duration
# 解析 TextGrid 文件获取每个音素的时间戳
```

### 3. 为什么需要 Mask？

**必须使用 Mask 的原因**:
1. Batch 中序列长度不同，需要 padding
2. Padding 位置不应参与注意力计算
3. 损失计算要准确（只计算有效位置）
4. 不用 Mask 会学到虚假的 padding 模式

### 4. 改进声码器

当前使用 Griffin-Lim 算法，音质一般。建议使用神经声码器：

**推荐选择**:
- **HiFi-GAN**: 快速，音质好
- **WaveGlow**: 音质优秀
- **Parallel WaveGAN**: 平衡速度和质量

```python
# 使用 HiFi-GAN (示例)
from vocoder import HiFiGAN

vocoder = HiFiGAN(checkpoint='hifigan_checkpoint.pth')
audio = vocoder.mel_to_audio(mel_spectrogram)
```

## 📝 性能优化

### 训练加速

1. **使用混合精度训练**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. **增加 batch size** (如果显存允许)

3. **使用多 GPU**:
```python
model = nn.DataParallel(model)
```

### 推理加速

1. **量化模型**:
```python
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **使用 ONNX**:
```python
torch.onnx.export(model, dummy_input, "fastspeech2.onnx")
```

## 🎯 下一步改进

### 短期目标

- [ ] 使用 MFA 进行精确的强制对齐
- [ ] 集成 HiFi-GAN 声码器
- [ ] 添加说话人 embedding（多说话人支持）
- [ ] 实现 Conformer 替代 Transformer

### 长期目标

- [ ] 支持中文 TTS
- [ ] 添加情感控制
- [ ] 实时流式合成
- [ ] Web 界面

## 📚 参考文献

1. **FastSpeech 2**: Ren, Y., et al. (2020). "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"
2. **LJSpeech**: Keith Ito and Linda Johnson. "The LJ Speech Dataset"
3. **Transformer**: Vaswani, A., et al. (2017). "Attention is All You Need"

## 📄 许可证

本项目仅用于学习和研究目的。

## 🙏 致谢

- HuggingFace 提供的 LJSpeech 数据集
- FastSpeech2 原作者
- 开源社区的贡献

---

**作者**: FastSpeech2 实现团队  
**更新日期**: 2025-10-16