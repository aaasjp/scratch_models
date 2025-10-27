# FastSpeech2 语音合成模型

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个从零开始实现的 FastSpeech2 语音合成模型，基于 PyTorch 框架，用于学习和研究目的。本项目实现了完整的 TTS 流水线，包括数据预处理、模型训练和推理生成。

## 🚀 特性

- **完整的 FastSpeech2 实现**：包含编码器、解码器、长度调节器等核心组件
- **端到端训练流程**：从原始音频到最终语音合成的完整流程
- **高质量声码器**：集成 HiFi-GAN 声码器，生成高质量音频
- **灵活的数据处理**：支持多种音频格式和文本预处理
- **详细的训练监控**：完整的训练日志和可视化
- **易于使用的推理接口**：简单的文本到语音转换
- **🎨 模型可视化**：自动生成架构图和层次结构图
- **📊 参数统计**：详细的模型参数分析和统计信息

## 📋 系统要求

- Python 3.8+
- PyTorch 2.8.0+
- CUDA 11.0+ (推荐使用 GPU 训练)
- 至少 8GB RAM
- 至少 10GB 可用磁盘空间

## 🛠️ 安装

### 1. 克隆项目

```bash
git clone https://github.com/aaasjp/scratch_models.git
cd stratch_models/train_model/audio/fastspeech2
```

### 2. 创建 Conda 环境

```bash
# 创建数据预处理环境
conda create -n aligner python=3.8
conda activate aligner
conda install -c conda-forge montreal-forced-aligner

# 创建训练环境
conda create -n scratch-models python=3.8
conda activate scratch-models
```

### 3. 安装依赖

```bash
# 安装训练环境依赖
conda activate scratch-models
pip install -r requirements_scratch-models.txt

# 安装数据预处理环境依赖
conda activate aligner
pip install -r requirements_aligner.txt
```

## 📁 项目结构

```
fastspeech2/
├── 📄 fastspeech2.py              # FastSpeech2 模型定义
├── 📄 fastspeech2_train.py        # 训练脚本
├── 📄 fastspeech2_inference.py    # 推理脚本
├── 📄 fastspeech2_dataset.py      # 数据集处理
├── 📄 hifigan_vocoder.py          # HiFi-GAN 声码器
├── 📄 length_regulator.py         # 长度调节器
├── 📄 audio_to_mel_spectrogram.py # 音频处理工具
├── 📄 test_model_visualization.py # 模型可视化测试脚本
├── 📁 corpus/                     # 原始音频数据
├── 📁 corpus_aligned/             # 音素对齐数据
├── 📁 processed_data/             # 处理后的训练数据
├── 📁 checkpoints/                # 模型检查点
├── 📁 outputs/                    # 推理输出
├── 📁 model_structure/            # 模型结构图
│   ├── fastspeech2_architecture.png  # 架构图
│   └── fastspeech2_hierarchy.png      # 层次结构图
├── 📁 hifi_gan/                   # HiFi-GAN 声码器实现
├── 📁 test_files/                 # 测试音频文件
└── 📄 模型训练推理操作步骤说明.md  # 详细操作指南
```
**注意：hifi_gan是从https://github.com/jik876/hifi-gan.git下载的，并且改动了代码。**

## 🚀 快速开始

### 1. 数据预处理

```bash
# 激活数据预处理环境
conda activate aligner

# 下载 MFA 模型和词典
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 执行音素对齐
mfa align ./corpus english_us_arpa english_us_arpa ./corpus_aligned
```

### 2. 构建数据集

```bash
# 激活训练环境
conda activate scratch-models

# 构建训练数据集
python fastspeech2_dataset.py > dataset.log 2>&1
```

#### 数据处理详解

数据集构建过程包含以下关键步骤：

##### 音频特征提取
- **梅尔频谱图**：使用 librosa 提取 80 维梅尔频谱特征
- **基频 (Pitch)**：使用 YIN 算法提取基频，范围 65.4Hz-2093Hz
- **能量 (Energy)**：使用 RMS 计算能量，更贴近人耳感知

##### Pitch 处理流程
1. **提取阶段**：
   - 使用 `librosa.yin()` 提取基频
   - 过滤无效值：超出范围或接近0的值设为0（表示静音）
   - 仅对有声段进行对数转换：`f0[f0 > 0] = np.log(f0[f0 > 0] + 1)`

2. **归一化阶段**：
   - 只对非零值进行标准化：`(x - mean) / std`
   - 保持静音帧的0值不变
   - 计算统计信息：min, max, mean, std

3. **逆归一化阶段**（推理时）：
   - 对非零值进行逆标准化：`x * std + mean`
   - 超出范围处理：
     - 如果 < pitch_min：设置为 0（静音）
     - 如果 > pitch_max：设置为 pitch_max（上限）

##### Energy 处理流程
1. **提取阶段**：
   - 使用 `librosa.feature.rms()` 计算 RMS 能量
   - 设置能量阈值：小于全局最大值1%视为静音
   - 仅对有声段进行对数转换：`energy[energy > 0] = np.log(energy[energy > 0] + 1)`

2. **归一化阶段**：
   - 只对非零值进行标准化
   - 保持静音帧的0值不变
   - 计算统计信息：min, max, mean, std

3. **逆归一化阶段**（推理时）：
   - 对非零值进行逆标准化
   - 超出范围处理：
     - 如果 < energy_min：设置为 0（静音）
     - 如果 > energy_max：设置为 energy_max（上限）

##### 统计信息保存
构建过程中会生成 `processed_data/stats.json`，包含：
```json
{
  "pitch_min": 0.0,
  "pitch_max": 8.5,
  "pitch_mean": 4.2,
  "pitch_std": 1.8,
  "energy_min": 0.0,
  "energy_max": 6.1,
  "energy_mean": 2.3,
  "energy_std": 1.2,
  "duration_min": 0.1,
  "duration_max": 15.0,
  "duration_mean": 2.5,
  "duration_std": 1.8,
  "mel_min": -12.0,
  "mel_max": 2.0,
  "mel_mean": -5.2,
  "mel_std": 2.1
}
```

### 3. 训练模型

```bash
# 开始训练
python fastspeech2_train.py > train.log 2>&1

# 监控训练进度
tail -f train.log
```

### 4. 模型推理

```bash
# 准备输入文本（小写格式）
echo "hello world" > input_texts.txt

# 文本转音素
conda activate aligner
mfa g2p input_texts.txt english_us_arpa output_phonemes.txt --num_pronunciations 1

# 执行推理
conda activate scratch-models
python fastspeech2_inference.py
```

生成的音频文件将保存在 `./outputs/output_from_phonemes.wav`。

### 5. 模型可视化

```bash
# 生成模型结构图
python test_model_visualization.py
```

这将生成以下可视化文件：
- `./model_structure/fastspeech2_architecture.png` - 模型架构图
- `./model_structure/fastspeech2_hierarchy.png` - 模型层次结构图

## 📊 模型架构

### FastSpeech2 核心组件

- **编码器 (Encoder)**：基于 Transformer 的文本编码器
- **长度调节器 (Length Regulator)**：对齐文本和音频长度
- **解码器 (Decoder)**：生成梅尔频谱图
- **方差适配器 (Variance Adaptor)**：预测音调和能量

#### 数据处理核心特性

- **智能静音处理**：区分有声段和静音段，分别处理
- **对数域转换**：对 pitch 和 energy 进行对数转换，符合人耳感知特性
- **统计归一化**：基于训练数据统计进行标准化，提高训练稳定性
- **范围限制**：推理时自动限制输出范围，确保生成质量
- **零值保持**：静音帧保持为0，避免引入噪声

### 声码器

- **HiFi-GAN**：高质量声码器，将梅尔频谱图转换为音频波形

### 🎨 模型可视化

项目提供了自动化的模型结构可视化功能：

#### 架构图 (Architecture Diagram)
- 清晰展示数据流向
- 显示各组件之间的连接关系
- 包含中英文双语标签

#### 层次结构图 (Hierarchy Diagram)
- 详细的模型层次结构
- 参数统计信息
- 组件功能说明

#### 使用方法
```python
from fastspeech2 import FastSpeech2, print_model_info, visualize_model_structure

# 创建模型
model = FastSpeech2(vocab_size=100, d_model=256)

# 打印模型信息
print_model_info(model)

# 生成可视化图片
visualize_model_structure(model, "./model_structure")
```

## 🔧 配置选项

### 训练参数

```python
# 在 fastspeech2_train.py 中可调整的参数
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--save_interval', type=int, default=10)
```

### 模型参数

```python
# 在 fastspeech2.py 中的模型配置
n_mel_channels=80,      # 梅尔频谱维度
n_phoneme_vocab=100,    # 音素词汇表大小
encoder_dim=256,        # 编码器维度
decoder_dim=256,        # 解码器维度
```

### 数据处理参数

```python
# 音频处理参数
sample_rate=22050,      # 采样率
n_fft=1024,            # FFT窗口大小
hop_length=256,        # 跳跃长度
win_length=1024,       # 窗口长度
n_mels=80,             # 梅尔频谱维度

# Pitch 提取参数
fmin=65.4,             # 最小基频 (C2)
fmax=2093.0,           # 最大基频 (C7)
energy_threshold=0.01, # 能量阈值（全局最大值的1%）

# 归一化参数
eps=1e-6,              # 数值稳定性参数
```

## 📈 训练监控

训练过程中会生成以下文件：

- `train.log`：训练日志
- `logs/training_log.jsonl`：结构化训练数据
- `checkpoints/`：模型检查点
- `mel_output/`：梅尔频谱图可视化
- `model_structure/`：模型结构图
  - `fastspeech2_architecture.png`：架构图
  - `fastspeech2_hierarchy.png`：层次结构图

## 🎯 性能指标

- **训练时间**：约 2-3 小时（GPU 训练）
- **模型大小**：约 50MB
- **推理速度**：实时生成（< 1秒）
- **音频质量**：高保真度语音合成

## 🐛 故障排除

### 常见问题

1. **MFA 安装问题**
   ```bash
   # 确保使用 conda 安装
   conda install -c conda-forge montreal-forced-aligner
   ```

2. **CUDA 内存不足**
   ```bash
   # 减少批次大小
   python fastspeech2_train.py --batch_size 8
   ```

3. **音素对齐失败**
   - 检查音频文件格式（推荐 WAV）
   - 确保文本为小写
   - 验证 MFA 模型是否正确下载

4. **模型可视化问题**
   ```bash
   # 确保 matplotlib 正确安装
   pip install matplotlib
   
   # 如果中文字体显示有问题
   python -c "import matplotlib.pyplot as plt; print(plt.rcParams['font.sans-serif'])"
   ```

5. **数据处理问题**
   ```bash
   # 如果 stats.json 文件损坏或缺失
   rm -rf processed_data/
   python fastspeech2_dataset.py
   
   # 如果 pitch/energy 提取异常
   # 检查音频文件质量，确保采样率正确
   python -c "import librosa; print(librosa.get_samplerate('corpus/audio_000001.wav'))"
   
   # 如果归一化后数值异常
   # 检查 stats.json 中的统计值是否合理
   cat processed_data/stats.json
   ```

6. **推理质量问题**
   ```bash
   # 如果生成的音频质量差
   # 检查 pitch_control 和 energy_control 参数
   python fastspeech2_inference.py --pitch_control 1.0 --energy_control 1.0
   
   # 如果出现数值溢出
   # 检查逆归一化后的数值范围
   python -c "
   import json
   with open('processed_data/stats.json', 'r') as f:
       stats = json.load(f)
   print('Pitch range:', stats['pitch_min'], 'to', stats['pitch_max'])
   print('Energy range:', stats['energy_min'], 'to', stats['energy_max'])
   "
   ```

### 日志分析

```bash
# 查看训练日志
tail -f train.log

# 查看数据集构建日志
tail -f dataset.log
```

## 🎨 模型可视化示例

### 快速生成模型结构图

```bash
# 运行可视化测试脚本
python test_model_visualization.py
```

### 在代码中使用可视化功能

```python
import torch
from fastspeech2 import FastSpeech2, print_model_info, visualize_model_structure

# 创建模型
model = FastSpeech2(
    vocab_size=100,
    d_model=256,
    n_layers=8,
    n_heads=2,
    d_ff=1024,
    n_mel_channels=80
)

# 打印详细的模型信息
print_model_info(model)

# 生成模型结构图
visualize_model_structure(model, "./my_model_structure")
```

### 可视化输出

运行后会生成：
- **架构图**：展示数据流向和组件连接
- **层次结构图**：详细的模型层次和参数统计
- **文本信息**：完整的模型参数分析

## 📚 参考资料

- [FastSpeech2 论文](https://arxiv.org/abs/2006.04558)
- [HiFi-GAN 论文](https://arxiv.org/abs/2010.05646)
- [Montreal Forced Alignment](https://montreal-forced-aligner.readthedocs.io/en/v3.3.0/installation.html)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**：本项目仅用于学习和研究目的。商业使用请确保遵守相关许可证要求。
