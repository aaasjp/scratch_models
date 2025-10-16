from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import torch

# 1. 数据准备（同上）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")
train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["test"].select(range(200))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 2. 模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. 定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 4. 检测设备并设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 根据设备类型决定是否使用fp16
use_fp16 = torch.cuda.is_available()  # 只有CUDA GPU才使用fp16
print(f"是否使用fp16混合精度: {use_fp16}")

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./saved_models/bert_imdb_hf",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=use_fp16,  # 根据设备类型决定是否使用混合精度
    report_to="none"  # 关闭 wandb/tensorboard
)

# 5. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 6. 开始训练
trainer.train()