import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# 1. 准备数据
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")
train_dataset = dataset["train"].select(range(1000))  # 小样本测试

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 2. 模型与优化器
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,output_loading_info=True)
loading_info = model[1]
print(f"loading_info: {loading_info}")
print("Missing keys (not in checkpoint):", loading_info["missing_keys"])
print("Unexpected keys (in checkpoint but not in model):", loading_info["unexpected_keys"])

model = model[0]

# 打印模型结构
print("=" * 50)
print("模型结构:")
print("=" * 50)
print(model)
print("\n")

# 打印模型参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print("\n")

# 打印各层详细信息
print("=" * 50)
print("各层详细信息:")
print("=" * 50)
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # 只打印叶子节点（实际层）
        param_count = sum(p.numel() for p in module.parameters())
        print(f"{name:40} | 参数数量: {param_count:>10,} | 类型: {type(module).__name__}")

print("\n")

# 打印模型参数名称和形状
print("=" * 50)
print("参数名称和形状:")
print("=" * 50)
for name, param in model.named_parameters():
    print(f"{name:50} | 形状: {str(param.shape):>20} | 参数数量: {param.numel():>10,}")

print("\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 训练循环（手动写）
model.train()
for epoch in range(1):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")

# 4. 保存训练好的模型
model_save_path = "./saved_models/bert_imdb"
os.makedirs(model_save_path, exist_ok=True)

# 保存模型和tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"模型已保存到: {model_save_path}")
print("保存的文件包括:")
print("- config.json (模型配置)")
print("- model.safetensors (模型权重)")
print("- tokenizer.json (分词器)")
print("- tokenizer_config.json (分词器配置)")
print("- vocab.txt (词汇表)")