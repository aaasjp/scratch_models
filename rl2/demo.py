import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class PPOConfig:
    """PPO训练配置"""
    model_name: str = "gpt2"  # 基础模型
    learning_rate: float = 1e-5
    batch_size: int = 4
    ppo_epochs: int = 4
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO裁剪范围
    vf_coef: float = 0.1  # 价值函数损失系数
    entropy_coef: float = 0.01  # 熵正则化系数
    max_grad_norm: float = 0.5  # 梯度裁剪
    kl_penalty: float = 0.1  # KL散度惩罚
    target_kl: float = 0.1  # 目标KL散度
    max_length: int = 512


class ValueHead(nn.Module):
    """价值函数头，用于估计状态价值"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - 隐藏状态
        Returns:
            value: [batch_size, seq_len] - 价值估计
        """
        x = F.relu(self.linear1(hidden_states))
        value = self.linear2(x)
        return value.squeeze(-1)


class PPOModel(nn.Module):
    """PPO模型，包含策略网络和价值网络"""
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练语言模型作为策略网络
        self.policy_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.hidden_size = self.policy_model.config.hidden_size
        
        # 价值函数头
        self.value_head = ValueHead(self.hidden_size)
        
        # 参考模型（冻结，用于KL散度计算）
        self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ):
        """
        前向传播，返回logits和价值估计
        Args:
            input_ids: [batch_size, seq_len] - 输入token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
            return_dict: 是否返回字典格式
        Returns:
            dict或tuple: logits [batch_size, seq_len, vocab_size], values [batch_size, seq_len]
        """
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 计算价值估计
        values = self.value_head(hidden_states)  # [batch_size, seq_len]
        
        if return_dict:
            return {
                'logits': logits,
                'values': values,
                'hidden_states': hidden_states
            }
        return logits, values
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        计算动作的对数概率
        Args:
            input_ids: [batch_size, seq_len] - 输入token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
            labels: [batch_size, seq_len] - 标签token IDs
        Returns:
            gathered_log_probs: [batch_size, seq_len-1] - 每个位置的对数概率
            values: [batch_size, seq_len] - 价值估计
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
        
        # 移位：预测下一个token
        shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]
        
        # 计算对数概率
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)  # [batch_size, seq_len-1, 1]
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        return gathered_log_probs, outputs['values']  # values: [batch_size, seq_len]
    
    @torch.no_grad()
    def get_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        使用参考模型计算对数概率
        Args:
            input_ids: [batch_size, seq_len] - 输入token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
            labels: [batch_size, seq_len] - 标签token IDs
        Returns:
            gathered_log_probs: [batch_size, seq_len-1] - 每个位置的对数概率
        """
        outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)  # [batch_size, seq_len-1, 1]
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        return gathered_log_probs


class RLHFDataset(Dataset):
    """RLHF数据集"""
    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        tokenizer
    ):
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            dict: 包含 'input_ids' [seq_len], 'attention_mask' [seq_len], 'reward' [] (标量)
        """
        prompt = self.prompts[idx]
        response = self.responses[idx]
        reward = self.rewards[idx]
        
        # 合并prompt和response
        full_text = prompt + response
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 调试：查看tokenizer后的文本
        if idx == 0:  # 只打印第一个样本，避免输出过多
            print(f"\n{'='*70}")
            print(f"样本 {idx}:")
            print(f"原始文本: {full_text}")
            print(f"\nToken IDs: {encoding['input_ids'][0].tolist()}")
            print(f"Attention Mask: {encoding['attention_mask'][0].tolist()}")
            
            # 解码回文本
            decoded_text = self.tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=False)
            print(f"\n解码后的文本（包含特殊token）:\n{decoded_text}")
            
            decoded_text_clean = self.tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
            print(f"\n解码后的文本（不含特殊token）:\n{decoded_text_clean}")
            
            # 显示每个token对应的文本
            print(f"\n前10个token的对应文本:")
            for i in range(min(10, len(encoding['input_ids'][0]))):
                token_id = encoding['input_ids'][0][i].item()
                token_text = self.tokenizer.decode([token_id])
                print(f"  位置{i}: ID={token_id:5d} → '{token_text}'")
            print(f"{'='*70}\n")
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [seq_len]
            'reward': torch.tensor(reward, dtype=torch.float)  # [] (标量)
        }


class PPOTrainer:
    """PPO训练器"""
    def __init__(self, model: PPOModel, config: PPOConfig, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.policy_model.parameters()},
                {'params': model.value_head.parameters()}
            ],
            lr=config.learning_rate
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor
    ):
        """计算GAE优势函数
        Args:
            rewards: [batch_size, seq_len-1] - 每个token位置的奖励（已shift）
            values: [batch_size, seq_len-1] - 每个token位置的价值估计（已shift）
            masks: [batch_size, seq_len-1] - 掩码，1表示有效位置，0表示padding（已shift）
        Returns:
            advantages: [batch_size, seq_len-1] - 优势函数
            returns: [batch_size, seq_len-1] - 回报值
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # 对每个batch样本分别计算GAE
        for b in range(batch_size):
            last_advantage = 0
            last_value = 0  # 序列结束后的价值为0（从后往前遍历，所以是"下一个"位置的值）
            
            # 从后往前遍历序列
            for t in reversed(range(seq_len)):
                if masks[b, t] == 0:  # padding位置，跳过
                    advantages[b, t] = 0
                    returns[b, t] = 0
                    last_advantage = 0  # padding位置后重置
                    last_value = 0
                    continue
                
                # 计算TD误差
                # 从后往前遍历，last_value是序列中更靠后位置的值（即下一个时间步的值）
                # 如果当前是最后一个有效位置，next_value为0（序列结束）
                next_value = last_value
                delta = rewards[b, t] + self.config.gamma * next_value - values[b, t]
                
                # 计算GAE优势
                advantages[b, t] = delta + self.config.gamma * self.config.lam * last_advantage
                
                # 更新last_advantage和last_value（用于下一个更早的位置）
                last_advantage = advantages[b, t]
                last_value = values[b, t]
            
            # 计算returns = advantages + values（只对有效位置）
            returns[b] = advantages[b] + values[b]
            
        return advantages, returns
    
    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        ref_log_probs: torch.Tensor,
        masks: torch.Tensor
    ):
        """
        计算PPO损失
        Args:
            log_probs: [batch_size, seq_len-1] - 当前策略的对数概率
            old_log_probs: [batch_size, seq_len-1] - 旧策略的对数概率
            advantages: [batch_size, seq_len-1] - 优势函数
            values: [batch_size, seq_len-1] - 价值估计
            returns: [batch_size, seq_len-1] - 回报值
            ref_log_probs: [batch_size, seq_len-1] - 参考模型的对数概率
            masks: [batch_size, seq_len-1] - 掩码，1表示有效位置，0表示padding
        Returns:
            loss: 总损失
            metrics: 损失指标字典
        """
        # 只对有效位置计算损失
        valid_mask = masks.float()  # [batch_size, seq_len-1]
        num_valid = valid_mask.sum() + 1e-8
        
        # 策略损失（带裁剪）
        ratio = torch.exp(log_probs - old_log_probs)  # [batch_size, seq_len-1]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss_unreduced = -torch.min(surr1, surr2)  # [batch_size, seq_len-1]
        policy_loss = (policy_loss_unreduced * valid_mask).sum() / num_valid
        
        # 价值函数损失（只对有效位置）
        value_loss_unreduced = (values - returns) ** 2  # [batch_size, seq_len-1]
        value_loss = (value_loss_unreduced * valid_mask).sum() / num_valid
        
        # KL散度惩罚（只对有效位置）
        kl_div_unreduced = old_log_probs - ref_log_probs  # [batch_size, seq_len-1]
        kl_div = (kl_div_unreduced * valid_mask).sum() / num_valid
        
        # 熵奖励（鼓励探索，只对有效位置）
        entropy_unreduced = -(torch.exp(log_probs) * log_probs)  # [batch_size, seq_len-1]
        entropy = (entropy_unreduced * valid_mask).sum() / num_valid
        
        # 总损失
        loss = (
            policy_loss +
            self.config.vf_coef * value_loss -
            self.config.entropy_coef * entropy +
            self.config.kl_penalty * kl_div
        )
        
        return loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.item()
        }
    
    def train_step(self, batch: Dict):
        """
        单步训练
        Args:
            batch: 包含 'input_ids' [batch_size, seq_len], 
                   'attention_mask' [batch_size, seq_len],
                   'reward' [batch_size] 的字典
        Returns:
            avg_loss: 平均损失
            metrics: 损失指标字典
        """
        input_ids = batch['input_ids'].to(self.device)  # [batch_size, seq_len]
        attention_mask = batch['attention_mask'].to(self.device)  # [batch_size, seq_len]
        rewards = batch['reward'].to(self.device)  # [batch_size]
        
        # 获取旧的对数概率和价值估计
        with torch.no_grad():
            old_log_probs, old_values = self.model.get_log_probs(
                input_ids, attention_mask, input_ids
            )
            # old_log_probs: [batch_size, seq_len-1]
            # old_values: [batch_size, seq_len]
            ref_log_probs = self.model.get_ref_log_probs(
                input_ids, attention_mask, input_ids
            )  # [batch_size, seq_len-1]
        
        # 计算优势函数
        masks = attention_mask[:, 1:].float()  # [batch_size, seq_len-1]，shift后对应log_probs
        # 将sentence级别的reward广播到每个token位置
        token_rewards = rewards.unsqueeze(-1).expand_as(old_values[:, :-1])  # [batch_size, seq_len-1]
        advantages, returns = self.compute_advantages(
            token_rewards,  # [batch_size, seq_len-1]
            old_values[:, :-1],  # [batch_size, seq_len-1]
            masks  # [batch_size, seq_len-1]
        )
        # advantages: [batch_size, seq_len-1], returns: [batch_size, seq_len-1]
        
        # 标准化优势函数（只对有效位置标准化）
        valid_advantages = advantages * masks
        advantages_mean = valid_advantages.sum() / (masks.sum() + 1e-8)
        advantages_std = ((valid_advantages - advantages_mean) ** 2 * masks).sum() / (masks.sum() + 1e-8)
        advantages = (advantages - advantages_mean) / (advantages_std.sqrt() + 1e-8)
        advantages = advantages * masks  # 确保padding位置为0
        
        # PPO更新
        total_loss = 0
        metrics = {}
        
        for _ in range(self.config.ppo_epochs):
            log_probs, values = self.model.get_log_probs(
                input_ids, attention_mask, input_ids
            )
            # log_probs: [batch_size, seq_len-1]
            # values: [batch_size, seq_len]
            
            loss, step_metrics = self.ppo_loss(
                log_probs,  # [batch_size, seq_len-1]
                old_log_probs,  # [batch_size, seq_len-1]
                advantages,  # [batch_size, seq_len-1]
                values[:, :-1],  # [batch_size, seq_len-1] - values需要shift
                returns,  # [batch_size, seq_len-1]
                ref_log_probs,  # [batch_size, seq_len-1]
                masks  # [batch_size, seq_len-1]
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in step_metrics.items():
                metrics[k] = metrics.get(k, 0) + v
        
        # 平均指标
        avg_loss = total_loss / self.config.ppo_epochs
        for k in metrics:
            metrics[k] /= self.config.ppo_epochs
            
        return avg_loss, metrics
    
    def train(self, dataset: RLHFDataset, num_epochs: int = 3):
        """训练循环"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            all_metrics = {}
            
            for batch_idx, batch in enumerate(dataloader):
                loss, metrics = self.train_step(batch)
                total_loss += loss
                
                for k, v in metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0) + v
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss:.4f}")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            for k, v in all_metrics.items():
                print(f"  Average {k}: {v/len(dataloader):.4f}")
            print()


class PPOInference:
    """PPO模型推理"""
    def __init__(self, model: PPOModel, tokenizer, config: PPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        生成文本
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus采样参数
            top_k: top-k采样参数
        Returns:
            generated_text: 生成的文本
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True
        ).to(self.device)
        
        input_ids = inputs['input_ids']  # [1, prompt_len]
        attention_mask = inputs['attention_mask']  # [1, prompt_len]
        
        for _ in range(max_length):
            outputs = self.model.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits[:, -1, :] / temperature  # [1, vocab_size]
            
            # Top-k采样
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # [1, vocab_size]
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # [1, vocab_size]
                
                sorted_indices_to_remove = cumulative_probs > top_p  # [1, vocab_size]
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # [1, seq_len+1]
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=self.device)  # [1, 1]
            ], dim=-1)  # [1, seq_len+1]
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text


# 使用示例
if __name__ == "__main__":
    # 配置
    config = PPOConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=2,
        ppo_epochs=4
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型
    model = PPOModel(config)
    
    # 准备示例数据
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature."
    ]
    
    responses = [
        " The capital of France is Paris.",
        " Quantum computing uses quantum mechanics to process information.",
        " Trees whisper in the breeze, nature's symphony."
    ]
    
    rewards = [1.0, 0.8, 0.9]  # 来自奖励模型的评分
    
    # 创建数据集
    dataset = RLHFDataset(prompts, responses, rewards, tokenizer)
    
    # 训练
    trainer = PPOTrainer(model, config, tokenizer)
    print("开始训练...")
    trainer.train(dataset, num_epochs=2)
    
    # 推理
    print("\n开始推理...")
    inference = PPOInference(model, tokenizer, config)
    
    test_prompt = "What is artificial intelligence?"
    generated = inference.generate(test_prompt, max_length=50)
    print(f"\nPrompt: {test_prompt}")
    print(f"Generated: {generated}")
    
    # 保存模型
    torch.save(model.state_dict(), 'ppo_model.pt')
    print("\n模型已保存到 ppo_model.pt")