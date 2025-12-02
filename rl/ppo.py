"""
PPO (Proximal Policy Optimization) 算法实现
支持离散和连续动作空间
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from collections import deque

from .networks import Actor, Critic, ActorCritic
from .replay_buffer import RolloutBuffer, PPOBuffer


class PPO:
    """
    PPO 算法实现
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        action_bound: float = 1.0,
        use_shared_network: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            continuous: 是否为连续动作空间
            lr_actor: Actor 学习率
            lr_critic: Critic 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            clip_epsilon: PPO clip 参数
            value_clip: 是否对价值函数进行 clip
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪的最大范数
            update_epochs: 每次更新的轮数
            batch_size: 批次大小
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            action_bound: 连续动作的边界值
            use_shared_network: 是否使用共享网络的 Actor-Critic
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device
        
        # 创建网络
        if use_shared_network:
            self.ac_network = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                continuous=continuous,
                action_bound=action_bound
            ).to(device)
            self.actor = None
            self.critic = None
        else:
            self.actor = Actor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                continuous=continuous,
                action_bound=action_bound
            ).to(device)
            self.critic = Critic(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                activation=activation
            ).to(device)
            self.ac_network = None
        
        # 优化器
        if use_shared_network:
            self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr_actor)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 训练统计
        self.training_stats = {
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy_loss": deque(maxlen=100),
            "total_loss": deque(maxlen=100),
            "clip_fraction": deque(maxlen=100)
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        选择动作
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.ac_network is not None:
                # 使用共享网络
                actor_output, value = self.ac_network(state_tensor)
                
                if self.continuous:
                    mean = actor_output["mean"]
                    std = actor_output["std"]
                    action_bound = self.ac_network.action_bound
                    if deterministic:
                        action = torch.tanh(mean) * action_bound
                        log_prob = torch.zeros(1)
                    else:
                        normal_dist = torch.distributions.Normal(mean, std)
                        action_raw = normal_dist.sample()
                        action = torch.tanh(action_raw) * action_bound
                        log_prob = normal_dist.log_prob(action_raw).sum(dim=-1)
                        log_prob -= torch.log(1 - action.pow(2) / (action_bound ** 2 + 1e-6)).sum(dim=-1)
                else:
                    probs = actor_output["probs"]
                    if deterministic:
                        action = torch.argmax(probs, dim=-1)
                    else:
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()
                    logits = actor_output["logits"]
                    log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
                
                value = value.squeeze()
            else:
                # 使用分离的网络
                action, log_prob = self.actor.sample(state_tensor, deterministic)
                value = self.critic(state_tensor).squeeze()
        
        action = action.cpu().numpy()
        if not self.continuous:
            action = action[0] if len(action.shape) > 0 else action
        else:
            action = action[0] if len(action.shape) > 1 else action
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()
        
        return action, log_prob, value
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        更新策略和价值网络
        Args:
            buffer: 经验回放缓冲区
        Returns:
            训练统计信息
        """
        if buffer.size == 0:
            return {}
        
        # 计算优势和回报
        buffer.compute_advantages_and_returns(
            last_value=0.0,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # 获取所有数据
        data = buffer.get_batch(batch_size=None)
        for key in data:
            data[key] = data[key].to(self.device)
        
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        
        total_size = states.size(0)
        
        # 多轮更新
        for epoch in range(self.update_epochs):
            # 打乱数据
            indices = torch.randperm(total_size, device=self.device)
            
            for start_idx in range(0, total_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                if self.ac_network is not None:
                    actor_output, values = self.ac_network(batch_states)
                    
                    if self.continuous:
                        mean = actor_output["mean"]
                        std = actor_output["std"]
                        action_bound = self.ac_network.action_bound
                        action_raw = torch.atanh(torch.clamp(batch_actions / action_bound, -0.999, 0.999))
                        normal_dist = torch.distributions.Normal(mean, std)
                        log_probs = normal_dist.log_prob(action_raw).sum(dim=-1)
                        log_probs -= torch.log(1 - batch_actions.pow(2) / (action_bound ** 2 + 1e-6)).sum(dim=-1)
                        entropy = normal_dist.entropy().sum(dim=-1)
                    else:
                        logits = actor_output["logits"]
                        dist = torch.distributions.Categorical(logits=logits)
                        log_probs = dist.log_prob(batch_actions.squeeze())
                        entropy = dist.entropy()
                    
                    values = values.squeeze()
                else:
                    log_probs, entropy, _ = self.actor.evaluate_actions(batch_states, batch_actions)
                    values = self.critic(batch_states).squeeze()
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO clip 目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算 clip 比例（用于监控）
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                
                # 价值损失
                if self.value_clip:
                    values_clipped = batch_old_log_probs.new_zeros(batch_old_log_probs.shape)  # 占位
                    # 需要旧的价值估计，这里简化处理
                    value_loss = F.mse_loss(values, batch_returns)
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                if self.ac_network is not None:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    # 分别更新 Actor 和 Critic
                    # Actor 更新
                    self.actor_optimizer.zero_grad()
                    actor_loss = policy_loss + self.entropy_coef * entropy_loss
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    # Critic 更新
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                
                # 记录统计信息
                self.training_stats["policy_loss"].append(policy_loss.item())
                self.training_stats["value_loss"].append(value_loss.item())
                self.training_stats["entropy_loss"].append(entropy_loss.item())
                self.training_stats["total_loss"].append(total_loss.item())
                self.training_stats["clip_fraction"].append(clip_fraction.item())
        
        # 返回平均统计信息
        stats = {
            "policy_loss": np.mean(self.training_stats["policy_loss"]),
            "value_loss": np.mean(self.training_stats["value_loss"]),
            "entropy_loss": np.mean(self.training_stats["entropy_loss"]),
            "total_loss": np.mean(self.training_stats["total_loss"]),
            "clip_fraction": np.mean(self.training_stats["clip_fraction"])
        }
        
        return stats
    
    def update_from_buffer(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        从 PPOBuffer 更新（支持批量并行环境）
        Args:
            buffer: PPOBuffer 实例
        Returns:
            训练统计信息
        """
        # 计算优势和回报
        if buffer.ptr < buffer.num_steps:
            # 缓冲区未满，只使用已有数据
            last_values = buffer.values[buffer.ptr - 1] if buffer.ptr > 0 else torch.zeros(buffer.num_envs, device=self.device)
        else:
            # 需要从网络中获取最后一个状态的价值
            last_states = buffer.states[-1]
            with torch.no_grad():
                if self.ac_network is not None:
                    _, last_values = self.ac_network(last_states)
                    last_values = last_values.squeeze()
                else:
                    last_values = self.critic(last_states).squeeze()
        
        buffer.compute_advantages_and_returns(
            last_values=last_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # 获取所有数据
        data = buffer.get()
        total_size = data["states"].size(0)
        
        # 多轮更新
        for epoch in range(self.update_epochs):
            indices = torch.randperm(total_size, device=self.device)
            
            for start_idx in range(0, total_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = data["states"][batch_indices]
                batch_actions = data["actions"][batch_indices]
                batch_old_log_probs = data["old_log_probs"][batch_indices]
                batch_advantages = data["advantages"][batch_indices]
                batch_returns = data["returns"][batch_indices]
                
                # 计算损失（同 update 方法）
                if self.ac_network is not None:
                    actor_output, values = self.ac_network(batch_states)
                    
                    if self.continuous:
                        mean = actor_output["mean"]
                        std = actor_output["std"]
                        action_bound = self.ac_network.action_bound
                        action_raw = torch.atanh(torch.clamp(batch_actions / action_bound, -0.999, 0.999))
                        normal_dist = torch.distributions.Normal(mean, std)
                        log_probs = normal_dist.log_prob(action_raw).sum(dim=-1)
                        log_probs -= torch.log(1 - batch_actions.pow(2) / (action_bound ** 2 + 1e-6)).sum(dim=-1)
                        entropy = normal_dist.entropy().sum(dim=-1)
                    else:
                        logits = actor_output["logits"]
                        dist = torch.distributions.Categorical(logits=logits)
                        log_probs = dist.log_prob(batch_actions.squeeze() if batch_actions.dim() > 1 else batch_actions)
                        entropy = dist.entropy()
                    
                    values = values.squeeze()
                else:
                    log_probs, entropy, _ = self.actor.evaluate_actions(batch_states, batch_actions)
                    values = self.critic(batch_states).squeeze()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 更新
                if self.ac_network is not None:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    self.actor_optimizer.zero_grad()
                    actor_loss = policy_loss + self.entropy_coef * entropy_loss
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                
                # 记录统计
                self.training_stats["policy_loss"].append(policy_loss.item())
                self.training_stats["value_loss"].append(value_loss.item())
                self.training_stats["entropy_loss"].append(entropy_loss.item())
                self.training_stats["total_loss"].append(total_loss.item())
                self.training_stats["clip_fraction"].append(clip_fraction.item())
        
        stats = {
            "policy_loss": np.mean(self.training_stats["policy_loss"]),
            "value_loss": np.mean(self.training_stats["value_loss"]),
            "entropy_loss": np.mean(self.training_stats["entropy_loss"]),
            "total_loss": np.mean(self.training_stats["total_loss"]),
            "clip_fraction": np.mean(self.training_stats["clip_fraction"])
        }
        
        return stats
    
    def save(self, filepath: str):
        """保存模型"""
        save_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous
        }
        
        if self.ac_network is not None:
            save_dict["ac_network"] = self.ac_network.state_dict()
            save_dict["optimizer"] = self.optimizer.state_dict()
        else:
            save_dict["actor"] = self.actor.state_dict()
            save_dict["critic"] = self.critic.state_dict()
            save_dict["actor_optimizer"] = self.actor_optimizer.state_dict()
            save_dict["critic_optimizer"] = self.critic_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.ac_network is not None:
            self.ac_network.load_state_dict(checkpoint["ac_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        print(f"模型已从 {filepath} 加载")

