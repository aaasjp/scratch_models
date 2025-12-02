"""
经验回放缓冲区
用于存储和采样 PPO 训练数据
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class RolloutBuffer:
    """
    用于 PPO 算法的经验回放缓冲区
    存储一个 episode 或一段轨迹的数据
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int, continuous: bool = False):
        """
        Args:
            capacity: 缓冲区容量
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            continuous: 是否为连续动作空间
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.ptr = 0
        self.size = 0
        
        # 存储数据
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        if continuous:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        存储一步经验
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作的对数概率
            done: 是否结束
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_advantages_and_returns(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        计算优势函数和回报
        使用 GAE (Generalized Advantage Estimation)
        Args:
            last_value: 最后一个状态的价值估计
            gamma: 折扣因子
            gae_lambda: GAE 的 lambda 参数
        """
        # 初始化
        advantages = np.zeros_like(self.rewards)
        last_gae = 0
        
        # 从后向前计算
        for step in reversed(range(self.size)):
            if self.dones[step]:
                # episode 结束，下一个状态价值为 0
                next_value = 0.0
            else:
                if step == self.size - 1:
                    next_value = last_value
                else:
                    next_value = self.values[step + 1]
            
            # TD 误差
            delta = self.rewards[step] + gamma * next_value - self.values[step]
            
            # GAE
            advantages[step] = last_gae = delta + gamma * gae_lambda * last_gae
            
            # 计算回报
            self.returns[step] = advantages[step] + self.values[step]
        
        self.advantages = advantages
        # 标准化优势函数（可选，但在实践中常用）
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        获取一个批次的数据
        Args:
            batch_size: 批次大小，如果为 None 则返回所有数据
        Returns:
            包含所有数据的字典
        """
        if batch_size is None:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            "states": torch.FloatTensor(self.states[indices]),
            "actions": torch.FloatTensor(self.actions[indices]) if self.continuous else torch.LongTensor(self.actions[indices]),
            "old_log_probs": torch.FloatTensor(self.log_probs[indices]),
            "advantages": torch.FloatTensor(self.advantages[indices]),
            "returns": torch.FloatTensor(self.returns[indices])
        }
        
        return batch
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size


class PPOBuffer:
    """
    支持批量处理的 PPO 缓冲区
    可以存储多个并行的经验流
    """
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            num_envs: 并行环境数量
            num_steps: 每个环境的步数
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            continuous: 是否为连续动作空间
            device: 计算设备
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.device = device
        self.ptr = 0
        
        # 存储数据
        self.states = torch.zeros((num_steps, num_envs, state_dim), dtype=torch.float32, device=device)
        if continuous:
            self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        else:
            self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.advantages = None
        self.returns = None
    
    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        存储一步经验（批量）
        Args:
            states: [num_envs, state_dim]
            actions: [num_envs, action_dim] 或 [num_envs]
            rewards: [num_envs]
            values: [num_envs]
            log_probs: [num_envs]
            dones: [num_envs]
        """
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones
        self.ptr += 1
    
    def compute_advantages_and_returns(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        计算优势函数和回报（批量）
        Args:
            last_values: [num_envs] 最后一个状态的价值估计
            gamma: 折扣因子
            gae_lambda: GAE 的 lambda 参数
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            
            # 处理 done 的情况
            next_values = next_values * (1.0 - self.dones[step].float())
            
            # TD 误差
            delta = self.rewards[step] + gamma * next_values - self.values[step]
            
            # GAE
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[step] = last_gae
        
        # 计算回报
        self.returns = advantages + self.values
        
        # 标准化优势函数
        advantages_mean = advantages.mean()
        advantages_std = advantages.std() + 1e-8
        self.advantages = (advantages - advantages_mean) / advantages_std
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据（扁平化）
        Returns:
            包含所有数据的字典
        """
        # 扁平化：[num_steps * num_envs, ...]
        batch = {
            "states": self.states.view(-1, self.state_dim),
            "actions": self.actions.view(-1, self.action_dim if self.continuous else -1),
            "old_log_probs": self.log_probs.view(-1),
            "advantages": self.advantages.view(-1),
            "returns": self.returns.view(-1)
        }
        return batch
    
    def get_minibatch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        获取一个 mini-batch
        Args:
            batch_size: 批次大小
        Returns:
            包含批次数据的字典
        """
        total_size = self.num_steps * self.num_envs
        indices = torch.randperm(total_size, device=self.device)[:batch_size]
        
        batch = {
            "states": self.states.view(-1, self.state_dim)[indices],
            "actions": self.actions.view(-1, self.action_dim if self.continuous else -1)[indices],
            "old_log_probs": self.log_probs.view(-1)[indices],
            "advantages": self.advantages.view(-1)[indices],
            "returns": self.returns.view(-1)[indices]
        }
        return batch
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0

