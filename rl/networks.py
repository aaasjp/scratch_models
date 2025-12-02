"""
Actor-Critic 网络结构
用于 PPO 算法的策略网络和价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Actor(nn.Module):
    """
    策略网络（Actor）
    输出动作的概率分布
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        continuous: bool = False,
        action_bound: float = 1.0
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数类型 ("tanh", "relu", "elu")
            continuous: 是否为连续动作空间
            action_bound: 连续动作的边界值（仅在 continuous=True 时使用）
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bound = action_bound
        
        # 选择激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出层
        if continuous:
            # 连续动作空间：输出均值和标准差
            self.mean_layer = nn.Linear(prev_dim, action_dim)
            self.log_std_layer = nn.Linear(prev_dim, action_dim)
            # 初始化
            nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
            self.mean_layer.bias.data.zero_()
            self.log_std_layer.bias.data.zero_()
        else:
            # 离散动作空间：输出动作概率分布
            self.action_head = nn.Linear(prev_dim, action_dim)
            nn.init.orthogonal_(self.action_head.weight, gain=0.01)
            self.action_head.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
        Returns:
            包含动作分布信息的字典
        """
        # 特征提取
        features = state
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                features = layer(features)
                features = self.activation(features)
            else:
                features = layer(features)
        
        if self.continuous:
            # 连续动作空间
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            # 限制 log_std 的范围，保证数值稳定性
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            
            return {
                "mean": mean,
                "std": std,
                "log_std": log_std
            }
        else:
            # 离散动作空间
            logits = self.action_head(features)
            probs = F.softmax(logits, dim=-1)
            
            return {
                "logits": logits,
                "probs": probs
            }
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从策略分布中采样动作
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
            deterministic: 是否使用确定性策略（连续动作时取均值，离散动作时取最大概率）
        Returns:
            action: 采样得到的动作
            log_prob: 动作的对数概率
        """
        dist_info = self.forward(state)
        
        if self.continuous:
            mean = dist_info["mean"]
            std = dist_info["std"]
            log_std = dist_info["log_std"]
            
            if deterministic:
                action = torch.tanh(mean) * self.action_bound
                # 确定性动作的 log_prob 需要特殊处理
                log_prob = torch.zeros_like(mean[:, 0] if len(mean.shape) > 1 else mean[0])
            else:
                # 从正态分布采样
                normal_dist = torch.distributions.Normal(mean, std)
                action_raw = normal_dist.sample()
                # 使用 tanh 将动作限制到 [-action_bound, action_bound]
                action = torch.tanh(action_raw) * self.action_bound
                
                # 计算 log_prob（考虑 tanh 变换）
                log_prob = normal_dist.log_prob(action_raw).sum(dim=-1)
                log_prob -= torch.log(1 - action.pow(2) / (self.action_bound ** 2 + 1e-6)).sum(dim=-1)
            
            return action, log_prob
        else:
            logits = dist_info["logits"]
            probs = dist_info["probs"]
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            log_prob = F.log_softmax(logits, dim=-1)
            log_prob = log_prob.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
            
            return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态和动作的对数概率和熵
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim] (连续) 或 [batch_size] (离散)
        Returns:
            log_prob: 动作的对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
            dist_info: 分布信息字典
        """
        dist_info = self.forward(state)
        
        if self.continuous:
            mean = dist_info["mean"]
            std = dist_info["std"]
            
            # 将动作从 [-action_bound, action_bound] 映射回原始空间
            action_raw = torch.atanh(torch.clamp(action / self.action_bound, -0.999, 0.999))
            
            normal_dist = torch.distributions.Normal(mean, std)
            log_prob = normal_dist.log_prob(action_raw).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) / (self.action_bound ** 2 + 1e-6)).sum(dim=-1)
            entropy = normal_dist.entropy().sum(dim=-1)
        else:
            logits = dist_info["logits"]
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return log_prob, entropy, dist_info


class Critic(nn.Module):
    """
    价值网络（Critic）
    估计状态价值函数 V(s)
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh"
    ):
        """
        Args:
            state_dim: 状态空间维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数类型 ("tanh", "relu", "elu")
        """
        super().__init__()
        self.state_dim = state_dim
        
        # 选择激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出层：输出状态价值
        self.value_head = nn.Linear(prev_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        self.value_head.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
        Returns:
            value: 状态价值 [batch_size, 1] 或 [1]
        """
        features = state
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                features = layer(features)
                features = self.activation(features)
            else:
                features = layer(features)
        
        value = self.value_head(features)
        return value


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络（共享特征提取层）
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        continuous: bool = False,
        action_bound: float = 1.0,
        shared_layers: int = 1
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数类型
            continuous: 是否为连续动作空间
            action_bound: 连续动作的边界值
            shared_layers: 共享的层数（从第一层开始）
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bound = action_bound
        
        # 选择激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 共享特征提取层
        shared_dims = hidden_dims[:shared_layers]
        shared_layers_list = []
        prev_dim = state_dim
        for hidden_dim in shared_dims:
            shared_layers_list.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers_list.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers_list) if shared_layers_list else nn.Identity()
        
        # Actor 特有层
        actor_dims = hidden_dims[shared_layers:]
        actor_layers = []
        for hidden_dim in actor_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.actor_feature_layers = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()
        
        # Critic 特有层
        critic_dims = hidden_dims[shared_layers:]
        critic_layers = []
        prev_dim_for_critic = prev_dim if shared_layers > 0 else state_dim
        for hidden_dim in critic_dims:
            critic_layers.append(nn.Linear(prev_dim_for_critic, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim_for_critic = hidden_dim
        
        self.critic_feature_layers = nn.Sequential(*critic_layers) if critic_layers else nn.Identity()
        
        # 输出层
        actor_output_dim = prev_dim if shared_layers > 0 else state_dim
        for dim in actor_dims:
            actor_output_dim = dim
        
        if continuous:
            self.actor_mean = nn.Linear(actor_output_dim, action_dim)
            self.actor_log_std = nn.Linear(actor_output_dim, action_dim)
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
            self.actor_mean.bias.data.zero_()
            self.actor_log_std.bias.data.zero_()
        else:
            self.actor_head = nn.Linear(actor_output_dim, action_dim)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            self.actor_head.bias.data.zero_()
        
        critic_output_dim = prev_dim_for_critic if shared_layers > 0 else state_dim
        for dim in critic_dims:
            critic_output_dim = dim
        
        self.critic_head = nn.Linear(critic_output_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        self.critic_head.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        前向传播
        Returns:
            actor_output: Actor 输出字典
            value: Critic 输出的状态价值
        """
        # 共享特征
        shared_features = state
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                shared_features = layer(shared_features)
                shared_features = self.activation(shared_features)
            elif not isinstance(layer, nn.Identity):
                shared_features = layer(shared_features)
        
        # Actor 分支
        actor_features = shared_features
        for layer in self.actor_feature_layers:
            if isinstance(layer, nn.Linear):
                actor_features = layer(actor_features)
                actor_features = self.activation(actor_features)
            elif not isinstance(layer, nn.Identity):
                actor_features = layer(actor_features)
        
        if self.continuous:
            mean = self.actor_mean(actor_features)
            log_std = self.actor_log_std(actor_features)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            actor_output = {"mean": mean, "std": std, "log_std": log_std}
        else:
            logits = self.actor_head(actor_features)
            probs = F.softmax(logits, dim=-1)
            actor_output = {"logits": logits, "probs": probs}
        
        # Critic 分支
        critic_features = shared_features
        for layer in self.critic_feature_layers:
            if isinstance(layer, nn.Linear):
                critic_features = layer(critic_features)
                critic_features = self.activation(critic_features)
            elif not isinstance(layer, nn.Identity):
                critic_features = layer(critic_features)
        
        value = self.critic_head(critic_features)
        
        return actor_output, value

