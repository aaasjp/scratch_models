# PPO (Proximal Policy Optimization) 算法实现

本项目实现了从零开始的 PPO 强化学习算法，支持离散和连续动作空间。

## 文件结构

```
rl/
├── __init__.py          # 模块初始化
├── networks.py          # Actor-Critic 网络结构
├── replay_buffer.py     # 经验回放缓冲区
├── ppo.py              # PPO 核心算法实现
├── example.py          # 使用示例
└── README.md           # 说明文档
```

## 主要特性

- ✅ 支持离散和连续动作空间
- ✅ 实现 GAE (Generalized Advantage Estimation)
- ✅ PPO-Clip 算法
- ✅ 支持分离的 Actor-Critic 和共享网络两种架构
- ✅ 完整的训练统计和监控
- ✅ 模型保存和加载功能

## 快速开始

### 基本使用

```python
import numpy as np
import gymnasium as gym
from rl import PPO, RolloutBuffer

# 创建环境
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建 PPO 智能体
ppo = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=False,  # 离散动作空间
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 创建经验缓冲区
buffer = RolloutBuffer(
    capacity=10000,
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=False
)

# 训练循环
for episode in range(1000):
    state, _ = env.reset()
    done = False
    
    while not done:
        # 选择动作
        action, log_prob, value = ppo.select_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        buffer.store(
            state=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )
        
        state = next_state
    
    # 更新策略
    if buffer.size >= 1000:
        stats = ppo.update(buffer)
        buffer.clear()
```

### 运行示例

```bash
# 训练离散动作空间的 PPO（CartPole-v1）
python rl/example.py discrete

# 训练连续动作空间的 PPO（Pendulum-v1）
python rl/example.py continuous
```

## API 文档

### PPO 类

#### 初始化参数

- `state_dim` (int): 状态空间维度
- `action_dim` (int): 动作空间维度
- `continuous` (bool): 是否为连续动作空间，默认 `False`
- `lr_actor` (float): Actor 学习率，默认 `3e-4`
- `lr_critic` (float): Critic 学习率，默认 `3e-4`
- `gamma` (float): 折扣因子，默认 `0.99`
- `gae_lambda` (float): GAE lambda 参数，默认 `0.95`
- `clip_epsilon` (float): PPO clip 参数，默认 `0.2`
- `entropy_coef` (float): 熵正则化系数，默认 `0.01`
- `value_coef` (float): 价值损失系数，默认 `0.5`
- `max_grad_norm` (float): 梯度裁剪的最大范数，默认 `0.5`
- `update_epochs` (int): 每次更新的轮数，默认 `10`
- `batch_size` (int): 批次大小，默认 `64`
- `hidden_dims` (list): 隐藏层维度列表，默认 `[64, 64]`
- `activation` (str): 激活函数 ("tanh", "relu", "elu")，默认 `"tanh"`
- `action_bound` (float): 连续动作的边界值，默认 `1.0`
- `use_shared_network` (bool): 是否使用共享网络，默认 `False`
- `device` (str): 计算设备，默认 `"cpu"`

#### 主要方法

##### `select_action(state, deterministic=False)`

选择动作。

**参数：**
- `state` (np.ndarray): 当前状态
- `deterministic` (bool): 是否使用确定性策略

**返回：**
- `action`: 选择的动作
- `log_prob`: 动作的对数概率
- `value`: 状态价值估计

##### `update(buffer)`

更新策略和价值网络。

**参数：**
- `buffer` (RolloutBuffer): 经验回放缓冲区

**返回：**
- `stats` (dict): 训练统计信息（policy_loss, value_loss, entropy_loss, total_loss, clip_fraction）

##### `save(filepath)`

保存模型到文件。

##### `load(filepath)`

从文件加载模型。

### RolloutBuffer 类

#### 初始化参数

- `capacity` (int): 缓冲区容量
- `state_dim` (int): 状态空间维度
- `action_dim` (int): 动作空间维度
- `continuous` (bool): 是否为连续动作空间

#### 主要方法

##### `store(state, action, reward, value, log_prob, done)`

存储一步经验。

##### `compute_advantages_and_returns(last_value, gamma, gae_lambda)`

计算优势函数和回报（使用 GAE）。

##### `get_batch(batch_size=None)`

获取一个批次的数据。

##### `clear()`

清空缓冲区。

## 算法说明

### PPO-Clip

PPO 使用 clipped surrogate objective 来更新策略：

```
L^CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```

其中：
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` 是重要性采样比率
- `A` 是优势函数估计
- `ε` 是 clip 参数（通常为 0.2）

### GAE (Generalized Advantage Estimation)

使用 GAE 来估计优势函数，减少方差：

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = δ_t + (γ * λ) * δ_{t+1} + (γ * λ)^2 * δ_{t+2} + ...
```

### 网络结构

#### Actor 网络（策略网络）
- 输入：状态 `s`
- 输出：
  - 离散：动作概率分布 `π(a|s)`
  - 连续：动作分布的均值和标准差 `μ(s), σ(s)`

#### Critic 网络（价值网络）
- 输入：状态 `s`
- 输出：状态价值估计 `V(s)`

## 示例结果

训练完成后会生成：
- 训练奖励曲线图 (`ppo_training_curve.png`)
- 控制台输出的训练统计信息

## 依赖

- PyTorch
- NumPy
- Gymnasium (可选，用于示例)
- Matplotlib (可选，用于可视化)

## 参考文献

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

