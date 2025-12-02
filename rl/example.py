"""
PPO 算法使用示例
演示如何训练 PPO 智能体
"""

import numpy as np
import torch
import sys
import os

# 添加父目录到路径，以便导入 rl 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium as gym
except ImportError:
    print("警告: 未安装 gymnasium，请安装: pip install gymnasium")
    gym = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("警告: 未安装 matplotlib，绘图功能将不可用")
    plt = None

from rl.ppo import PPO
from rl.replay_buffer import RolloutBuffer


def train_ppo_discrete(env_name: str = "CartPole-v1", max_episodes: int = 500):
    """
    训练离散动作空间的 PPO 智能体
    Args:
        env_name: 环境名称
        max_episodes: 最大训练回合数
    """
    if gym is None:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 创建 PPO 智能体
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
        hidden_dims=[64, 64],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    
    # 创建经验缓冲区
    buffer = RolloutBuffer(
        capacity=10000,
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False
    )
    
    # 训练循环
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # 收集一个 episode 的经验
        while not done:
            # 选择动作
            action, log_prob, value = ppo.select_action(state, deterministic=False)
            
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
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 每隔一定步数更新一次
        if buffer.size >= 1000 or episode % 10 == 0:
            stats = ppo.update(buffer)
            buffer.clear()
            
            if stats:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f}")
        
        # 定期评估
        if (episode + 1) % 50 == 0:
            eval_reward = evaluate_agent(ppo, env, num_episodes=5)
            print(f"评估 (Episode {episode+1}): 平均奖励 = {eval_reward:.2f}")
    
    env.close()
    
    # 绘制训练曲线
    plot_training_curve(episode_rewards, episode_lengths)
    
    return ppo, episode_rewards


def train_ppo_continuous(env_name: str = "Pendulum-v1", max_episodes: int = 500):
    """
    训练连续动作空间的 PPO 智能体
    Args:
        env_name: 环境名称
        max_episodes: 最大训练回合数
    """
    if gym is None:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作边界: [-{action_bound}, {action_bound}]")
    
    # 创建 PPO 智能体
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=True,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
        hidden_dims=[64, 64],
        action_bound=action_bound,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    
    # 创建经验缓冲区
    buffer = RolloutBuffer(
        capacity=10000,
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=True
    )
    
    # 训练循环
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # 收集一个 episode 的经验
        while not done:
            # 选择动作
            action, log_prob, value = ppo.select_action(state, deterministic=False)
            
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
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 每隔一定步数更新一次
        if buffer.size >= 1000 or episode % 10 == 0:
            stats = ppo.update(buffer)
            buffer.clear()
            
            if stats:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f}")
        
        # 定期评估
        if (episode + 1) % 50 == 0:
            eval_reward = evaluate_agent(ppo, env, num_episodes=5)
            print(f"评估 (Episode {episode+1}): 平均奖励 = {eval_reward:.2f}")
    
    env.close()
    
    # 绘制训练曲线
    plot_training_curve(episode_rewards, episode_lengths)
    
    return ppo, episode_rewards


def evaluate_agent(ppo: PPO, env: gym.Env, num_episodes: int = 10) -> float:
    """
    评估智能体性能
    Args:
        ppo: PPO 智能体
        env: 环境
        num_episodes: 评估回合数
    Returns:
        平均奖励
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = ppo.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def plot_training_curve(rewards: list, lengths: list):
    """
    绘制训练曲线
    Args:
        rewards: 奖励列表
        lengths: episode 长度列表
    """
    if plt is None:
        print("matplotlib 未安装，跳过绘图")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 奖励曲线
    axes[0].plot(rewards, alpha=0.6, label="Episode Reward")
    if len(rewards) > 20:
        # 移动平均
        window_size = 20
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0].plot(range(window_size-1, len(rewards)), moving_avg, label="Moving Average", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("训练奖励曲线")
    axes[0].legend()
    axes[0].grid(True)
    
    # Episode 长度曲线
    axes[1].plot(lengths, alpha=0.6, label="Episode Length")
    if len(lengths) > 20:
        window_size = 20
        moving_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        axes[1].plot(range(window_size-1, len(lengths)), moving_avg, label="Moving Average", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Length")
    axes[1].set_title("Episode 长度曲线")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("ppo_training_curve.png", dpi=150)
    print("训练曲线已保存到: ppo_training_curve.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PPO 算法训练示例")
    print("=" * 60)
    
    # 选择环境类型
    import sys
    
    if len(sys.argv) > 1:
        env_type = sys.argv[1]
    else:
        env_type = "discrete"  # 默认使用离散动作空间
    
    if env_type == "discrete":
        print("\n训练离散动作空间的 PPO 智能体 (CartPole-v1)")
        print("-" * 60)
        ppo, rewards = train_ppo_discrete(env_name="CartPole-v1", max_episodes=500)
    elif env_type == "continuous":
        print("\n训练连续动作空间的 PPO 智能体 (Pendulum-v1)")
        print("-" * 60)
        ppo, rewards = train_ppo_continuous(env_name="Pendulum-v1", max_episodes=500)
    else:
        print(f"未知的环境类型: {env_type}")
        print("使用方法: python example.py [discrete|continuous]")
    
    print("\n训练完成!")

