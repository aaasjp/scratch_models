"""
FastSpeech2 完整训练脚本
使用 HuggingFace LJSpeech 数据集

运行方式:
python fastspeech2_train.py --epochs 100 --batch_size 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json

# 导入自定义模块
from fastspeech2 import FastSpeech2, get_mask_from_lengths
from fastspeech2_dataset import LJSpeechDataset, collate_fn


def load_stats(processed_dir: str):
    """从 processed_dir 读取 stats.json 统计信息并返回字典。"""
    stats_path = os.path.join(processed_dir, 'stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"未找到统计文件: {stats_path}")
    with open(stats_path, 'r') as f:
        return json.load(f)


def clamp_and_standardize(x: torch.Tensor,
                          min_value: float,
                          max_value: float,
                          mean_value: float,
                          std_value: float,
                          eps: float = 1e-6) -> torch.Tensor:
    """按[min,max]裁剪后做标准化 (x - mean) / (std + eps)。"""
    x = torch.clamp(x, min=float(min_value), max=float(max_value))
    return (x - float(mean_value)) / (float(std_value) + eps)


def normalize_targets(mel_target: torch.Tensor,
                      duration_target: torch.Tensor,
                      pitch_target: torch.Tensor,
                      energy_target: torch.Tensor,
                      processed_dir: str = None,
                      stats: dict = None,
                      eps: float = 1e-6):
    """根据 stats.json 的 mean/min/max/std 对四个 target 执行裁剪+标准化归一化。

    参数:
        mel_target: [B, T_mel, n_mels] 或 [T_mel, n_mels]
        duration_target: [B, L_text] 或 [L_text]
        pitch_target: [B, L_text] 或 [L_text]
        energy_target: [B, L_text] 或 [L_text]
        processed_dir: 存放 stats.json 的目录（当 stats 未显式传入时必须提供）
        stats: 预先加载好的统计字典（可选），键包含 mel/pitch/energy/duration 的 min/max/mean/std
        eps: 数值稳定项

    返回:
        (mel_norm, duration_norm, pitch_norm, energy_norm)

    注意:
        目前训练中 duration 损失使用对数域目标；若改用本函数标准化后的 duration，需同步调整损失计算逻辑。
    """
    if stats is None:
        if processed_dir is None:
            raise ValueError("normalize_targets 需要提供 processed_dir 或者 stats 字典")
        stats = load_stats(processed_dir)

    mel_norm = clamp_and_standardize(
        mel_target, stats['mel_min'], stats['mel_max'], stats['mel_mean'], stats['mel_std'], eps
    )
    duration_norm = clamp_and_standardize(
        duration_target, stats['duration_min'], stats['duration_max'], stats['duration_mean'], stats['duration_std'], eps
    )
    pitch_norm = clamp_and_standardize(
        pitch_target, stats['pitch_min'], stats['pitch_max'], stats['pitch_mean'], stats['pitch_std'], eps
    )
    energy_norm = clamp_and_standardize(
        energy_target, stats['energy_min'], stats['energy_max'], stats['energy_mean'], stats['energy_std'], eps
    )

    return mel_norm, duration_norm, pitch_norm, energy_norm


class FastSpeech2Loss(nn.Module):
    """FastSpeech2 损失函数"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        
    def forward(self, predictions, targets, text_mask, mel_mask):
        """
        计算损失，正确处理 mask
        """
        '''
        print("============LOSS START============")
        print(f"predictions: {predictions}")
        print(f"targets: {targets}")
        print(f"text_mask: {text_mask}")
        print(f"mel_mask: {mel_mask}")
        print("============LOSS END============")
        '''


        mel_pred, duration_pred, pitch_pred, energy_pred = predictions
        mel_target, duration_target, pitch_target, energy_target = normalize_targets(
            targets[0], targets[1], targets[2], targets[3], processed_dir='./processed_data'
        )
        
        # 1. Mel Loss (MAE)
        mel_loss_unreduced = self.mae_loss(mel_pred, mel_target)
        mel_loss_masked = mel_loss_unreduced * mel_mask.unsqueeze(-1).float()
        mel_loss = mel_loss_masked.sum() / (mel_mask.sum() * mel_pred.size(-1) + 1e-6)
        
        # 2. Duration Loss (MSE)
        duration_loss_unreduced = self.mse_loss(duration_pred, duration_target)
        duration_loss_masked = duration_loss_unreduced * text_mask.float()
        duration_loss = duration_loss_masked.sum() / (text_mask.sum() + 1e-6)
        
        # 3. Pitch Loss (MSE)
        pitch_loss_unreduced = self.mse_loss(pitch_pred, pitch_target)
        pitch_loss_masked = pitch_loss_unreduced * text_mask.float()
        pitch_loss = pitch_loss_masked.sum() / (text_mask.sum() + 1e-6)
        
        # 4. Energy Loss (MSE)
        energy_loss_unreduced = self.mse_loss(energy_pred, energy_target)
        energy_loss_masked = energy_loss_unreduced * text_mask.float()
        energy_loss = energy_loss_masked.sum() / (text_mask.sum() + 1e-6)
        
        # 总损失
        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'duration_loss': duration_loss,
            'pitch_loss': pitch_loss,
            'energy_loss': energy_loss
        }


class Trainer:
    """训练器"""
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 损失函数
        self.criterion = FastSpeech2Loss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 记录
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        losses_dict = {
            'mel_loss': 0,
            'duration_loss': 0,
            'pitch_loss': 0,
            'energy_loss': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            #打印第一个batch的第一条数据的详细信息
            if batch_idx == 0:
                print("============BATCH START============")
                print(f"batch_idx: {batch_idx}")
                print(f"text: {batch['text'][0]}")
                print(f"text_lengths: {batch['text_lengths'][0]}")
                print(f"mel: {batch['mel'][0]}")
                print(f"mel_lengths: {batch['mel_lengths'][0]}")
                print(f"duration: {batch['duration'][0]}")
                print(f"pitch: {batch['pitch'][0]}")
                print(f"energy: {batch['energy'][0]}")
                print("============BATCH END============")
            # 数据移到设备
            text = batch['text'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            mel_target = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            duration_target = batch['duration'].to(self.device)
            pitch_target = batch['pitch'].to(self.device)
            energy_target = batch['energy'].to(self.device)
            
            # 生成 mask
            text_mask = get_mask_from_lengths(text_lengths, text.size(1))
            mel_mask_target = get_mask_from_lengths(mel_lengths, mel_target.size(1))
            
            # 前向传播
            mel_pred, mel_mask_pred, duration_pred, pitch_pred, energy_pred, _ = self.model(
                text=text,
                text_lengths=text_lengths,
                duration=duration_target,
                pitch=pitch_target,
                energy=energy_target,
                max_mel_len=mel_target.size(1)
            )
            
            # 计算损失
            predictions = (mel_pred, duration_pred, pitch_pred, energy_pred)
            targets = (mel_target, duration_target, pitch_target, energy_target)
            losses = self.criterion(predictions, targets, text_mask, mel_mask_target)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.global_step += 1
            
            # 记录损失
            total_loss += losses['total_loss'].item()
            for key in losses_dict.keys():
                losses_dict[key] += losses[key].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'mel': f"{losses['mel_loss'].item():.4f}",
                'dur': f"{losses['duration_loss'].item():.4f}",
                'pit': f"{losses['pitch_loss'].item():.4f}",
                'ene': f"{losses['energy_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_losses = {k: v / len(self.train_loader) for k, v in losses_dict.items()}
        
        return avg_loss, avg_losses
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        losses_dict = {
            'mel_loss': 0,
            'duration_loss': 0,
            'pitch_loss': 0,
            'energy_loss': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                text = batch['text'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                mel_target = batch['mel'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                duration_target = batch['duration'].to(self.device)
                pitch_target = batch['pitch'].to(self.device)
                energy_target = batch['energy'].to(self.device)
                
                text_mask = get_mask_from_lengths(text_lengths, text.size(1))
                mel_mask_target = get_mask_from_lengths(mel_lengths, mel_target.size(1))
                
                mel_pred, mel_mask_pred, duration_pred, pitch_pred, energy_pred, _ = self.model(
                    text=text,
                    text_lengths=text_lengths,
                    duration=duration_target,
                    pitch=pitch_target,
                    energy=energy_target,
                    max_mel_len=mel_target.size(1)
                )
                
                predictions = (mel_pred, duration_pred, pitch_pred, energy_pred)
                targets = (mel_target, duration_target, pitch_target, energy_target)
                losses = self.criterion(predictions, targets, text_mask, mel_mask_target)
                
                total_loss += losses['total_loss'].item()
                for key in losses_dict.keys():
                    losses_dict[key] += losses[key].item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_losses = {k: v / len(self.val_loader) for k, v in losses_dict.items()}
        
        return avg_loss, avg_losses
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'config': vars(self.config)
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f'\n✓ 保存最佳模型: {path} (val_loss: {val_loss:.4f})')
        
        # 定期保存
        if epoch % self.config.save_interval == 0:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
            print(f'✓ 保存检查点: {path}')
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print("开始训练 FastSpeech2")
        print(f"{'='*70}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"总 epochs: {self.config.num_epochs}")
        print(f"设备: {self.device}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f'\n{"="*70}')
            print(f'Epoch {epoch}/{self.config.num_epochs}')
            print(f'{"="*70}')
            
            # 训练
            train_loss, train_losses = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_losses = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 打印结果
            print(f'\n{"="*70}')
            print(f'训练结果:')
            print(f'  Total Loss:    {train_loss:.4f}')
            print(f'  Mel Loss:      {train_losses["mel_loss"]:.4f}')
            print(f'  Duration Loss: {train_losses["duration_loss"]:.4f}')
            print(f'  Pitch Loss:    {train_losses["pitch_loss"]:.4f}')
            print(f'  Energy Loss:   {train_losses["energy_loss"]:.4f}')
            
            print(f'\n验证结果:')
            print(f'  Total Loss:    {val_loss:.4f}')
            print(f'  Mel Loss:      {val_losses["mel_loss"]:.4f}')
            print(f'  Duration Loss: {val_losses["duration_loss"]:.4f}')
            print(f'  Pitch Loss:    {val_losses["pitch_loss"]:.4f}')
            print(f'  Energy Loss:   {val_losses["energy_loss"]:.4f}')
            print(f'{"="*70}')
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 保存训练日志
            log_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            log_file = os.path.join(self.config.log_dir, 'training_log.jsonl')
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        
        print(f'\n{"="*70}')
        print('训练完成！')
        print(f'最佳验证损失: {self.best_val_loss:.4f}')
        print(f'{"="*70}\n')


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Train FastSpeech2')
    
    # 数据参数
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='HuggingFace 数据集缓存目录（当前未使用）')
    parser.add_argument('--corpus_dir', type=str, default='./corpus',
                        help='原始语音与标注(.lab)所在目录')
    parser.add_argument('--aligned_dir', type=str, default='./corpus_aligned',
                        help='MFA 对齐后的 TextGrid 文件目录')
    parser.add_argument('--processed_dir', type=str, default='./processed_data',
                        help='预处理数据保存目录')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='最大训练样本数（用于快速测试）')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='最大验证样本数')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='强制重新预处理数据')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Transformer 层数')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 率')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    
    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的间隔')
    
    # 其他
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    config = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f'使用设备: {device}')
    
    # 加载数据集
    print("\n" + "="*70)
    print("加载数据集...")
    print("="*70)
    
    train_dataset = LJSpeechDataset(
        split='train',
        corpus_dir=config.corpus_dir,
        aligned_dir=config.aligned_dir,
        processed_dir=config.processed_dir,
        max_samples=config.max_train_samples,
        force_preprocess=False
    )
    
    val_dataset = LJSpeechDataset(
        split='validation',
        corpus_dir=config.corpus_dir,
        aligned_dir=config.aligned_dir,
        processed_dir=config.processed_dir,
        max_samples=config.max_val_samples,
        force_preprocess=False
    )
    
    # 打印一共多少条训练数据和验证数据
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    print("\n" + "="*70)
    print("创建模型...")
    print("="*70)
    
    vocab_size = len(train_dataset.tokenizer)
    model = FastSpeech2(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        n_mel_channels=80,
        stats_path=config.processed_dir+'/stats.json'
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"词表大小: {vocab_size}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # 从检查点恢复（如果指定）
    if config.resume:
        print(f"\n从检查点恢复: {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint['global_step']
        print(f"✓ 从 epoch {checkpoint['epoch']} 恢复训练")
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()