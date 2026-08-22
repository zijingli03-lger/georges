"""
训练主脚本
类似 YOLO 的一行命令训练入口

使用示例:
    # 快速测试（2个epoch）
    python train.py --epochs 2 --backbone mobilenet_v2
    
    # 完整训练
    python train.py --epochs 10 --backbone resnet50
    
    # 断点续训
    python train.py --resume checkpoint.pth
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Windows 控制台默认 GBK 编码无法打印 emoji（如 ✨），强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import create_model
from data.dataset import create_dataloaders


def set_seed(seed):
    """设置所有随机源，保证训练可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """训练器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据路径
        data_dir = Path(args.data_dir)
        
        # 创建 DataLoader
        print("=" * 60)
        print("初始化训练环境")
        print("=" * 60)
        
        print("\n[1/3] 加载数据...")
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = \
            create_dataloaders(data_dir, args.batch_size, args.num_workers, args.img_size)
        
        # 创建模型
        print(f"\n[2/3] 创建模型: {args.backbone}")
        self.model = create_model(args.backbone, pretrained=True, 
                                  num_classes=2, dropout_rate=args.dropout)
        self.model = self.model.to(self.device)
        
        # 冻结策略
        if args.freeze_backbone:
            print("  冻结骨干网络，只训练分类头")
            self.model.freeze_backbone()
        else:
            print("  全模型训练")
        
        # 损失函数（带类别权重）
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        
        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_precision': [], 'val_recall': [],
            'lr': []
        }
        
        self.start_epoch = 0
        self.best_val_acc = 0
        
        # 断点续训
        if args.resume:
            self._load_checkpoint(args.resume)
        
        print(f"\n[3/3] 训练配置:")
        print(f"  设备: {self.device}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Learning Rate: {args.lr}")
        print(f"  开始 Epoch: {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Train {epoch+1}/{self.args.epochs}', 
             leave=False, ncols=100)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += images.size(0)
            
            # 更新进度条
            batch_acc = correct / total
            batch_loss = total_loss / total
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch=None):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        desc = 'Val' if epoch is None else f'Val {epoch+1}/{self.args.epochs}'
        pbar = tqdm(self.val_loader, desc=desc, leave=False, ncols=100)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            batch_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
            batch_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        avg_loss = total_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1, precision, recall
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_f1, val_precision, val_recall = self.validate(epoch)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 更新学习率
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # 打印日志
            print(f"\nEpoch {epoch+1}/{self.args.epochs}:")
            print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"  [Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(f"  [LR]    {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  [Time]  {epoch_time:.1f}s")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ✨ 新的最佳模型! (Val Acc: {val_acc:.4f})")
            
            # 定期保存断点
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"训练完成!")
        print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"最佳模型保存在: {self.output_dir / 'best_model.pth'}")
        print("=" * 60)
        
        # 保存训练历史
        self._save_history()
        
        # 绘制训练曲线
        self._plot_curves()
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'args': vars(self.args)
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def _load_checkpoint(self, path):
        """加载检查点"""
        if not os.path.exists(path):
            print(f"警告: 检查点文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"✅ 加载检查点: {path}")
        print(f"   从 Epoch {self.start_epoch} 继续训练")
        print(f"   最佳 Val Acc: {self.best_val_acc:.4f}")
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n训练历史已保存: {history_path}")
    
    def _plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss 曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy 曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 曲线
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].plot(self.history['val_precision'], label='Precision', color='blue')
        axes[1, 0].plot(self.history['val_recall'], label='Recall', color='orange')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(self.history['lr'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='图像分类训练脚本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'data' / 'splits'),
                        help='数据划分文件目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='目标图像尺寸')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                                 'efficientnet_b0', 'mobilenet_v2', 'vit_base'],
                        help='骨干网络')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比率')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='冻结骨干网络')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default=str(project_root / 'results'),
                        help='输出目录')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='保存检查点间隔')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（保证可复现）')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()