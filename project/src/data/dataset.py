"""
数据集类和 DataLoader 构建
用途：加载图像数据，应用数据增强，创建 PyTorch DataLoader
"""

import os
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BinaryClassificationDataset(Dataset):
    """二分类图像数据集"""
    
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: CSV 文件路径，包含 image_path, label, label_code
            transform: 图像变换
        """
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'image_path': row['image_path'],
                    'label': row['label'],
                    'label_code': int(row['label_code'])
                })
        
        self.transform = transform
        
        # 统计类别分布
        self.class_counts = {}
        for item in self.data:
            label = item['label']
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label_code']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load {image_path}: {e}")
            # 返回黑色占位图，但保留原始标签，避免静默污染数据
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """计算类别权重（用于处理不均衡）"""
        total = len(self.data)
        weights = []
        for label_code in [0, 1]:
            count = sum(1 for item in self.data if item['label_code'] == label_code)
            if count > 0:
                weights.append(total / (2 * count))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)


def get_transforms(train=True, img_size=224):
    """
    获取数据变换
    
    Args:
        train: 是否为训练模式（使用数据增强）
        img_size: 目标图像尺寸
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            # 注：不使用 RandomVerticalFlip——圣乔治题材多为画作/雕塑，垂直翻转生成
            # 上下颠倒的图像，不符合真实场景，可能让模型学到错误的对称性
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),  # 256 for 224 target
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(data_dir, batch_size=32, num_workers=2, img_size=224):
    """
    创建训练、验证、测试的 DataLoader
    
    Args:
        data_dir: 包含 train.csv, val.csv, test.csv 的目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        img_size: 目标图像尺寸
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    data_dir = Path(data_dir)
    
    # 创建数据集
    train_dataset = BinaryClassificationDataset(
        data_dir / 'train.csv',
        transform=get_transforms(train=True, img_size=img_size)
    )
    
    val_dataset = BinaryClassificationDataset(
        data_dir / 'val.csv',
        transform=get_transforms(train=False, img_size=img_size)
    )
    
    test_dataset = BinaryClassificationDataset(
        data_dir / 'test.csv',
        transform=get_transforms(train=False, img_size=img_size)
    )
    
    # 获取类别权重
    class_weights = train_dataset.get_class_weights()
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃最后一个不完整的 batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 打印信息
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")
    print(f"类别权重: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == '__main__':
    # 测试代码
    data_dir = Path(r'f:\bifu\project\data\splits')
    
    if data_dir.exists():
        train_loader, val_loader, test_loader, class_weights = create_dataloaders(
            data_dir,
            batch_size=32,
            num_workers=2,
            img_size=224
        )
        
        # 测试一个 batch
        images, labels = next(iter(train_loader))
        print(f"\n测试 batch:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签值: {labels.unique()}")
    else:
        print(f"数据目录不存在: {data_dir}")
        print("请先运行 split_data.py 进行数据划分")