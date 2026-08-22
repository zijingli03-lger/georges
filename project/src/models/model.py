"""
模型定义文件
支持多种预训练模型用于二分类任务
"""

import torch
import torch.nn as nn
from torchvision import models


class BinaryClassifier(nn.Module):
    """二分类模型基类"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # 加载预训练模型
        self.model, num_features = self._get_backbone(backbone, pretrained)
        
        # 替换分类头
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def _get_backbone(self, backbone_name, pretrained):
        """获取预训练骨干网络"""
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        models_dict = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048),
            'resnet101': (models.resnet101, 2048),
            'resnet152': (models.resnet152, 2048),
            'efficientnet_b0': (models.efficientnet_b0, 1280),
            'efficientnet_b1': (models.efficientnet_b1, 1280),
            'efficientnet_b2': (models.efficientnet_b2, 1408),
            'mobilenet_v2': (models.mobilenet_v2, 1280),
            'vit_base': (models.vit_b_16, 768),
        }
        
        if backbone_name not in models_dict:
            raise ValueError(f"不支持的模型: {backbone_name}")
        
        model_fn, num_features = models_dict[backbone_name]
        
        if pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        
        return model, num_features
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """冻结骨干网络，只训练分类头"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻分类头
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True


def create_model(backbone='resnet50', pretrained=True, num_classes=2, dropout_rate=0.3):
    """
    创建模型的便捷函数
    
    Args:
        backbone: 骨干网络名称
        pretrained: 是否使用预训练权重
        num_classes: 分类数量
        dropout_rate: Dropout 比率
    
    Returns:
        model: BinaryClassifier 实例
    """
    return BinaryClassifier(backbone, pretrained, num_classes, dropout_rate)


def get_available_models():
    """获取可用的模型列表"""
    return [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'mobilenet_v2', 'vit_base'
    ]


if __name__ == '__main__':
    # 测试代码
    print("可用模型列表:", get_available_models())
    print("\n测试创建模型...")
    
    # 测试 MobileNetV2（轻量，适合快速验证）
    model = create_model('mobilenet_v2', pretrained=True)
    print(f"MobileNetV2 创建成功")
    
    # 测试 ResNet50
    model = create_model('resnet50', pretrained=True)
    print(f"ResNet50 创建成功")
    
    # 测试冻结
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"冻结后可训练参数: {trainable}/{total}")
    
    # 测试解冻
    model.unfreeze_all()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"全部解冻后可训练参数: {trainable}/{total}")