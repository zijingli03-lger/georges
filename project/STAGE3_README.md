# 阶段 3：Baseline 模型训练指南

## 概述

本阶段将完成图像分类模型的训练、评估和优化。

## ⭐ 阶段 3 已完成

### 完成时间
2026-08-22 18:58

### 最终结果
| 指标 | 数值 |
|------|------|
| 测试集 Accuracy | **92.98%** |
| 测试集 F1 Score | **92.79%** |
| 测试集 AUC | **97.61%** |
| 最佳验证 Accuracy | 92.28% |

### 输出文件
- `project/results/best_model.pth` - 最佳模型
- `project/results/evaluation_report.json` - 评估报告
- `project/results/confusion_matrix.png` - 混淆矩阵
- `project/results/roc_curve.png` - ROC 曲线

### 可选的后续优化
如需进一步提升，参考文档末尾的"进阶选项"部分。

---

---

## 环境准备

### 1. 激活 Conda 环境

```bash
# 方法一：使用 conda activate（需要先初始化 conda）
conda activate f:\bifu\.conda

# 方法二：直接使用 conda run（推荐，无需激活）
conda run -p f:\bifu\.conda <command>
```

### 2. 验证环境

```bash
conda run -p f:\bifu\.conda python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**预期输出**：
```
CUDA: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

---

## 训练步骤

### Step 1：快速验证（ResNet18，2 epoch）

**目的**：确认代码流程正确，数据加载没问题

```bash
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 2 --backbone resnet18
```

**预期耗时**：约 2-5 分钟

**成功标志**：
- 输出显示 `Epoch 1/2` 和 `Epoch 2/2`
- 生成 `project/results/best_model.pth`
- 生成 `project/results/training_curves.png`

**如果失败**：
- 检查错误信息
- 参考本文档末尾的"常见问题"部分

---

### Step 2：正式训练（ResNet50，10 epoch）

**目的**：获得较好的 baseline 效果

```bash
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet50
```

**预期耗时**：约 10-20 分钟

**参数说明**：
- `--epochs 10`：训练 10 轮
- `--backbone resnet50`：使用 ResNet50 骨干网络
- 其他参数使用默认值即可

---

### Step 3：评估模型

**训练完成后**，运行评估脚本：

```bash
conda run -p f:\bifu\.conda python f:\bifu\project\src\eval.py --model_path f:\bifu\project\results\best_model.pth
```

**生成的文件**：
- `project/results/evaluation_report.json`：详细指标
- `project/results/confusion_matrix.png`：混淆矩阵
- `project/results/roc_curve.png`：ROC 曲线

---

### Step 4：查看结果

检查以下文件：

```bash
# 查看训练历史
cat f:\bifu\project\results\training_history.json

# 查看评估报告
cat f:\bifu\project\results\evaluation_report.json
```

或者直接查看图片：
- `f:\bifu\project\results\training_curves.png`
- `f:\bifu\project\results\confusion_matrix.png`
- `f:\bifu\project\results\roc_curve.png`

---

## 进阶选项

### 选项 A：断点续训

如果训练中断，继续训练：

```bash
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --resume f:\bifu\project\results\checkpoint_epoch_5.pth
```

### 选项 B：调整超参数

```bash
# 修改学习率
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet50 --lr 0.001

# 修改 batch size
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet50 --batch_size 16

# 冻结骨干网络（只训练分类头）
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet50 --freeze_backbone
```

### 选项 C：尝试其他模型

```bash
# ResNet18（更快）
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet18

# ResNet101（更大，效果可能更好）
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet101

# EfficientNet B0（高效）
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone efficientnet_b0
```

---

## 文件结构

```
f:\bifu\
├── project\
│   ├── src\
│   │   ├── models\
│   │   │   └── model.py              # 模型定义
│   │   ├── data\
│   │   │   ├── split_data.py         # 数据划分
│   │   │   └── dataset.py            # Dataset + DataLoader
│   │   ├── train.py                  # 训练脚本 ⭐
│   │   └── eval.py                   # 评估脚本
│   ├── data\
│   │   └── splits\
│   │       ├── train.csv             # 训练集索引
│   │       ├── val.csv               # 验证集索引
│   │       └── test.csv              # 测试集索引
│   └── results\
│       ├── best_model.pth            # 最佳模型（训练后生成）
│       ├── training_history.json     # 训练历史（训练后生成）
│       └── ...
└── ...
```

---

## 常见问题

### Q1: 训练命令没反应？

**原因**：Windows 的 Python 输出缓冲问题

**解决**：在命令后加 `-u` 参数禁用缓冲
```bash
conda run -p f:\bifu\.conda python -u f:\bifu\project\src\train.py --epochs 2 --backbone resnet18
```

### Q2: CUDA OOM（显存不足）？

**解决**：减小 batch size
```bash
conda run -p f:\bifu\.conda python f:\bifu\project\src\train.py --epochs 10 --backbone resnet50 --batch_size 8
```

### Q3: 预训练模型下载慢？

**解决**：手动下载到缓存目录
```bash
# ResNet50 直接下载链接
# 下载到: C:\Users\Lenovo\.cache\torch\hub\checkpoints\
```

### Q4: 想停掉训练？

**方法**：按 `Ctrl + C`

已保存的检查点不会丢失，可以用 `--resume` 继续。

---

## 预期输出示例

```
============================================================
初始化训练环境
============================================================

[1/3] 加载数据...
训练集: 3990 样本, 124 batches
验证集: 855 样本, 27 batches
测试集: 855 样本, 27 batches
类别权重: tensor([0.8533, 1.2076])

[2/3] 创建模型: resnet18
  全模型训练

[3/3] 训练配置:
  设备: cuda
  Epochs: 2
  Batch Size: 32
  Learning Rate: 0.0001
  开始 Epoch: 0

============================================================
开始训练
============================================================

Epoch 1/2:
  [Train] Loss: 0.5234 | Acc: 0.8234
  [Val]   Loss: 0.4123 | Acc: 0.8712 | F1: 0.8654
  [LR]    0.000100
  [Time]  15.2s
  ✨ 新的最佳模型! (Val Acc: 0.8712)

Epoch 2/2:
  [Train] Loss: 0.3876 | Acc: 0.8921
  [Val]   Loss: 0.3567 | Acc: 0.8956 | F1: 0.8892
  [LR]    0.000010
  [Time]  14.8s
  ✨ 新的最佳模型! (Val Acc: 0.8956)

============================================================
训练完成!
总耗时: 30.5s (0.5min)
最佳验证准确率: 0.8956
最佳模型保存在: f:\bifu\project\results\best_model.pth
============================================================
```

---

## 下一步

完成训练后，将结果反馈给 AI 助手进行分析和优化。