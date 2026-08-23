# 圣乔治图像检测系统

基于深度学习的图像分类项目，用于检测图像中是否包含"圣乔治"。

## 项目概述

本项目使用 ResNet50 预训练模型进行迁移学习，实现二分类任务：
- **正类 (georges)**: 包含圣乔治的图像
- **负类 (non_georges)**: 不包含圣乔治的图像

### 性能指标

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 92.98% |
| F1 Score | 0.9279 |
| AUC | 0.9761 |

### 数据获取

原始图像数据集（`georges.zip` / `non_georges.zip`）因体积较大（约 550 MB），不包含在本仓库中。
请按以下任一方式获取：

1. 联系作者（微信/邮件）获取压缩包
2. 从 GitHub Releases 页面下载（如有上传）
3. 自行准备数据：正类图片放入 `georges/`，负类图片放入 `non_georges/`

获取后将两个压缩包放到仓库根目录解压，或直接放置图片文件夹，再运行数据划分脚本即可。

> 模型权重 `best_model.pth`（281 MB）因超过 GitHub 单文件 100MB 限制，不直接包含在仓库中，
> 请从 **GitHub Releases 页面**下载，或联系作者获取。下载后放到 `project/results/best_model.pth` 即可运行评估。

---

## 环境配置

### 系统要求
- Python 3.10+
- CUDA 12.1 (GPU 版本，可选——没有 GPU 也能用 CPU 运行，只是训练更慢)
- 建议 GPU 显存 >= 4GB

### 安装依赖（一键）

```bash
# 1. (可选但推荐) 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. 一键安装全部依赖（默认安装 PyTorch CPU 版，无 GPU 也能运行）
pip install -r requirements.txt
```

> 💡 需要 GPU 加速训练？先执行下面这行，再执行上面的 `pip install -r requirements.txt`：
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
> ```

### 验证环境

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

---

## 项目结构

```
bifu/
├── georges/                  # 正类图像 (包含圣乔治)
├── non_georges/              # 负类图像 (不包含圣乔治)
├── project/
│   ├── src/
│   │   ├── models/
│   │   │   └── model.py      # 模型定义
│   │   ├── data/
│   │   │   ├── explore_data.py   # 数据探索脚本
│   │   │   ├── split_data.py     # 数据划分脚本
│   │   │   └── dataset.py        # Dataset + DataLoader
│   │   ├── train.py          # 训练脚本
│   │   ├── eval.py           # 评估脚本
│   │   ├── predict.py        # 单张图片预测
│   │   └── demo.py           # Gradio Web Demo
│   ├── data/
│   │   └── splits/           # 划分后的数据索引 (CSV)
│   ├── results/
│   │   ├── best_model.pth    # 最佳模型权重 (从 Releases 下载)
│   │   ├── training_history.json  # 训练历史
│   │   ├── training_curves.png    # 训练曲线
│   │   ├── evaluation_report.json # 评估报告（含误分类样本清单）
│   │   ├── confusion_matrix.png   # 混淆矩阵
│   │   ├── roc_curve.png          # ROC 曲线
│   │   ├── misclassified_samples.csv   # 误分类样本清单（路径+标签+置信度）
│   │   └── misclassified_samples.png   # 误分类样本网格图
│   └── docs/
│       ├── EXPERIMENT_REPORT.md # 实验报告
│       └── RESULTS.md          # 结果报告
├── requirements.txt          # 依赖列表
└── README.md
```

---

## 使用指南

### 1. 数据探索

查看数据分布和样本：
```bash
python project/src/data/explore_data.py
```

生成的图表保存在 `project/results/` 目录下。

### 2. 数据划分

将数据按 7:1.5:1.5 划分：
```bash
python project/src/data/split_data.py
```

划分结果保存在 `project/data/splits/` 目录下。

### 3. 模型训练

```bash
# 基础训练 (ResNet50, 10 epochs)
python project/src/train.py --epochs 10 --backbone resnet50

# 快速测试 (ResNet18, 2 epochs)
python project/src/train.py --epochs 2 --backbone resnet18

# 从检查点续训
python project/src/train.py --resume project/results/checkpoint_epoch_5.pth --epochs 10
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--backbone` | resnet50 | 骨干网络 (resnet18, resnet50, mobilenet_v2) |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 0.0001 | 学习率 |
| `--img_size` | 224 | 输入图像尺寸 |
| `--seed` | 42 | 随机种子（保证可复现） |
| `--resume` | None | 从检查点续训 |

### 4. 模型评估

```bash
python project/src/eval.py --model_path project/results/best_model.pth
```

评估结果保存在 `project/results/` 目录下，包括：
- `evaluation_report.json`：核心指标 + 误分类样本清单
- `confusion_matrix.png` / `roc_curve.png`：混淆矩阵与 ROC 曲线
- `misclassified_samples.csv`：所有误分类样本（图片路径 + 真实/预测标签 + 置信度）
- `misclassified_samples.png`：误分类样本网格图（按置信度排序）

> 提示：Windows 下如遇多进程报错（`PermissionError` / `WinError 5`），加 `--num_workers 0` 即可。

### 5. 单张图片预测

```bash
python project/src/predict.py --image "path/to/image.jpg"

# 显示预测图像
python project/src/predict.py --image "path/to/image.jpg" --show
```

### 6. 启动 Web Demo

```bash
python project/src/demo.py
```

浏览器访问: http://localhost:7860

---

## 模型架构

### 骨干网络
- ResNet50 (默认)
- ResNet18 (轻量版本)
- MobileNetV2 (移动端优化)

### 分类头
```python
class Classifier(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
```

### 数据增强
训练时使用以下增强策略：
- RandomResizedCrop (0.8-1.0)
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomRotation (±15°)

---

## 常见问题

### Q1: 如何使用自己的数据？
1. 将图片分别放入 `georges/` 和 `non_georges/` 文件夹
2. 运行 `split_data.py` 重新划分
3. 运行 `train.py` 重新训练

### Q2: 训练时间太长？
- 减少 epochs 数量
- 使用更小的模型 (resnet18)
- 减小图像尺寸 (--img_size 128)
- 使用 CPU 训练会慢很多

### Q3: 如何提升模型效果？
- 增加训练数据量
- 使用更强的骨干网络 (resnet101, efficientnet)
- 增加训练 epochs
- 尝试不同的超参数

### Q4: GPU 内存不足？
- 减小 batch_size (--batch_size 16 或 8)
- 使用更小的模型

---

## 技术栈

- Python 3.10
- PyTorch 2.5.1
- torchvision 0.20.1
- CUDA 12.1
- scikit-learn 1.7.2
- Gradio 6.25.0

---

## 项目历史

详细的实验过程、决策记录和结果分析请参考 [实验报告](project/docs/EXPERIMENT_REPORT.md) 与 [结果报告](project/docs/RESULTS.md)。

---

## License

本项目仅供学习和面试使用。
