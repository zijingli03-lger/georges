# 项目工作流程记录

## 项目信息
- **项目名称**: Bifu - 圣乔治图像二分类
- **任务**: 构建二元图像分类模型，判断图像是否包含圣乔治
- **数据**: 5700 张图像（2360 正类 / 3340 负类）
- **开始时间**: 2026-08-22 14:37
- **完成时间**: 2026-08-22 20:30
- **总跨度**: ~5.5 小时（含前期规划、环境问题排查、文档整理）
- **纯执行耗时**: ~2 小时（各阶段耗时明细见 `EXPERIMENT_REPORT.md` §10）

> ⚠️ 注意：本文件创建于项目启动阶段，部分"规划"任务实际未执行。
> 已完成项以 ✅ 为准；未执行项已标注 ⚠️/⏳。最终结果以 `results/` 与 `docs/RESULTS.md` 为准。

## 最终成果
- **最佳模型**: ResNet50
- **测试准确率**: 92.98%
- **F1 Score**: 0.9279
- **AUC**: 0.9761
- **所有阶段**: ✅ 已完成

---

## 阶段 1：环境搭建

### 任务 1.1：创建 Conda 虚拟环境
- **时间**: 2026-08-22 14:37
- **操作**: 使用 conda 创建独立虚拟环境
- **环境名称**: bifu (使用 --prefix f:\bifu\.conda)
- **Python 版本**: 3.10.20
- **状态**: ✅ 已完成
- **备注**: 
  - 初始尝试 `conda create -n bifu` 失败，因沙箱限制无法写入 D:\anaconda\envs\
  - 解决方案: 使用 `--prefix f:\bifu\.conda` 在项目目录下创建环境
  - 清华镜像源不可用(403)，已移除，使用默认源成功

### 任务 1.2：安装 PyTorch
- **时间**: 2026-08-22 15:05
- **操作**: 安装 PyTorch 及相关库
- **硬件**: NVIDIA GeForce RTX 3060 (6GB 显存)
- **CUDA 版本**: 13.2 (驱动) → 安装 CUDA 12.1 版本 PyTorch
- **状态**: ✅ 已完成（最终由用户手动通过 conda 安装成功，实际版本 torch 2.5.1+cu121）
- **版本历史**:
  - 15:01 - 安装 CPU 版本成功
  - 15:05 - 用户关闭代理，配置清华源，开始安装 CUDA 版本
- **遇到的问题**:
  - 之前 CUDA 12.1 版本 (2.4GB) 下载因网络/代理不稳定多次中断
  - 解决方案: 关闭代理，配置清华源，重新安装
- **配置项**:
  - pip 清华源: https://pypi.tuna.tsinghua.edu.cn/simple
  - pip 额外源: https://download.pytorch.org/whl/cu121 (PyTorch CUDA 轮子)
  - conda 清华源已添加

### 任务 1.3：安装其他依赖
- **时间**: 2026-08-22 14:48
- **操作**: 安装数据处理、可视化等库
- **依赖列表**: numpy, pandas, Pillow, scikit-learn, matplotlib, seaborn, jupyter, ipywidgets, tqdm, pyyaml
- **状态**: ✅ 已完成

### 任务 1.4：环境验证
- **时间**: 2026-08-22 15:50
- **操作**: 验证 PyTorch 能否正常导入，检查 CUDA 可用性
- **验证结果**:
  - PyTorch: 2.5.1+cu121
  - torchvision: 0.20.1+cu121
  - CUDA: 12.1 ✅
  - GPU: NVIDIA GeForce RTX 3060 Laptop GPU ✅
  - numpy: 2.2.6
  - pandas: 2.3.3
  - scikit-learn: 1.7.2
  - matplotlib: 3.10.9
- **状态**: ✅ 已完成
- **备注**: 用户手动使用 conda 安装成功 CUDA 版本

---

## 阶段 2：数据准备

### 任务 2.1：数据探索
- **时间**: 2026-08-22 17:25
- **操作**: 
  - 查看各类别样本图像
  - 统计图像尺寸分布
  - 检查数据质量
- **状态**: ✅ 已完成
- **结果摘要**:
  - 正类 (georges): 2360 张
  - 负类 (non_georges): 3340 张
  - 总计: 5700 张
  - 正负比例: 1 : 1.42
  - 平均尺寸: 550 x 732 px
  - 98.5% 为 RGB 图像，少量灰度图
  - ⚠️ 4065 种不同尺寸（差异极大）
  - ✅ 无损坏文件
- **修正记录**:
  - 初版脚本因 Windows 大小写不敏感导致重复统计，已修正
  - 修正后数据准确

- **生成的文件**:
  - `project/results/dimension_distribution.png` - 尺寸分布图
  - `project/results/georges_samples.png` - 正类样本
  - `project/results/non_georges_samples.png` - 负类样本

### 任务 2.2：数据划分
- **时间**: 2026-08-22 17:30
- **操作**: 
  - 将数据按 7:1.5:1.5 划分为 train/val/test
  - 保持类别比例
  - 分别对正负类独立划分
- **状态**: ✅ 已完成
- **划分结果**:
  - TRAIN: 正类 1652 + 负类 2338 = 3990 张 (比例 1:1.42)
  - VAL: 正类 354 + 负类 501 = 855 张 (比例 1:1.42)
  - TEST: 正类 354 + 负类 501 = 855 张 (比例 1:1.42)
- **验证**: ✅ 三个子集无重叠
- **输出文件**:
  - `project/data/splits/train.csv`
  - `project/data/splits/val.csv`
  - `project/data/splits/test.csv`

### 任务 2.3：数据增强策略
- **时间**: 2026-08-22 17:30
- **操作**: 
  - 设计基础增强：随机裁剪、水平翻转、颜色抖动
  - 设计进阶增强：RandAugment 或 Mixup
- **状态**: ✅ 已完成（基础增强已实现，进阶未执行）
- **训练时增强**:
  1. RandomResizedCrop(224, scale=0.8-1.0)
  2. RandomHorizontalFlip(p=0.5)
  3. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  4. RandomRotation(±15°)
  5. Normalize(ImageNet 均值/标准差)
  - 注：不使用 RandomVerticalFlip（圣乔治题材语义不允许上下颠倒，见 EXPERIMENT_REPORT.md §4.1）
- **验证/测试时预处理**:
  1. Resize(256)
  2. CenterCrop(224)
  3. Normalize(ImageNet 均值/标准差)

### 任务 2.4：构建 DataLoader
- **时间**: 2026-08-22 17:32
- **操作**: 
  - 使用自定义 BinaryClassificationDataset
  - 设置 batch_size=32
  - 配置 num_workers=2
  - 计算类别权重处理不均衡
- **状态**: ✅ 已完成
- **验证结果**:
  - 训练集: 3990 样本, 124 batches
  - 验证集: 855 样本, 27 batches
  - 测试集: 855 样本, 27 batches
  - 类别权重: [0.8533, 1.2076]
  - 图像形状: [32, 3, 224, 224] ✅
- **输出文件**:
  - `project/src/data/split_data.py`
  - `project/src/data/dataset.py`

---

## 阶段 3：Baseline 模型

### 任务 3.1：选择预训练模型
- **时间**: 2026-08-22 18:30
- **操作**: 
  - 选用 ResNet50 作为 baseline
  - 使用 ImageNet 预训练权重
- **状态**: ✅ 已完成
- **决策理由**: ResNet50 结构成熟、预训练效果好、社区支持完善

### 任务 3.2：迁移学习实现
- **时间**: 2026-08-22 18:30
- **操作**: 
  - 自定义分类头 (Linear → ReLU → Dropout → Linear)
  - 全模型训练（不冻结 backbone）
  - 使用 AdamW 优化器 + CosineAnnealingLR 调度器
- **状态**: ✅ 已完成
- **实现文件**: `project/src/models/model.py`

### 任务 3.3：训练循环
- **时间**: 2026-08-22 18:32 - 18:50
- **操作**: 
  - 实现标准训练循环（带 tqdm 进度条）
  - 添加验证环节
  - 记录 loss 和指标
  - 支持断点续训
  - 自动保存最佳模型
- **状态**: ✅ 已完成
- **实现文件**: `project/src/train.py`

### 任务 3.4：快速验证（ResNet18）
- **时间**: 2026-08-22 18:32 - 18:38
- **目的**: 验证代码流程正确
- **配置**: ResNet18, 2 epoch, batch_size=32
- **结果**:
  - 验证 Acc: 88.07%
  - F1 Score: 0.8771
- **状态**: ✅ 已完成
- **备注**: 代码流程验证通过

### 任务 3.5：正式训练（ResNet50）
- **时间**: 2026-08-22 18:42 - 18:57
- **配置**: ResNet50, 10 epoch, batch_size=32, lr=1e-4
- **训练结果**:
  - Epoch 1: Train Acc 81.78%, Val Acc 86.90%
  - Epoch 2: Train Acc 88.58%, Val Acc 89.59%
  - Epoch 4: Train Acc 93.75%, Val Acc 90.76%
  - Epoch 6: Train Acc 96.12%, Val Acc 91.70%
  - Epoch 9: Train Acc 97.96%, Val Acc 92.05%
  - **Epoch 10: Train Acc 98.66%, Val Acc 92.28%** ⭐ 最佳
- **总耗时**: 14.8 分钟
- **状态**: ✅ 已完成
- **输出文件**:
  - `project/results/best_model.pth` (281.49 MB)
  - `project/results/training_history.json`
  - `project/results/training_curves.png`

### 任务 3.6：模型评估
- **时间**: 2026-08-22 18:58
- **评估数据**: TEST 集 (855 张图像)
- **评估结果**:
  - Accuracy: **92.98%** ⭐
  - Precision: 92.69%
  - Recall: 92.89%
  - F1 Score: **92.79%** ⭐
  - AUC: **97.61%** ⭐⭐⭐
- **分类报告**:
  - non_georges: Precision 0.95, Recall 0.93, F1 0.94
  - georges: Precision 0.91, Recall 0.92, F1 0.92
- **错误分析**:
  - 总样本: 855, 错误: 60, 错误率: 7.02%
  - 假阳性 (FP): 33 个
  - 假阴性 (FN): 27 个
- **状态**: ✅ 已完成
- **输出文件**:
  - `project/results/evaluation_report.json`
  - `project/results/confusion_matrix.png`
  - `project/results/roc_curve.png`
- **评估脚本**: `project/src/eval.py`

---

## 阶段 4：模型优化

> ⚠️ 本阶段为规划项，**未实际执行**（本次交付以阶段 1-3 的完整 pipeline + 阶段 5-6 的文档/工程完善为准）。

### 任务 4.1：超参数调优
- **时间**: 待定
- **操作**: 
  - 学习率调整
  - Batch size 调整
  - 训练轮数优化
- **状态**: ⏳ 未执行（规划项）

### 任务 4.2：尝试其他架构
- **时间**: 待定
- **操作**: 
  - EfficientNet-B0
  - Vision Transformer (ViT)
  - ConvNeXt
- **状态**: ⏳ 未执行（规划项）

### 任务 4.3：进阶数据增强
- **时间**: 待定
- **操作**: 
  - 实现 Mixup
  - 实现 CutMix
  - 实现 RandAugment
- **状态**: ⏳ 未执行（规划项）

### 任务 4.4：误差分析
- **时间**: 待定
- **操作**: 
  - 找出分类错误的样本
  - 分析错误模式
  - 指导后续优化方向
- **状态**: ⏳ 未执行（规划项；错误样本清单已由 eval.py 自动生成，见 results/misclassified_samples.csv）

---

## 阶段 5：工程完善

### 任务 5.1：代码重构
- **时间**: 2026-08-22 20:00
- **操作**: 
  - 模块化代码结构
  - 添加类型注解
  - 编写 docstring
- **状态**: ✅ 已完成
- **文件结构**:
  ```
  project/src/
  ├── models/model.py      # 模型定义
  ├── data/dataset.py      # Dataset + DataLoader
  ├── data/split_data.py   # 数据划分
  ├── train.py             # 训练脚本
  ├── eval.py              # 评估脚本
  ├── predict.py           # 预测脚本
  └── demo.py              # Gradio Demo
  ```

### 任务 5.2：实验日志整理
- **时间**: 2026-08-22 20:15
- **操作**: 
  - 整理所有实验结果
  - 对比不同方案
  - 记录关键发现
- **状态**: ✅ 已完成
- **输出文件**: `project/docs/EXPERIMENT_REPORT.md`

### 任务 5.3：结果可视化
- **时间**: 2026-08-22 20:15
- **操作**: 
  - 训练曲线图表
  - 混淆矩阵热力图
  - ROC 曲线
- **状态**: ✅ 已完成
- **输出文件**:
  - `project/results/training_curves.png`
  - `project/results/confusion_matrix.png`
  - `project/results/roc_curve.png`

### 任务 5.4：撰写最终报告
- **时间**: 2026-08-22 20:20
- **操作**: 
  - 总结项目过程
  - 展示最终结果
  - 分析改进方向
- **状态**: ✅ 已完成
- **输出文件**: `project/docs/RESULTS.md`

---

## 阶段 6：交付准备

### 任务 6.1：README 编写
- **时间**: 2026-08-22 20:25
- **操作**: 
  - 项目简介
  - 安装说明
  - 运行指南
  - 架构概述
- **状态**: ✅ 已完成
- **输出文件**: `README.md`

### 任务 6.2：环境配置文档
- **时间**: 2026-08-22 15:00
- **操作**: 
  - 完善 requirements.txt
  - 完善 environment.yml
- **状态**: ✅ 已完成
- **输出文件**:
  - `requirements.txt`
  - `environment.yml`

### 任务 6.3：代码提交
- **时间**: 2026-08-22 20:30
- **操作**: 
  - 配置 .gitignore
  - 准备提交
- **状态**: ⚠️ 部分完成（.gitignore 已配置；尚未初始化 git 仓库、未提交到 GitHub/GitLab，待交付前执行）

### 任务 6.4：准备演示材料
- **时间**: 2026-08-22 19:00
- **操作**: 
  - Gradio Web Demo
  - 预测脚本验证
- **状态**: ✅ 已完成
- **演示方式**:
  - 命令行: `python project/src/predict.py --image "path.jpg"`
  - Web UI: `python project/src/demo.py` → http://localhost:7860

---

## 问题与经验记录

### 已解决问题

#### 问题 1：Conda 环境创建位置问题
- **日期**: 2026-08-22
- **问题描述**: `conda create -n bifu` 无法写入 D:\anaconda\envs\
- **尝试的解决方案**: 使用 `conda create -n bifu`
- **最终解决方案**: 使用 `--prefix f:\bifu\.conda` 在项目目录下创建
- **经验教训**: Windows 沙箱环境下注意写权限，建议在项目目录下创建虚拟环境

#### 问题 2：PyTorch CUDA 版本下载不稳定
- **日期**: 2026-08-22
- **问题描述**: 2.4GB 的 CUDA 版本多次下载中断
- **尝试的解决方案**: pip 直接安装、清华源
- **最终解决方案**: 用户手动使用 conda 安装成功
- **经验教训**: 大文件下载不稳定时，建议用户手动执行

#### 问题 3：Windows Python 输出缓冲问题
- **日期**: 2026-08-22
- **问题描述**: 训练脚本运行后没有输出，命令行看起来"黑屏"
- **尝试的解决方案**: 使用 `conda run` 执行
- **最终解决方案**: 添加 `-u` 参数禁用缓冲；在 Anaconda Prompt 中直接激活环境运行
- **经验教训**: Windows 下运行 Python 脚本时，使用 `python -u` 确保实时输出

#### 问题 4：数据重复统计
- **日期**: 2026-08-22
- **问题描述**: 初版数据探索脚本因 Windows 大小写不敏感导致重复统计，数量翻倍
- **尝试的解决方案**: 直接统计文件数
- **最终解决方案**: 使用 `set()` 去重，修正后数据准确
- **经验教训**: Windows 文件系统大小写不敏感，处理文件名时注意去重

#### 问题 5：训练脚本缺少进度条
- **日期**: 2026-08-22
- **问题描述**: 原训练脚本只在 epoch 结束后输出，没有 batch 级进度显示
- **尝试的解决方案**: 等待 epoch 结束
- **最终解决方案**: 添加 tqdm 进度条，支持实时显示 loss 和 acc
- **经验教训**: 深度学习训练脚本应包含 tqdm 进度条，提升用户体验

### 待解决问题
（暂无）

---

## 决策日志

| 日期 | 决策内容 | 决策理由 | 影响范围 |
|------|---------|---------|---------|
| 2026-08-22 | 选择 Conda 而非 Docker | Windows 环境兼容性更好，快速上手 | 环境管理 |
| 2026-08-22 | 选用 PyTorch 作为框架 | 用户熟悉 PyTorch，生态成熟 | 技术栈 |
| 2026-08-22 | ResNet50 作为 baseline | 结构成熟、迁移学习效果好 | 模型选择 |
| 2026-08-22 | 目标图像尺寸 224×224 | 预训练模型标准输入，训练速度快 | 数据预处理 |
| 2026-08-22 | 分阶段训练策略 | 先用 ResNet18 验证流程，再用 ResNet50 正式训练 | 训练流程 |
| 2026-08-22 | 全模型训练（不冻结 backbone） | 数据量适中，全模型训练效果更好 | 模型训练 |
| 2026-08-22 | 使用加权损失函数 | 处理类别不均衡（1:1.42） | 损失函数 |

---

## 项目统计

### 数据统计
- 正类 (georges): 2360 张
- 负类 (non_georges): 3340 张
- 总计: 5700 张

### 训练配置
- 骨干网络: ResNet50
- Epochs: 10
- Batch Size: 32
- Learning Rate: 1e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- 损失函数: CrossEntropyLoss (加权)

### 最终结果
| 指标 | 数值 |
|------|------|
| 测试集 Accuracy | 92.98% |
| 测试集 F1 Score | 92.79% |
| 测试集 AUC | 97.61% |
| 最佳验证 Accuracy | 92.28% |
| 训练总耗时 | 14.8 分钟 |

### 生成的文件
```
project/
├── src/
│   ├── models/model.py           # 模型定义
│   ├── data/split_data.py        # 数据划分
│   ├── data/dataset.py           # Dataset + DataLoader
│   ├── train.py                  # 训练脚本
│   └── eval.py                   # 评估脚本
├── data/splits/
│   ├── train.csv                 # 3990 条
│   ├── val.csv                   # 855 条
│   └── test.csv                  # 855 条
├── results/
│   ├── best_model.pth            # 最佳模型 (281.49 MB)
│   ├── training_history.json     # 训练历史
│   ├── training_curves.png       # 训练曲线
│   ├── evaluation_report.json   # 评估报告
│   ├── confusion_matrix.png      # 混淆矩阵
│   └── roc_curve.png             # ROC 曲线
├── docs/workflow.md              # 工作流程（本文件）
├── STAGE3_README.md              # 阶段 3 指南
└── PROJECT_SUMMARY.md            # 项目总结
```

---

*最后更新: 2026-08-22 19:00*