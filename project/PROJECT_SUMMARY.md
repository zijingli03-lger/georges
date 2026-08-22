# 项目阶段性总结与注意事项

## 📅 最后更新
2026-08-22

---

## 1. 项目概述

### 任务描述
创建一个二分类图像分类模型，判断图像中是否包含"圣乔治"。

### 数据集
- 正类 (georges): 2360 张
- 负类 (non_georges): 3340 张
- 总计: 5700 张
- 比例: 1:1.42（轻微不均衡）

### 目标
获得高质量的分类模型，展示完整的工程能力。

---

## 2. 已完成工作

### ✅ 阶段 1：环境搭建
- Conda 环境：`f:\bifu\.conda` (Python 3.10.20)
- PyTorch：2.5.1+cu121
- CUDA：12.1
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- 基础依赖：numpy, pandas, matplotlib, jupyter, scikit-learn

### ✅ 阶段 2：数据准备

#### 2.1 数据探索
- 平均尺寸：550 × 732 px
- 98.5% 为 RGB 图像
- 4065 种不同尺寸（差异极大）
- 无损坏文件

#### 2.2 数据划分
- 按 7:1.5:1.5 划分，保持类别比例
- TRAIN: 3990 张 (1:1.42)
- VAL: 855 张 (1:1.42)
- TEST: 855 张 (1:1.42)
- 三个子集无重叠

#### 2.3 数据增强
- 训练时：RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation
- 验证/测试时：Resize, CenterCrop, Normalize
- 目标尺寸：224×224
- 注：不使用垂直翻转（圣乔治题材语义不允许上下颠倒）

#### 2.4 DataLoader
- 正常工作，图像形状 [32, 3, 224, 224]
- 类别权重：[0.8533, 1.2076]

### ✅ 代码文件
```
project/src/
├── models/model.py          # 模型定义（支持多个骨干网络）
├── data/
│   ├── explore_data.py      # 数据探索
│   ├── split_data.py        # 数据划分
│   └── dataset.py           # Dataset + DataLoader
├── train.py                 # 训练主脚本
└── eval.py                  # 评估脚本
```

### ✅ 数据文件
```
project/data/splits/
├── train.csv               # 3990 条
├── val.csv                 # 855 条
└── test.csv                # 855 条
```

### ✅ 结果文件
```
project/results/
├── dimension_distribution.png  # 尺寸分布图
├── georges_samples.png         # 正类样本
└── non_georges_samples.png     # 负类样本
```

---

## 3. 已完成工作

### ✅ 阶段 3：Baseline 模型训练

#### 3.1 快速验证（ResNet18）
- **状态**: ✅ 已完成
- **结果**: Val Acc 88.07%, F1 0.8771
- **耗时**: 5.2 分钟

#### 3.2 正式训练（ResNet50）
- **状态**: ✅ 已完成
- **配置**: ResNet50, 10 epoch, batch_size=32, lr=1e-4
- **最佳结果**: Val Acc 92.28% (Epoch 10)
- **耗时**: 14.8 分钟

#### 3.3 模型评估
- **状态**: ✅ 已完成
- **评估数据**: TEST 集 (855 张)
- **结果**:
  - Accuracy: 92.98% ⭐
  - F1 Score: 92.79% ⭐
  - AUC: 97.61% ⭐⭐
- **错误分析**: 错误率 7.02% (60/855)

#### 3.4 输出文件
```
project/results/
├── best_model.pth            # 最佳模型 (281.49 MB)
├── training_history.json     # 训练历史
├── training_curves.png       # 训练曲线
├── evaluation_report.json    # 评估报告
├── confusion_matrix.png      # 混淆矩阵
└── roc_curve.png             # ROC 曲线
```

### ⏳ 阶段 4：模型优化（可选）
- 尝试更多 epoch（当前 10 epoch，可增加到 20-30）
- 尝试其他模型（EfficientNet, ViT）
- 进阶数据增强（Mixup, RandAugment）
- 误差分析深入（查看错误样本）

### ⏳ 阶段 5：工程完善（待规划）
- 代码重构
- 撰写 README
- 整理实验日志
- 准备 GitHub 仓库

---

## 4. ⚠️ 注意事项

### 4.1 Windows 环境问题

#### 问题 1：Python 输出缓冲
**现象**：训练脚本运行后没有输出
**解决方案**：使用 `-u` 参数
```bash
conda run -p f:\bifu\.conda python -u script.py
```

#### 问题 2：conda 环境激活
**现象**：`conda activate` 命令失败
**解决方案**：使用 `conda run` 代替
```bash
# 不要用这个
conda activate f:\bifu\.conda

# 用这个
conda run -p f:\bifu\.conda python script.py
```

#### 问题 3：torchvision 大小写不敏感导致重复统计
**已修复**：`split_data.py` 中使用 `set()` 去重

### 4.2 数据相关

#### 数据划分已完成
- 不要重新运行 `split_data.py`
- 如果重新划分，需要删除 `project/data/splits/` 下的 CSV 文件

#### 数据增强策略
- 训练时 4 种随机增强（RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation）
- 验证/测试时只做固定预处理
- 这是正确的做法，不需要修改

### 4.3 训练相关

#### 断点续训
```bash
conda run -p f:\bifu\.conda python -u f:\bifu\project\src\train.py --resume <checkpoint_path>
```

#### 停止训练
- 按 `Ctrl + C`
- 已保存的检查点不会丢失

#### 预训练模型缓存
位置：`C:\Users\Lenovo\.cache\torch\hub\checkpoints\`

已缓存：
- `resnet152-394f9c45.pth` (230MB)
- `resnet18-f37072fd.pth` (44MB)
- `resnet50-0676ba61.pth` (98MB)

### 4.4 GPU 相关

#### GPU 信息
- 型号：NVIDIA GeForce RTX 3060 Laptop GPU
- 显存：6GB
- CUDA 版本：12.1

#### 显存不足时
减小 batch size：
```bash
--batch_size 16  # 或更小
```

---

## 5. 🔗 相关文档

- 工作流程记录：`project/docs/workflow.md`
- 阶段 3 执行指南：`project/STAGE3_README.md`
- 需求文档：`shuoming.txt`

---

## 6. 📞 在新对话中如何开始

### 开场白建议
```
继续图像分类项目的训练任务。
已完成阶段 1 和 2，相关信息见 f:\bifu\project\PROJECT_SUMMARY.md。
现在需要执行阶段 3：Baseline 模型训练。
```

### 需要告诉 AI 的信息
1. 项目位置：`f:\bifu`
2. 环境：`conda run -p f:\bifu\.conda`
3. 数据已准备好：`project/data/splits/`
4. 代码已编写：`project/src/`
5. 需要执行的命令：参考 `project/STAGE3_README.md`

### 关键文件
- 项目总结：`f:\bifu\project\PROJECT_SUMMARY.md`
- 阶段 3 指南：`f:\bifu\project\STAGE3_README.md`
- 工作流程：`f:\bifu\project\docs\workflow.md`

---

## 7. 🎯 成功标准

### 最低要求
- [x] 完成数据探索
- [x] 完成数据划分
- [x] 运行基线模型训练
- [x] 获得准确率 > 85% (92.98%)

### 目标要求
- [x] 准确率 > 90% (92.98%)
- [x] F1 Score > 0.9 (0.9279)
- [ ] 完整的实验报告
- [ ] GitHub 仓库

---

## 8. 备注

### 训练脚本参数说明
```
--epochs       训练轮数 (默认: 10)
--backbone     骨干网络 (默认: resnet50)
--batch_size   批大小 (默认: 32)
--lr           学习率 (默认: 1e-4)
--img_size     图像尺寸 (默认: 224)
--freeze_backbone 冻结骨干网络 (store_true)
--resume       断点续训路径
--output_dir   输出目录 (默认: f:/bifu/project/results)
```

### 可用的骨干网络
- resnet18, resnet34, resnet50, resnet101, resnet152
- efficientnet_b0, efficientnet_b1, efficientnet_b2
- mobilenet_v2
- vit_base

---

**最后更新时间**: 2026-08-22 19:00