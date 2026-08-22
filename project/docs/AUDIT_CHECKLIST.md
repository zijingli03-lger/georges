# 项目审计清单（交付前必读）

> 创建日期：2026-08-22
> 最后更新：2026-08-22（一轮修复后）
> 审计依据：`shuoming.txt` 需求文档 vs 现有 `project/` 实现
> 使用说明：每完成一项，把 `[ ]` 改为 `[x]` 并填上日期。

---

## 0. 已核实为真实的部分（无需担心，勿重复修改）

- [x] 指标真实：`training_history.json` 10 个 epoch 的 loss/acc 与文档逐项吻合；`evaluation_report.json`（acc 92.98% / F1 0.9279 / AUC 0.9761）与所有文档一致
- [x] 模型可用：`.conda` 环境成功加载 `best_model.pth`（ResNet50），前向推理正常
- [x] 数据划分干净：train 3990 / val 855 / test 855，三集互不重叠，5700 个路径全部存在，比例 1:1.42
- [x] 类别权重正确：[0.8533, 1.2076]，与公式 `total/(2*count)` 吻合
- [x] `eval.py` 已实际运行验证（2026-08-22）：指标与文档完全一致，说明模型和评估流程真实可靠

---

## 1. 🔴 关键问题

### 1.1 错误样本示例（需求明确要求）—— ✅ 已完成
- [x] `eval.py` 重写 `_analyze_errors`：从测试集 DataLoader 自动找出预测错误的样本（无需手动找）
- [x] 生成 `results/misclassified_samples.csv`：60 条，含图片路径、真实/预测标签、置信度、错误类型
- [x] 生成 `results/misclassified_samples.png`：误分类样本网格图（按置信度排序）
- [x] 误分类样本已纳入 `evaluation_report.json`（`num_errors` / `error_rate` / `misclassified_samples`）
- [x] 已实际运行验证：错误 60 个（FP 33 / FN 27），与旧报告一致

> 💡 原理备忘（给未来的自己）：
> 错误样本不用去训练集里找。数据在划分时留出了独立的**测试集 855 张**（`test.csv`），模型训练时没见过这些图。
> eval 时把 855 张图喂给模型，预测结果和真实标签对比，不一样的就是"误分类样本"，自动记录路径。

### 1.2 依赖清单可复现 —— ✅ 已完成
- [x] `requirements.txt` 补全：`torch==2.5.1`、`torchvision==0.20.1`、`gradio==6.25.0`（与实际环境版本一致）
- [x] `environment.yml` 补全：锁版本 `pytorch=2.5.1` / `torchvision=0.20.1`，gradio 通过 pip 段安装
- [ ] （加分，可选）补一个 `Dockerfile` —— 未做，不做也不影响评分

### 1.3 训练随机种子 —— ✅ 已完成（顺手加，成本极低）
- [x] `train.py` 新增 `set_seed()` 函数 + `--seed 42` 参数（默认 42），训练前统一设置所有随机源
- [x] 说明：用户接受"不能比特级复现"，但加 seed 后数据划分 + 权重初始化都确定，跨机器复现更可靠

---

## 2. 🟡 文档可信度

### 2.1 实验对比表 —— ✅ 已完成
- [x] `EXPERIMENT_REPORT.md` §3.1 删除无数据支持的 MobileNetV2 / ResNet101 行
- [x] 只保留真实训练过的两组：ResNet18 (88.07%) 与 ResNet50 (92.28%)，并加说明

### 2.2 时间记录统一 —— ✅ 已完成
- [x] `workflow.md` 顶部：总跨度 ~5.5 小时（含规划、排查、文档）
- [x] `EXPERIMENT_REPORT.md` §10：纯执行耗时 ~118 min
- [x] 两处口径已区分说明，不再矛盾

### 2.3 eval.py docstring —— ✅ 已完成
- [x] 删除不存在的 `--split` 参数示例，改为 `--num_workers 0` 示例

### 2.4 workflow.md 未执行项标注 —— ✅ 已完成
- [x] 任务 6.3"代码提交"改为 ⚠️ 部分完成（.gitignore 已配，git 仓库未建，待发布前执行）
- [x] 任务 1.2 状态改为 ✅ 已完成（最终由用户手动安装成功）
- [x] 阶段 4 全部标注"未执行（规划项）"
- [x] 文件顶部加提示：规划 ≠ 已完成，以 results/ 和 RESULTS.md 为准

---

## 3. 🟢 清理与代码健壮性

### 3.1 根目录垃圾文件 —— ✅ 已完成
- [x] 删除两个 `u=...webp` 残留文件

### 3.2 空目录 —— ✅ 已完成
- [x] 删除 `project/configs/`、`project/notebooks/`、`project/src/utils/`（均为空）

### 3.3 代码修复 —— ✅ 已完成
- [x] `dataset.py`：图片加载失败不再静默改标签为 0（改为保留原标签 + 黑图占位），避免数据污染
- [x] `dataset.py`：删除 `RandomVerticalFlip(p=0.3)`，理由：圣乔治题材（画作/雕塑）上下颠倒不符合真实场景（已在代码注释和 EXPERIMENT_REPORT §4.1 说明）
- [x] `demo.py`：界面指标从 `evaluation_report.json` 动态读取，重训后不会过时
- [x] `eval.py` / `train.py` / `predict.py` / `demo.py`：新增 UTF-8 输出强制（`sys.stdout.reconfigure`），修复 Windows GBK 控制台打印 emoji（✅✨）崩溃问题

### 3.4 Windows 环境注意（本次实际踩坑）—— ⚠️ 记录
- [x] **GBK 编码崩溃**：`eval.py` 原来在 Windows 控制台打印 `✅` 直接 `UnicodeEncodeError` 崩溃。已通过强制 UTF-8 修复。**教训：Windows 下任何脚本打印非 ASCII 都要做此处理**
- [x] **DataLoader 多进程报错**：`num_workers>0` 时 `PermissionError: [WinError 5]`。`eval.py` 已加 `--num_workers` 参数，报错时设 0 即可（README 已注明）
- [ ] 用户本机若正常运行（Anaconda Prompt），可保持默认 `num_workers=2` 更快

### 3.5 drop_last 说明（用户要求记录，留给以后理解）
> 💡 原理备忘：
> `train.py` 的 DataLoader 用了 `drop_last=True`，意思是：训练集 3990 张 ÷ batch 32 = 124 批余 22 张，
> 最后不足 32 张的那一批**直接丢弃不训练**。所以每个 epoch 实际只用 3968 张。
> 好处：保证每个 batch 形状一致，训练稳定；坏处：每 epoch 少看 22 张图（占 0.5%，影响可忽略）。
> 当初设计如此，可接受，无需修改。

### 3.6 发布注意事项 —— ✅ 已完成
- [x] `.gitignore` 例外放行 `project/results/best_model.pth`（需求要求可复现，权重随仓库提供，281 MB）
- [ ] 注意：`.gitignore` 仍忽略 `*.jpg/*.png/*.pth`（除 best_model 外）——checkpoint 和图片产物不会提交，靠运行脚本重新生成，README 已说明

---

## 4. 加分项（可选，未做）

- [ ] 补充 `Dockerfile`（需求推荐项）
- [ ] 交付一份**英文版** README / 最终报告（原始需求为英文）
- [ ] 在 `EXPERIMENT_REPORT.md` §7 错误分析中引用具体误分类样本文件名（产物已生成，随时可补）

---

## 4.5 已创建的个人资料（非对外交付物）

- [x] `docs/LEARNING_NOTES.md`：**个人学习笔记**——项目每一步的大白话讲解 + 面试追问清单 + 术语对照表。面试前必读，**不要发布到仓库**
- [x] `docs/DELIVERY_GUIDE.md`：**交付操作手册**——shuoming.txt 逐条中文翻译对照、对外发布内容清单、git 发布傻瓜步骤、面试前 48h 检查单。**不要发布到仓库**

---

## 5. 需求交付物对照（最终检查）

| 交付物 | 状态 |
|--------|------|
| Python 代码（数据准备/训练/推理/评估） | ✅ 已有且已验证 |
| README（安装/运行/架构） | ✅ 已同步 |
| 结果/日志/checkpoint 文件 | ✅ 已有（含误分类产物） |
| 报告（Markdown） | ✅ 已修正 |
| 错误样本示例 | ✅ 已补（CSV + 网格图） |
| 改进建议 | ✅ 已有 |
| Dockerfile（推荐） | ❌ 未做（可选） |
| requirements/environment 可复现 | ✅ 已补全 |
| 公开仓库（GitHub/GitLab） | ❌ **待发布**（无 `.git`，下一步） |

---

## 6. 修复完成后自检

- [x] `eval.py` 实际跑通，指标与文档一致，误分类产物生成
- [x] 所有脚本 `py_compile` 通过
- [ ] 待用户本机验证：`pip install -r requirements.txt` 后 `python project/src/demo.py` 可启动
- [ ] 待用户本机验证：`python project/src/predict.py --image "某张图片.jpg"` 可预测
- [ ] 待办：初始化 git 并推送到 GitHub/GitLab（任务 6.3）
