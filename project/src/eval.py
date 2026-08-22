"""
评估脚本
用于评估训练好的模型，生成详细的报告

使用示例:
    python eval.py --model_path best_model.pth
    python eval.py --model_path best_model.pth --num_workers 0
"""

import os
import sys
import csv
import argparse
import json
from pathlib import Path

# Windows 控制台默认 GBK 编码无法打印 emoji（如 ✅），强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import create_model
from data.dataset import create_dataloaders


class Evaluator:
    """评估器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        checkpoint = torch.load(args.model_path, map_location=self.device, weights_only=False)
        
        # 创建模型
        self.model = create_model(
            backbone=checkpoint['args']['backbone'],
            pretrained=False,
            num_classes=2,
            dropout_rate=0.0  # 评估时不使用 Dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载数据
        data_dir = Path(args.data_dir)
        _, _, self.test_loader, _ = create_dataloaders(
            data_dir, batch_size=32, num_workers=args.num_workers, img_size=224
        )
        
        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 模型加载成功: {args.model_path}")
        print(f"   最佳验证准确率: {checkpoint['best_val_acc']:.4f}")
    
    @torch.no_grad()
    def predict(self, loader):
        """预测"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def evaluate(self):
        """完整评估"""
        print("\n" + "=" * 60)
        print("开始评估")
        print("=" * 60)
        
        # 预测
        predictions, labels, probabilities = self.predict(self.test_loader)
        
        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        auc = roc_auc_score(labels, probabilities)
        
        # 打印报告
        print("\n【分类报告】")
        print(classification_report(labels, predictions, 
                                    target_names=['non_georges', 'georges']))
        
        print("\n【核心指标】")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(labels, predictions)
        
        # 绘制 ROC 曲线
        self._plot_roc_curve(labels, probabilities, auc)
        
        # 分析错误案例（返回误分类样本清单）
        misclassified = self._analyze_errors(labels, predictions, probabilities)
        
        # 保存报告
        report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'num_errors': int(len(misclassified)),
            'error_rate': float(len(misclassified) / len(labels)),
            'misclassified_samples': misclassified,
            'detailed_report': classification_report(labels, predictions, 
                                                      target_names=['non_georges', 'georges'],
                                                      output_dict=True)
        }
        
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n评估报告已保存: {report_path}")
        
        return report
    
    def _plot_confusion_matrix(self, labels, predictions):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['non_georges', 'georges'],
               yticklabels=['non_georges', 'georges'],
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # 在格子内显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plot_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  混淆矩阵已保存: {plot_path}")
    
    def _plot_roc_curve(self, labels, probabilities, auc):
        """绘制 ROC 曲线"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'roc_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ROC 曲线已保存: {plot_path}")
    
    def _analyze_errors(self, labels, predictions, probabilities):
        """分析错误案例：保存误分类样本清单（CSV + JSON）与可视化网格图"""
        errors = np.where(labels != predictions)[0]
        
        print(f"\n【错误分析】")
        print(f"  总样本数: {len(labels)}")
        print(f"  错误数: {len(errors)}")
        print(f"  错误率: {len(errors)/len(labels):.4f}")
        
        # 假阳性（实际负类，预测为正类）
        false_positives = np.where((labels == 0) & (predictions == 1))[0]
        # 假阴性（实际正类，预测为负类）
        false_negatives = np.where((labels == 1) & (predictions == 0))[0]
        
        print(f"\n  假阳性 (FP): {len(false_positives)} 个")
        print(f"  假阴性 (FN): {len(false_negatives)} 个")
        
        # 测试集 DataLoader 使用 shuffle=False，索引顺序与 dataset 一致，
        # 因此可以直接通过 dataset.data 拿到每个样本的图片路径
        dataset = self.test_loader.dataset
        class_names = ['non_georges', 'georges']
        
        misclassified = []
        for idx in errors:
            item = dataset.data[idx]
            misclassified.append({
                'index': int(idx),
                'image_path': item['image_path'],
                'true_label': class_names[int(labels[idx])],
                'predicted_label': class_names[int(predictions[idx])],
                'probability': float(probabilities[idx]),
                'error_type': 'false_positive' if labels[idx] == 0 else 'false_negative'
            })
        
        # 按置信度从高到低排序（高置信度仍出错 = 最值得关注的案例）
        misclassified.sort(key=lambda x: x['probability'], reverse=True)
        
        # 保存 CSV 清单
        csv_path = self.output_dir / 'misclassified_samples.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'index', 'image_path', 'true_label', 'predicted_label',
                'probability', 'error_type'
            ])
            writer.writeheader()
            writer.writerows(misclassified)
        print(f"\n  误分类样本清单已保存: {csv_path} ({len(misclassified)} 条)")
        
        # 保存误分类样本网格图
        self._plot_misclassified(misclassified)
        
        return misclassified
    
    def _plot_misclassified(self, misclassified, max_samples=16):
        """绘制误分类样本网格图（最多 16 张，按置信度从高到低）"""
        if len(misclassified) == 0:
            print("  无错误样本，跳过可视化")
            return
        
        from PIL import Image
        
        samples = misclassified[:max_samples]
        cols = 4
        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, s in enumerate(samples):
            ax = axes[i // cols, i % cols]
            try:
                img = Image.open(s['image_path']).convert('RGB')
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f'load error:\n{e}', ha='center', va='center')
            ax.axis('off')
            color = 'red' if s['error_type'] == 'false_positive' else 'blue'
            ax.set_title(
                f"True: {s['true_label']}\nPred: {s['predicted_label']} | P={s['probability']:.3f}",
                fontsize=9, color=color
            )
        
        # 隐藏多余的子图
        for i in range(len(samples), rows * cols):
            axes[i // cols, i % cols].axis('off')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'misclassified_samples.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  误分类样本网格图已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'data' / 'splits'),
                        help='数据划分文件目录')
    parser.add_argument('--output_dir', type=str, default=str(project_root / 'results'),
                        help='输出目录')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数（Windows 下如遇多进程报错可设为 0）')
    
    args = parser.parse_args()
    
    # 评估
    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == '__main__':
    main()