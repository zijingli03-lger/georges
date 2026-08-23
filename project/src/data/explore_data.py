"""
数据集探索脚本
用途：统计数据分布、分析图像尺寸、查看样本
"""

import os
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Windows 控制台默认 GBK 编码无法打印 emoji（如 ✅），强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def get_image_files(folder_path, extensions=None):
    """获取文件夹中所有图像文件（Windows 下自动去重）"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    folder = Path(folder_path)
    files_set = set()
    for ext in extensions:
        # Windows 不区分大小写，只需匹配一次
        for f in folder.glob(f'*{ext}'):
            files_set.add(f)
    return sorted(files_set)


def analyze_image_dimensions(image_files, sample_size=None):
    """分析图像尺寸分布"""
    widths = []
    heights = []
    channels = []
    corrupted = []
    
    files_to_check = image_files
    if sample_size and sample_size < len(image_files):
        indices = np.random.choice(len(image_files), sample_size, replace=False)
        files_to_check = [image_files[i] for i in indices]
    
    print(f"  正在分析 {len(files_to_check)} 张图像...")
    
    for img_path in files_to_check:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                widths.append(width)
                heights.append(height)
                
                if mode == 'RGB':
                    channels.append(3)
                elif mode == 'L':
                    channels.append(1)
                elif mode == 'RGBA':
                    channels.append(4)
                else:
                    channels.append(0)
                    
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
    
    return {
        'widths': widths,
        'heights': heights,
        'channels': channels,
        'corrupted': corrupted,
        'sampled': len(files_to_check) < len(image_files)
    }


def plot_dimension_distribution( widths, heights, save_path=None):
    """绘制尺寸分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Width distribution
    axes[0, 0].hist(widths, bins=50, edgecolor='black', alpha=0.7, color='#2196F3')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(widths):.0f}')
    axes[0, 0].legend()
    
    # Height distribution
    axes[0, 1].hist(heights, bins=50, edgecolor='black', alpha=0.7, color='#4CAF50')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--',
                       label=f'Mean: {np.mean(heights):.0f}')
    axes[0, 1].legend()
    
    # Width vs Height scatter
    axes[1, 0].scatter(widths, heights, alpha=0.5, s=10, color='#FF9800')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Height (pixels)')
    axes[1, 0].set_title('Width vs Height')
    
    # Aspect ratio distribution
    aspect_ratios = [w / h for w, h in zip(widths, heights)]
    axes[1, 1].hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7, color='#9C27B0')
    axes[1, 1].set_title('Aspect Ratio Distribution')
    axes[1, 1].set_xlabel('Aspect Ratio (Width/Height)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(aspect_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(aspect_ratios):.2f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  尺寸分布图已保存: {save_path}")
    plt.close()


def show_sample_images(image_files, n_samples=8, save_path=None):
    """显示样本图像"""
    indices = np.random.choice(len(image_files), min(n_samples, len(image_files)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, i in enumerate(indices):
        img_path = image_files[i]
        try:
            with Image.open(img_path) as img:
                axes[idx].imshow(img)
                axes[idx].set_title(f'{img_path.stem[:16]}...\n{img.size}')
                axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  样本图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    # 自动推导项目根目录（本文件位于 project/src/data/ 下，向上 3 级即项目根）
    base_dir = Path(__file__).resolve().parents[3]
    georges_dir = base_dir / 'georges'
    non_georges_dir = base_dir / 'non_georges'
    output_dir = base_dir / 'project' / 'results'
    
    print("=" * 60)
    print("数据集探索报告")
    print("=" * 60)
    
    # 1. 数据统计
    print("\n【1. 数据统计】")
    georges_files = get_image_files(georges_dir)
    non_georges_files = get_image_files(non_georges_dir)
    
    print(f"  正类 (georges): {len(georges_files)} 张")
    print(f"  负类 (non_georges): {len(non_georges_files)} 张")
    print(f"  总计: {len(georges_files) + len(non_georges_files)} 张")
    print(f"  正负比例: 1 : {len(non_georges_files)/len(georges_files):.2f}")
    
    # 2. 图像尺寸分析
    print("\n【2. 图像尺寸分析】")
    print("  分析正类...")
    georges_stats = analyze_image_dimensions(georges_files)
    
    print("  分析负类...")
    non_georges_stats = analyze_image_dimensions(non_georges_files)
    
    for label, stats in [("正类 (georges)", georges_stats), 
                          ("负类 (non_georges)", non_georges_stats)]:
        if stats['widths']:
            print(f"\n  {label}:")
            print(f"    宽度: {min(stats['widths'])} - {max(stats['widths'])} px, "
                  f"平均: {np.mean(stats['widths']):.0f}, 中位数: {np.median(stats['widths']):.0f}")
            print(f"    高度: {min(stats['heights'])} - {max(stats['heights'])} px, "
                  f"平均: {np.mean(stats['heights']):.0f}, 中位数: {np.median(stats['heights']):.0f}")
            
            modes = Counter(stats['channels'])
            mode_desc = {1: '灰度', 3: 'RGB', 4: 'RGBA', 0: '其他'}
            print(f"    通道类型: {dict(modes)}")
            for ch, count in modes.items():
                if ch in mode_desc:
                    print(f"      {mode_desc[ch]}: {count} 张")
    
    # 3. 损坏文件检查
    print("\n【3. 损坏文件检查】")
    all_corrupted = georges_stats['corrupted'] + non_georges_stats['corrupted']
    if all_corrupted:
        print(f"  ⚠️ 发现 {len(all_corrupted)} 个损坏文件:")
        for path, error in all_corrupted:
            print(f"    - {Path(path).name}: {error}")
    else:
        print("  ✅ 所有文件均正常")
    
    # 4. 可视化
    print("\n【4. 生成可视化图表】")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并尺寸数据用于对比
    all_widths = georges_stats['widths'] + non_georges_stats['widths']
    all_heights = georges_stats['heights'] + non_georges_stats['heights']
    plot_dimension_distribution(
        all_widths, all_heights,
        save_path=str(output_dir / 'dimension_distribution.png')
    )
    
    # 各类别样本
    show_sample_images(
        georges_files, n_samples=8,
        save_path=str(output_dir / 'georges_samples.png')
    )
    show_sample_images(
        non_georges_files, n_samples=8,
        save_path=str(output_dir / 'non_georges_samples.png')
    )
    
    # 5. 建议
    print("\n【5. 数据准备建议】")
    if all_widths:
        avg_h = np.mean(all_heights)
        avg_w = np.mean(all_widths)
        print(f"  建议的输入尺寸: {224}x{224} (标准) 或 {320}x{320} (大尺寸)")
        print(f"  当前平均尺寸: {avg_w:.0f} x {avg_h:.0f}")
        
        unique_sizes = len(set(zip(all_widths, all_heights)))
        if unique_sizes > 10:
            print(f"  ⚠️ 图像尺寸不一致 ({unique_sizes}种不同尺寸)")
            print(f"  建议: 使用 resize + center crop 统一尺寸")
        
        print(f"\n  数据划分建议 (7:1.5:1.5):")
        total = len(georges_files) + len(non_georges_files)
        train_count = int(total * 0.7)
        val_count = int(total * 0.15)
        test_count = total - train_count - val_count
        print(f"    训练集: {train_count} 张")
        print(f"    验证集: {val_count} 张")
        print(f"    测试集: {test_count} 张")
        
        print(f"\n  类别不均衡提醒:")
        if len(georges_files) != len(non_georges_files):
            ratio = max(len(georges_files), len(non_georges_files)) / \
                     min(len(georges_files), len(non_georges_files))
            print(f"    比例约 {ratio:.2f}:1, 建议使用加权损失函数或数据平衡策略")
    
    print("\n" + "=" * 60)
    print("探索完成！")


if __name__ == '__main__':
    main()