"""
数据划分脚本
用途：将数据集按 7:1.5:1.5 划分为 train/val/test，保持类别比例
"""

import os
import random
import csv
from pathlib import Path
from collections import defaultdict


def get_image_files(folder_path, extensions=None):
    """获取文件夹中所有图像文件"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    folder = Path(folder_path)
    files_set = set()
    for ext in extensions:
        for f in folder.glob(f'*{ext}'):
            files_set.add(f)
    return sorted(files_set)


def split_dataset_by_class(files, train_ratio, val_ratio, test_ratio, seed=42):
    """
    按比例划分单个类别的文件列表
    
    Args:
        files: 文件路径列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1.0，当前为 {train_ratio + val_ratio + test_ratio}"
    
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = shuffled[:train_end]
    val_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def main():
    """主函数"""
    base_dir = Path(r'f:\bifu')
    georges_dir = base_dir / 'georges'
    non_georges_dir = base_dir / 'non_georges'
    output_dir = base_dir / 'project' / 'data' / 'splits'
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    seed = 42
    
    print("=" * 60)
    print("数据划分脚本")
    print("=" * 60)
    
    # 1. 收集文件
    print("\n【1. 收集文件】")
    georges_files = get_image_files(georges_dir)
    non_georges_files = get_image_files(non_georges_dir)
    
    print(f"  正类 (georges): {len(georges_files)} 张")
    print(f"  负类 (non_georges): {len(non_georges_files)} 张")
    print(f"  总计: {len(georges_files) + len(non_georges_files)} 张")
    
    # 2. 按类别独立划分
    print("\n【2. 按类别独立划分】")
    print(f"  划分比例: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"  随机种子: {seed}")
    
    georges_split = split_dataset_by_class(georges_files, train_ratio, val_ratio, test_ratio, seed)
    non_georges_split = split_dataset_by_class(non_georges_files, train_ratio, val_ratio, test_ratio, seed)
    
    # 3. 汇总结果
    print("\n【3. 划分结果】")
    
    summary = {}
    for split_name in ['train', 'val', 'test']:
        g_files = georges_split[split_name]
        n_files = non_georges_split[split_name]
        total = len(g_files) + len(n_files)
        ratio = len(g_files) / len(n_files) if len(n_files) > 0 else float('inf')
        
        summary[split_name] = {
            'georges': g_files,
            'non_georges': n_files
        }
        
        print(f"\n  {split_name.upper()} 集:")
        print(f"    正类: {len(g_files)} 张")
        print(f"    负类: {len(n_files)} 张")
        print(f"    总计: {total} 张")
        print(f"    比例 (正:负): 1 : {1/ratio:.2f}" if ratio > 0 else "")
    
    # 4. 保存为 CSV
    print("\n【4. 保存文件列表】")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        output_file = output_dir / f'{split_name}.csv'
        rows = []
        
        for label, files in [('georges', summary[split_name]['georges']),
                            ('non_georges', summary[split_name]['non_georges'])]:
            for f in files:
                rows.append({
                    'image_path': str(f),
                    'label': label,
                    'label_code': 1 if label == 'georges' else 0
                })
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['image_path', 'label', 'label_code'])
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  已保存: {output_file} ({len(rows)} 条记录)")
    
    # 5. 验证：检查无重叠
    print("\n【5. 验证：检查数据无重叠】")
    for i, split1 in enumerate(['train', 'val', 'test']):
        for j, split2 in enumerate(['train', 'val', 'test']):
            if i < j:
                paths1 = set(str(f) for f in summary[split1]['georges'] + summary[split1]['non_georges'])
                paths2 = set(str(f) for f in summary[split2]['georges'] + summary[split2]['non_georges'])
                overlap = paths1 & paths2
                if overlap:
                    print(f"  ⚠️ {split1} 和 {split2} 有 {len(overlap)} 个重叠文件!")
                else:
                    print(f"  ✅ {split1} 和 {split2} 无重叠")
    
    print("\n" + "=" * 60)
    print("划分完成！")


if __name__ == '__main__':
    main()