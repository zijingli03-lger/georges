"""
命令行预测脚本
用于单张图像预测

使用示例:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model path/to/model.pth
"""

import sys
import argparse
from pathlib import Path

# Windows 控制台默认 GBK 编码无法打印 emoji（如 ✅），强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
from torchvision import transforms
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import create_model


class ImagePredictor:
    """图像预测器"""

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 创建模型
        self.model = create_model(
            backbone=checkpoint['args']['backbone'],
            pretrained=False,
            num_classes=2,
            dropout_rate=0.0
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['non_georges', 'georges']
        print(f"✅ 模型加载成功: {model_path}")
        print(f"   设备: {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str):
        """预测单张图像"""
        # 加载图像
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        # 获取所有类别的概率
        probs_np = probs.cpu().numpy()[0]
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                'non_georges': float(probs_np[0]),
                'georges': float(probs_np[1])
            }
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description='单张图像预测脚本')
    parser.add_argument('--image', type=str, required=True,
                        help='图像路径')
    parser.add_argument('--model', type=str, 
                        default=str(project_root / 'results' / 'best_model.pth'),
                        help='模型路径')
    parser.add_argument('--show', action='store_true',
                        help='显示预测结果详情')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.image).exists():
        print(f"❌ 图像文件不存在: {args.image}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    # 加载模型并预测
    predictor = ImagePredictor(args.model)
    result = predictor.predict(args.image)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"图像: {result['image_path']}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']:.4f} ({result['confidence']:.2%})")
    
    if result['predicted_class'] == 'georges':
        print("结论: ✅ 检测到圣乔治")
    else:
        print("结论: ❌ 未检测到圣乔治")
    
    if args.show:
        print("\n各类别概率:")
        for class_name, prob in result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"  {class_name:15s}: {prob:.4f} {bar}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()