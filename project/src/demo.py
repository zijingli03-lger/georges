"""
交互式演示界面
使用 Gradio 进行单张图像预测

使用示例:
    python demo.py
    然后访问 http://localhost:7860
"""

import sys
from pathlib import Path

# Windows 控制台默认 GBK 编码无法打印 emoji（如 ✅），强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr
import numpy as np
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

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """预测单张图像"""
        # 预处理
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        # 获取所有类别的概率
        probs_np = probs.cpu().numpy()[0]
        
        result = {
            'predicted_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                'non_georges': float(probs_np[0]),
                'georges': float(probs_np[1])
            }
        }
        
        return result


def load_metrics():
    """从评估报告动态读取指标，避免界面数字过时"""
    report_path = project_root / 'results' / 'evaluation_report.json'
    try:
        import json
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)
        acc = report.get('accuracy')
        auc = report.get('auc')
        if acc is not None and auc is not None:
            return f"{acc*100:.2f}%", f"{auc*100:.2f}%"
    except Exception as e:
        print(f"警告: 无法读取评估报告，使用默认指标文本: {e}")
    return "92.98%", "97.61%"


def create_interface():
    """创建 Gradio 界面"""
    
    # 加载模型
    model_path = str(project_root / 'results' / 'best_model.pth')
    predictor = ImagePredictor(model_path)
    
    # 动态读取指标
    acc_text, auc_text = load_metrics()
    
    def predict_fn(image):
        """预测函数"""
        if image is None:
            return None, None
        
        result = predictor.predict(image)
        
        # 格式化输出
        label = result['predicted_class']
        confidence = result['confidence']
        
        # 生成标签文本
        if label == 'georges':
            label_text = f"✅ 检测到圣乔治 (置信度: {confidence:.2%})"
        else:
            label_text = f"❌ 未检测到圣乔治 (置信度: {confidence:.2%})"
        
        # 生成概率柱状图数据
        probs = result['probabilities']
        
        return label_text, probs
    
    # 创建界面
    with gr.Blocks(title="圣乔治图像检测", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🐉 圣乔治图像检测系统
        
        上传一张图片，模型将判断是否包含"圣乔治"。
        
        **模型**: ResNet50 | **准确率**: {acc_text} | **AUC**: {auc_text}
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                    height=400
                )
                predict_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_label = gr.Textbox(
                    label="检测结果",
                    lines=3
                )
                output_probs = gr.Label(
                    label="类别概率",
                    num_top_classes=2
                )
        
        gr.Markdown("""
        ---
        ### 使用说明
        1. 点击上方区域上传图片，或直接拖拽图片到框内
        2. 点击"开始检测"按钮
        3. 查看检测结果和各类别概率
        
        **支持的格式**: JPG, PNG, BMP, WebP 等
        """)
        
        predict_btn.click(
            fn=predict_fn,
            inputs=input_image,
            outputs=[output_label, output_probs]
        )
    
    return demo


if __name__ == '__main__':
    print("正在启动演示界面...")
    demo = create_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False
    )