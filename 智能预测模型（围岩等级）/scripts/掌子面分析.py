import numpy as np
import os
import logging
import cv2
from pathlib import Path
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChinesePathImageAnalyzer:
    """专门处理中文路径的图像分析器"""
    
    def __init__(self):
        self.device = 'cpu'
        logger.info(f"使用设备: {self.device}")
    
    def read_image_chinese_path(self, image_path):
        """安全读取包含中文字符的图像文件"""
        try:
            # 方法1：使用numpy和cv2的组合来处理中文路径
            # 先用numpy读取文件字节，再用cv2解码
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # 将字节转换为numpy数组
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # 使用cv2解码图像
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"无法解码图像: {image_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"读取图像失败: {e}")
            return None
    
    def preprocess_image(self, image):
        """图像预处理"""
        if image is None:
            return None
        
        try:
            # 转换为RGB（如果需要）
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 调整大小（保持宽高比）
            height, width = image_rgb.shape[:2]
            target_size = 512
            
            if max(height, width) > target_size:
                if height > width:
                    new_height = target_size
                    new_width = int(width * target_size / height)
                else:
                    new_width = target_size
                    new_height = int(height * target_size / width)
                
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
    
    def extract_geological_features(self, image):
        """提取地质特征"""
        if image is None:
            return {}
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算特征
            features = {
                '图像尺寸': f"{image.shape[1]}x{image.shape[0]}",
                '边缘像素数': int(np.sum(edges > 0)),
                '边缘密度': float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])),
                '平均亮度': float(np.mean(gray)),
                '亮度标准差': float(np.std(gray)),
                '对比度': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return {}
    
    def analyze_joint_development(self, features):
        """分析节理发育程度"""
        if not features:
            return "无法分析"
        
        edge_density = features.get('边缘密度', 0)
        contrast = features.get('对比度', 0)
        
        # 简单的节理发育程度评估
        if edge_density > 0.1 and contrast > 0.3:
            development_level = "高度发育"
        elif edge_density > 0.05 and contrast > 0.2:
            development_level = "中等发育"
        elif edge_density > 0.02:
            development_level = "轻微发育"
        else:
            development_level = "不发育"
        
        return development_level
    
    def analyze_image(self, image_path):
        """完整的图像分析流程"""
        try:
            # 验证文件存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"文件不存在: {image_path}")
            
            logger.info(f"开始分析图像: {image_path}")
            
            # 读取图像（处理中文路径）
            image = self.read_image_chinese_path(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            logger.info(f"成功读取图像，尺寸: {image.shape}")
            
            # 预处理
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                raise ValueError("图像预处理失败")
            
            # 特征提取
            features = self.extract_geological_features(processed_image)
            
            # 节理发育分析
            joint_development = self.analyze_joint_development(features)
            
            # 输出结果
            print("\n=== 图像分析结果 ===")
            print(f"文件路径: {image_path}")
            print(f"原始尺寸: {image.shape[1]}x{image.shape[0]}")
            print(f"处理后尺寸: {processed_image.shape[1]}x{processed_image.shape[0]}")
            print("\n=== 地质特征 ===")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            print(f"\n=== 节理发育评估 ===")
            print(f"节理发育程度: {joint_development}")
            
            return {
                'success': True,
                'features': features,
                'joint_development': joint_development,
                'image_shape': image.shape
            }
            
        except Exception as e:
            error_msg = f"图像分析失败: {str(e)}"
            logger.error(error_msg)
            print(f"\n错误: {error_msg}")
            return {'success': False, 'error': str(e)}

def main():
    """主函数"""
    # 创建分析器
    analyzer = ChinesePathImageAnalyzer()
    
    # 指定图像路径
    image_path = r"c:\Users\ASUS\Desktop\科研+论文\AI_Recognition\节理裂隙\节理发育.JPG"
    
    print(f"开始分析图像: {image_path}")
    
    # 执行分析
    result = analyzer.analyze_image(image_path)
    
    if result['success']:
        print("\n分析完成！")
    else:
        print(f"\n分析失败: {result['error']}")
        
        # 提供备选方案
        print("\n=== 备选解决方案 ===")
        print("1. 将图像文件复制到不包含中文字符的路径")
        print("2. 重命名文件为英文名称")
        print("3. 检查文件是否损坏")
        print("4. 确认文件格式是否支持")

if __name__ == "__main__":
    main()