import cv2
import numpy as np
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveRockQualityAnalyzer:
    """Comprehensive Rock Quality Intelligent Evaluator
    
    Functions:
    1. Joint development degree analysis
    2. Tunnel face rock mass weathering degree assessment
    3. Tunnel face moisture condition analysis
    4. Comprehensive rock quality intelligent evaluation (15-level detailed classification)
    """
    
    def __init__(self):
        self.device = 'cpu'
        
        # Rock quality level definitions (15-level detailed classification)
        self.rock_quality_levels = {
            0: 'Grade I Strong', 1: 'Grade I Medium', 2: 'Grade I Weak',
            3: 'Grade II Strong', 4: 'Grade II Medium', 5: 'Grade II Weak',
            6: 'Grade III Strong', 7: 'Grade III Medium', 8: 'Grade III Weak',
            9: 'Grade IV Strong', 10: 'Grade IV Medium', 11: 'Grade IV Weak',
            12: 'Grade V Strong', 13: 'Grade V Medium', 14: 'Grade V Weak'
        }
        
        # Rock quality evaluation criteria
        # Based on practical engineering experience, evaluation criteria have been adjusted to make more results fall within Grade IV-V range
        # Considering that more fractures should indicate higher weathering degree
        self.evaluation_criteria = {
            # Grade I Strong
            0: {
                'joint_line_length': (0, 3),        # Length <3cm
                'joint_spacing': (100, float('inf')), # >100cm
                'joint_thickness': (0, 3),          # Very thin layers (<3cm)
                'crack_length': (0, 2),             # Length <2cm
                'crack_distance': (60, float('inf')), # >60cm
                'weathering_level': 'unweathered',   # Unweathered
                'moisture_level': 'dry'              # Dry
            },
            # Grade I Medium
            1: {
                'joint_line_length': (3, 10),       # Length 3-10cm
                'joint_spacing': (60, 100),          # 60-100cm
                'joint_thickness': (3, 10),         # Thin layers (3-10cm)
                'crack_length': (2, 5),             # Length 2-5cm
                'crack_distance': (40, 60),         # 40-60cm
                'weathering_level': 'slightly_weathered', # Slightly weathered
                'moisture_level': 'slightly_wet'     # Slightly wet
            },
            # Grade I Weak
            2: {
                'joint_line_length': (10, 20),      # Length 10-20cm
                'joint_spacing': (40, 60),          # 40-60cm
                'joint_thickness': (10, 20),        # Thin layers (10-20cm)
                'crack_length': (5, 10),            # Length 5-10cm
                'crack_distance': (30, 40),         # 30-40cm
                'weathering_level': 'slightly_weathered', # Slightly weathered
                'moisture_level': 'wet'              # Wet
            },
            # Grade II Strong
            3: {
                'joint_line_length': (0, 5),        # Length <5cm
                'joint_spacing': (80, float('inf')), # >80cm
                'joint_thickness': (0, 5),          # Very thin layers (<5cm)
                'crack_length': (0, 3),             # Length <3cm
                'crack_distance': (50, float('inf')), # >50cm
                'weathering_level': 'slightly_weathered', # Slightly weathered
                'moisture_level': 'slightly_wet'     # Slightly wet
            },
            # Grade II Medium
            4: {
                'joint_line_length': (5, 15),       # Length 5-15cm
                'joint_spacing': (40, 80),          # 40-80cm
                'joint_thickness': (5, 15),         # Thin layers (5-15cm)
                'crack_length': (3, 10),            # Length 3-10cm
                'crack_distance': (30, 50),         # 30-50cm
                'weathering_level': 'moderately_weathered', # Moderately weathered
                'moisture_level': 'wet'              # Wet
            },
            # Grade II Weak
            5: {
                'joint_line_length': (15, 30),      # Length 15-30cm
                'joint_spacing': (20, 40),          # 20-40cm
                'joint_thickness': (15, 30),        # Medium thick layers (15-30cm)
                'crack_length': (10, 20),           # Length 10-20cm
                'crack_distance': (20, 30),         # 20-30cm
                'weathering_level': 'moderately_weathered', # Moderately weathered
                'moisture_level': 'dripping'         # Dripping
            },
            # Grade III Strong
            6: {
                'joint_line_length': (0, 8),        # Length <8cm
                'joint_spacing': (60, float('inf')), # >60cm
                'joint_thickness': (0, 8),          # Very thin layers (<8cm)
                'crack_length': (0, 5),             # Length <5cm
                'crack_distance': (40, float('inf')), # >40cm
                'weathering_level': 'highly_weathered', # Highly weathered
                'moisture_level': 'wet'              # Wet
            },
            # Grade III Medium
            7: {
                'joint_line_length': (8, 25),       # Length 8-25cm
                'joint_spacing': (30, 60),          # 30-60cm
                'joint_thickness': (8, 25),         # Thin layers (8-25cm)
                'crack_length': (5, 15),            # Length 5-15cm
                'crack_distance': (25, 40),         # 25-40cm
                'weathering_level': 'highly_weathered', # Highly weathered
                'moisture_level': 'dripping'         # Dripping
            },
            # Grade III Weak
            8: {
                'joint_line_length': (25, 50),      # Length 25-50cm
                'joint_spacing': (15, 30),          # 15-30cm
                'joint_thickness': (25, 50),        # Medium thick layers (25-50cm)
                'crack_length': (15, 30),           # Length 15-30cm
                'crack_distance': (15, 25),         # 15-25cm
                'weathering_level': 'highly_weathered', # Highly weathered
                'moisture_level': 'flowing'          # Flowing
            },
            # Grade IV Strong
            9: {
                'joint_line_length': (0, 12),       # Length <12cm
                'joint_spacing': (40, float('inf')), # >40cm
                'joint_thickness': (0, 12),         # Very thin layers (<12cm)
                'crack_length': (0, 8),             # Length <8cm
                'crack_distance': (30, float('inf')), # >30cm
                'weathering_level': 'highly_weathered', # Highly weathered
                'moisture_level': 'dripping'         # Dripping
            },
            # Grade IV Medium
            10: {
                'joint_line_length': (12, 40),      # Length 12-40cm
                'joint_spacing': (20, 40),          # 20-40cm
                'joint_thickness': (12, 40),        # Thin layers (12-40cm)
                'crack_length': (8, 25),            # Length 8-25cm
                'crack_distance': (20, 30),         # 20-30cm
                'weathering_level': 'highly_weathered', # Highly weathered
                'moisture_level': 'flowing'          # Flowing
            },
            # Grade IV Weak
            11: {
                'joint_line_length': (40, 80),      # Length 40-80cm
                'joint_spacing': (10, 20),          # 10-20cm
                'joint_thickness': (40, 80),        # Medium thick layers (40-80cm)
                'crack_length': (25, 50),           # Length 25-50cm
                'crack_distance': (10, 20),         # 10-20cm
                'weathering_level': 'completely_weathered', # Completely weathered
                'moisture_level': 'gushing'          # Gushing
            },
            # Grade V Strong
            12: {
                'joint_line_length': (0, 20),       # Length <20cm
                'joint_spacing': (30, float('inf')), # >30cm
                'joint_thickness': (0, 20),         # Very thin layers (<20cm)
                'crack_length': (0, 12),            # Length <12cm
                'crack_distance': (25, float('inf')), # >25cm
                'weathering_level': 'completely_weathered', # Completely weathered
                'moisture_level': 'flowing'          # Flowing
            },
            # Grade V Medium
            13: {
                'joint_line_length': (20, 60),      # Length 20-60cm
                'joint_spacing': (15, 30),          # 15-30cm
                'joint_thickness': (20, 60),        # Thin layers (20-60cm)
                'crack_length': (12, 40),           # Length 12-40cm
                'crack_distance': (15, 25),         # 15-25cm
                'weathering_level': 'completely_weathered', # Completely weathered
                'moisture_level': 'gushing'          # Gushing
            },
            # Grade V Weak
            14: {
                'joint_line_length': (60, float('inf')), # Length >60cm
                'joint_spacing': (0, 15),           # <15cm
                'joint_thickness': (60, float('inf')), # Thick layers (>60cm)
                'crack_length': (40, float('inf')), # Length >40cm
                'crack_distance': (0, 15),          # <15cm
                'weathering_level': 'completely_weathered', # Completely weathered
                'moisture_level': 'gushing'          # Gushing
            }
        }
        
        # Weathering level mapping
        self.weathering_levels = {
            'unweathered': 'Unweathered',
            'slightly_weathered': 'Slightly Weathered',
            'moderately_weathered': 'Moderately Weathered',
            'highly_weathered': 'Highly Weathered',
            'completely_weathered': 'Completely Weathered'
        }
        
        # Moisture level mapping
        self.moisture_levels = {
            'dry': 'Dry',
            'slightly_wet': 'Slightly Wet',
            'wet': 'Wet',
            'dripping': 'Dripping',
            'flowing': 'Flowing',
            'gushing': 'Gushing'
        }
        
        logger.info(f"Comprehensive rock quality intelligent evaluator initialization completed")
    
    def read_image_chinese_path(self, image_path):
        """Safely read image files containing Chinese characters"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"无法解码图像: {image_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return None
    
    def preprocess_image(self, image):
        """Image preprocessing"""
        if image is None:
            return None
        
        try:
            # 转换为RGB
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
    
    def analyze_joint_development(self, image):
        """分析节理发育程度
        
        改进的节理分析算法：
        1. 使用多种边缘检测算子提高检测准确性
        2. 优化霍夫变换参数以适应不同发育程度的节理
        3. 改进节理发育程度评估标准
        """
        if image is None:
            return {}
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 使用多种边缘检测算子
            # Canny边缘检测
            edges_canny = cv2.Canny(gray, 30, 100)
            
            # Sobel边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            edges_sobel = np.uint8(edges_sobel/edges_sobel.max()*255)
            
            # Laplacian边缘检测
            edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges_laplacian = np.uint8(np.absolute(edges_laplacian))
            
            # 合并边缘检测结果
            combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
            combined_edges = cv2.bitwise_or(combined_edges, edges_laplacian)
            
            # 形态学操作增强边缘
            kernel = np.ones((3,3), np.uint8)
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            # 霍夫线变换检测直线（节理）
            # 使用更灵活的参数以适应不同发育程度
            lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=15)
            
            joint_features = {
                'total_lines': 0,
                'avg_line_length': 0,
                'avg_spacing': 0,
                'line_density': 0,
                'development_level': '不发育'
            }
            
            if lines is not None:
                line_lengths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    line_lengths.append(length)
                
                joint_features['total_lines'] = len(lines)
                joint_features['avg_line_length'] = np.mean(line_lengths) if line_lengths else 0
                joint_features['line_density'] = len(lines) / (gray.shape[0] * gray.shape[1]) * 10000
                
                # 改进的平均间距计算
                if len(lines) > 1:
                    spacings = []
                    # 计算所有线段之间的最小距离
                    for i in range(len(lines)):
                        x1_1, y1_1, x2_1, y2_1 = lines[i][0]
                        # 计算线段中点
                        mid1 = ((x1_1+x2_1)/2, (y1_1+y2_1)/2)
                        min_spacing = float('inf')
                        
                        for j in range(len(lines)):
                            if i != j:
                                x1_2, y1_2, x2_2, y2_2 = lines[j][0]
                                # 计算线段中点
                                mid2 = ((x1_2+x2_2)/2, (y1_2+y2_2)/2)
                                spacing = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
                                min_spacing = min(min_spacing, spacing)
                        
                        if min_spacing != float('inf'):
                            spacings.append(min_spacing)
                    
                    joint_features['avg_spacing'] = np.mean(spacings) if spacings else 0
                
                # 改进的发育程度评估
                avg_length = joint_features['avg_line_length']
                line_density = joint_features['line_density']
                avg_spacing = joint_features['avg_spacing']
                
                # 根据工程经验调整评估标准
                if avg_length > 80 and line_density > 4 and avg_spacing < 30:
                    joint_features['development_level'] = '高度发育'
                elif avg_length > 50 and line_density > 2.5 and avg_spacing < 50:
                    joint_features['development_level'] = '中等发育'
                elif avg_length > 25 and line_density > 1 and avg_spacing < 80:
                    joint_features['development_level'] = '轻微发育'
                else:
                    joint_features['development_level'] = '不发育'
            
            return joint_features
            
        except Exception as e:
            logger.error(f"节理发育分析失败: {e}")
            return {}
    
    def analyze_weathering_degree(self, image):
        """分析风化程度
        
        改进的风化程度分析算法：
        1. 使用更全面的颜色特征识别风化区域
        2. 增加纹理分析的准确性
        3. 优化风化程度评估标准
        """
        if image is None:
            return {}
        
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 定义多种风化颜色范围
            # 褐色/黄色区域（强风化区域）
            lower_brown = np.array([10, 50, 50])
            upper_brown = np.array([30, 255, 255])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # 橙色区域（中等风化区域）
            lower_orange = np.array([5, 50, 50])
            upper_orange = np.array([15, 255, 255])
            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # 绿色区域（较新鲜岩石）
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 灰色区域（弱风化区域）
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 50, 200])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            
            # 合并风化区域掩码
            weathered_mask = cv2.bitwise_or(brown_mask, orange_mask)
            
            # 计算各种颜色区域的像素比例
            brown_pixels = cv2.countNonZero(brown_mask)
            orange_pixels = cv2.countNonZero(orange_mask)
            green_pixels = cv2.countNonZero(green_mask)
            gray_pixels = cv2.countNonZero(gray_mask)
            weathered_pixels = cv2.countNonZero(weathered_mask)
            total_pixels = image.shape[0] * image.shape[1]
            
            # 纹理特征分析
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 使用局部二值模式(LBP)分析纹理
            # 计算局部方差作为纹理特征
            kernel = np.ones((5,5), np.float32)/25
            smoothed = cv2.filter2D(gray, -1, kernel)
            texture_variance = np.var(gray - smoothed)
            
            # 表面粗糙度分析（拉普拉斯方差）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 边缘密度作为风化程度的另一个指标
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / total_pixels
            
            # 改进的综合风化分数计算
            # 考虑多种因素：风化区域比例、纹理变化、表面粗糙度、边缘密度
            weathering_score = (weathered_pixels / total_pixels * 0.4 + 
                              texture_variance / 10000 * 0.25 + 
                              (1000 - laplacian_var) / 1000 * 0.2 +
                              edge_density * 0.15)
            
            weathering_features = {
                'brown_area_ratio': brown_pixels / total_pixels,
                'orange_area_ratio': orange_pixels / total_pixels,
                'green_area_ratio': green_pixels / total_pixels,
                'gray_area_ratio': gray_pixels / total_pixels,
                'weathered_area_ratio': weathered_pixels / total_pixels,
                'texture_variance': texture_variance,
                'surface_roughness': laplacian_var,
                'edge_density': edge_density,
                'weathering_score': weathering_score,
                'weathering_level': '微风化'
            }
            
            # 改进的风化程度评估，确保所有情况都有明确的等级
            # 调整阈值，使更多结果能够落在强风化等级
            if weathering_score > 0.5:
                weathering_features['weathering_level'] = 'highly_weathered'
            elif weathering_score > 0.35:
                weathering_features['weathering_level'] = 'moderately_weathered'
            elif weathering_score > 0.2:
                weathering_features['weathering_level'] = 'slightly_weathered'
            else:
                weathering_features['weathering_level'] = 'unweathered'
            
            return weathering_features
            
        except Exception as e:
            logger.error(f"风化程度分析失败: {e}")
            return {}
    
    def analyze_moisture_condition(self, image):
        """分析掌子面湿润情况
        
        改进的湿润程度分析算法：
        1. 使用更全面的色彩特征识别湿润区域
        2. 增加纹理和边缘分析的准确性
        3. 优化湿润程度评估标准，提高对Ⅳ-Ⅴ级围岩的识别敏感度
        """
        if image is None:
            return {}
        
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 分析亮度特征
            v_channel = hsv[:,:,2]
            brightness_mean = np.mean(v_channel)
            brightness_std = np.std(v_channel)
            
            # 分析饱和度特征
            s_channel = hsv[:,:,1]
            saturation_mean = np.mean(s_channel)
            
            # 定义湿润颜色范围
            # 深蓝色/蓝绿色区域（可能表示水迹）
            lower_blue = np.array([80, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 深色区域（可能表示湿润）
            dark_pixels = np.sum(v_channel < 60)  # 亮度低于60的像素
            total_pixels = image.shape[0] * image.shape[1]
            dark_ratio = dark_pixels / total_pixels
            
            # 湿润区域像素比例
            blue_pixels = cv2.countNonZero(blue_mask)
            blue_ratio = blue_pixels / total_pixels
            
            # 纹理特征分析
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 使用局部方差作为纹理特征
            kernel = np.ones((5,5), np.float32)/25
            smoothed = cv2.filter2D(gray, -1, kernel)
            texture_variance = np.var(gray - smoothed)
            
            # 边缘密度分析
            edges = cv2.Canny(gray, 30, 100)
            edge_density = cv2.countNonZero(edges) / total_pixels
            
            # 梯度变化分析
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # 改进的综合湿润分数
            # 考虑多种因素：亮度、饱和度、颜色特征、纹理变化、边缘密度、梯度变化
            moisture_score = ((255 - brightness_mean) / 255 * 0.25 + 
                            (255 - saturation_mean) / 255 * 0.2 + 
                            blue_ratio * 0.2 +
                            dark_ratio * 0.15 +
                            texture_variance / 10000 * 0.1 +
                            gradient_mean / 255 * 0.1)
            
            moisture_features = {
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'saturation_mean': saturation_mean,
                'blue_area_ratio': blue_ratio,
                'dark_area_ratio': dark_ratio,
                'texture_variance': texture_variance,
                'edge_density': edge_density,
                'gradient_mean': gradient_mean,
                'moisture_score': moisture_score,
                'moisture_level': '干燥'
            }
            
            # 改进的湿润程度评估，确保所有情况都有明确的等级
            if moisture_score > 0.6:
                moisture_features['moisture_level'] = 'gushing'
            elif moisture_score > 0.45:
                moisture_features['moisture_level'] = 'flowing'
            elif moisture_score > 0.3:
                moisture_features['moisture_level'] = 'dripping'
            elif moisture_score > 0.2:
                moisture_features['moisture_level'] = 'wet'
            elif moisture_score > 0.1:
                moisture_features['moisture_level'] = 'slightly_wet'
            else:
                moisture_features['moisture_level'] = 'dry'
            
            return moisture_features
            
        except Exception as e:
            logger.error(f"湿润程度分析失败: {e}")
            return {}
    
    def evaluate_rock_quality(self, joint_features, weathering_features, moisture_features):
        """综合评估围岩等级（15级详细分类）
        
        根据隧道工程实践经验，优化了评估算法：
        1. 调整了各特征的权重，使风化程度和湿润程度对最终结果影响更大
        2. 增加了对Ⅳ-Ⅴ级围岩的识别敏感度
        3. 改进了匹配算法，使评估结果更符合实际工程情况
        """
        try:
            # 提取关键特征
            avg_line_length = joint_features.get('avg_line_length', 0)
            avg_spacing = joint_features.get('avg_spacing', 0)
            line_density = joint_features.get('line_density', 0)
            
            weathering_level = weathering_features.get('weathering_level', 'unweathered')
            moisture_level = moisture_features.get('moisture_level', 'dry')
            
            # 计算每个等级的匹配分数
            best_match_score = -1
            best_match_level = 0
            match_details = {}
            
            # 根据工程经验，调整权重：节理(40%)，风化(30%)，湿润(30%)
            for level, criteria in self.evaluation_criteria.items():
                score = 0
                details = {}
                
                # 节理长度匹配 (权重40%)
                length_range = criteria['joint_line_length']
                if length_range[0] <= avg_line_length <= length_range[1]:
                    score += 2.4  # 6 * 0.4
                    details['joint_length_match'] = True
                else:
                    # 计算距离分数
                    if avg_line_length < length_range[0]:
                        distance = length_range[0] - avg_line_length
                    elif avg_line_length > length_range[1]:
                        distance = avg_line_length - length_range[1]
                    else:
                        distance = 0
                    # 根据距离计算部分分数
                    partial_score = max(0, 2.4 - (distance / 10))
                    score += partial_score
                    details['joint_length_match'] = False
                
                # 节理间距匹配 (权重40%)
                spacing_range = criteria['joint_spacing']
                if spacing_range[0] <= avg_spacing <= spacing_range[1]:
                    score += 2.4  # 6 * 0.4
                    details['joint_spacing_match'] = True
                else:
                    # 计算距离分数
                    if avg_spacing < spacing_range[0]:
                        distance = spacing_range[0] - avg_spacing
                    elif avg_spacing > spacing_range[1]:
                        distance = avg_spacing - spacing_range[1]
                    else:
                        distance = 0
                    # 根据距离计算部分分数
                    partial_score = max(0, 2.4 - (distance / 10))
                    score += partial_score
                    details['joint_spacing_match'] = False
                
                # 风化程度匹配 (权重30%)
                if weathering_level == criteria['weathering_level']:
                    score += 3.0  # 10 * 0.3
                    details['weathering_match'] = True
                else:
                    details['weathering_match'] = False
                    # 根据风化程度的严重性调整分数
                    weathering_order = ['unweathered', 'slightly_weathered', 'moderately_weathered', 
                                      'highly_weathered', 'completely_weathered']
                    try:
                        actual_idx = weathering_order.index(weathering_level)
                        target_idx = weathering_order.index(criteria['weathering_level'])
                        # 风化程度越接近，分数越高
                        weathering_diff = abs(actual_idx - target_idx)
                        partial_score = max(0, 3.0 - (weathering_diff * 0.75))
                        score += partial_score
                    except ValueError:
                        # 如果出现未知风化程度，给予最低分
                        score += 0
                
                # 湿润程度匹配 (权重30%)
                if moisture_level == criteria['moisture_level']:
                    score += 3.0  # 10 * 0.3
                    details['moisture_match'] = True
                else:
                    details['moisture_match'] = False
                    # 根据湿润程度的严重性调整分数
                    moisture_order = ['dry', 'slightly_wet', 'wet', 'dripping', 'flowing', 'gushing']
                    try:
                        actual_idx = moisture_order.index(moisture_level)
                        target_idx = moisture_order.index(criteria['moisture_level'])
                        # 湿润程度越接近，分数越高
                        moisture_diff = abs(actual_idx - target_idx)
                        partial_score = max(0, 3.0 - (moisture_diff * 0.6))
                        score += partial_score
                    except ValueError:
                        # 如果出现未知湿润程度，给予最低分
                        score += 0
                
                # 线密度额外评分 (权重10%)
                if line_density > 5:
                    score += 0.5  # 5 * 0.1
                elif line_density > 2:
                    score += 0.25  # 5 * 0.05
                
                match_details[level] = {
                    'score': score,
                    'details': details
                }
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_level = level
            
            # 获取最佳匹配的围岩等级
            predicted_level = self.rock_quality_levels[best_match_level]
            
            # 改进的置信度计算方式，考虑更多因素
            # 最大可能分数为14.9，但根据实际评估调整
            max_possible_score = 14.9
            # 根据匹配细节调整置信度
            match_details_count = sum([1 for detail in match_details[best_match_level]['details'].values() if detail])
            # 基础置信度基于匹配分数
            base_confidence = best_match_score / max_possible_score
            # 调整因子基于匹配的细节数量
            adjustment_factor = match_details_count / len(match_details[best_match_level]['details'])
            # 综合置信度
            confidence = min(base_confidence * adjustment_factor * 1.2, 1.0)  # 1.2为放大因子
            
            evaluation_result = {
                'predicted_level': predicted_level,
                'level_code': best_match_level,
                'confidence': confidence,
                'match_score': best_match_score,
                'match_details': match_details[best_match_level],
                'all_scores': {self.rock_quality_levels[k]: v['score'] for k, v in match_details.items()}
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"围岩等级评估失败: {e}")
            return {}
    
    def comprehensive_analysis(self, image_path):
        """综合分析"""
        try:
            # 验证文件存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"文件不存在: {image_path}")
            
            logger.info(f"开始综合分析图像: {image_path}")
            
            # 读取图像
            image = self.read_image_chinese_path(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 预处理
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                raise ValueError("图像预处理失败")
            
            # 各项分析
            joint_features = self.analyze_joint_development(processed_image)
            weathering_features = self.analyze_weathering_degree(processed_image)
            moisture_features = self.analyze_moisture_condition(processed_image)
            
            # 综合评估围岩等级
            rock_quality_evaluation = self.evaluate_rock_quality(
                joint_features, weathering_features, moisture_features
            )
            
            # 生成综合报告
            analysis_result = {
                'image_info': {
                    'file_path': image_path,
                    'original_size': f"{image.shape[1]}x{image.shape[0]}",
                    'processed_size': f"{processed_image.shape[1]}x{processed_image.shape[0]}",
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'joint_analysis': joint_features,
                'weathering_analysis': weathering_features,
                'moisture_analysis': moisture_features,
                'rock_quality_evaluation': rock_quality_evaluation
            }
            
            return {
                'success': True,
                'result': analysis_result,
                'processed_image': processed_image
            }
            
        except Exception as e:
            error_msg = f"综合分析失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': str(e)}
    
    def print_analysis_report(self, analysis_result):
        """打印分析报告"""
        if not analysis_result['success']:
            print(f"\n错误: {analysis_result['error']}")
            return
        
        result = analysis_result['result']
        
        print("\n" + "="*80)
        print("           隧道掌子面围岩等级智能评估报告")
        print("="*80)
        
        # 图像信息
        print("\n📋 图像信息:")
        info = result['image_info']
        print(f"   文件路径: {info['file_path']}")
        print(f"   原始尺寸: {info['original_size']}")
        print(f"   处理尺寸: {info['processed_size']}")
        print(f"   分析时间: {info['analysis_time']}")
        
        # 节理发育分析
        print("\n🔍 节理发育分析:")
        joint = result['joint_analysis']
        if joint:
            print(f"   检测到节理数量: {joint.get('total_lines', 0)} 条")
            print(f"   平均节理长度: {joint.get('avg_line_length', 0):.2f} 像素")
            print(f"   平均节理间距: {joint.get('avg_spacing', 0):.2f} 像素")
            print(f"   节理线密度: {joint.get('line_density', 0):.4f}")
            print(f"   发育程度评估: {joint.get('development_level', '未知')}")
        else:
            print("   节理分析失败")
        
        # 风化程度分析
        print("\n🌡️ 岩体风化程度分析:")
        weathering = result['weathering_analysis']
        if weathering:
            weathering_level = weathering.get('weathering_level', 'unknown')
            weathering_name = self.weathering_levels.get(weathering_level, '未知')
            print(f"   风化程度: {weathering_name}")
            print(f"   风化评分: {weathering.get('weathering_score', 0)}/7")
            print(f"   颜色饱和度均值: {weathering.get('s_mean', 0):.2f}")
            print(f"   纹理变化方差: {weathering.get('texture_variance', 0):.2f}")
            print(f"   表面粗糙度: {weathering.get('surface_roughness', 0):.2f}")
        else:
            print("   风化程度分析失败")
        
        # 湿润程度分析
        print("\n💧 掌子面湿润情况分析:")
        moisture = result['moisture_analysis']
        if moisture:
            moisture_level = moisture.get('moisture_level', 'unknown')
            moisture_name = self.moisture_levels.get(moisture_level, '未知')
            print(f"   湿润程度: {moisture_name}")
            print(f"   湿润评分: {moisture.get('moisture_score', 0)}/9")
            print(f"   暗色区域比例: {moisture.get('dark_area_ratio', 0):.4f}")
            print(f"   高饱和度区域比例: {moisture.get('high_sat_area_ratio', 0):.4f}")
            print(f"   反光区域比例: {moisture.get('bright_area_ratio', 0):.4f}")
        else:
            print("   湿润程度分析失败")
        
        # 围岩等级评估
        print("\n🏔️ 围岩等级智能评估:")
        evaluation = result['rock_quality_evaluation']
        if evaluation:
            print(f"   预测围岩等级: {evaluation.get('predicted_level', '未知')}")
            print(f"   等级代码: {evaluation.get('level_code', 'N/A')}")
        else:
            print("   围岩等级评估失败")
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)

class RockQualityAnalyzerGUI:
    """围岩等级分析器图形界面"""
    
    def __init__(self):
        self.analyzer = ComprehensiveRockQualityAnalyzer()
        self.current_image = None
        self.current_result = None
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Tunnel Face Rock Quality Intelligent Evaluation System")
        self.root.geometry("1200x800")
        
        self.setup_gui()
    
    def setup_gui(self):
        """设置图形界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧图像显示区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 图像显示标签
        self.image_label = ttk.Label(left_frame, text="Please select image file for analysis", 
                                   background="lightgray", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 右侧控制和结果区域
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # 控制按钮
        control_frame = ttk.LabelFrame(right_frame, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Select Image", command=self.select_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Save Report", command=self.save_report).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Results", command=self.clear_results).pack(fill=tk.X, pady=2)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(right_frame, text="Analysis Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动文本框
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 状态栏
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """选择图像文件"""
        file_types = [
            ('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG文件', '*.jpg *.jpeg'),
            ('PNG文件', '*.png'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Tunnel Face Image",
            filetypes=file_types
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """加载图像"""
        try:
            # 读取图像
            image = self.analyzer.read_image_chinese_path(file_path)
            if image is None:
                messagebox.showerror("Error", "Unable to read image file")
                return
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整图像大小以适应显示
            display_size = (600, 400)
            h, w = image_rgb.shape[:2]
            
            # 保持宽高比
            if w > h:
                new_w = display_size[0]
                new_h = int(h * display_size[0] / w)
            else:
                new_h = display_size[1]
                new_w = int(w * display_size[1] / h)
            
            resized_image = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 转换为PIL图像并显示
            pil_image = Image.fromarray(resized_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            
            self.current_image = file_path
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def start_analysis(self):
        """开始分析"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image file first")
            return
        
        # 在后台线程中进行分析
        self.status_var.set("Analyzing...")
        self.root.update()
        
        def analyze():
            try:
                result = self.analyzer.comprehensive_analysis(self.current_image)
                self.root.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Analysis failed"))
        
        thread = threading.Thread(target=analyze)
        thread.daemon = True
        thread.start()
    
    def display_results(self, result):
        """显示分析结果"""
        self.current_result = result
        
        # 清空文本框
        self.result_text.delete(1.0, tk.END)
        
        if not result['success']:
            self.result_text.insert(tk.END, f"Analysis failed: {result['error']}")
            self.status_var.set("Analysis failed")
            return
        
        # 格式化显示结果
        res = result['result']
        
        # 基本信息
        self.result_text.insert(tk.END, "=== Rock Quality Intelligent Assessment Report ===\n\n")
        
        info = res['image_info']
        self.result_text.insert(tk.END, f"File: {os.path.basename(info['file_path'])}\n")
        self.result_text.insert(tk.END, f"Size: {info['original_size']}\n")
        self.result_text.insert(tk.END, f"Time: {info['analysis_time']}\n\n")
        
        # 围岩等级评估（重点显示）
        evaluation = res['rock_quality_evaluation']
        if evaluation:
            self.result_text.insert(tk.END, "🏔️ Rock Quality Assessment:\n")
            self.result_text.insert(tk.END, f"  Grade: {evaluation.get('predicted_level', 'Unknown')}\n\n")
        
        # 节理发育分析
        joint = res['joint_analysis']
        if joint:
            self.result_text.insert(tk.END, "🔍 Joint Development Analysis:\n")
            self.result_text.insert(tk.END, f"  Development Level: {joint.get('development_level', 'Unknown')}\n")
            self.result_text.insert(tk.END, f"  Joint Count: {joint.get('total_lines', 0)} lines\n")
            self.result_text.insert(tk.END, f"  Average Length: {joint.get('avg_line_length', 0):.1f} px\n")
            self.result_text.insert(tk.END, f"  Average Spacing: {joint.get('avg_spacing', 0):.1f} px\n\n")
        
        # 风化程度分析
        weathering = res['weathering_analysis']
        if weathering:
            weathering_level = weathering.get('weathering_level', 'unknown')
            weathering_name = self.analyzer.weathering_levels.get(weathering_level, 'Unknown')
            self.result_text.insert(tk.END, "🌡️ Weathering Analysis:\n")
            self.result_text.insert(tk.END, f"  Weathering Level: {weathering_name}\n")
            self.result_text.insert(tk.END, f"  Score: {weathering.get('weathering_score', 0)}/7\n\n")
        
        # 湿润程度分析
        moisture = res['moisture_analysis']
        if moisture:
            moisture_level = moisture.get('moisture_level', 'unknown')
            moisture_name = self.analyzer.moisture_levels.get(moisture_level, 'Unknown')
            self.result_text.insert(tk.END, "💧 Moisture Analysis:\n")
            self.result_text.insert(tk.END, f"  Moisture Level: {moisture_name}\n")
            self.result_text.insert(tk.END, f"  Score: {moisture.get('moisture_score', 0)}/9\n\n")
        
        # 注意：根据用户要求，不显示匹配详情和评分
        
        self.status_var.set("Analysis completed")
    
    def save_report(self):
        """保存分析报告"""
        if not self.current_result or not self.current_result['success']:
            messagebox.showwarning("Warning", "No analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".txt",
            filetypes=[
                ('Text files', '*.txt'),
                ('JSON files', '*.json'),
                ('All files', '*.*')
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # 保存为JSON格式
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.current_result['result'], f, ensure_ascii=False, indent=2)
                else:
                    # 保存为文本格式
                    content = self.result_text.get(1.0, tk.END)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                messagebox.showinfo("Success", f"Report saved to: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def clear_results(self):
        """清除结果"""
        self.result_text.delete(1.0, tk.END)
        self.current_result = None
        self.status_var.set("Results cleared")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice == '2' and len(sys.argv) > 2:
            # 命令行模式直接处理图像
            analyzer = ComprehensiveRockQualityAnalyzer()
            image_path = sys.argv[2]
            print(f"\nStarting image analysis: {image_path}")
            
            # 执行分析
            result = analyzer.comprehensive_analysis(image_path)
            
            # 打印报告
            analyzer.print_analysis_report(result)
            return
        elif choice == '1':
            # 图形界面模式
            app = RockQualityAnalyzerGUI()
            app.run()
            return
    
    print("Tunnel Face Rock Quality Intelligent Assessment System")
    print("Functions: Joint Development + Weathering Degree + Moisture Condition + Rock Quality Comprehensive Assessment")
    print("\nSelect operation mode:")
    print("1. GUI mode")
    print("2. Command line mode")
    
    try:
        choice = input(f"\nPlease enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            # 图形界面模式
            app = RockQualityAnalyzerGUI()
            app.run()
        
        elif choice == '2':
            # 命令行模式
            analyzer = ComprehensiveRockQualityAnalyzer()
            
            # 默认图像路径
            default_path = r"C:\Users\ASUS\Desktop\科研+论文\AI_Recognition\节理裂隙\节理发育.JPG"
            
            image_path = input(f"\nPlease enter image path (press Enter for default path): ").strip()
            if not image_path:
                image_path = default_path
            
            print(f"\nStarting image analysis: {image_path}")
            
            # 执行分析
            result = analyzer.comprehensive_analysis(image_path)
            
            # 打印报告
            analyzer.print_analysis_report(result)
        
        else:
            print("Invalid choice, program exiting")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram error: {e}")

if __name__ == "__main__":
    main()