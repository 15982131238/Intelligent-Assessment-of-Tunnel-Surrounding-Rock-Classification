import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from typing import Tuple, List, Dict, Optional
from skimage import feature, filters, morphology, measure
from scipy import ndimage

# --- 配置参数 ---
class PreprocessConfig:
    """预处理配置类 - 专为围岩等级判断优化"""
    def __init__(self):
        # 基础路径配置 - 修正为从多个尺寸目录读取
        self.input_base_folder = r"C:\Users\ASUS\Desktop\AI_Recognition\processed_tunnel_face_photos\1"
        self.output_base_folder = r"C:\Users\ASUS\Desktop\AI_Recognition\processed_tunnel_face_photos\2"
        
        # 输入尺寸目录配置 - 从这些目录读取已处理的图像
        self.input_size_folders = {
            512: "size_512x512",
            1024: "size_1024x1024", 
            2048: "size_2048x2048",
            3024: "size_3024x3024"
        }
        
        # 选择处理哪个尺寸的图像作为输入（推荐使用1024）
        self.process_from_size = 1024  # 从1024x1024尺寸的图像开始处理
        
        # 多尺寸输出配置 - 针对围岩等级判断的不同精度需求
        self.target_sizes = [512, 1024, 2048, 3024]  # 多尺寸输出
        self.padding_color = (128, 128, 128)  # 灰色填充，更适合围岩图像
        
        # 图像质量增强配置 - 调整为更保守的策略
        self.enable_noise_reduction = True  # 噪声减少
        self.enable_gentle_enhancement = True  # 温和增强
        self.noise_reduction_method = 'gaussian'  # 改为高斯滤波，更温和
        self.bilateral_d = 5  # 减小滤波核大小
        self.bilateral_sigma_color = 50  # 降低颜色相似性阈值
        self.bilateral_sigma_space = 50  # 降低空间相似性阈值
        
        # 地质特征增强配置 - 大幅降低增强强度
        self.enable_edge_enhancement = True  # 边缘增强（突出节理）
        self.enable_texture_enhancement = True  # 纹理增强
        self.edge_enhancement_strength = 0.15  # 从0.3降低到0.15
        self.texture_enhancement_strength = 0.1  # 从0.2降低到0.1
        
        # 颜色空间配置 - 保持原始BGR，避免失真
        self.processing_color_space = 'BGR'  # 保持原始颜色空间
        self.output_color_space = 'BGR'  # 输出时的颜色空间
        self.enable_normalization = True
        self.enable_standardization = False  # 关闭标准化，保持原始特征
        
        # 围岩特征保护配置
        self.preserve_rock_texture = True  # 保护岩体纹理
        self.preserve_joint_features = True  # 保护节理特征
        self.preserve_color_information = True  # 保护颜色信息
        
        # 数据增强配置 - 针对地质图像的特殊增强
        self.enable_augmentation = True
        self.augmentation_params = {
            'rotation_angles': [-2, -1, 1, 2],  # 极小角度旋转，保持地质方向性
            'flip_horizontal': False,  # 关闭水平翻转，保持地质构造方向
            'brightness_range': (0.95, 1.05),  # 极小亮度调整
            'contrast_range': (0.98, 1.02),  # 极小对比度调整
            'geological_augmentation': True,  # 地质专用增强
        }
        
        # 日志配置
        self.enable_logging = True
        self.log_level = logging.INFO
        
        # 统计信息保存路径
        self.stats_file = r"C:\Users\ASUS\Desktop\AI_Recognition\dataset_statistics_v2.json"
        
        # 性能配置
        self.batch_size = 10  # 批处理大小
        self.save_intermediate_results = True  # 保存中间结果
        
        # 围岩等级相关配置
        self.rock_quality_assessment = True  # 启用岩体质量评估
        self.joint_detection = True  # 启用节理检测
        self.weathering_analysis = True  # 启用风化程度分析
        
        # 高级边缘和纹理增强配置 - 禁用可能导致过度处理的功能
        self.enable_advanced_edge_detection = False  # 禁用高级边缘检测
        self.enable_canny_edge_enhancement = False  # 禁用Canny边缘检测
        self.enable_gabor_texture_enhancement = False  # 禁用Gabor纹理增强
        self.enable_lbp_texture_analysis = False  # 禁用局部二值模式纹理分析
        self.enable_adaptive_enhancement = False  # 禁用自适应增强
        self.enable_multiscale_analysis = False  # 禁用多尺度分析
        
        # Canny边缘检测参数
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.canny_aperture_size = 3
        self.canny_l2_gradient = True
        
        # Gabor滤波器参数
        self.gabor_frequencies = [0.1, 0.3, 0.5]  # 多频率
        self.gabor_orientations = [0, 45, 90, 135]  # 多方向
        self.gabor_sigma_x = 2
        self.gabor_sigma_y = 2
        
        # 局部二值模式参数
        self.lbp_radius = 3
        self.lbp_n_points = 24
        self.lbp_method = 'uniform'
        
        # 自适应增强参数
        self.adaptive_block_size = 64  # 自适应块大小
        self.adaptive_overlap = 16  # 块重叠
        
        # 多尺度分析参数
        self.multiscale_levels = 3  # 尺度层数
        self.multiscale_sigma_base = 1.0  # 基础sigma值
        
        # 增强强度控制 - 全面降低所有增强强度
        self.edge_enhancement_strength = 0.2  # 从0.5降低到0.2
        self.texture_enhancement_strength = 0.15  # 从0.4降低到0.15
        self.canny_enhancement_strength = 0.1  # 从0.3降低到0.1
        self.gabor_enhancement_strength = 0.1  # 从0.25降低到0.1
        self.adaptive_enhancement_strength = 0.1  # 从0.3降低到0.1
        
        # CLAHE参数优化
        self.clahe_clip_limit = 1.2  # 降低CLAHE剪切限制
        self.clahe_tile_grid_size = (20, 20)  # 增大网格尺寸，减少局部对比度变化
        
        # 特征保护参数
        self.feature_preservation_threshold = 0.1  # 特征保护阈值
        self.geological_authenticity_weight = 0.7  # 地质真实性权重
        
class TunnelFaceImagePreprocessor:
    """隧道掌子面图像预处理器 - 专为围岩等级判断优化"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.dataset_stats = {'mean': None, 'std': None}
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'augmented_generated': 0,
            'processing_times': [],
            'geological_features_detected': 0
        }
        
        # 初始化地质统计信息
        self.geological_stats = {
            'total_joints': 0,
            'total_fractures': 0,
            'avg_rock_quality': 0.0,
            'processed_images': 0
        }
        
        # 设置日志
        self._setup_logging()
        
        # 创建输出目录结构
        self._create_output_directories()
        
        self.logger.info("隧道掌子面图像预处理器初始化完成（围岩等级专用版）")
        self.logger.info(f"输入尺寸: {self.config.process_from_size}x{self.config.process_from_size}")
        self.logger.info(f"目标尺寸: {self.config.target_sizes}")
        self.logger.info(f"处理颜色空间: {self.config.processing_color_space}")
        self.logger.info(f"地质特征保护: 纹理={self.config.preserve_rock_texture}, "
                        f"节理={self.config.preserve_joint_features}, "
                        f"颜色={self.config.preserve_color_information}")
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        if not self.config.enable_logging:
            return
            
        # 创建日志目录
        log_dir = os.path.join(self.config.output_base_folder, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        log_filename = f"geological_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _create_output_directories(self) -> None:
        """创建多尺寸输出目录结构"""
        # 创建基础输出目录
        os.makedirs(self.config.output_base_folder, exist_ok=True)
        
        # 为每个尺寸创建子目录
        for size in self.config.target_sizes:
            size_dir = os.path.join(self.config.output_base_folder, f'size_{size}x{size}')
            os.makedirs(size_dir, exist_ok=True)
            
            # 创建增强图像目录
            aug_dir = os.path.join(size_dir, 'augmented')
            os.makedirs(aug_dir, exist_ok=True)
            
            # 创建地质特征分析目录
            feature_dir = os.path.join(size_dir, 'geological_features')
            os.makedirs(feature_dir, exist_ok=True)
        
        # 创建报告目录
        report_dir = os.path.join(self.config.output_base_folder, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        self.logger.info(f"输出目录结构创建完成: {self.config.output_base_folder}")
    
    def _safe_cv2_findcontours(self, image, mode, method):
        """安全的cv2.findContours调用，兼容所有OpenCV版本"""
        try:
            # 尝试获取OpenCV版本
            cv_version = cv2.__version__.split('.')[0]
            
            # 调用findContours
            result = cv2.findContours(image, mode, method)
            
            # 根据OpenCV版本处理返回值
            if cv_version == '3':
                # OpenCV 3.x 返回 (image, contours, hierarchy)
                if len(result) == 3:
                    _, contours, hierarchy = result
                    return contours, hierarchy
                else:
                    # 备用方案
                    contours = result[-2] if len(result) >= 2 else result[0]
                    hierarchy = result[-1] if len(result) >= 3 else None
                    return contours, hierarchy
            else:
                # OpenCV 4.x 返回 (contours, hierarchy)
                if len(result) == 2:
                    contours, hierarchy = result
                    return contours, hierarchy
                else:
                    # 备用方案
                    contours = result[-2] if len(result) >= 2 else result[0]
                    hierarchy = result[-1] if len(result) >= 2 else None
                    return contours, hierarchy
                    
        except Exception as e:
            self.logger.warning(f"findContours调用失败: {e}，使用备用方案")
            # 最安全的备用方案
            result = cv2.findContours(image, mode, method)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[-2], result[-1] if len(result) >= 3 else None
            else:
                return [], None
    
    def _safe_cv2_houghlines(self, image, rho, theta, threshold):
        """安全的cv2.HoughLines调用"""
        try:
            lines = cv2.HoughLines(image, rho, theta, threshold)
            if lines is not None and len(lines) > 0:
                return lines
            else:
                return None
        except Exception as e:
            self.logger.warning(f"HoughLines调用失败: {e}")
            return None
    
    def _safe_line_extraction(self, lines):
        """安全的线条参数提取"""
        extracted_lines = []
        if lines is None:
            return extracted_lines
            
        try:
            for line in lines:
                if line is not None:
                    # 处理不同的线条数据结构
                    if isinstance(line, (list, tuple, np.ndarray)):
                        if len(line) > 0:
                            # 获取第一个元素
                            line_data = line[0] if hasattr(line, '__len__') and len(line) > 0 else line
                            
                            # 确保line_data是可索引的且有足够的元素
                            if hasattr(line_data, '__len__') and len(line_data) >= 2:
                                try:
                                    rho = float(line_data[0])
                                    theta = float(line_data[1])
                                    extracted_lines.append((rho, theta))
                                except (ValueError, TypeError, IndexError) as e:
                                    self.logger.debug(f"跳过无效线条数据: {e}")
                                    continue
        except Exception as e:
            self.logger.warning(f"线条参数提取失败: {e}")
            
        return extracted_lines
    
    def _geological_noise_reduction(self, img: np.ndarray) -> np.ndarray:
        """地质专用噪声减少 - 保持岩体纹理特征"""
        if not self.config.enable_noise_reduction:
            return img
        
        try:
            if self.config.noise_reduction_method == 'bilateral':
                # 使用更温和的双边滤波参数
                denoised = cv2.bilateralFilter(
                    img, 
                    d=3,  # 进一步减小邻域直径
                    sigmaColor=30,  # 降低颜色标准差
                    sigmaSpace=30   # 降低空间标准差
                )
            elif self.config.noise_reduction_method == 'gaussian':
                # 使用轻微的高斯模糊
                denoised = cv2.GaussianBlur(img, (3, 3), 0.5)
            else:
                # 使用非局部均值去噪，保持更多细节
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            
            # 与原图融合，保持更多原始细节
            result = cv2.addWeighted(img, 0.9, denoised, 0.1, 0)
            
            self.logger.debug(f"应用{self.config.noise_reduction_method}噪声减少")
            return result
        except Exception as e:
            self.logger.warning(f"噪声减少失败: {e}，返回原图")
            return img
    
    def _enhance_geological_features(self, img: np.ndarray) -> np.ndarray:
        """增强地质特征 - 突出节理、层理等结构"""
        try:
            enhanced_img = img.copy()
            
            # 边缘增强 - 突出节理
            if self.config.enable_edge_enhancement:
                enhanced_img = self._edge_enhancement(enhanced_img)
            
            # 纹理增强 - 突出岩体表面特征
            if self.config.enable_texture_enhancement:
                enhanced_img = self._texture_enhancement(enhanced_img)
            
            # 高级特征增强
            if self.config.enable_advanced_edge_detection:
                enhanced_img = self._advanced_edge_enhancement(enhanced_img)
            
            return enhanced_img
        except Exception as e:
            self.logger.warning(f"地质特征增强失败: {e}，返回原图")
            return img
    
    def _edge_enhancement(self, img: np.ndarray) -> np.ndarray:
        """边缘增强 - 使用Sobel算子突出节理"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Sobel边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 归一化
            sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 融合到原图
            enhanced_img = img.copy().astype(np.float32)
            edge_weight = self.config.edge_enhancement_strength
            
            for i in range(3):
                channel = enhanced_img[:, :, i]
                enhanced_channel = channel + edge_weight * sobel_normalized
                enhanced_img[:, :, i] = np.clip(enhanced_channel, 0, 255)
            
            self.logger.debug("应用边缘增强")
            return enhanced_img.astype(np.uint8)
        except Exception as e:
            self.logger.warning(f"边缘增强失败: {e}")
            return img
    
    def _texture_enhancement(self, img: np.ndarray) -> np.ndarray:
        """纹理增强 - 增强岩体表面纹理"""
        try:
            # 使用拉普拉斯算子增强纹理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 融合到原图
            enhanced_img = img.copy().astype(np.float32)
            texture_weight = self.config.texture_enhancement_strength
            
            for i in range(3):
                channel = enhanced_img[:, :, i]
                enhanced_channel = channel + texture_weight * laplacian_normalized
                enhanced_img[:, :, i] = np.clip(enhanced_channel, 0, 255)
            
            self.logger.debug("应用纹理增强")
            return enhanced_img.astype(np.uint8)
        except Exception as e:
            self.logger.warning(f"纹理增强失败: {e}")
            return img
    
    def _advanced_edge_enhancement(self, img: np.ndarray) -> np.ndarray:
        """高级边缘增强 - 组合多种方法"""
        try:
            enhanced_img = img.copy()
            
            # Canny边缘增强
            if self.config.enable_canny_edge_enhancement:
                enhanced_img = self._canny_edge_enhancement(enhanced_img)
            
            # Gabor纹理增强
            if self.config.enable_gabor_texture_enhancement:
                enhanced_img = self._gabor_texture_enhancement(enhanced_img)
            
            # LBP纹理分析
            if self.config.enable_lbp_texture_analysis:
                enhanced_img = self._lbp_texture_analysis(enhanced_img)
            
            # 多尺度特征增强
            if self.config.enable_multiscale_analysis:
                enhanced_img = self._multiscale_feature_enhancement(enhanced_img)
            
            return enhanced_img
        except Exception as e:
            self.logger.warning(f"高级边缘增强失败: {e}")
            return img
    
    def _gentle_contrast_enhancement(self, img: np.ndarray) -> np.ndarray:
        """温和的对比度增强 - 专为保持地质特征清晰度优化"""
        if not self.config.enable_gentle_enhancement:
            return img
        
        try:
            # 转换为LAB颜色空间进行亮度调整
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # 使用更温和的CLAHE参数
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,  # 进一步降低剪切限制
                tileGridSize=self.config.clahe_tile_grid_size  # 增大网格尺寸
            )
            l_channel = clahe.apply(l_channel)
            
            # 重新组合通道
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 与原图进行更温和的融合
            result = cv2.addWeighted(img, 0.8, enhanced_img, 0.2, 0)  # 降低增强权重
            
            self.logger.debug("应用温和对比度增强")
            return result
        except Exception as e:
            self.logger.warning(f"对比度增强失败: {e}")
            return img
    
    def _detect_geological_features(self, img: np.ndarray, filename: str, size: int) -> Dict:
        """检测地质特征 - 保守模式，避免过度处理"""
        features = {
            'joints_detected': 0,
            'fractures_detected': 0,
            'rock_quality_index': 0.0,
            'weathering_degree': 'unknown'
        }
        
        if not (self.config.rock_quality_assessment or self.config.joint_detection):
            return features
        
        try:
            # 转换为灰度图进行分析
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 使用更保守的边缘检测参数
            edges = cv2.Canny(gray, 30, 80, apertureSize=3)  # 降低阈值，检测更多细节
            
            # 简化的节理检测
            if self.config.joint_detection:
                lines = self._safe_cv2_houghlines(edges, 1, np.pi/180, 50)  # 降低阈值
                if lines is not None:
                    extracted_lines = self._safe_line_extraction(lines)
                    features['joints_detected'] = len(extracted_lines)
                    self.logger.debug(f"检测到 {len(extracted_lines)} 条潜在节理")
            
            # 简化的裂隙检测
            contours, _ = self._safe_cv2_findcontours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 过滤小轮廓
                significant_contours = []
                for c in contours:
                    try:
                        area = cv2.contourArea(c)
                        if area > 100:
                            significant_contours.append(c)
                    except Exception as e:
                        self.logger.debug(f"跳过无效轮廓: {e}")
                        continue
                features['fractures_detected'] = len(significant_contours)
            
            # 简化的岩体质量评估
            if self.config.rock_quality_assessment:
                # 基于边缘密度的简单评估
                edge_density = np.sum(edges > 0) / edges.size
                features['rock_quality_index'] = max(0.0, 1.0 - edge_density * 10)
            
            # 简化的风化程度评估
            color_variance = np.var(img)
            if color_variance < 1000:
                features['weathering_degree'] = 'high'
            elif color_variance < 3000:
                features['weathering_degree'] = 'medium'
            else:
                features['weathering_degree'] = 'low'
            
            # 保存特征分析图像（简化版）
            if self.config.save_intermediate_results:
                try:
                    feature_dir = os.path.join(
                        self.config.output_base_folder, 
                        f'size_{size}x{size}', 
                        'geological_features'
                    )
                    
                    # 只保存边缘检测结果
                    base_name = os.path.splitext(filename)[0]
                    edge_path = os.path.join(feature_dir, f"{base_name}_edges.jpg")
                    cv2.imwrite(edge_path, edges)
                except Exception as e:
                    self.logger.warning(f"保存特征分析图像失败: {e}")
            
            # 更新地质特征统计
            if not hasattr(self, 'geological_stats'):
                self.geological_stats = {
                    'total_joints': 0,
                    'total_fractures': 0,
                    'avg_rock_quality': 0.0,
                    'processed_images': 0
                }
            
            self.geological_stats['processed_images'] += 1
            self.geological_stats['total_joints'] += features.get('joints_detected', 0)
            self.geological_stats['total_fractures'] += features.get('fractures_detected', 0)
            self.geological_stats['avg_rock_quality'] = (
                (self.geological_stats['avg_rock_quality'] * (self.geological_stats['processed_images'] - 1) + 
                 features.get('rock_quality_index', 0)) / self.geological_stats['processed_images']
            )
            
            self.processing_stats['geological_features_detected'] += 1
            
            self.logger.debug(f"地质特征检测完成: {features}")
            return features
            
        except Exception as e:
            self.logger.warning(f"地质特征检测失败: {e}")
            return features
    
    def _process_image_pipeline(self, img: np.ndarray, for_statistics: bool = False) -> np.ndarray:
        """完整的地质图像处理流水线 - 优化为保持清晰度"""
        try:
            processed_img = img.copy()
            
            # 步骤1: 轻微噪声减少（如果启用）
            if self.config.enable_noise_reduction:
                processed_img = self._geological_noise_reduction(processed_img)
            
            # 步骤2: 温和对比度增强（如果启用）
            if self.config.enable_gentle_enhancement:
                processed_img = self._gentle_contrast_enhancement(processed_img)
            
            # 步骤3: 跳过其他可能导致过度处理的步骤
            # 注释掉边缘和纹理增强，保持原始清晰度
            # if self.config.preserve_joint_features:
            #     processed_img = self._enhance_geological_features(processed_img)
            
            self.logger.debug("图像处理流水线完成（清晰度优化版）")
            return processed_img
        except Exception as e:
            self.logger.warning(f"图像处理流水线失败: {e}，返回原图")
            return img
    
    def _resize_and_pad(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """调整尺寸并填充为正方形 - 保持地质特征比例"""
        try:
            original_height, original_width = img.shape[:2]
            ratio = float(target_size) / max(original_height, original_width)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            
            # 使用高质量插值，保持地质纹理
            if ratio > 1:
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
            else:
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            
            # 填充到正方形 - 使用灰色填充
            delta_w = target_size - new_size[0]
            delta_h = target_size - new_size[1]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            padded_img = cv2.copyMakeBorder(
                resized_img, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=self.config.padding_color
            )
            
            return padded_img
        except Exception as e:
            self.logger.error(f"图像尺寸调整失败: {e}")
            # 返回一个默认尺寸的图像
            return np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    
    def _generate_geological_augmented_images(self, img: np.ndarray, base_filename: str) -> List[Tuple[np.ndarray, str]]:
        """生成地质专用数据增强图像 - 保持地质构造特征"""
        augmented_images = []
        
        if not self.config.enable_augmentation:
            return augmented_images
        
        try:
            # 极小角度旋转 - 保持地质方向性
            for angle in self.config.augmentation_params['rotation_angles']:
                try:
                    center = (img.shape[1] // 2, img.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), 
                                           borderValue=self.config.padding_color)
                    augmented_images.append((rotated, f"{base_filename}_rot{angle}"))
                except Exception as e:
                    self.logger.debug(f"旋转增强失败 {angle}度: {e}")
                    continue
            
            # 亮度和对比度的极小调整
            brightness_min, brightness_max = self.config.augmentation_params['brightness_range']
            contrast_min, contrast_max = self.config.augmentation_params['contrast_range']
            
            # 极小亮度调整
            try:
                bright_factor = np.random.uniform(brightness_min, brightness_max)
                brightened = cv2.convertScaleAbs(img, alpha=1.0, beta=(bright_factor - 1.0) * 10)
                augmented_images.append((brightened, f"{base_filename}_bright"))
            except Exception as e:
                self.logger.debug(f"亮度增强失败: {e}")
            
            # 极小对比度调整
            try:
                contrast_factor = np.random.uniform(contrast_min, contrast_max)
                contrasted = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
                augmented_images.append((contrasted, f"{base_filename}_contrast"))
            except Exception as e:
                self.logger.debug(f"对比度增强失败: {e}")
            
            # 地质专用增强 - 局部对比度增强
            if self.config.augmentation_params.get('geological_augmentation', False):
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l_channel = lab[:, :, 0]
                    
                    # 应用局部直方图均衡化
                    clahe_local = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
                    l_enhanced = clahe_local.apply(l_channel)
                    
                    lab[:, :, 0] = l_enhanced
                    geological_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    augmented_images.append((geological_enhanced, f"{base_filename}_geological"))
                except Exception as e:
                    self.logger.debug(f"地质专用增强失败: {e}")
            
        except Exception as e:
            self.logger.warning(f"数据增强过程失败: {e}")
        
        return augmented_images
    
    def process_single_image(self, img_path: str, filename: str) -> Dict:
        """处理单张图像 - 地质专用处理流程"""
        start_time = time.time()
        result = {
            'filename': filename,
            'success': False,
            'sizes_processed': [],
            'augmented_count': 0,
            'geological_features': {},
            'error_message': None,
            'processing_time': 0
        }
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            self.logger.info(f"开始处理地质图像: {filename} (尺寸: {img.shape})")
            
            # 应用地质图像处理流水线
            processed_img = self._process_image_pipeline(img)
            
            # 处理每个目标尺寸
            for target_size in self.config.target_sizes:
                try:
                    # 调整尺寸和填充
                    resized_img = self._resize_and_pad(processed_img, target_size)
                    
                    # 地质特征检测
                    geological_features = self._detect_geological_features(
                        resized_img, filename, target_size
                    )
                    result['geological_features'][f'size_{target_size}'] = geological_features
                    
                    # 归一化（保持在BGR空间）
                    if self.config.enable_normalization:
                        normalized_img = resized_img.astype(np.float32) / 255.0
                        final_img = normalized_img
                    else:
                        final_img = resized_img.astype(np.float32)
                    
                    # 保存主图像
                    base_name, ext = os.path.splitext(filename)
                    output_filename = f"{base_name}_geological{ext}"
                    size_dir = os.path.join(self.config.output_base_folder, f'size_{target_size}x{target_size}')
                    output_path = os.path.join(size_dir, output_filename)
                    
                    # 转换回保存格式
                    if self.config.enable_normalization:
                        save_img = (final_img * 255).astype(np.uint8)
                    else:
                        save_img = final_img.astype(np.uint8)
                    
                    cv2.imwrite(output_path, save_img)
                    result['sizes_processed'].append(target_size)
                    
                    self.logger.debug(f"已保存地质处理版本 {target_size}x{target_size}: {output_path}")
                    
                    # 生成地质专用数据增强图像
                    if self.config.enable_augmentation:
                        augmented_images = self._generate_geological_augmented_images(resized_img, base_name)
                        aug_dir = os.path.join(size_dir, 'augmented')
                        
                        for aug_img, aug_name in augmented_images:
                            try:
                                # 应用相同的归一化
                                if self.config.enable_normalization:
                                    aug_normalized = aug_img.astype(np.float32) / 255.0
                                    aug_final = aug_normalized
                                else:
                                    aug_final = aug_img.astype(np.float32)
                                
                                # 保存增强图像
                                aug_output_filename = f"{aug_name}{ext}"
                                aug_output_path = os.path.join(aug_dir, aug_output_filename)
                                
                                if self.config.enable_normalization:
                                    aug_save_img = (aug_final * 255).astype(np.uint8)
                                else:
                                    aug_save_img = aug_final.astype(np.uint8)
                                
                                cv2.imwrite(aug_output_path, aug_save_img)
                                result['augmented_count'] += 1
                            except Exception as e:
                                self.logger.debug(f"保存增强图像失败: {e}")
                                continue
                    
                except Exception as e:
                    self.logger.error(f"处理尺寸 {target_size}x{target_size} 时出错: {e}")
                    continue
            
            result['success'] = len(result['sizes_processed']) > 0
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"处理地质图像 {filename} 时出错: {e}")
        
        finally:
            result['processing_time'] = time.time() - start_time
            self.processing_stats['processing_times'].append(result['processing_time'])
        
        return result
    
    def process_all_images(self) -> None:
        """处理所有地质图像 - 从指定尺寸目录读取"""
        self.logger.info("开始批量处理地质图像...")
        
        # 构建输入目录路径
        size_folder_name = self.config.input_size_folders[self.config.process_from_size]
        input_folder = os.path.join(self.config.input_base_folder, size_folder_name)
        
        if not os.path.exists(input_folder):
            self.logger.error(f"输入目录不存在: {input_folder}")
            return
        
        # 获取所有图像文件（排除metadata文件）
        all_files = os.listdir(input_folder)
        image_files = [f for f in all_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) 
                      and not f.endswith('_metadata.json')]
        
        if not image_files:
            self.logger.error(f"在目录 {input_folder} 中未找到有效的图像文件")
            return
        
        self.logger.info(f"从 {input_folder} 找到 {len(image_files)} 张地质图像，开始处理...")
        
        # 处理每张图像
        for index, filename in enumerate(image_files):
            input_path = os.path.join(input_folder, filename)
            
            self.logger.info(f"地质处理进度: ({index+1}/{len(image_files)}) - {filename}")
            
            result = self.process_single_image(input_path, filename)
            
            # 更新统计信息
            self.processing_stats['total_processed'] += 1
            if result['success']:
                self.processing_stats['successful_processed'] += 1
                self.processing_stats['augmented_generated'] += result['augmented_count']
            else:
                self.processing_stats['failed_processed'] += 1
            
            # 记录处理结果
            if result['success']:
                geological_summary = ""
                for size, features in result['geological_features'].items():
                    geological_summary += f"{size}: 节理{features['joints_detected']}, 裂隙{features['fractures_detected']}, RQI{features['rock_quality_index']:.2f}; "
                
                self.logger.info(f"✓ {filename} 地质处理成功 - 尺寸: {result['sizes_processed']}, "
                               f"增强: {result['augmented_count']}, 地质特征: {geological_summary}"
                               f"耗时: {result['processing_time']:.2f}s")
            else:
                self.logger.error(f"✗ {filename} 地质处理失败 - {result['error_message']}")
        
        # 生成最终报告
        self._generate_geological_report()
        
        self.logger.info("所有地质图像处理完成！")
    
    def _generate_geological_report(self) -> None:
        """生成地质专用处理报告"""
        try:
            # 计算统计信息
            total_processed = self.processing_stats['total_processed']
            successful = self.processing_stats['successful_processed']
            failed = self.processing_stats['failed_processed']
            success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
            avg_time = np.mean(self.processing_stats['processing_times']) if self.processing_stats['processing_times'] else 0
            
            # 地质统计信息
            avg_joints = self.geological_stats['total_joints'] / max(1, self.geological_stats['processed_images'])
            avg_fractures = self.geological_stats['total_fractures'] / max(1, self.geological_stats['processed_images'])
            
            # 创建详细报告
            report = {
                'processing_summary': {
                    'total_images': total_processed,
                    'successful_processed': successful,
                    'failed_processed': failed,
                    'success_rate_percent': round(success_rate, 2),
                    'augmented_images_generated': self.processing_stats['augmented_generated'],
                    'average_processing_time_seconds': round(avg_time, 3)
                },
                'geological_analysis': {
                    'total_images_analyzed': self.geological_stats['processed_images'],
                    'total_joints_detected': self.geological_stats['total_joints'],
                    'total_fractures_detected': self.geological_stats['total_fractures'],
                    'average_joints_per_image': round(avg_joints, 2),
                    'average_fractures_per_image': round(avg_fractures, 2),
                    'average_rock_quality_index': round(self.geological_stats['avg_rock_quality'], 3)
                },
                'configuration': {
                    'input_size': f"{self.config.process_from_size}x{self.config.process_from_size}",
                    'target_sizes': self.config.target_sizes,
                    'geological_features_enabled': {
                        'rock_texture_preservation': self.config.preserve_rock_texture,
                        'joint_features_preservation': self.config.preserve_joint_features,
                        'color_information_preservation': self.config.preserve_color_information,
                        'rock_quality_assessment': self.config.rock_quality_assessment,
                        'joint_detection': self.config.joint_detection
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存详细报告
            report_dir = os.path.join(self.config.output_base_folder, 'reports')
            report_filename = f"geological_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(report_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 生成简要摘要
            summary_filename = f"geological_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            summary_path = os.path.join(report_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("隧道掌子面地质图像处理报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图像数: {total_processed}\n")
                f.write(f"成功处理: {successful}\n")
                f.write(f"处理失败: {failed}\n")
                f.write(f"成功率: {success_rate:.1f}%\n")
                f.write(f"生成增强图像: {self.processing_stats['augmented_generated']}\n")
                f.write(f"地质特征分析: {self.geological_stats['processed_images']}\n")
                f.write(f"平均节理数: {avg_joints:.1f}\n")
                f.write(f"平均裂隙数: {avg_fractures:.1f}\n")
                f.write(f"平均岩体质量指数: {self.geological_stats['avg_rock_quality']:.3f}\n")
                f.write(f"平均处理时间: {avg_time:.2f}秒\n")
                f.write(f"输出目录: {self.config.output_base_folder}\n")
            
            self.logger.info(f"地质处理报告已保存: {report_path}")
            self.logger.info(f"地质处理摘要已保存: {summary_path}")
            
            # 打印摘要到控制台
            print("\n" + "=" * 70)
            print("地质图像处理完成摘要（围岩等级专用版）")
            print("=" * 70)
            print(f"总图像数: {total_processed}")
            print(f"成功处理: {successful}")
            print(f"处理失败: {failed}")
            print(f"成功率: {success_rate:.1f}%")
            print(f"生成增强图像: {self.processing_stats['augmented_generated']}")
            print(f"地质特征分析: {self.geological_stats['processed_images']}")
            print(f"平均处理时间: {avg_time:.2f}秒")
            print(f"输出目录: {self.config.output_base_folder}")
            print("=" * 70)
            
        except Exception as e:
            self.logger.error(f"生成地质报告失败: {e}")
    
    # 高级特征增强方法的安全实现
    def _canny_edge_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Canny边缘增强 - 突出地质结构边界"""
        if not self.config.enable_canny_edge_enhancement:
            return img
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Canny边缘检测
            edges = cv2.Canny(
                gray, 
                self.config.canny_low_threshold, 
                self.config.canny_high_threshold,
                apertureSize=self.config.canny_aperture_size,
                L2gradient=self.config.canny_l2_gradient
            )
            
            # 膨胀操作增强边缘
            kernel = np.ones((2, 2), np.uint8)
            edges_enhanced = cv2.dilate(edges, kernel, iterations=1)
            
            # 融合到原图
            enhanced_img = img.copy().astype(np.float32)
            edge_weight = self.config.canny_enhancement_strength
            
            for i in range(3):
                channel = enhanced_img[:, :, i]
                # 在边缘位置增强对比度
                edge_mask = edges_enhanced.astype(np.float32) / 255.0
                enhanced_channel = channel + edge_weight * edge_mask * 255
                enhanced_img[:, :, i] = np.clip(enhanced_channel, 0, 255)
            
            self.logger.debug("应用Canny边缘增强")
            return enhanced_img.astype(np.uint8)
        except Exception as e:
            self.logger.warning(f"Canny边缘增强失败: {e}")
            return img
    
    def _gabor_texture_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Gabor滤波器纹理增强 - 增强层理结构"""
        if not self.config.enable_gabor_texture_enhancement:
            return img
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            enhanced_img = img.copy().astype(np.float32)
            
            # 多方向多频率Gabor滤波
            gabor_responses = []
            
            for freq in self.config.gabor_frequencies:
                for angle in self.config.gabor_orientations:
                    try:
                        # 创建Gabor滤波器
                        real, _ = cv2.getGaborKernel(
                            (21, 21), 
                            self.config.gabor_sigma_x, 
                            np.radians(angle), 
                            2 * np.pi * freq, 
                            0.5, 0, ktype=cv2.CV_32F
                        )
                        
                        # 应用滤波器
                        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, real)
                        gabor_responses.append(gabor_response)
                    except Exception as e:
                        self.logger.debug(f"Gabor滤波器创建失败 freq={freq}, angle={angle}: {e}")
                        continue
            
            if gabor_responses:
                # 合并所有Gabor响应
                gabor_combined = np.zeros_like(gray)
                for response in gabor_responses:
                    gabor_combined += np.abs(response)
                
                # 归一化
                gabor_combined = cv2.normalize(gabor_combined, None, 0, 255, cv2.NORM_MINMAX)
                
                # 融合到原图
                gabor_weight = self.config.gabor_enhancement_strength
                for i in range(3):
                    channel = enhanced_img[:, :, i]
                    enhanced_channel = channel + gabor_weight * gabor_combined
                    enhanced_img[:, :, i] = np.clip(enhanced_channel, 0, 255)
            
            self.logger.debug("应用Gabor纹理增强")
            return enhanced_img.astype(np.uint8)
        except Exception as e:
            self.logger.warning(f"Gabor纹理增强失败: {e}")
            return img
    
    def _lbp_texture_analysis(self, img: np.ndarray) -> np.ndarray:
        """局部二值模式纹理分析 - 增强岩石表面纹理"""
        if not self.config.enable_lbp_texture_analysis:
            return img
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 计算LBP
            lbp = feature.local_binary_pattern(
                gray, 
                self.config.lbp_n_points, 
                self.config.lbp_radius, 
                method=self.config.lbp_method
            )
            
            # 归一化LBP
            lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 应用高斯滤波平滑LBP
            lbp_smooth = cv2.GaussianBlur(lbp_normalized, (5, 5), 0)
            
            # 融合到原图
            enhanced_img = img.copy().astype(np.float32)
            lbp_weight = self.config.texture_enhancement_strength * 0.5
            
            for i in range(3):
                channel = enhanced_img[:, :, i]
                enhanced_channel = channel + lbp_weight * lbp_smooth
                enhanced_img[:, :, i] = np.clip(enhanced_channel, 0, 255)
            
            self.logger.debug("应用LBP纹理分析")
            return enhanced_img.astype(np.uint8)
        except Exception as e:
            self.logger.warning(f"LBP纹理分析失败: {e}")
            return img
    
    def _multiscale_feature_enhancement(self, img: np.ndarray) -> np.ndarray:
        """多尺度特征增强 - 在不同尺度下增强地质特征"""
        if not self.config.enable_multiscale_analysis:
            return img
        
        try:
            # 转换为灰度图像进行处理
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 定义多个尺度
            scales = [0.5, 1.0, 1.5, 2.0]
            enhanced_features = []
            
            original_height, original_width = gray.shape
            
            for scale in scales:
                # 计算缩放后的尺寸
                new_height = int(original_height * scale)
                new_width = int(original_width * scale)
                
                # 缩放图像
                if scale != 1.0:
                    scaled_img = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                else:
                    scaled_img = gray.copy()
                
                # 在当前尺度下进行特征增强
                # 1. Gabor滤波器增强纹理
                gabor_kernel = cv2.getGaborKernel((21, 21), 5, np.pi/4, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
                gabor_filtered = cv2.filter2D(scaled_img, cv2.CV_8UC3, gabor_kernel)
                
                # 2. 拉普拉斯算子增强边缘
                laplacian = cv2.Laplacian(scaled_img, cv2.CV_64F, ksize=3)
                laplacian = np.uint8(np.absolute(laplacian))
                
                # 3. Sobel算子检测梯度
                sobelx = cv2.Sobel(scaled_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(scaled_img, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = np.sqrt(sobelx**2 + sobely**2)
                sobel_combined = np.uint8(sobel_combined)
                
                # 4. 形态学操作增强结构特征
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                morph_gradient = cv2.morphologyEx(scaled_img, cv2.MORPH_GRADIENT, kernel)
                
                # 组合多种特征
                combined_features = cv2.addWeighted(gabor_filtered, 0.3, laplacian, 0.3, 0)
                combined_features = cv2.addWeighted(combined_features, 0.7, sobel_combined, 0.2, 0)
                combined_features = cv2.addWeighted(combined_features, 0.8, morph_gradient, 0.2, 0)
                
                # 将结果缩放回原始尺寸
                if scale != 1.0:
                    combined_features = cv2.resize(combined_features, (original_width, original_height), 
                                                 interpolation=cv2.INTER_CUBIC)
                
                enhanced_features.append(combined_features)
            
            # 融合所有尺度的特征
            final_enhanced = np.zeros_like(enhanced_features[0], dtype=np.float64)
            weights = [0.2, 0.4, 0.25, 0.15]  # 不同尺度的权重
            
            for i, feature in enumerate(enhanced_features):
                final_enhanced += feature.astype(np.float64) * weights[i]
            
            # 归一化到0-255范围
            final_enhanced = np.clip(final_enhanced, 0, 255).astype(np.uint8)
            
            # 如果原图是彩色的，将增强结果应用到原图
            if len(img.shape) == 3:
                # 将增强的灰度图转换为3通道
                enhanced_3ch = cv2.cvtColor(final_enhanced, cv2.COLOR_GRAY2BGR)
                # 与原图融合
                result = cv2.addWeighted(img, 0.7, enhanced_3ch, 0.3, 0)
            else:
                result = final_enhanced
            
            self.logger.debug(f"多尺度特征增强完成，使用{len(scales)}个尺度")
            return result
            
        except Exception as e:
            self.logger.warning(f"多尺度特征增强失败: {e}")
            return img

if __name__ == "__main__":
    try:
        # 创建配置
        config = PreprocessConfig()
        
        # 显示配置信息
        print("=" * 60)
        print("隧道掌子面图像预处理系统 - 增强版")
        print("=" * 60)
        print(f"输入基础目录: {config.input_base_folder}")
        print(f"输出基础目录: {config.output_base_folder}")
        print(f"处理尺寸: {config.process_from_size}x{config.process_from_size}")
        print(f"目标尺寸: {config.target_sizes}")
        print(f"增强参数: {config.augmentation_params}")
        print(f"岩体质量评估: {'启用' if config.rock_quality_assessment else '禁用'}")
        print(f"节理检测: {'启用' if config.joint_detection else '禁用'}")
        print(f"多尺度分析: {'启用' if config.enable_multiscale_analysis else '禁用'}")
        print("=" * 60)
        
        # 创建预处理器
        preprocessor = TunnelFaceImagePreprocessor(config)
        
        # 处理所有图像
        preprocessor.process_all_images()
        
        print("\n图像预处理完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        
