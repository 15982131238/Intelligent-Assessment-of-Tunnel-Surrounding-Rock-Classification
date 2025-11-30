import cv2
import numpy as np
import os
import logging
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import shutil

class TunnelFaceImagePreprocessor:
    """隧道掌子面图像预处理器 - 专为围岩等级判断优化"""
    
    def __init__(self, config: Dict = None):
        """初始化预处理器
        
        Args:
            config: 配置字典，包含各种预处理参数
        """
        # 默认配置
        self.default_config = {
            'input_folder': r"C:\Users\ASUS\Desktop\AI_Recognition\tunnel_face_images",
            'output_folder': r"C:\Users\ASUS\Desktop\AI_Recognition\processed_tunnel_face_photos\1",
            'target_sizes': [512, 1024, 2048, 3024],  # 增加3024尺寸
            'padding_color': (0, 0, 0),
            'enable_histogram_equalization': True,
            'enable_median_filter': True,
            'median_filter_kernel': 5,
            'enable_data_augmentation': True,
            'augmentation_count': 3,
            'rotation_angles': [-15, -10, -5, 5, 10, 15],
            'brightness_factors': [0.8, 0.9, 1.1, 1.2],
            'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'],
            'quality_threshold': 0.3,  # 图像质量阈值
            'log_level': 'INFO'
        }
        
        # 合并用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 设置日志
        self._setup_logging()
        
        # 创建输出目录
        self._create_output_directories()
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'augmented_images': 0,
            'processing_times': [],
            'quality_scores': [],
            'best_copied_images': 0
        }
    
    def _setup_logging(self):
        """设置日志记录"""
        log_dir = Path(self.config['output_folder']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("隧道掌子面图像预处理器初始化完成")
        self.logger.info(f"配置参数: {json.dumps(self.config, ensure_ascii=False, indent=2)}")
    
    def _create_output_directories(self):
        """创建输出目录结构"""
        base_output = Path(self.config['output_folder'])
        
        # 创建best文件夹
        best_dir = base_output / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        
        # 为不同尺寸创建子目录
        for size in self.config['target_sizes']:
            size_dir = base_output / f"size_{size}x{size}"
            size_dir.mkdir(parents=True, exist_ok=True)
            
            # 如果启用数据增强，创建增强子目录
            if self.config['enable_data_augmentation']:
                aug_dir = size_dir / "augmented"
                aug_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"输出目录结构创建完成: {base_output}")
        self.logger.info(f"Best文件夹创建完成: {best_dir}")
    
    def _calculate_adaptive_size(self, image_shape: Tuple[int, int]) -> int:
        """根据图像尺寸自适应选择目标尺寸
        
        Args:
            image_shape: (height, width)
            
        Returns:
            最适合的目标尺寸
        """
        height, width = image_shape
        max_dimension = max(height, width)
        
        # 根据原始图像尺寸选择合适的目标尺寸
        if max_dimension <= 1024:
            return 512
        elif max_dimension <= 2048:
            return 1024
        elif max_dimension <= 3024:
            return 2048
        else:
            return 3024
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """评估图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            质量分数 (0-1)
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算拉普拉斯方差（清晰度指标）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算对比度
            contrast = gray.std()
            
            # 计算亮度分布均匀性
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
            # 综合质量分数
            quality_score = (
                min(laplacian_var / 1000, 1.0) * 0.4 +  # 清晰度权重40%
                min(contrast / 100, 1.0) * 0.3 +        # 对比度权重30%
                entropy / 8.0 * 0.3                     # 信息熵权重30%
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"质量评估失败: {e}")
            return 0.5  # 默认中等质量
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """增强图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        try:
            # 1. 中值滤波去噪
            if self.config['enable_median_filter']:
                kernel_size = self.config['median_filter_kernel']
                enhanced = cv2.medianBlur(enhanced, kernel_size)
                self.logger.debug(f"应用中值滤波，核大小: {kernel_size}")
            
            # 2. 直方图均衡化增强对比度
            if self.config['enable_histogram_equalization']:
                # 转换到LAB色彩空间，只对L通道进行直方图均衡化
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                # 应用CLAHE（限制对比度自适应直方图均衡化）
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                
                # 重新合并通道
                enhanced = cv2.merge([l_channel, a_channel, b_channel])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                self.logger.debug("应用CLAHE直方图均衡化")
            
            # 3. 轻微的锐化处理（针对隧道环境的模糊问题）
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {e}")
            return image
    
    def _resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """调整图像尺寸并填充
        
        Args:
            image: 输入图像
            target_size: 目标正方形尺寸
            
        Returns:
            调整后的图像
        """
        height, width = image.shape[:2]
        
        # 计算缩放比例
        ratio = float(target_size) / max(height, width)
        new_size = (int(width * ratio), int(height * ratio))
        
        # 缩放图像
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # 计算填充
        delta_w = target_size - new_size[0]
        delta_h = target_size - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # 应用填充
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=self.config['padding_color']
        )
        
        return padded
    
    def _generate_augmented_images(self, image: np.ndarray, base_filename: str) -> List[Tuple[np.ndarray, str]]:
        """生成数据增强图像
        
        Args:
            image: 原始图像
            base_filename: 基础文件名
            
        Returns:
            增强图像列表，每个元素为(图像, 文件名后缀)
        """
        augmented_images = []
        
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # 1. 旋转增强（小角度，适合隧道掌子面）
            for angle in self.config['rotation_angles']:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                       borderMode=cv2.BORDER_REFLECT_101)
                augmented_images.append((rotated, f"_rot{angle}"))
            
            # 2. 翻转增强
            # 水平翻转（对隧道掌子面合理）
            h_flipped = cv2.flip(image, 1)
            augmented_images.append((h_flipped, "_hflip"))
            
            # 注意：不进行垂直翻转，因为会违反重力方向
            
            # 3. 亮度调整
            for factor in self.config['brightness_factors']:
                # 转换到HSV色彩空间调整亮度
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 2] = hsv[:, :, 2] * factor  # 调整V通道
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                hsv = hsv.astype(np.uint8)
                brightness_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                augmented_images.append((brightness_adjusted, f"_bright{factor}"))
            
            # 4. 对比度调整
            for alpha in [0.8, 1.2]:  # 对比度因子
                contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                augmented_images.append((contrast_adjusted, f"_contrast{alpha}"))
            
            # 限制增强数量
            if len(augmented_images) > self.config['augmentation_count']:
                # 随机选择指定数量的增强
                import random
                augmented_images = random.sample(augmented_images, self.config['augmentation_count'])
            
            self.logger.debug(f"为 {base_filename} 生成了 {len(augmented_images)} 个增强图像")
            return augmented_images
            
        except Exception as e:
            self.logger.error(f"数据增强失败 {base_filename}: {e}")
            return []
    
    def _save_image_with_metadata(self, image: np.ndarray, filepath: str, metadata: Dict) -> bool:
        """保存图像并记录元数据
        
        Args:
            image: 要保存的图像
            filepath: 保存路径
            metadata: 元数据信息
            
        Returns:
            是否保存成功
        """
        try:
            # 保存图像
            success = cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                # 保存元数据
                metadata_file = filepath.replace('.jpg', '_metadata.json').replace('.JPG', '_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                self.logger.debug(f"成功保存: {filepath}")
                return True
            else:
                self.logger.error(f"保存失败: {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"保存图像时出错 {filepath}: {e}")
            return False
    
    def _copy_to_best_folder(self, source_dir: Path, best_dir: Path) -> int:
        """将处理后的图像复制到best文件夹
        
        Args:
            source_dir: 源目录
            best_dir: best目录
            
        Returns:
            复制的文件数量
        """
        copied_count = 0
        try:
            if source_dir.exists():
                # 复制所有图像文件
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        # 计算相对路径
                        relative_path = file_path.relative_to(source_dir)
                        dest_path = best_dir / relative_path
                        
                        # 创建目标目录
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 复制文件
                        shutil.copy2(file_path, dest_path)
                        copied_count += 1
                        self.logger.debug(f"复制到best文件夹: {dest_path}")
                        
                        # 同时复制对应的元数据文件
                        metadata_file = file_path.with_suffix('_metadata.json')
                        if metadata_file.exists():
                            dest_metadata = dest_path.with_suffix('_metadata.json')
                            shutil.copy2(metadata_file, dest_metadata)
                            
        except Exception as e:
            self.logger.error(f"复制到best文件夹失败: {e}")
            
        return copied_count
    
    def process_single_image(self, image_path: str) -> Dict:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果字典
        """
        start_time = datetime.now()
        result = {
            'filename': os.path.basename(image_path),
            'success': False,
            'error': None,
            'quality_score': 0.0,
            'target_sizes_processed': [],
            'augmented_count': 0,
            'processing_time': 0.0
        }
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            self.logger.info(f"开始处理: {result['filename']}")
            
            # 评估原始图像质量
            quality_score = self._assess_image_quality(image)
            result['quality_score'] = quality_score
            self.stats['quality_scores'].append(quality_score)
            
            # 检查质量阈值
            if quality_score < self.config['quality_threshold']:
                self.logger.warning(f"图像质量较低 ({quality_score:.3f}): {result['filename']}")
            
            # 增强图像质量
            enhanced_image = self._enhance_image_quality(image)
            
            # 处理所有目标尺寸
            base_name, ext = os.path.splitext(result['filename'])
            
            for target_size in self.config['target_sizes']:
                try:
                    # 调整尺寸
                    processed_image = self._resize_with_padding(enhanced_image, target_size)
                    
                    # 归一化
                    normalized_image = processed_image.astype(np.float32) / 255.0
                    saveable_image = (normalized_image * 255).astype(np.uint8)
                    
                    # 准备保存路径和元数据
                    output_dir = Path(self.config['output_folder']) / f"size_{target_size}x{target_size}"
                    output_filename = f"{base_name}_{target_size}x{target_size}_processed{ext}"
                    output_path = output_dir / output_filename
                    
                    metadata = {
                        'original_file': result['filename'],
                        'original_size': image.shape[:2],
                        'target_size': [target_size, target_size],
                        'quality_score': quality_score,
                        'processing_timestamp': datetime.now().isoformat(),
                        'enhancements_applied': {
                            'histogram_equalization': self.config['enable_histogram_equalization'],
                            'median_filter': self.config['enable_median_filter'],
                            'sharpening': True
                        }
                    }
                    
                    # 保存主图像
                    if self._save_image_with_metadata(saveable_image, str(output_path), metadata):
                        result['target_sizes_processed'].append(target_size)
                        self.stats['processed_images'] += 1
                        self.logger.info(f"成功处理尺寸 {target_size}x{target_size}: {result['filename']}")
                    
                    # 数据增强
                    if self.config['enable_data_augmentation'] and quality_score >= self.config['quality_threshold']:
                        augmented_images = self._generate_augmented_images(saveable_image, base_name)
                        aug_dir = output_dir / "augmented"
                        
                        for aug_image, suffix in augmented_images:
                            aug_filename = f"{base_name}_{target_size}x{target_size}{suffix}{ext}"
                            aug_path = aug_dir / aug_filename
                            
                            aug_metadata = metadata.copy()
                            aug_metadata['augmentation_type'] = suffix
                            aug_metadata['is_augmented'] = True
                            
                            if self._save_image_with_metadata(aug_image, str(aug_path), aug_metadata):
                                result['augmented_count'] += 1
                                self.stats['augmented_images'] += 1
                                
                except Exception as e:
                    self.logger.error(f"处理尺寸 {target_size}x{target_size} 失败: {e}")
                    continue
            
            # 如果至少有一个尺寸处理成功，标记为成功
            if result['target_sizes_processed']:
                result['success'] = True
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            self.stats['processing_times'].append(processing_time)
            
            self.logger.info(f"处理完成: {result['filename']} (质量: {quality_score:.3f}, 时间: {processing_time:.2f}s, 尺寸: {result['target_sizes_processed']})")
            
        except Exception as e:
            result['error'] = str(e)
            self.stats['failed_images'] += 1
            self.logger.error(f"处理失败 {result['filename']}: {e}")
        
        return result
    
    def process_dataset(self) -> Dict:
        """处理整个数据集
        
        Returns:
            处理统计信息
        """
        input_folder = Path(self.config['input_folder'])
        
        if not input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 获取所有支持的图像文件
        image_files = []
        for ext in self.config['supported_formats']:
            image_files.extend(input_folder.glob(f"*{ext}"))
            image_files.extend(input_folder.glob(f"*{ext.upper()}"))
        
        self.stats['total_images'] = len(image_files)
        
        if not image_files:
            self.logger.warning(f"在 {input_folder} 中未找到支持的图像文件")
            return self.stats
        
        self.logger.info(f"开始处理 {len(image_files)} 张图像...")
        
        # 处理每张图像
        results = []
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"进度: {i}/{len(image_files)}")
            result = self.process_single_image(str(image_file))
            results.append(result)
        
        # 复制所有处理后的图像到best文件夹
        self.logger.info("开始复制图像到best文件夹...")
        base_output = Path(self.config['output_folder'])
        best_dir = base_output / "best"
        
        total_copied = 0
        for size in self.config['target_sizes']:
            source_dir = base_output / f"size_{size}x{size}"
            copied_count = self._copy_to_best_folder(source_dir, best_dir)
            total_copied += copied_count
            self.logger.info(f"复制 {size}x{size} 尺寸图像: {copied_count} 个文件")
        
        self.stats['best_copied_images'] = total_copied
        self.logger.info(f"总共复制到best文件夹: {total_copied} 个文件")
        
        # 生成最终统计报告
        self._generate_final_report(results)
        
        return self.stats
    
    def _generate_final_report(self, results: List[Dict]):
        """生成最终处理报告
        
        Args:
            results: 所有图像的处理结果
        """
        report_dir = Path(self.config['output_folder']) / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 详细报告
        detailed_report = {
            'processing_summary': {
                'total_images': self.stats['total_images'],
                'processed_successfully': self.stats['processed_images'],
                'failed_images': self.stats['failed_images'],
                'augmented_images': self.stats['augmented_images'],
                'best_copied_images': self.stats['best_copied_images'],
                'success_rate': self.stats['processed_images'] / self.stats['total_images'] if self.stats['total_images'] > 0 else 0
            },
            'quality_analysis': {
                'average_quality': np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
                'min_quality': np.min(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
                'max_quality': np.max(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
                'quality_threshold': self.config['quality_threshold']
            },
            'performance_metrics': {
                'average_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'total_processing_time': sum(self.stats['processing_times']),
                'images_per_second': len(self.stats['processing_times']) / sum(self.stats['processing_times']) if sum(self.stats['processing_times']) > 0 else 0
            },
            'configuration_used': self.config,
            'individual_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存详细报告
        report_file = report_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)
        
        # 生成简要文本报告
        summary_file = report_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("隧道掌子面图像预处理报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总图像数: {self.stats['total_images']}\n")
            f.write(f"成功处理: {self.stats['processed_images']}\n")
            f.write(f"处理失败: {self.stats['failed_images']}\n")
            f.write(f"生成增强: {self.stats['augmented_images']}\n")
            f.write(f"复制到best: {self.stats['best_copied_images']}\n")
            f.write(f"成功率: {detailed_report['processing_summary']['success_rate']:.2%}\n")
            f.write(f"平均质量分数: {detailed_report['quality_analysis']['average_quality']:.3f}\n")
            f.write(f"平均处理时间: {detailed_report['performance_metrics']['average_processing_time']:.2f}秒\n")
            f.write(f"处理速度: {detailed_report['performance_metrics']['images_per_second']:.2f}张/秒\n")
            f.write(f"支持尺寸: {', '.join([f'{s}x{s}' for s in self.config['target_sizes']])}\n")
        
        self.logger.info(f"处理报告已保存: {report_file}")
        self.logger.info(f"简要报告已保存: {summary_file}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'input_folder': r"C:\Users\ASUS\Desktop\AI_Recognition\tunnel_face_images",
        'output_folder': r"C:\Users\ASUS\Desktop\AI_Recognition\processed_tunnel_face_photos\1",
        'target_sizes': [512, 1024, 2048, 3024],  # 增加3024尺寸
        'enable_histogram_equalization': True,
        'enable_median_filter': True,
        'median_filter_kernel': 5,
        'enable_data_augmentation': True,
        'augmentation_count': 4,
        'rotation_angles': [-10, -5, 5, 10],
        'brightness_factors': [0.8, 0.9, 1.1, 1.2],
        'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'],
        'quality_threshold': 0.3,
        'log_level': 'INFO'
    }
    
    try:
        # 创建预处理器
        preprocessor = TunnelFaceImagePreprocessor(config)
        
        print("隧道掌子面图像预处理系统")
        print("=" * 50)
        print("功能特点:")
        print("✓ 多尺寸支持 (512, 1024, 2048, 3024)")
        print("✓ 自动复制所有图像到best文件夹")
        print("✓ 自适应尺寸调整")
        print("✓ 直方图均衡化增强")
        print("✓ 中值滤波去噪")
        print("✓ 智能数据增强")
        print("✓ 图像质量评估")
        print("✓ 详细日志记录")
        print("✓ 支持多种格式")
        print("=" * 50)
        
        # 开始处理
        stats = preprocessor.process_dataset()
        
        # 显示最终统计
        print("\n处理完成！")
        print(f"总计: {stats['total_images']} 张")
        print(f"成功: {stats['processed_images']} 张")
        print(f"失败: {stats['failed_images']} 张")
        print(f"增强: {stats['augmented_images']} 张")
        print(f"复制到best: {stats['best_copied_images']} 张")
        print(f"平均质量: {np.mean(stats['quality_scores']):.3f}" if stats['quality_scores'] else "N/A")
        print(f"平均耗时: {np.mean(stats['processing_times']):.2f}秒" if stats['processing_times'] else "N/A")
        
        print("\n输出目录结构:")
        print("├── size_512x512/")
        print("├── size_1024x1024/")
        print("├── size_2048x2048/")
        print("├── size_3024x3024/")
        print("└── best/ (包含所有尺寸的副本)")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        logging.error(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()