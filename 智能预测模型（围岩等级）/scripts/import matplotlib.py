import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns

# 创建多模型性能对比数据
def create_comprehensive_model_comparison_data():
    models = ['AlexNet', 'VGG-16', 'VGG-19', 'MobileNet-V2', 'MobileNet-V3', 
              'ResNet-50', 'ResNet-101', 'ResNeXt-50', 'DenseNet-121', 
              'DenseNet-169', 'EfficientNet-B0', 'EfficientNet-B2', 
              'SENet-50', '改进ResNet']
    
    accuracy = [82.45, 85.67, 86.23, 87.89, 88.67, 89.34, 90.12, 90.78, 
                91.23, 91.67, 92.45, 92.89, 91.45, 94.27]
    
    params = [61.1, 138.4, 143.7, 3.5, 5.4, 25.6, 44.5, 25.0, 8.0, 
              14.1, 5.3, 9.1, 28.1, 28.3]  # 参数量(M)
    
    flops = [0.7, 15.5, 19.6, 0.3, 0.2, 4.1, 7.8, 4.3, 2.9, 
             3.4, 0.4, 1.0, 4.1, 4.8]  # FLOPs(G)
    
    training_time = [2.1, 4.8, 5.2, 2.1, 2.3, 3.2, 4.6, 3.5, 3.6, 
                     4.1, 2.9, 3.4, 3.7, 3.5]  # 训练时间(h)
    
    return {
        'models': models,
        'accuracy': accuracy,
        'params': params,
        'flops': flops,
        'training_time': training_time
    }
# 绘制多模型性能对比与回归分析图
def plot_comprehensive_model_comparison():
    data = create_comprehensive_model_comparison_data()
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 定义颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(data['models'])))
    
    # 1. 准确率与参数量的关系
    x1 = np.array(data['params']).reshape(-1, 1)
    y = np.array(data['accuracy'])
    
    # 多项式回归拟合
    poly_features = PolynomialFeatures(degree=2)
    x1_poly = poly_features.fit_transform(x1)
    poly_reg = LinearRegression()
    poly_reg.fit(x1_poly, y)
    
    # 生成拟合曲线
    x1_range = np.linspace(0, 150, 300).reshape(-1, 1)
    x1_range_poly = poly_features.transform(x1_range)
    y1_pred = poly_reg.predict(x1_range_poly)
    
    # 绘制散点图和拟合曲线
    for i, (model, acc, param) in enumerate(zip(data['models'], data['accuracy'], data['params'])):
        if model == '改进ResNet':
            ax1.scatter(param, acc, s=150, c='red', alpha=0.9, edgecolors='black', 
                       linewidth=2, marker='*', label=model, zorder=5)
        else:
            ax1.scatter(param, acc, s=100, c=[colors[i]], alpha=0.8, edgecolors='black', linewidth=1)
        
        # 添加模型名称标注
        if model in ['改进ResNet', 'VGG-19', 'EfficientNet-B2']:
            ax1.annotate(model, (param, acc), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold' if model == '改进ResNet' else 'normal')
    
    ax1.plot(x1_range, y1_pred, 'r--', linewidth=2, alpha=0.8, 
            label=f'多项式拟合 (R²={r2_score(y, poly_reg.predict(x1_poly)):.3f})')
    ax1.set_xlabel('模型参数量 (M)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('模型准确率与参数量关系回归分析', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 150)
    ax1.set_ylim(80, 96)
    
    # 2. 准确率与FLOPs的关系
    x2 = np.array(data['flops']).reshape(-1, 1)
    
    # 对数回归拟合
    x2_log = np.log(x2 + 0.1)  # 避免log(0)
    linear_reg = LinearRegression()
    linear_reg.fit(x2_log, y)
    
    # 生成拟合曲线
    x2_range = np.linspace(0.1, 20, 300).reshape(-1, 1)
    x2_range_log = np.log(x2_range + 0.1)
    y2_pred = linear_reg.predict(x2_range_log)
    
    # 绘制散点图和拟合曲线
    for i, (model, acc, flop) in enumerate(zip(data['models'], data['accuracy'], data['flops'])):
        if model == '改进ResNet':
            ax2.scatter(flop, acc, s=150, c='red', alpha=0.9, edgecolors='black', 
                       linewidth=2, marker='*', label=model, zorder=5)
        else:
            ax2.scatter(flop, acc, s=100, c=[colors[i]], alpha=0.8, edgecolors='black', linewidth=1)
        
        if model in ['改进ResNet', 'VGG-19', 'MobileNet-V2']:
            ax2.annotate(model, (flop, acc), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold' if model == '改进ResNet' else 'normal')
    
    ax2.plot(x2_range, y2_pred, 'r--', linewidth=2, alpha=0.8, 
            label=f'对数拟合 (R²={r2_score(y, linear_reg.predict(x2_log)):.3f})')
    ax2.set_xlabel('计算复杂度 FLOPs (G)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('模型准确率与计算复杂度关系回归分析', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(80, 96)
    
    # 3. 模型性能综合对比柱状图
    model_names_short = [name[:8] + '...' if len(name) > 8 else name for name in data['models']]
    x_pos = np.arange(len(model_names_short))
    
    bars = ax3.bar(x_pos, data['accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 突出显示改进ResNet
    improved_resnet_idx = data['models'].index('改进ResNet')
    bars[improved_resnet_idx].set_color('red')
    bars[improved_resnet_idx].set_alpha(0.9)
    bars[improved_resnet_idx].set_linewidth(2)
    
    ax3.set_xlabel('模型名称', fontsize=12, fontweight='bold')
    ax3.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('各模型准确率对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names_short, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(80, 96)
    
    # 添加数值标注
    for i, (bar, acc) in enumerate(zip(bars, data['accuracy'])):
        height = bar.get_height()
        ax3.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, fontweight='bold' if i == improved_resnet_idx else 'normal')
    
    # 4. 效率-性能权衡分析
    # 计算效率指标（准确率/训练时间）
    efficiency = [acc/time for acc, time in zip(data['accuracy'], data['training_time'])]
    
    bubble_sizes = [param * 3 for param in data['params']]  # 气泡大小表示参数量
    
    for i, (model, acc, time, size) in enumerate(zip(data['models'], data['accuracy'], 
                                                     data['training_time'], bubble_sizes)):
        if model == '改进ResNet':
            ax4.scatter(time, acc, s=size, c='red', alpha=0.7, edgecolors='black', 
                       linewidth=2, label=model, zorder=5)
        else:
            ax4.scatter(time, acc, s=size, c=[colors[i]], alpha=0.6, edgecolors='black', linewidth=1)
        
        if model in ['改进ResNet', 'VGG-19', 'AlexNet', 'EfficientNet-B0']:
            ax4.annotate(model, (time, acc), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold' if model == '改进ResNet' else 'normal')
    
    ax4.set_xlabel('训练时间 (小时)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('模型效率-性能权衡分析\n(气泡大小表示参数量)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim(1.5, 5.5)
    ax4.set_ylim(80, 96)
    
    plt.tight_layout()
    plt.savefig('多模型综合性能对比分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return data

# 生成Excel数据文件
def create_comprehensive_comparison_excel_data():
    data = create_comprehensive_model_comparison_data()
    
    df = pd.DataFrame({
        '模型名称': data['models'],
        '准确率(%)': data['accuracy'],
        '参数量(M)': data['params'],
        'FLOPs(G)': data['flops'],
        '训练时间(h)': data['training_time']
    })
    
    # 计算效率指标
    df['效率指标(准确率/训练时间)'] = df['准确率(%)'] / df['训练时间(h)']
    df['参数效率(准确率/参数量)'] = df['准确率(%)'] / df['参数量(M)']
    
    df.to_excel('多模型综合性能对比数据.xlsx', index=False)
    print("多模型综合对比数据已保存至：多模型综合性能对比数据.xlsx")
    
    return df

# 执行绘图和数据生成
if __name__ == "__main__":
    comparison_data = plot_comprehensive_model_comparison()
    excel_data = create_comprehensive_comparison_excel_data()