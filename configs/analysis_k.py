import matplotlib.pyplot as plt
import numpy as np

# 数据
percentile_values = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 1]
ninco_auroc = [78.90, 78.95, 79.49, 80.36, 80.05, 79.65, 74.36]  # Near-OOD

# 创建自定义x轴位置（让0.12和1之间的间距等于前面的间距）
x_positions = [0, 1, 2, 3, 4, 5, 6]  # 等间距位置
x_labels = ['0.02', '0.04', '0.06', '0.08', '0.10', '0.12', '1']  # 对应的标签

# 设置图形
plt.figure(figsize=(14, 12))
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 36

# 绘制折线图（使用自定义x位置）
line2, = plt.plot(x_positions, ninco_auroc, 's-', linewidth=8, markersize=20,
                 color='#9BD0E2', label='Near-OOD')

# 设置标题和标签
plt.ylabel('AUROC', weight='bold')

# 自定义x轴
plt.xticks(x_positions, x_labels, weight='bold')
plt.yticks([72, 75, 80, 85], weight='bold')
plt.ylim(72, 85)
plt.xlim(-0.5, 6.8)  # 留出一些边距

# 添加断点指示
plt.annotate('', xy=(5.5, 72), xytext=(5.1, 72),
             arrowprops=dict(arrowstyle='-', color='gray', linestyle='--', linewidth=3))
# plt.text(5.8, 72, 'scale break', fontsize=30, ha='center', va='center', color='gray')

# 图例
legend = plt.legend(loc='upper right',
                   handlelength=1.5, handletextpad=0.5)

# 数据点标注
for x, y in zip(x_positions[:-1], ninco_auroc[:-1]):
    plt.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 30),
                textcoords='offset points', weight='bold', ha='center')
for x, y in zip([x_positions[-1]], [ninco_auroc[-1]]):
    plt.annotate(f'{y:.2f}', xy=(x, y), xytext=(10, 120),
                textcoords='offset points', weight='bold', ha='center')    

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_k.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()