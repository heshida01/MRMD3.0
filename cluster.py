import numpy as np
import matplotlib.pyplot as plt

# 示例数据
labels = ['A', 'B', 'C', 'D', 'E']
values1 = [3, 4, 2, 5, 4]
values2 = [4, 3, 4, 3, 2]

# 计算雷达图的角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 闭合图形
values1 += values1[:1]
values2 += values2[:1]
angles += angles[:1]

# 设置画布
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制雷达图 - 第一组数据
ax.plot(angles, values1, color='blue', linewidth=2, label='Group 1')
ax.fill(angles, values1, color='blue', alpha=0.25)

# 绘制雷达图 - 第二组数据
ax.plot(angles, values2, color='red', linewidth=2, label='Group 2')
ax.fill(angles, values2, color='red', alpha=0.25)

# 设置轴和刻度
max_value = max(max(values1), max(values2))
num_circles = 5  # 设置圆圈的数量

# 设置y轴刻度和标签
yticks = np.linspace(0, max_value, num_circles)
yticklabels = [str(int(y)) for y in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

# 设置x轴刻度和标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# 显示图表
plt.show()
