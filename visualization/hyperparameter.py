import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Times New Roman，字体大小为 14 号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 14

# X-axis values (from 0 to 1 with a step of 0.1)
x_values = [i/10 for i in range(11)]

# 数据集值（NMI、ARI、ACC）
klein_values_nmi = [0.9202, 0.9202, 0.8507, 0.8442, 0.9272, 0.9352, 0.9352, 0.9182, 0.8501, 0.8308, 0.8308]
qs_trachea_nmi = [0.68, 0.68, 0.6764, 0.6862, 0.7119, 0.719, 0.7124, 0.7124, 0.7168, 0.6968, 0.6888]
quake_10x_spleen_nmi = [0.7785, 0.8116, 0.8004, 0.7525, 0.7519, 0.8086, 0.7857, 0.7872, 0.7853, 0.7622, 0.7543]

klein_values_ari = [0.9463, 0.9461, 0.8212, 0.8061, 0.8549, 0.9604, 0.9604, 0.9182, 0.8199, 0.8109, 0.8031]
qs_trachea_ari = [0.5575, 0.5575, 0.5683, 0.5601, 0.5865, 0.6493, 0.569, 0.5964, 0.5888, 0.5768, 0.5688]
quake_10x_spleen_ari = [0.7685, 0.8292, 0.7999, 0.6833, 0.678, 0.8393, 0.7794, 0.7343, 0.7875, 0.7622, 0.6888]

klein_values_acc = [0.9775, 0.9772, 0.8579, 0.8423, 0.9805, 0.9823, 0.9823, 0.9113, 0.8868, 0.8528, 0.8528]
qs_trachea_acc = [0.7156, 0.7156, 0.7284, 0.7163, 0.7742, 0.8161, 0.7831, 0.7377, 0.7431, 0.7376, 0.7238]
quake_10x_spleen_acc = [0.9037, 0.9329, 0.9206, 0.8619, 0.8586, 0.9373, 0.9117, 0.9115, 0.9099, 0.8822, 0.8688]

# 调整 figure 的高度和宽度
plt.figure(figsize=(15, 12))

# 使用 seaborn 颜色调色板
sns.set_palette("deep")
colors = sns.color_palette("deep", 3)

# 绘制 NMI 图
plt.subplot(3, 1, 1)
plt.plot(x_values, klein_values_nmi, marker='o', linestyle='-', label='Klein', color=colors[0])
plt.plot(x_values, qs_trachea_nmi, marker='^', linestyle='-', label='Qs_Trachea', color=colors[1])
plt.plot(x_values, quake_10x_spleen_nmi, marker='d', linestyle=':', label='Qx_Spleen', color=colors[2])
plt.xlabel(r'Moment Coefficient $\alpha$')
plt.ylabel('NMI')
plt.xticks(x_values)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.grid(True, which='both', axis='both', linestyle='-', color='white', linewidth=0.5, zorder=0)
plt.gca().set_facecolor('none')
plt.legend(loc='upper right')

# 绘制 ARI 图
plt.subplot(3, 1, 2)
plt.plot(x_values, klein_values_ari, marker='o', linestyle='-', label='Klein', color=colors[0])
plt.plot(x_values, qs_trachea_ari, marker='^', linestyle='-', label='Qs_Trachea', color=colors[1])
plt.plot(x_values, quake_10x_spleen_ari, marker='d', linestyle=':', label='Qx_Spleen', color=colors[2])
plt.xlabel(r'Moment Coefficient $\alpha$')
plt.ylabel('ARI')
plt.xticks(x_values)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.grid(True, which='both', axis='both', linestyle='-', color='white', linewidth=0.5, zorder=0)
plt.gca().set_facecolor('none')
plt.legend(loc='upper right')

# 绘制 ACC 图
plt.subplot(3, 1, 3)
plt.plot(x_values, klein_values_acc, marker='o', linestyle='-', label='Klein', color=colors[0])
plt.plot(x_values, qs_trachea_acc, marker='^', linestyle='-', label='Qs_Trachea', color=colors[1])
plt.plot(x_values, quake_10x_spleen_acc, marker='d', linestyle=':', label='Qx_Spleen', color=colors[2])
plt.xlabel(r'Moment Coefficient $\alpha$')
plt.ylabel('ACC')
plt.xticks(x_values)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.grid(True, which='both', axis='both', linestyle='-', color='white', linewidth=0.5, zorder=0)
plt.gca().set_facecolor('none')
plt.legend(loc='upper right')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig("hyperparameter.png", dpi=600, bbox_inches='tight')

# 显示图像
plt.show()
