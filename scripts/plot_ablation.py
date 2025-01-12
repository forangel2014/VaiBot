import matplotlib.pyplot as plt
import numpy as np

# 数据
data = [
    [33.26, 85.56, 21.11, 44.44, 48.67, 78.33, 58.10, 28.57],
    [31.65, 0.53, 16.67, 0.00, 45.67, 11.67, 54.29, 4.76],
    [32.94, 84.49, 4.44, 22.22, 48.22, 80.00, 56.19, 33.33],
    [29.52, 1.59, 14.74, 0.00, 38.00, 7.78, 49.52, 4.76],
    [30.37, 36.36, 32.22, 50.00, 38.22, 20.00, 49.52, 19.05],
    [28.77, 0.53, 27.78, 0.00, 37.00, 0.56, 47.62, 0.00],
    [18.29, 36.90, 10.00, 44.44, 24.44, 19.44, 31.43, 28.57],
    [28.98, 2.14, 26.67, 0.00, 40.72, 4.42, 48.57, 3.57]
]

# 分别对应8行
methods = [
    'NesyFlow-in-domain',
    'w/o xy',
    'w/o k',
    'finetune w/o encoder',
    'NesyFlow-pretrain *',
    'w/o xy *',
    'w/o k *',
    'finetune w/o encoder *'
]

# 分别对应1-4列和5-8列
tasks = ["SNI", "P3"]

# 分别对应1和5，2和6列，3和7，4和8列
settings = [
    'Seen Deduction', 'Seen Induction', 'Unseen Deduction', 'Unseen Induction',
]

# 绘制消融实验图
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(methods))

# 绘制每个任务的柱状图
for i, task in enumerate(tasks):
    ax.bar(index + i * bar_width, [data[j][i] for j in range(len(data))], bar_width, label=task)

# 添加标签和标题
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title('Ablation Study Results')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(methods)
ax.legend()

plt.tight_layout()
plt.savefig('ablation.pdf')