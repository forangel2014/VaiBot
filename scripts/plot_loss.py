import os
import matplotlib.pyplot as plt
import numpy as np
meta_exp_dir = "../exp_training_sample/"

all_total_loss = []
all_recon_loss = []
all_reg_loss = []
all_task_loss = []

for exp_dir in os.listdir(meta_exp_dir):
    # 读取日志文件
    log_file_path = os.path.join(meta_exp_dir, exp_dir, "train.log")
    total_loss = []
    recon_loss = []
    reg_loss = []
    task_loss = []

    with open(log_file_path, 'r') as f:
        for line in f:
            if "recon_loss" in line:
                parts = line.split(',')
                total_loss.append(float(parts[0].split('=')[1]))
                recon_loss.append(float(parts[1].split('=')[1]))
                reg_loss.append(float(parts[2].split('=')[1]))
                task_loss.append(float(parts[3].split('=')[1]))

    all_total_loss.append(total_loss)
    all_recon_loss.append(recon_loss)
    all_reg_loss.append(reg_loss)
    all_task_loss.append(task_loss)

#对于所有exp_dir，按最短的total_loss的长度进行截断
min_length = min([len(total_loss) for total_loss in all_total_loss])
all_total_loss = [total_loss[:min_length] for total_loss in all_total_loss]
all_recon_loss = [recon_loss[:min_length] for recon_loss in all_recon_loss]
all_reg_loss = [reg_loss[:min_length] for reg_loss in all_reg_loss]
all_task_loss = [task_loss[:min_length] for task_loss in all_task_loss]
#计算均值和方差
total_loss_mean = np.mean(all_total_loss, axis=0)
recon_loss_mean = np.mean(all_recon_loss, axis=0)
reg_loss_mean = np.mean(all_reg_loss, axis=0)
task_loss_mean = np.mean(all_task_loss, axis=0)
total_loss_std = np.std(all_total_loss, axis=0)
recon_loss_std = np.std(all_recon_loss, axis=0)
reg_loss_std = np.std(all_reg_loss, axis=0)
task_loss_std = np.std(all_task_loss, axis=0)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
#plt.plot(total_loss_mean, label='Total Loss', color='red')
plt.plot(recon_loss_mean, label='Reconstruction Loss', color='blue')
plt.plot(reg_loss_mean, label='Regularization Loss', color='orange')
plt.plot(task_loss_mean, label='Task Loss', color='green')
# 绘制带状区域
#plt.fill_between(range(len(total_loss_mean)), total_loss_mean - total_loss_std, total_loss_mean + total_loss_std, alpha=0.2, color='red')
plt.fill_between(range(len(recon_loss_mean)), recon_loss_mean - recon_loss_std, recon_loss_mean + recon_loss_std, alpha=0.2, color='blue')
plt.fill_between(range(len(reg_loss_mean)), reg_loss_mean - reg_loss_std, reg_loss_mean + reg_loss_std, alpha=0.2, color='orange')
plt.fill_between(range(len(task_loss_mean)), task_loss_mean - task_loss_std, task_loss_mean + task_loss_std, alpha=0.2, color='green')
plt.title('Loss Curves')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
plt.savefig(f"loss_curve.pdf")