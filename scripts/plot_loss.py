import os
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'

# plt.rcParams['axes.facecolor'] = '#2c3e50'
# plt.rcParams['figure.facecolor'] = '#34495e'

plt.rcParams['grid.color'] = '#7f8c8d'
plt.rcParams['grid.alpha'] = 0.3

meta_exp_dir = "../exp_training_sample/"

for type in ["domain", "pretrain"]:

    all_total_loss = []
    all_recon_loss = []
    all_reg_loss = []
    all_task_loss = []

    for exp_dir in os.listdir(meta_exp_dir):

        if type in exp_dir:
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
    min_length = 100#min([len(total_loss) for total_loss in all_total_loss])
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
    plt.figure(figsize=(10, 5), facecolor='#f8f8f8')
    ax = plt.gca()
    ax.set_facecolor('#ebebf2')

    #plt.plot(total_loss_mean, label='Total Loss', color='red')
    plt.plot(recon_loss_mean, label='Reconstruction Loss', color='blue')
    plt.plot(reg_loss_mean, label='Regularization Loss', color='purple')
    plt.plot(task_loss_mean, label='Task Loss', color='green')
    # 绘制带状区域
    #plt.fill_between(range(len(total_loss_mean)), total_loss_mean - total_loss_std, total_loss_mean + total_loss_std, alpha=0.2, color='red')
    plt.fill_between(range(len(recon_loss_mean)), recon_loss_mean - recon_loss_std, recon_loss_mean + recon_loss_std, alpha=0.2, color='blue')
    plt.fill_between(range(len(reg_loss_mean)), reg_loss_mean - reg_loss_std, reg_loss_mean + reg_loss_std, alpha=0.2, color='purple')
    plt.fill_between(range(len(task_loss_mean)), task_loss_mean - task_loss_std, task_loss_mean + task_loss_std, alpha=0.2, color='green')
    #plt.title('Loss Curves')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    # 设置x轴从0开始
    plt.xlim(0, len(recon_loss_mean)-1)
    
    # 修改x轴刻度，使每个标注点均乘以10
    locs, labels = plt.xticks()
    new_labels = [int(loc * 10) for loc in locs if loc >= 0]  # 确保只有非负值
    plt.xticks([loc for loc in locs if loc >= 0], new_labels)
    
    plt.show()
    plt.savefig(f"loss_curve_{type}.pdf", bbox_inches='tight')