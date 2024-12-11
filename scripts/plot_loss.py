import os
import matplotlib.pyplot as plt

exp_dir = "../exp_final/vae-pretrain"

# 读取日志文件
log_file_path = os.path.join(exp_dir, "train.log")
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

# 绘制损失曲线
plt.figure(figsize=(10, 5))
# 只绘制前500步
plt.plot(recon_loss[:500], label='Reconstruction Loss', color='blue')
plt.plot(reg_loss[:500], label='Regularization Loss', color='orange')
plt.plot(task_loss[:500], label='Task Loss', color='green')
plt.title('Loss Curves')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
plt.savefig(f"{exp_dir}/loss_curve.pdf")