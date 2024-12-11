# 读取../exp_generalize/下所有文件夹，每个文件夹下的epoch1下有一个neural2symbolic.log和一个symbolic2neural.log文件，读取这两个文件中的accuracy，绘制accuracy随training ratio的变化曲线

import os
import matplotlib.pyplot as plt

exp_dir = "../exp_generalize"
selected_epoch = 1

pretrain_ratios = []
seen_neural2symbolic_accuracy = []
seen_symbolic2neural_accuracy = []
unseen_neural2symbolic_accuracy = []
unseen_symbolic2neural_accuracy = []

for exp_name in os.listdir(exp_dir):
    exp_path = os.path.join(exp_dir, exp_name)
    train_ratio = exp_name.split("pretrain-")[1]
    find_seen_neural2symbolic_accuracy = False
    find_unseen_neural2symbolic_accuracy = False
    find_seen_symbolic2neural_accuracy = False
    find_unseen_symbolic2neural_accuracy = False
    if os.path.isdir(exp_path):
        neural2symbolic_path = os.path.join(exp_path, f"epoch{selected_epoch}", "neural2symbolic.log")
        symbolic2neural_path = os.path.join(exp_path, f"epoch{selected_epoch}", "symbolic2neural.log")
        try:
            with open(neural2symbolic_path, "r") as f:
                # 分别找到seen task accuracy和unseen task accuracy
                lines = f.readlines()
                for line in lines:
                    if "accuracy on seen task" in line:
                        find_seen_neural2symbolic_accuracy = True
                        this_seen_neural2symbolic_accuracy = float(line.split("samples: ")[1].split(" ")[0])
                    elif "accuracy on unseen task" in line:
                        find_unseen_neural2symbolic_accuracy = True
                        this_unseen_neural2symbolic_accuracy = float(line.split("samples: ")[1].split(" ")[0])
            with open(symbolic2neural_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "accuracy on seen task" in line:
                        find_seen_symbolic2neural_accuracy = True
                        this_seen_symbolic2neural_accuracy = float(line.split("samples: ")[1].split(" ")[0])
                    elif "accuracy on unseen task" in line:
                        find_unseen_symbolic2neural_accuracy = True
                        this_unseen_symbolic2neural_accuracy = float(line.split("samples: ")[1].split(" ")[0])
        except Exception as e:
            print(f"Error reading file {neural2symbolic_path} or {symbolic2neural_path}: {e}")
            continue
        
        if find_seen_neural2symbolic_accuracy and find_unseen_neural2symbolic_accuracy and find_seen_symbolic2neural_accuracy and find_unseen_symbolic2neural_accuracy:
            pretrain_ratios.append(float(train_ratio))
            seen_neural2symbolic_accuracy.append(this_seen_neural2symbolic_accuracy)
            unseen_neural2symbolic_accuracy.append(this_unseen_neural2symbolic_accuracy)
            seen_symbolic2neural_accuracy.append(this_seen_symbolic2neural_accuracy)
            unseen_symbolic2neural_accuracy.append(this_unseen_symbolic2neural_accuracy)

# 将所有列表按pretrain_ratios升序排序
pretrain_ratios, seen_neural2symbolic_accuracy, unseen_neural2symbolic_accuracy, seen_symbolic2neural_accuracy, unseen_symbolic2neural_accuracy \
= zip(*sorted(zip(pretrain_ratios, seen_neural2symbolic_accuracy, unseen_neural2symbolic_accuracy, seen_symbolic2neural_accuracy, unseen_symbolic2neural_accuracy), key=lambda x: x[0]))

plt.plot(pretrain_ratios, seen_neural2symbolic_accuracy, label="seen neural2symbolic", color='#2ecc71')
plt.plot(pretrain_ratios, unseen_neural2symbolic_accuracy, label="unseen neural2symbolic", color='#3498db')
plt.plot(pretrain_ratios, seen_symbolic2neural_accuracy, label="seen symbolic2neural", color='#e74c3c')
plt.plot(pretrain_ratios, unseen_symbolic2neural_accuracy, label="unseen symbolic2neural", color='#f39c12')

plt.legend(loc='upper right')
plt.savefig("accuracy.pdf")

