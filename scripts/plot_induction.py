import os
import matplotlib.pyplot as plt

exp_dir = "../exp_induction"
selected_epoch = 1

observed_samples = []
seen_induction_accuracy = []
unseen_induction_accuracy = []

for exp_name in os.listdir(exp_dir):
    exp_path = os.path.join(exp_dir, exp_name)
    observed_sample = int(exp_name.split("induction-")[1]) - 1
    find_seen_induction_accuracy = False
    find_unseen_induction_accuracy = False
    find_seen_deduction_accuracy = False
    find_unseen_deduction_accuracy = False
    if os.path.isdir(exp_path):
        induction_path = os.path.join(exp_path, f"epoch{selected_epoch}", "neural2symbolic.log")
        try:
            with open(induction_path, "r") as f:
                # 分别找到seen task accuracy和unseen task accuracy
                lines = f.readlines()
                for line in lines:
                    if "accuracy on seen task" in line:
                        find_seen_induction_accuracy = True
                        this_seen_induction_accuracy = float(line.split("samples: ")[1].split(" ")[0])
                    elif "accuracy on unseen task" in line:
                        find_unseen_induction_accuracy = True
                        this_unseen_induction_accuracy = float(line.split("samples: ")[1].split(" ")[0])

        except Exception as e:
            print(f"Error reading file {induction_path} or {deduction_path}: {e}")
            continue
        
        if find_seen_induction_accuracy and find_unseen_induction_accuracy:
            observed_samples.append(observed_sample)
            seen_induction_accuracy.append(this_seen_induction_accuracy)
            unseen_induction_accuracy.append(this_unseen_induction_accuracy)


# 将所有列表按pretrain_ratios升序排序
observed_samples, seen_induction_accuracy, unseen_induction_accuracy \
= zip(*sorted(zip(observed_samples, seen_induction_accuracy, unseen_induction_accuracy), key=lambda x: x[0]))

# 将seen和unseen的accuracy按0.9和0.1的比例混合，得到induction的accuracy
induction_accuracy = [0.9 * seen + 0.1 * unseen for seen, unseen in zip(seen_induction_accuracy, unseen_induction_accuracy)]

plt.plot(observed_samples, induction_accuracy, label="induction", color='#2ecc71', marker='o')

#横轴标签：pretrain ratio
plt.xlabel("observed samples")
plt.ylabel("accuracy")

plt.legend(loc='lower right')
plt.savefig("induction.pdf")

