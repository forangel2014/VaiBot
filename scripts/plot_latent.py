import os
import matplotlib.pyplot as plt

nesy_exp_dir = "../exp_restart"
selected_epoch = 10

plt.figure(facecolor='#f8f8f8')
ax = plt.gca()
ax.set_facecolor('#ebebf2')

nesy_latent_induction = []
nesy_seen_induction_accuracy = []
nesy_unseen_induction_accuracy = []
nesy_latent_deduction = []
nesy_seen_deduction_accuracy = []
nesy_unseen_deduction_accuracy = []

pretrain_latent_induction = []
pretrain_seen_induction_accuracy = []
pretrain_unseen_induction_accuracy = []
pretrain_latent_deduction = []
pretrain_seen_deduction_accuracy = []
pretrain_unseen_deduction_accuracy = []


sft_latent_induction = []
sft_seen_induction_accuracy = []
sft_unseen_induction_accuracy = []
sft_latent_deduction = []
sft_seen_deduction_accuracy = []
sft_unseen_deduction_accuracy = []

for exp_name in os.listdir(nesy_exp_dir):
    if "qwen-domain" in exp_name:
        exp_path = os.path.join(nesy_exp_dir, exp_name)
        name = exp_name.split("domain-")[1]
        if name == "fixtask-delta":
            name = "10"
        latent_z = int(name)
        find_seen_induction_accuracy = False
        find_unseen_induction_accuracy = False
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
                print(f"Error reading file {induction_path}: {e}")
                continue

            if find_seen_induction_accuracy and find_unseen_induction_accuracy:
                nesy_latent_induction.append(latent_z)
                nesy_seen_induction_accuracy.append(this_seen_induction_accuracy)
                nesy_unseen_induction_accuracy.append(this_unseen_induction_accuracy)
            
            deduction_path = os.path.join(exp_path, f"epoch{selected_epoch}", "symbolic2neural.log")
            try:
                with open(deduction_path, "r") as f:
                    # 分别找到seen task accuracy和unseen task accuracy
                    lines = f.readlines()
                    for line in lines:
                        if "accuracy on seen task" in line:
                            find_seen_deduction_accuracy = True
                            this_seen_deduction_accuracy = float(line.split("samples: ")[1].split(" ")[0])
                        elif "accuracy on unseen task" in line:
                            find_unseen_deduction_accuracy = True
                            this_unseen_deduction_accuracy = float(line.split("samples: ")[1].split(" ")[0])

            except Exception as e:
                print(f"Error reading file {induction_path}: {e}")
                continue

            if find_seen_deduction_accuracy and find_unseen_deduction_accuracy:
                nesy_latent_deduction.append(latent_z)
                nesy_seen_deduction_accuracy.append(this_seen_deduction_accuracy)
                nesy_unseen_deduction_accuracy.append(this_unseen_deduction_accuracy)

    if "vae-pretrain-z" in exp_name:
        exp_path = os.path.join(nesy_exp_dir, exp_name)
        latent_z = int(exp_name.split("-z")[1])
        find_seen_induction_accuracy = False
        find_unseen_induction_accuracy = False
        if os.path.isdir(exp_path):
            induction_path = os.path.join(exp_path, f"epoch1", "neural2symbolic.log")
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
                print(f"Error reading file {induction_path}: {e}")
                continue

            if find_seen_induction_accuracy and find_unseen_induction_accuracy:
                pretrain_latent_induction.append(latent_z)
                pretrain_seen_induction_accuracy.append(this_seen_induction_accuracy)
                pretrain_unseen_induction_accuracy.append(this_unseen_induction_accuracy)
            
            deduction_path = os.path.join(exp_path, f"epoch1", "symbolic2neural.log")
            try:
                with open(deduction_path, "r") as f:
                    # 分别找到seen task accuracy和unseen task accuracy
                    lines = f.readlines()
                    for line in lines:
                        if "accuracy on seen task" in line:
                            find_seen_deduction_accuracy = True
                            this_seen_deduction_accuracy = float(line.split("samples: ")[1].split(" ")[0])
                        elif "accuracy on unseen task" in line:
                            find_unseen_deduction_accuracy = True
                            this_unseen_deduction_accuracy = float(line.split("samples: ")[1].split(" ")[0])

            except Exception as e:
                print(f"Error reading file {induction_path}: {e}")
                continue

            if find_seen_deduction_accuracy and find_unseen_deduction_accuracy:
                pretrain_latent_deduction.append(latent_z)
                pretrain_seen_deduction_accuracy.append(this_seen_deduction_accuracy)
                pretrain_unseen_deduction_accuracy.append(this_unseen_deduction_accuracy)

    if "ft-domain-z" in exp_name:
        exp_path = os.path.join(nesy_exp_dir, exp_name)
        latent_z = int(exp_name.split("-z")[1])
        find_seen_induction_accuracy = False
        find_unseen_induction_accuracy = False
        if os.path.isdir(exp_path):
            induction_path = os.path.join(exp_path, "symbolic_task.log")
            try:
                with open(induction_path, "r") as f:
                    # 分别找到seen task accuracy和unseen task accuracy
                    lines = f.readlines()
                    for line in lines:
                        if "symbolic seen task accuracy" in line:
                            find_seen_induction_accuracy = True
                            this_seen_induction_accuracy = float(line.split("finetuning: ")[1].split(" ")[0])
                        elif "symbolic unseen task accuracy" in line:
                            find_unseen_induction_accuracy = True
                            this_unseen_induction_accuracy = float(line.split("finetuning: ")[1].split(" ")[0])

            except Exception as e:
                print(f"Error reading file {induction_path}: {e}")
                continue
            
            if find_seen_induction_accuracy and find_unseen_induction_accuracy:
                sft_latent_induction.append(latent_z)
                sft_seen_induction_accuracy.append(this_seen_induction_accuracy)
                sft_unseen_induction_accuracy.append(this_unseen_induction_accuracy)

            deduction_path = os.path.join(exp_path, "neural_task.log")
            try:
                with open(deduction_path, "r") as f:
                    # 分别找到seen task accuracy和unseen task accuracy
                    lines = f.readlines()
                    for line in lines:
                        if "neural seen task accuracy" in line:
                            find_seen_deduction_accuracy = True
                            this_seen_deduction_accuracy = float(line.split("finetuning: ")[1].split(" ")[0])
                        elif "neural unseen task accuracy" in line:
                            find_unseen_deduction_accuracy = True
                            this_unseen_deduction_accuracy = float(line.split("finetuning: ")[1].split(" ")[0])

            except Exception as e:
                print(f"Error reading file {induction_path}: {e}")
                continue
            
            if find_seen_deduction_accuracy and find_unseen_deduction_accuracy:
                sft_latent_deduction.append(latent_z)
                sft_seen_deduction_accuracy.append(this_seen_deduction_accuracy)
                sft_unseen_deduction_accuracy.append(this_unseen_deduction_accuracy)


if len(nesy_latent_induction) > 0:
    try:
        nesy_latent_induction, nesy_seen_induction_accuracy, nesy_unseen_induction_accuracy \
        = zip(*sorted(zip(nesy_latent_induction, nesy_seen_induction_accuracy, nesy_unseen_induction_accuracy), key=lambda x: x[0]))
        plt.plot(nesy_latent_induction, nesy_seen_induction_accuracy, label="SHIP Induction Domain Seen",)
        plt.plot(nesy_latent_induction, nesy_unseen_induction_accuracy, label="SHIP Induction Domain Unseen")
    except Exception as e:
        print(f"Error plotting induction: {e}")

if len(nesy_latent_deduction) > 0:
    try:
        nesy_latent_deduction, nesy_seen_deduction_accuracy, nesy_unseen_deduction_accuracy \
        = zip(*sorted(zip(nesy_latent_deduction, nesy_seen_deduction_accuracy, nesy_unseen_deduction_accuracy), key=lambda x: x[0]))
        plt.plot(nesy_latent_deduction, nesy_seen_deduction_accuracy, label="SHIP Deduction Domain Seen")
        plt.plot(nesy_latent_deduction, nesy_unseen_deduction_accuracy, label="SHIP Deduction Domain Unseen")
    except Exception as e:
        print(f"Error plotting deduction: {e}")

if len(pretrain_latent_induction) > 0:
    try:
        pretrain_latent_induction, pretrain_seen_induction_accuracy, pretrain_unseen_induction_accuracy \
        = zip(*sorted(zip(pretrain_latent_induction, pretrain_seen_induction_accuracy, pretrain_unseen_induction_accuracy), key=lambda x: x[0]))
        plt.plot(pretrain_latent_induction, pretrain_seen_induction_accuracy, label="SHIP Induction Pretrain Seen",)
        plt.plot(pretrain_latent_induction, pretrain_unseen_induction_accuracy, label="SHIP Induction Pretrain Unseen")
    except Exception as e:
        print(f"Error plotting induction: {e}")

if len(pretrain_latent_deduction) > 0:
    try:
        pretrain_latent_deduction, pretrain_seen_deduction_accuracy, pretrain_unseen_deduction_accuracy \
        = zip(*sorted(zip(pretrain_latent_deduction, pretrain_seen_deduction_accuracy, pretrain_unseen_deduction_accuracy), key=lambda x: x[0]))
        plt.plot(pretrain_latent_deduction, pretrain_seen_deduction_accuracy, label="SHIP Deduction Pretrain Seen")
        plt.plot(pretrain_latent_deduction, pretrain_unseen_deduction_accuracy, label="SHIP Deduction Pretrain Unseen")
    except Exception as e:
        print(f"Error plotting deduction: {e}")


# if len(sft_latent_induction) > 0:
#     try:
#         sft_latent_induction, sft_seen_induction_accuracy, sft_unseen_induction_accuracy \
#         = zip(*sorted(zip(sft_latent_induction, sft_seen_induction_accuracy, sft_unseen_induction_accuracy), key=lambda x: x[0]))
#         plt.plot(sft_latent_induction, sft_seen_induction_accuracy, label="SFT Induction Seen",)
#         plt.plot(sft_latent_induction, sft_unseen_induction_accuracy, label="SFT Induction Unseen")
#     except Exception as e:
#         print(f"Error plotting induction: {e}")

# if len(sft_latent_deduction) > 0:
#     try:
#         sft_latent_deduction, sft_seen_deduction_accuracy, sft_unseen_deduction_accuracy \
#         = zip(*sorted(zip(sft_latent_deduction, sft_seen_deduction_accuracy, sft_unseen_deduction_accuracy), key=lambda x: x[0]))
#         plt.plot(sft_latent_deduction, sft_seen_deduction_accuracy, label="SFT Deduction Seen")
#         plt.plot(sft_latent_deduction, sft_unseen_deduction_accuracy, label="SFT Deduction Unseen")
#     except Exception as e:
#         print(f"Error plotting deduction: {e}")

#横轴标签：pretrain ratio
plt.xlabel("observed samples")
plt.ylabel("accuracy")
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("latent.pdf", bbox_inches='tight')

# 生成markdown表格
def generate_markdown_table():
    # 合并所有数据，确保完整性
    all_latent_values = sorted(list(set(nesy_latent_induction + nesy_latent_deduction)))
    
    if len(all_latent_values) == 0:
        print("没有找到可用的数据来生成表格")
        return
    
    # 创建数据字典方便查找
    pretrain_induction_data = {}
    pretrain_deduction_data = {}
    domain_induction_data = {}
    domain_deduction_data = {}
    
    for i, latent in enumerate(nesy_latent_induction):
        domain_induction_data[latent] = {
            'seen': nesy_seen_induction_accuracy[i],
            'unseen': nesy_unseen_induction_accuracy[i]
        }
    
    for i, latent in enumerate(nesy_latent_deduction):
        domain_deduction_data[latent] = {
            'seen': nesy_seen_deduction_accuracy[i],
            'unseen': nesy_unseen_deduction_accuracy[i]
        }
    
    for i, latent in enumerate(pretrain_latent_induction):
        pretrain_induction_data[latent] = {
            'seen': pretrain_seen_induction_accuracy[i],
            'unseen': pretrain_unseen_induction_accuracy[i]
        }
    
    for i, latent in enumerate(pretrain_latent_deduction):
        pretrain_deduction_data[latent] = {
            'seen': pretrain_seen_deduction_accuracy[i],
            'unseen': pretrain_unseen_deduction_accuracy[i]
        }
    
    # 生成markdown表格
    markdown_content = "# 实验结果表格\n\n"
    
    # 表头
    header = "|  | " + " | ".join([str(val) for val in all_latent_values]) + " |\n"
    separator = "|" + "---|" * (len(all_latent_values) + 1) + "\n"
    
    markdown_content += header
    markdown_content += separator
    
    # 第一行：number of soft tokens
    row1 = "| number of soft tokens | " + " | ".join([str(val) for val in all_latent_values]) + " |\n"
    markdown_content += row1
    
    # 第二行：dimension of latent
    row2 = "| dimension of latent | " + " | ".join([str(val * 4096) for val in all_latent_values]) + " |\n"
    markdown_content += row2
    
    # 第三行：nesy_seen_induction_accuracy
    row3_values = []
    for val in all_latent_values:
        if val in domain_induction_data:
            row3_values.append(f"{domain_induction_data[val]['seen']*100:.2f}")
        else:
            row3_values.append("N/A")
    row3 = "| nesy_seen_induction_accuracy | " + " | ".join(row3_values) + " |\n"
    markdown_content += row3
    
    # 第四行：nesy_unseen_induction_accuracy
    row4_values = []
    for val in all_latent_values:
        if val in domain_induction_data:
            row4_values.append(f"{domain_induction_data[val]['unseen']*100:.2f}")
        else:
            row4_values.append("N/A")
    row4 = "| nesy_unseen_induction_accuracy | " + " | ".join(row4_values) + " |\n"
    markdown_content += row4
    
    # 第五行：nesy_seen_deduction_accuracy
    row5_values = []
    for val in all_latent_values:
        if val in domain_deduction_data:
            row5_values.append(f"{domain_deduction_data[val]['seen']*100:.2f}")
        else:
            row5_values.append("N/A")
    row5 = "| nesy_seen_deduction_accuracy | " + " | ".join(row5_values) + " |\n"
    markdown_content += row5
    
    # 第六行：nesy_unseen_deduction_accuracy
    row6_values = []
    for val in all_latent_values:
        if val in domain_deduction_data:
            row6_values.append(f"{domain_deduction_data[val]['unseen']*100:.2f}")
        else:
            row6_values.append("N/A")
    row6 = "| nesy_unseen_deduction_accuracy | " + " | ".join(row6_values) + " |\n"
    markdown_content += row6

    # 第七行：pretrain_seen_induction_accuracy
    row7_values = []
    for val in all_latent_values:
        if val in pretrain_induction_data:
            row7_values.append(f"{pretrain_induction_data[val]['seen']*100:.2f}")
        else:
            row7_values.append("N/A")
    row7 = "| pretrain_seen_induction_accuracy | " + " | ".join(row7_values) + " |\n"
    markdown_content += row7

    # 第八行：pretrain_unseen_induction_accuracy
    row8_values = []
    for val in all_latent_values:
        if val in pretrain_induction_data:
            row8_values.append(f"{pretrain_induction_data[val]['unseen']*100:.2f}")
        else:
            row8_values.append("N/A")
    row8 = "| pretrain_unseen_induction_accuracy | " + " | ".join(row8_values) + " |\n"
    markdown_content += row8

    # 第九行：pretrain_seen_deduction_accuracy
    row9_values = []
    for val in all_latent_values:
        if val in pretrain_deduction_data:
            row9_values.append(f"{pretrain_deduction_data[val]['seen']*100:.2f}")
        else:
            row9_values.append("N/A")
    row9 = "| pretrain_seen_deduction_accuracy | " + " | ".join(row9_values) + " |\n"
    markdown_content += row9

    # 第十行：pretrain_unseen_deduction_accuracy
    row10_values = []
    for val in all_latent_values:
        if val in pretrain_deduction_data:
            row10_values.append(f"{pretrain_deduction_data[val]['unseen']*100:.2f}")
        else:
            row10_values.append("N/A")
    row10 = "| pretrain_unseen_deduction_accuracy | " + " | ".join(row10_values) + " |\n"
    markdown_content += row10
    
    # 保存到文件
    with open("latent_results.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print("Markdown表格已保存到 latent_results.md")
    print("\n生成的表格内容：")
    print(markdown_content)

# 调用函数生成表格
generate_markdown_table()



