import sys
import os

from transformers import AutoTokenizer

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_pretrain_data_hf

train_dataset, valid_dataset = load_pretrain_data_hf(pretrain_data_ratio=1.0)
all_pretrain_knowledge = [data["knowledge"] for data in train_dataset]

f = open("./exp_final/vae-pretrain/epoch1/neural2symbolic.log", "r")
lines = f.readlines()

all_groundtruth_knowledge = []
all_predicted_knowledge = []

for line in lines:
    if "groundtruth knowledge" in line:
        all_groundtruth_knowledge.append(line.split("groundtruth knowledge\": \"")[1].strip())
    if "predicted knowledge" in line:
        all_predicted_knowledge.append(line.split("predicted knowledge\": \"")[1].strip())

all_groundtruth_knowledge = [knowledge.replace("<instruction>", "").replace("</instruction>", "") for knowledge in all_groundtruth_knowledge]
all_predicted_knowledge = [knowledge.replace("<instruction>", "").replace("</instruction>", "") for knowledge in all_predicted_knowledge]
all_pretrain_knowledge = [knowledge.replace("<instruction>", "").replace("</instruction>", "") for knowledge in all_pretrain_knowledge]


all_groundtruth_knowledge_tokens = [tokenizer.encode(knowledge, add_special_tokens=False) for knowledge in all_groundtruth_knowledge]
all_predicted_knowledge_tokens = [tokenizer.encode(knowledge, add_special_tokens=False) for knowledge in all_predicted_knowledge]
all_pretrain_knowledge_tokens = [tokenizer.encode(knowledge, add_special_tokens=False) for knowledge in all_pretrain_knowledge]

average_groundtruth_knowledge_tokens = sum(len(tokens) for tokens in all_groundtruth_knowledge_tokens) / len(all_groundtruth_knowledge_tokens)
average_predicted_knowledge_tokens = sum(len(tokens) for tokens in all_predicted_knowledge_tokens) / len(all_predicted_knowledge_tokens)
average_pretrain_knowledge_tokens = sum(len(tokens) for tokens in all_pretrain_knowledge_tokens) / len(all_pretrain_knowledge_tokens)

print(f"Average groundtruth knowledge tokens: {average_groundtruth_knowledge_tokens}")
print(f"Average predicted knowledge tokens: {average_predicted_knowledge_tokens}")
print(f"Average pretrain knowledge tokens: {average_pretrain_knowledge_tokens}")

pass
