import copy
import openai
import os
import torch
import re
import json
import numpy as np
import subprocess
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import string
from src.rouge import rouge_scorer
openai.api_key = "sk-GX5fQitXHKizUe4iF8Ed3375A72847A8807c9dAb0290C1Bc"
        # openai.base_url = url
openai.base_url = 'https://chatapi.onechats.top/v1/'

with open('src/data_dict.json', 'r') as f:
    data_dict = json.load(f)
    data_map = data_dict['data_map']

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def hook(grad, name=None):
    if name:
        print(name)
    print(grad)

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    result = f"{hours}h{minutes}m{remaining_seconds}s"
    return result

def get_gpu_memory_usage():
    # 使用nvidia-smi命令获取显存信息
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    # 解析输出
    output = result.stdout.decode('utf-8')
    lines = output.strip().split('\n')
    
    # 格式化输出
    for i, line in enumerate(lines):
        used, total = line.split(',')
        print(f"GPU {i}: {used} MiB / {total} MiB")

def plot_loss_curve(loss_dict, name):
    # 创建一个新的图形
    plt.figure()

    # 遍历损失字典中的每个损失
    for loss_name, loss_values in loss_dict.items():
        # 绘制损失曲线
        plt.plot(range(len(loss_values)), loss_values, label=loss_name)

    # 添加图例
    plt.legend()

    # 设置图形标题和轴标签
    plt.title('Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # 显示图形
    plt.savefig(f"{name}.pdf")

def tsne(encoded_latent, trained_latent, randomn_latent, filename):

    # Assume encoded_latent and trained_latent are lists of tensors
    # Convert the lists to numpy arrays
    encoded_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in encoded_latent]
    trained_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in trained_latent]
    randomn_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in randomn_latent]

    encoded_latent_np = np.array(encoded_latent)
    trained_latent_np = np.array(trained_latent)
    randomn_latent_np = np.array(randomn_latent)

    # Flatten the tensors to 2D arrays
    encoded_latent_flat = encoded_latent_np.reshape((len(encoded_latent), -1))
    trained_latent_flat = trained_latent_np.reshape((len(trained_latent), -1))
    randomn_latent_flat = randomn_latent_np.reshape((len(randomn_latent), -1))

    # Combine the flattened arrays
    combined_data = np.vstack((encoded_latent_flat, trained_latent_flat, randomn_latent_flat))

    # Apply t-SNE to reduce the dimensions to 2
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_result = tsne.fit_transform(combined_data)

    # Plotting the t-SNE visualization
    plt.figure(figsize=(8, 6))

    # Plot the encoded_latent points
    plt.scatter(tsne_result[:5, 0], tsne_result[:5, 1], label='encoded_latent')

    # Plot the trained_latent points
    plt.scatter(tsne_result[5:10, 0], tsne_result[5:10, 1], label='trained_latent')
    
    plt.scatter(tsne_result[10:, 0], tsne_result[10:, 1], label='randomn_latent', color="gray")

    # Add legend
    plt.legend()

    # Show the plot
    mkdir(os.path.dirname(filename))
    plt.savefig(filename)

def create_task_data_lookup(data):
    
    seen_train_knowledge_base = []
    
    for train_sample in data["seen_tasks"]["train"]:
        
        seen_train_knowledge_base.append(train_sample["knowledge"])
        
    lookup = dict(zip(seen_train_knowledge_base, list(range(len(seen_train_knowledge_base)))))
    
    return seen_train_knowledge_base, lookup

def load_task_data(task, unseen_task_ratio=None, unseen_task_num=None, test_sample_ratio=None, test_sample_num=None,
                   num_words=32, num_pertask=1000, task_fields=None):
    
    all_data = {
        "seen_tasks": {
            "train": [],
            "test": []
        },
        "unseen_tasks": {
            "train": [],
            "test": []
        },
        "prompt_template": None,
        "neural_evaluater": None,
        "symbolic_evaluater": None,
        "task_num": None,
        "seen_task_num": None
    }
    
    if task == "list_functions":
        
        prompt_template = """please predict the output list given the input list.
input: {}
output: """

        all_data["prompt_template"] = prompt_template
        
        all_task_id = list(range(1, 251))
        task_num = len(all_task_id)
        random.shuffle(all_task_id)
        
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
            
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in range(1, 251):

            task_type = task_id2task_type[sub_task_id]
            
            task_dir = f"./data/{task}/c{sub_task_id:03d}"
            task_file = os.path.join(task_dir, "task.json")

            with open(task_file) as f:
                sub_task_data = json.load(f)

            description = sub_task_data["description"]
            examples = sub_task_data["examples"]

            pattern = r'"([^"]*)"'
            rule = re.findall(pattern, description)[0]
        
            sample_num = len(examples)
            all_sample_id = list(range(sample_num))
            random.shuffle(all_sample_id)
            
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")

            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
        
                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": example["input"],
                    "target": example["target"],
                    "knowledge": rule,
                    #"metadata": task_data,
                })
        
            if sub_task_id == 1:
                
                output_regex = sub_task_data["output_regex"]
                
                def neural_evaluater(y_pred, y_true):
                    matched = re.findall(output_regex, y_pred)
                    if len(matched):
                        return int(matched[0] == str(y_true))
                    else:
                        return 0

                def symbolic_evaluater(knowledge_pred, knowledge_true):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are two transformations described in natural language for lists. 
Please help me determine if these two transformations are equivalent.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
                        except:
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score

                all_data["neural_evaluater"] = neural_evaluater
                all_data["symbolic_evaluater"] = symbolic_evaluater

    elif task == "sni":
        
        prompt_template = """Please complete the following task given the input and only return the output without other words.
Input: {}
Output: """

        all_data["prompt_template"] = prompt_template
        with open(f"./data/{task}/tasks_{num_words}.txt") as f:
            train_tasks = f.readlines()
            
        if task_fields is not None:
            task_fields = [data_map[t] for t in task_fields.split(',')]
            tasks = copy.deepcopy(train_tasks)
            train_tasks = []
            for task_name in tasks:
                task_file = f"./data/{task}/tasks/{task_name.strip()}.json"
                with open(task_file) as f:
                    sub_task_data = json.load(f)
                    if sub_task_data["Categories"][0] in task_fields:
                        train_tasks.append(task_name)
                
        
        all_task_id = list(range(1, len(train_tasks) + 1))
        task_num = len(all_task_id)
        print(f"task_num: {task_num}")
        random.shuffle(all_task_id)
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in range(1, len(train_tasks) + 1):

            task_type = task_id2task_type[sub_task_id]
            task_file = f"./data/{task}/tasks/{train_tasks[sub_task_id - 1].strip()}.json"
            with open(task_file) as f:
                sub_task_data = json.load(f)

            description = sub_task_data["Definition"][0]
            examples = []
            for ex in sub_task_data["Instances"]:
                if len(ex['input'].split(' ')) < 20:
                    examples.append(ex)
                # else:
                #     print(len(ex['input'].split(' ')))
                if len(examples) == num_pertask:
                    break
            if len(examples) != num_pertask:
                print(f"task_name: {train_tasks[sub_task_id - 1].strip()}, task_num: {len(examples)}")
                if len(examples) < 60:
                    continue
            # examples = sub_task_data["Instances"][:num_pertask]
            rule = description
            
            all_sample_id = list(range(len(examples)))
            sample_num = len(all_sample_id)
            random.shuffle(all_sample_id)
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")
            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
                output = random.choice(example["output"])
                if output == '':
                    continue
                if not output[-1] in string.punctuation:
                    output += "."

                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": example["input"] + "." if not example["input"][-1] in string.punctuation else example["input"],
                    "target": output,
                    "knowledge": rule,
                    #"metadata": task_data,
                })
        
            if sub_task_id == 1:
                def symbolic_evaluater(knowledge_pred, knowledge_true):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are two instructions described in natural language. 
Please help me determine if these two instructions are equivalent.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
                        except:
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score
                
                def neural_evaluater(y_pred, y_true):
                    return (normalize_answer(y_pred.split('\n')[0]) == normalize_answer(y_true))
                
                all_data["neural_evaluater"] = neural_evaluater
                all_data["symbolic_evaluater"] = symbolic_evaluater

    all_data["seen_task_num"] = seen_task_num
    all_data["task_num"] = task_num
    print(f"seen_tasks: {seen_task_num}, unseen_tasks: {task_num - seen_task_num}")
    print(f"seen_tasks train number: {len(all_data['seen_tasks']['train'])}")
    print(all_data['seen_tasks']['train'][0])
    # import pdb; pdb.set_trace()
    return all_data

def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

# load_task_data('sni', test_ratio=0.1, unseen_task_ratio=0.1, num_words=32, num_pertask=1000, task_fields='Translation')