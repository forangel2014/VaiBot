import json
import os
import datasets
from tqdm import tqdm
import pandas as pd
import re
import random
from datetime import datetime, timedelta
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def date_understanding_gen(num_records=10000):
    # 定义函数：生成指定范围内的随机天数
    def generate_random_days(min_days=-1000, max_days=1000):
        return random.randint(min_days, max_days)

    # 定义函数：根据输入日期和天数计算目标日期
    def calculate_target_date(input_date, days_offset):
        input_date_obj = datetime.strptime(input_date, "%m/%d/%Y")
        target_date_obj = input_date_obj + timedelta(days=days_offset)
        return target_date_obj.strftime("%m/%d/%Y")
    data_list = []
    for _ in range(num_records):
        # 随机生成输入日期（假设在 2020-2025 年之间）
        start_year = random.randint(1000, 2500)
        start_month = random.randint(1, 12)
        start_day = random.randint(1, 28)  # 简化处理，避免月末日期问题
        input_date = f"{start_month:02d}/{start_day:02d}/{start_year}"

        # 随机生成天数偏移量
        days_offset = generate_random_days()

        # 计算目标日期
        output_date = calculate_target_date(input_date, days_offset)

        # 构造数据字典
        record = {
            "knowledge": f"{abs(days_offset)} days {'before' if days_offset < 0 else 'after'} this date.",
            "input": input_date,
            "output": output_date
        }

        # 添加到列表
        data_list.append(record)

    return data_list

def dyck_languages_gen(num_records=1000):
    def generate_random_bracket_sequence(max_length=10):
        brackets = ["{", "[", "("]
        sequence = []
        length = random.randint(1, max_length)  # 随机生成序列长度
        for _ in range(length):
            sequence.append(random.choice(brackets))
        return "".join(sequence)
    def fix_bracket_sequence(sequence):
        stack = []
        result = []
        opening = {"}": "{", "]": "[", ")": "("}
        closing = {"{": "}", "[": "]", "(": ")"}

        # 遍历输入序列，尝试修复
        for char in sequence:
            if char in opening.values():  # 如果是左括号，入栈
                stack.append(char)
                result.append(char)
            elif char in opening:  # 如果是右括号
                if stack and stack[-1] == opening[char]:  # 栈顶匹配
                    stack.pop()
                    result.append(char)
                else:  # 不匹配，忽略该右括号
                    continue
            else:  # 非括号字符，直接添加
                result.append(char)

        # 添加剩余未闭合的括号
        while stack:
            last_open = stack.pop()
            result.append(closing[last_open])

        return "".join(result)
    data_list = []
    for _ in range(num_records):
        # 生成随机括号序列作为输入
        input_sequence = generate_random_bracket_sequence()

        # 修复括号序列，确保闭合正确
        output_sequence = fix_bracket_sequence(input_sequence)

        # 构造数据字典
        record = {
            "knowledge": "Complete the rest of the sequence, making sure that the parentheses are closed properly.",
            "input": input_sequence,
            "output": output_sequence[len(input_sequence):]
        }

        # 添加到列表
        data_list.append(record)

    return data_list

def load_data(load_from_local=False, save=True):
    mkdir(f"./data/ood_data")
    all_samples = {}

    print("loading: corypaik-prost")
    path = "corypaik/prost"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/corypaik-prost.json"))
    else:
        dataset = datasets.load_dataset(path,split='explicit_questions')
        dataset_samples = []
    #This dataset only has test split
        for sample in tqdm(list(dataset["test"])):
            my_sample = {
                "knowledge": sample["question"].replace("[MASK]", sample[chr(ord('A') + sample['label'])]),
                "input": sample["context"],
                "target": sample[chr(ord('A') + sample['label'])],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (corypaik-prost): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/corypaik-prost.json", "w"))

    print("loading: NEWTONReasoning-NEWTON")
    path = "NEWTONReasoning/NEWTON"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/NEWTONReasoning-NEWTON.json"))
    else:
    # download from: https://huggingface.co/datasets/NEWTONReasoning/NEWTON/resolve/main/explicit_questions.csv
        raw_dataset = pd.read_csv('./data/ood_data/raw/NEWTON-explicit_questions.csv')
        dataset = raw_dataset[raw_dataset['q_type'] == 'MC']
        dataset_samples = []
        for index, sample in dataset.iterrows():
            my_sample = {
                "knowledge": sample['question'].replace('Which object', sample['gt']).replace('?','.'),
                "input": ', '.join([sample['choice_1'],sample['choice_2'],sample['choice_3'],sample['choice_4']]),
                "target": sample['gt'],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (NEWTONReasoning-NEWTON): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/NEWTONReasoning-NEWTON.json", "w"))

    print("loading: allenai-openbookqa")
    path = "allenai/openbookqa"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/allenai-openbookqa.json"))
    else:
        dataset = datasets.load_dataset(path,"additional")
        dataset_samples = []
        for sample in tqdm(list(dataset["train"])):
            my_sample = {
                "knowledge": sample['fact1'],
                "input": sample['question_stem'],
                "target": sample['choices']['text'][ord(sample['answerKey'])-ord('A')],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (allenai-openbookqa): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/allenai-openbookqa.json", "w"))

    print("loading: e-CARE")
    path = "e-CARE"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/e-CARE.json"))
    else:
        # download from: https://raw.githubusercontent.com/Waste-Wood/e-CARE/refs/heads/main/dataset/Explanation_Generation/train.jsonl
        dataset = [json.loads(line) for line in open('./data/ood_data/raw/e-care-train.jsonl', 'r', encoding='utf-8')]
        dataset_samples = []
        for sample in tqdm(list(dataset)):
            my_sample = {
                "knowledge": sample['conceptual_explanation'],
                "input": sample['cause'],
                "target": sample['effect'],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (e-CARE): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/e-CARE.json", "w"))

    print("loading: derek-thomas-ScienceQA")
    path = "derek-thomas/ScienceQA"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/derek-thomas-ScienceQA.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            if sample['topic'] not in ['biology','chemistry', 'earth-science', 'geography', 'physics', 'science-and-engineering-practices', 'units-and-measurement']\
                    or len(sample['hint'])==0 or len(sample['lecture'])==0:
                continue
            my_sample = {
                "knowledge": sample['lecture'],
                "input": sample['hint']+" "+sample['question'],
                "target": sample['choices'][sample['answer']],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (derek-thomas-ScienceQA): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/derek-thomas-ScienceQA.json", "w"))

    print("loading: openlifescienceai-medmcqa")
    path = "openlifescienceai/medmcqa"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/openlifescienceai-medmcqa.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            my_sample = {
                "knowledge": sample['exp'],
                "input": sample['question'],
                "target": sample[f'op{chr(ord("a")+sample["cop"])}'],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (openlifescienceai-medmcqa): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/openlifescienceai-medmcqa.json", "w"))

    print("loading: qiaojin-PubMedQA")
    path = "qiaojin/PubMedQA"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/qiaojin-PubMedQA.json"))
    else:
        dataset = datasets.load_dataset(path,"pqa_labeled")
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            my_sample = {
                "knowledge": sample['long_answer'],
                "input": sample['question'],
                "target": sample['final_decision'],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (qiaojin-PubMedQA): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/qiaojin-PubMedQA.json", "w"))

    print("loading: fzkuji-CMExam")
    path = "fzkuji/CMExam"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/fzkuji-CMExam.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            if "下列" in sample['Question']:continue
            my_sample = {
                "knowledge": sample['Explanation'],
                "input": sample['Question'],
                "target": next((item["value"] for item in sample['Options'] if item["key"] == sample['Answer']), None),
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (fzkuji-CMExam): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/fzkuji-CMExam.json", "w"))

    print("loading: allenai-multi_lexsum")
    path = "allenai/multi_lexsum"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/allenai-multi_lexsum.json"))
    else:
        dataset=datasets.load_dataset("allenai/multi_lexsum", name="v20230518",trust_remote_code=True)
        dataset_samples = []
        instructions={
            "long":"Given a large collection of legal documents related to a civil rights lawsuit, generate a long summary (around 650 words) covering key events and outcomes.",
            "short":"Given a large collection of legal documents related to a civil rights lawsuit, generate a short summary (around 130 words) providing a brief overview.",
            "tiny":"Given a large collection of legal documents related to a civil rights lawsuit, generate a tiny summary (around 25 words) suitable for quick updates."
        }
        for sample in tqdm(list(dataset['train'])):
            for sum_len in ["long", "short", "tiny"]:
                if sample["summary/" + sum_len]==None:continue
                my_sample = {
                    "knowledge":instructions[sum_len] ,
                    "input": '\n'.join(sample['sources']),
                    "target": sample["summary/" + sum_len],
                }
                dataset_samples.append(my_sample)
    print(f"有效样本数量 (allenai-multi_lexsum): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/allenai-multi_lexsum.json", "w"))

    print("loading: ContractNLI")
    path = "ContractNLI"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/ContractNLI.json"))
    else:
        # download from: https://stanfordnlp.github.io/contract-nli/#download
        dataset = json.load(open('./data/ood_data/raw/ContractNLI-train.json', 'r', encoding='utf-8'))
        dataset_samples = []
        for sample in tqdm(list(dataset['documents'])):
            for index,annotation in sample['annotation_sets'][0]['annotations'].items():
                if annotation['choice']=='NotMentioned':continue
                my_sample = {
                    "knowledge": '\n'.join([sample['text'][evidence_idx[0]:evidence_idx[1]+1] for evidence_idx in [sample['spans'][i] for i in annotation['spans']]]),
                    "input": "Document:\n"+sample['text']+"\nHypothesis:\n"+dataset['labels'][index]['hypothesis'],
                    "target": annotation['choice'],
                }
                dataset_samples.append(my_sample)
    print(f"有效样本数量 (ContractNLI): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/ContractNLI.json", "w"))

    print("loading: maveriq/bigbenchhard - date_understanding")
    path="date_understanding"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/date_understanding.json"))
    else:
        dataset_samples=date_understanding_gen(10000)
    print(f"有效样本数量 (date_understanding): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/date_understanding.json", "w"))

    print("loading: maveriq/bigbenchhard - dyck_languages")
    path="dyck_languages"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/dyck_languages.json"))
    else:
        dataset_samples=dyck_languages_gen(10000)
    print(f"有效样本数量 (dyck_languages): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/dyck_languages.json", "w"))

    print("loading: maveriq/bigbenchhard - geometric_shapes")
    path = "maveriq/bigbenchhard"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/geometric_shapes.json"))
    else:
        dataset = datasets.load_dataset(path,"geometric_shapes")
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            my_sample = {
                "knowledge": "Identify the shape it represents based on the number of sides it connects.",
                "input": sample['input'].split('draws a')[0],
                "target": re.search(rf"\({sample['target'][1]}\)\s*([a-zA-Z]+)",sample['input'].split('Options')[1]).group(1)
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (geometric_shapes): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/geometric_shapes.json", "w"))

    print("loading: maveriq/bigbenchhard - word_sorting")
    path = "maveriq/bigbenchhard"
    if load_from_local:
        dataset_samples = json.load(open(f"./data/ood_data/word_sorting.json"))
    else:
        dataset = datasets.load_dataset(path,"word_sorting")
        dataset_samples = []
        for sample in tqdm(list(dataset['train'])):
            my_sample = {
                "knowledge": "Sort the following words alphabetically.",
                "input": sample['input'].split('List: ')[1],
                "target": sample['target']
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (word_sorting): {len(dataset_samples)}")
    all_samples[path]=dataset_samples
    if save:
        json.dump(dataset_samples, open(f"./data/ood_data/word_sorting.json", "w"))


if __name__ == '__main__':
    load_data()