import os
import time
import shutil
#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
import argparse
import random
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from utils import mkdir, convert_seconds, load_task_data, plot_loss_curve, tsne, create_task_data_lookup, get_gpu_memory_usage, load_pretrain_data_hf
from tqdm import tqdm
random.seed(73)
torch.manual_seed(73)

def train_subtask(args, nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data):

    if args.zero_init:
        params = torch.normal(mean=0, std=1e-2, size=(1, nesy.args.latent_size), requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16)
    else:
        params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16)
    
    optimizer = torch.optim.Adam([params], lr=args.task_finetune_lr)
    keep_training = True
    test_loss_ls = []
    
    #return params, test_loss_ls
    
    while keep_training:

        for i, batch in tqdm(enumerate(subtask_train_data_loader)):

            if i % 100 == 0:
                test_loss = 0
                with torch.no_grad():
                    for batch in subtask_test_data_loader:
                        x_batch = batch["input"]
                        x_batch = [prompt_template.format(x) for x in x_batch]
                        y_batch = batch["target"]
                        expanded_params = params.repeat_interleave(len(x_batch), dim=0)
                        test_loss += nesy.compute_task_loss(expanded_params, x_batch, y_batch)

                    test_loss /= len(subtask_test_data)
                    test_loss_ls.append(test_loss.tolist())
                    if len(test_loss_ls) > args.task_finetune_step:
                        if test_loss_ls[-1] > test_loss_ls[-2]:
                            keep_training = False
                            break

            optimizer.zero_grad()
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]
            expanded_params = params.repeat_interleave(len(x_batch), dim=0)
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
            task_loss.backward()
            optimizer.step()
    
    return params, test_loss_ls

def pretrain_subtask(args, train_data, test_data, nesy, prompt_template, log):
    
    all_tasks_ids = list(set([sample["sub_task_id"] for sample in test_data]))
    pretrained_params = []
    
    all_tasks_ids = all_tasks_ids[180:210]

    for task_id in tqdm(all_tasks_ids):
        
        log.writelines(f"training subtask {task_id}\n")
        log.flush()

        subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]
        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_test_data[0]["knowledge"]
        num_samples = 10
        
        optimal_params = []

        for i in range(num_samples):
            
            params, test_loss_ls = train_subtask(args, nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data)
            
            log.writelines(f"subtask train loss: {str(test_loss_ls)} \n")
            log.flush()
            
            optimal_params.append(params.detach().cpu())
            
        # pretrained_params.append({
        #     "task_id": task_id,
        #     "optimal_params": optimal_params
        # })

        save_dir = f"./exp/sni-pretrain/pretrain/{task_id}"
        mkdir(save_dir)
        #torch.save(pretrained_params, f"{args.exp_dir}/pretrain/{task_id}/optimal_params.pth")
        torch.save(pretrained_params, f"{save_dir}/optimal_params.pth")
    
    if args.fuse_method == "delta":
        json.dump(nesy.llm.param_info, open(f"{args.exp_dir}/params_info.json", "w"))

def valid_symbolic2neural(args, epoch, data_loader, nesy, prompt_template, evaluater, log, name):
    
    log.writelines(f"epoch {epoch} \n")

    start_time = time.time()
    num_correct = 0
    num_test = 0
    num_batches = 0  # 初始化一个计数器

    for batch in data_loader:
        
        if num_batches >= 5:  # 如果已经处理了10个batch，跳出循环
            break
        
        with torch.no_grad():
            knowledge_batch = batch["knowledge"]
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]
            results = nesy.eval_task(knowledge_batch, x_batch, y_batch, evaluater)
            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct += result["score"]
                num_test += 1
                log.flush()
                
        num_batches += 1
        #break
        
    accuracy = num_correct / num_test
    log.writelines(f"symbolic2neural accuracy on {name} samples: {accuracy} \n")
    end_time = time.time()
    cost_time = convert_seconds(end_time-start_time)
    log.writelines(f"symbolic2neural validation on {name} finished, time cost {cost_time} \n")
    log.flush()

def valid_neural2symbolic(args, epoch, train_data, test_data, nesy, prompt_template, evaluater, log, name):

    log.writelines(f"epoch {epoch} \n")

    start_time = time.time()
    all_tasks_ids = list(set([sample["sub_task_id"] for sample in test_data]))
    
    all_tasks_ids = random.sample(all_tasks_ids, 10)
    
    num_correct_symbolic = 0
    num_test_symbolic = 0
    num_correct_neural = 0
    num_test_neural = 0
    

    for task_id in all_tasks_ids:

        subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]
        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_test_data[0]["knowledge"]
        num_samples = 1

        knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
        encoded_latent = [nesy.reparameterize(*nesy.encode(knowledge_ids)) for i in range(num_samples)]
        randomn_latent = [torch.randn([1, nesy.args.latent_size]) for i in range(num_samples)]
        trained_latent = []
        
        # with torch.no_grad():
            
        #     if args.method == "vaeflow":
        #         params = []
        #         for latent in encoded_latent:
        #             latent = latent.to(nesy.args.flow_device)
        #             param = nesy.flow_forward(latent)
        #             param = param.to(nesy.args.task_device)
        #             params.append(param)
        #     else:
        #         params = [latent.to(nesy.args.task_device) for latent in encoded_latent]

        #     print("encoded params")
            
        #     batch = next(iter(subtask_train_data_loader))

        #     x_batch = batch["input"]
        #     x_batch = [prompt_template.format(x) for x in x_batch]
        #     y_batch = batch["target"]
            
        #     print("params 0")
        #     expanded_params = params[0].repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("params 1")
        #     expanded_params = params[1].repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)
            
        #     print("params 0+1 /2")
        #     expanded_params = ((params[0]+params[1])/2).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("params 2+3+4 /3")
        #     expanded_params = ((params[3]+params[4]+params[2])/3).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("random params")
        #     expanded_params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)
    
        for i in range(num_samples):
            
            trained_params, test_loss_ls = train_subtask(args, nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data)

            trained_latent.append(trained_params)

            with torch.no_grad():

                if args.method == "vaeflow":
                    trained_params = trained_params.to(nesy.args.flow_device)
                    trained_params = nesy.flow_backward(trained_params).to(nesy.args.decoder_device)
                else:
                    trained_params = trained_params.to(nesy.args.decoder_device)

                predicted_knowledge = nesy.sample(trained_params, sample_from_guassian=False)
                encoded_params = encoded_latent[i].to(nesy.args.decoder_device)
                encode_decode_knowledge = nesy.sample(encoded_params, sample_from_guassian=False)

            log.writelines(f"prediction on {name} subtask {task_id}: \n")
            log.writelines(f"subtask train loss: {str(test_loss_ls)} \n")
            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)
            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            result = nesy.eval_knowledge(knowledge, encode_decode_knowledge, evaluater)
            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_test_symbolic += 1
            log.flush()

        # with torch.no_grad():

        #     print("trained params")
            
        #     batch = next(iter(subtask_train_data_loader))

        #     x_batch = batch["input"]
        #     x_batch = [prompt_template.format(x) for x in x_batch]
        #     y_batch = batch["target"]
            
        #     print("params 0")
        #     expanded_params = trained_latent[0].repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("params 1")
        #     expanded_params = trained_latent[1].repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)
            
        #     print("params 0+1 /2")
        #     expanded_params = ((trained_latent[0]+trained_latent[1])/2).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("params 2+3+4 /3")
        #     expanded_params = ((trained_latent[3]+trained_latent[4]+trained_latent[2])/3).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)

        #     print("random params")
        #     expanded_params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16).repeat_interleave(len(x_batch), dim=0)
        #     test_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch)
        #     print(test_loss)


        #tsne(encoded_latent, trained_latent, randomn_latent, filename=f"{args.exp_dir}/epoch{epoch}/tsne/task{task_id}.pdf")
        #break

    # accuracy = num_correct_neural / num_test_neural
    # log.writelines(f"finetuned accuracy on {name} tasks: {accuracy} \n")

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"neural2symbolic accuracy on {name} samples: {accuracy} \n")
    end_time = time.time()
    cost_time = convert_seconds(end_time-start_time)
    log.writelines(f"neural2symbolic validation on {name} finished, time cost {cost_time} \n")
    log.flush()

def test_neural_task(args, seen_task_train_data_loader, seen_task_test_data_loader, unseen_task_test_data_loader, nesy, prompt_template, evaluater, log, method):

    log.writelines(f"neural task testing for method: {method} \n")
    log.flush()

    num_correct_neural = 0
    num_test_neural = 0

    if method == "finetuning-all":

        params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16)
        optimizer = torch.optim.Adam([params], lr=args.task_finetune_lr)
        keep_training = True
        test_loss_ls = []
        
        while keep_training:

            for i, batch in tqdm(enumerate(seen_task_train_data_loader)):

                if i % 100 == 0:
                    test_loss = 0
                    with torch.no_grad():
                        for batch in seen_task_test_data_loader:
                            knowledge_batch = batch["knowledge"]
                            batch_size = len(knowledge_batch)
                            x_batch = batch["input"]
                            x_batch = [prompt_template.format(x) for x in x_batch]
                            # knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
                            # knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
                            # x_batch_prompt = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
                            y_batch = batch["target"]
                            test_loss += nesy.compute_task_loss(params, x_batch, y_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step*10:
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                # knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
                # knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
                # x_batch_prompt = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
                y_batch = batch["target"]
                task_loss = nesy.compute_task_loss(params, x_batch, y_batch)
                task_loss.backward()
                optimizer.step()

    if method == "finetuning-each":
        
        all_knowledge = list(set([sample["knowledge"] for sample in seen_task_train_data_loader.dataset]))
        print(len(all_knowledge))
        all_params = torch.randn(size=[len(all_knowledge), nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.bfloat16)
        optimizer = torch.optim.Adam([all_params], lr=args.task_finetune_lr)
        keep_training = True
        test_loss_ls = []
        
        while keep_training:

            for i, batch in tqdm(enumerate(seen_task_train_data_loader)):

                if i % 100 == 0:
                    test_loss = 0
                    with torch.no_grad():
                        for batch in seen_task_test_data_loader:
                            knowledge_batch = batch["knowledge"]
                            batch_size = len(knowledge_batch)
                            x_batch = batch["input"]
                            x_batch = [prompt_template.format(x) for x in x_batch]
                            # knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
                            # knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
                            # x_batch_prompt = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
                            y_batch = batch["target"]
                            params = all_params[[all_knowledge.index(k) for k in knowledge_batch]]
                            test_loss += nesy.compute_task_loss(params, x_batch, y_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step*len(all_knowledge):
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                # knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
                # knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
                # x_batch_prompt = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
                y_batch = batch["target"]
                params = all_params[[all_knowledge.index(k) for k in knowledge_batch]]
                task_loss = nesy.compute_task_loss(params, x_batch, y_batch)
                task_loss.backward()
                optimizer.step()

    # start testing neural task
    with torch.no_grad():
        for batch in seen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            if method == "prompting":
                knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
                knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
                x_batch = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
            y_batch = batch["target"]
            results = []
            for i in range(batch_size):
                x_id = nesy.llm.tokenizer(x_batch[i], return_tensors="pt").input_ids.to(nesy.args.task_device)
                
                if method == "prompting":
                    new_task_parameters = None
                if method == "finetuning-all":
                    new_task_parameters = nesy.llm.allocate(params[0])
                if method == "finetuning-each":
                    new_task_parameters = nesy.llm.allocate(all_params[all_knowledge.index(knowledge_batch[i])])
                
                y_pred = nesy.llm.predict_task(x_id, new_task_parameters)
                results.append({
                    "knowledge": knowledge_batch[i],
                    "x": x_batch[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred,
                    "score": evaluater(y_pred, y_batch[i])
                    })
            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_neural += result["score"]
                num_test_neural += 1
                log.flush()

    accuracy = num_correct_neural / num_test_neural
    log.writelines(f"neural seen task accuracy of method {method}: {accuracy} \n")
    log.flush()
    
    with torch.no_grad():
        for batch in unseen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            knowledge_prompt = "Here is the instruction of this task: given the input list, the output list is {}\n"
            knowledge_prompt_batch = [knowledge_prompt.format(k) for k in knowledge_batch]
            x_batch_prompt = [knowledge_prompt_batch[i] + x_batch[i] for i in range(batch_size)]
            y_batch = batch["target"]
            results = []
            for i in range(batch_size):
                x_id = nesy.llm.tokenizer(x_batch_prompt[i], return_tensors="pt").input_ids.to(nesy.args.task_device)
                
                if method == "prompting":
                    new_task_parameters = None
                if method == "finetuning-all":
                    new_task_parameters = nesy.llm.allocate(params[0])
                if method == "finetuning-each":
                    new_task_parameters = nesy.llm.allocate(all_params[all_knowledge.index(knowledge_batch[i])])
                
                y_pred = nesy.llm.predict_task(x_id, new_task_parameters)
                results.append({
                    "knowledge": knowledge_batch[i],
                    "x": x_batch_prompt[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred,
                    "score": evaluater(y_pred, y_batch[i])
                    })
            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_neural += result["score"]
                num_test_neural += 1
                log.flush()

    accuracy = num_correct_neural / num_test_neural
    log.writelines(f"neural unseen task accuracy of method {method}: {accuracy} \n")
    log.flush()

def test_symbolic_task(args, seen_task_train_data_loader, seen_task_test_data_loader, unseen_task_test_data_loader, nesy, prompt_template, evaluater, log, method):

    log.writelines(f"symbolic task testing for method: {method} \n")
    log.flush()

    num_correct_symbolic = 0
    num_test_symbolic = 0

    # start testing symbolic task
    with torch.no_grad():
        for batch in seen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]
            
            prompt = """
I gave a friend an instruction and an input. 
The friend read the instruction and wrote an output for the input.
The input is "{}".
The friend's output is "{}".
So my instruction is: 
"""

            induction_questions = [prompt.format(x_batch[i], y_batch[i]) for i in range(batch_size)]
            
            results = []
            for i in range(batch_size):
                input_ids = nesy.llm.tokenizer(induction_questions[i], return_tensors="pt").input_ids.to(nesy.args.task_device)
                predicted_knowledge = nesy.llm.predict_task(input_ids)
                result = nesy.eval_knowledge(knowledge_batch[i], predicted_knowledge, evaluater)
                results.append(result)

            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_symbolic += result["score"]
                num_test_symbolic += 1
                log.flush()

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"neural seen task accuracy of method {method}: {accuracy} \n")
    log.flush()
    
def main(args):

    if args.exp_name is None:
        current_time = datetime.now()
        args.exp_name = str(current_time)
    args.exp_dir = f"{args.meta_exp_dir}/{args.exp_name}"
    mkdir(args.exp_dir)
    
    if args.load_exp:
        if args.load_exp == "self":
            args.load_exp = args.exp_dir
        else:
            args.load_exp = f"{args.meta_exp_dir}/{args.load_exp}"
        with open(f"{args.load_exp}/args.json", "r") as f:
            loaded_args = json.load(f)
        for key in loaded_args:
            if key not in ["exp_dir", "load_exp", "load_epoch", "encoder_device", "decoder_device", "task_device", 
                           "flow_device", "noise_device", "task_finetune_step", "task_finetune_lr", "batch_size",
                           "zero_init", "dataset", "pretraining"]:
                args.__dict__[key] = loaded_args[key]
        args.load_nesy_ckpt = f"{args.load_exp}/epoch{args.load_epoch}/nesy_ckpt/"
        start_epoch = args.load_epoch
        file_mode = "a"
    else:
        # training from scratch
        args.load_nesy_ckpt = None
        start_epoch = 0
        file_mode = "w"

    if args.fuse_method == "p-tuning":
        from transformers import AutoConfig
        task_model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        args.latent_size = args.num_soft_token * task_model_config.hidden_size
        print(args.latent_size)

    args_dict = vars(args)
    output_file = f"{args.exp_dir}/args.json"
    with open(output_file, "w") as f:
        json.dump(args_dict, f, indent=4)
        f.flush()

    if args.pretraining:
        train_dataset, valid_dataset = load_pretrain_data_hf()
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
        prompt_template = "{}"
        print("pretraining")

    data = load_task_data(task=args.dataset, unseen_task_ratio=args.unseen_task_ratio, test_sample_num=1, num_words=args.num_words, num_pertask=args.num_pertask, task_fields=args.task_fields)
    args.task_id2knowledge, args.knowledge2task_id = create_task_data_lookup(data)
    prompt_template = data["prompt_template"]
    neural_evaluater = data["neural_evaluater"]
    symbolic_evaluater = data["symbolic_evaluater"]
    seen_train_data_loader = DataLoader(data["seen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    seen_test_data_loader = DataLoader(data["seen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)
    unseen_train_data_loader = DataLoader(data["unseen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    unseen_test_data_loader = DataLoader(data["unseen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)

    if args.prior == "gaussian":
        from vae import Nesy
    elif args.prior == "mog":
        from gmvae import Nesy
    elif args.prior == "gmg":
        from gmg import Nesy
    elif args.prior == "vaeflow":
        from vaeflow import Nesy
    else:
        raise Exception("undefined prior")
    
    if args.prior == "vaeflow":
        nesy = Nesy(args)#.to(torch.bfloat16)
    else:
        nesy = Nesy(args).to(torch.bfloat16)

    if args.method == "nesy-pretrain":
        
        pretrain_log = open(f"{args.exp_dir}/pretrain.log", "w")

        pretrain_subtask(args, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, pretrain_log)

    elif args.method == "nesy":
        optimizer = torch.optim.Adam([
            {'params': nesy.llm.encoder.parameters(), 'lr': args.lr},
            {'params': nesy.encoder_mlp.parameters(), 'lr': args.lr},
            {'params': nesy.llm.decoder.parameters(), 'lr': args.lr},
            {'params': nesy.decoder_mlp.parameters(), 'lr': args.lr},
            #{'params': nesy.flow_net.parameters(), 'lr': args.lr},
            #{'params': nesy.logZ, 'lr': args.lr}
                                    ], lr=args.lr)
        if args.prior == "vaeflow" and args.ebm_optim_method == "fce":
            optimizer_noise = torch.optim.Adam(nesy.noise_flow_net.parameters(), lr=args.lr*0.01)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
        train_log = open(f"{args.exp_dir}/train.log", file_mode)
        try:
            shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/neural2symbolic.log", f"{args.exp_dir}/neural2symbolic.log")
            shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/symbolic2neural.log", f"{args.exp_dir}/symbolic2neural.log")
        except:
            pass
        neural2symbolic_valid_log = open(f"{args.exp_dir}/neural2symbolic.log", file_mode)
        symbolic2neural_valid_log = open(f"{args.exp_dir}/symbolic2neural.log", file_mode)
        
        train_data_loader = seen_train_data_loader if not args.pretraining else train_data_loader

        for epoch in range(start_epoch, args.num_epochs):

            if epoch % 1 == 0 and epoch > 0:

                nesy.save(f"{args.exp_dir}/epoch{epoch}/nesy_ckpt/")

                if args.pretraining:

                    valid_neural2symbolic(args, epoch, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="seen task")
                    #valid_neural2symbolic(args, epoch, data["unseen_tasks"]["train"], data["unseen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="unseen task")

                    valid_symbolic2neural(args, epoch, seen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="seen task test")
                    #valid_symbolic2neural(args, epoch, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="unseen task test")

                else:

                    valid_neural2symbolic(args, epoch, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="seen task")
                    #valid_neural2symbolic(args, epoch, data["unseen_tasks"]["train"], data["unseen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="unseen task")

                    valid_symbolic2neural(args, epoch, seen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="seen task test")
                    #valid_symbolic2neural(args, epoch, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="unseen task test")
                
            #return 0


            for i, batch in tqdm(enumerate(train_data_loader), desc=f"epoch {epoch}"):

                knowledge_batch = batch["knowledge"]
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                optimizer.zero_grad()

                train_noise = False

                if args.prior == "gaussian":
                    kl_loss, recon_loss, task_loss = nesy.forward_batch(knowledge_batch, x_batch, y_batch)
                    loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss
                elif args.prior == "mog":
                    kl_loss, recon_loss, task_loss, entropy_loss = nesy.forward_batch(knowledge_batch, x_batch, y_batch)
                    loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss #+ args.entropy_loss_weight * entropy_loss
                elif args.prior in ["gmg", "vaeflow"]:
                    
                    if nesy.args.ebm_optim_method == "fce":
                
                        kl_loss, recon_loss, task_loss, flow_loss, noise_loss, acc = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss
                    
                        train_noise = acc > args.threshold
                        train_log.writelines(f"acc={acc}\n")
                        train_log.writelines(f"train_noise={train_noise}\n")

                    elif nesy.args.ebm_optim_method in ["entropy", "kl"]:
                
                        kl_loss, recon_loss, task_loss, flow_loss, entropy = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss - args.entropy_loss_weight * entropy
                    
                        train_log.writelines(f"entropy={entropy}\n")

                    else:
                        kl_loss, recon_loss, task_loss, flow_loss = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss
                        
                if train_noise:

                    loss = noise_loss
                    loss.backward()
                    optimizer_noise.step()
                    if i % 10 == 0:
                        train_log.writelines(f"noise_loss={loss}\n")
                        train_log.flush()

                else:
                    loss.backward()
                    optimizer.step()

                    if i % 10 == 0:
                        train_log.writelines(f"epoch {epoch} step {i} \n")
                        if args.prior == "gaussian":
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, task_loss={task_loss}\n")
                        elif args.prior == "mog":
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, task_loss={task_loss}, entropy_loss={entropy_loss}\n")
                        elif args.prior in ["gmg", "vaeflow"]:
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, flow_loss={flow_loss}\n")
                            train_log.writelines(f"task_loss={task_loss}\n")
                        train_log.flush()
                    
                if i % 100 == 0:
                    info = get_gpu_memory_usage()
                    train_log.writelines(f"{info}\n")
                    train_log.flush()
    else:
        symbolic_task_test_log = open(f"{args.exp_dir}/symbolic_task.log", "w")
        test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, symbolic_evaluater, symbolic_task_test_log, method=args.method)
        # neural_task_test_log = open(f"{args.exp_dir}/neural_task.log", "w")
        # test_neural_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, neural_task_test_log, method=args.method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="sni", help='name of dataset.')
    parser.add_argument('--meta_exp_dir', type=str, default="./exp", help='name of dataset.')
    parser.add_argument('--exp_name', type=str, default="debug", help='name of dataset.')
    parser.add_argument('--pretraining', action="store_true", default=True, help='Whether to pretrain the model.')

    parser.add_argument('--method', type=str, default="nesy", help='name of dataset.')
    parser.add_argument('--prior', type=str, default="gaussian", help='name of dataset.')
    # parser.add_argument('--fuse_method', type=str, default="delta", help='name of dataset.')
    parser.add_argument('--fuse_method', type=str, default="p-tuning", help='name of dataset.')

    parser.add_argument('--ebm_optim_method', type=str, default="entropy", help='name of dataset.')
    #parser.add_argument('--ebm_optim_method', type=str, default="nce", help='name of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='input batchsize.')
    parser.add_argument('--threshold', type=float, default=0.8, help='input batchsize.')

    parser.add_argument('--batch_size', type=int, default=4, help='input batchsize.')
    parser.add_argument('--latent_size', type=int, default=1000, help='input batchsize.')
    parser.add_argument('--selected_layers', type=int, default=2, help='input batchsize.')
    parser.add_argument('--num_latent_samples', type=int, default=2, help='input batchsize.')
    parser.add_argument('--num_peak', type=int, default=100, help='input batchsize.')
    parser.add_argument('--lr', type=float, default=1e-4, help='input batchsize.')
    parser.add_argument('--episilon', type=float, default=1e-5, help='input batchsize.')
    parser.add_argument('--num_epochs', type=int, default=100, help='input batchsize.')

    parser.add_argument('--task_finetune_step', type=int, default=20, help='input batchsize.')
    parser.add_argument('--task_finetune_lr', type=float, default=1e-2, help='input batchsize.')
    parser.add_argument('--zero_init', action="store_true", default=False, help='input batchsize.')

    parser.add_argument('--alignment_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--task_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--entropy_loss_weight', type=float, default=1e-5, help='input batchsize.')
    parser.add_argument('--kl_loss_weight', type=float, default=0.01, help='input batchsize.')
    parser.add_argument('--recon_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--flow_loss_weight', type=float, default=10, help='input batchsize.')
    
    parser.add_argument('--max_token', type=int, default=50, help='max number of tokens to generate.')
    parser.add_argument('--num_soft_token', type=int, default=2, help='max number of tokens to generate.')
    
    #parser.add_argument('--load_exp', type=str, default="gmvae-list-10peak-entropy-4", help='name of dataset.')
    parser.add_argument('--load_exp', type=str, default=None, help='name of dataset.')
    parser.add_argument('--load_epoch', type=int, default=20, help='input batchsize.')
    parser.add_argument('--ignore_exist', action="store_true", default=False, help='whether show results')
    parser.add_argument('--results_name', type=str, default=None, help='keywords must include in results')
    parser.add_argument('--model_name_or_path', type=str, default="/netcache/huggingface/llama-2-7b-chat-hf", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='finetuned model path')
    
    parser.add_argument('--encoder_device', type=int, default=0, help='device to use')
    parser.add_argument('--decoder_device', type=int, default=2, help='device to use')
    parser.add_argument('--task_device', type=int, default=1, help='device to use')
    parser.add_argument('--flow_device', type=int, default=2, help='device to use')
    parser.add_argument('--noise_device', type=int, default=4, help='device to use')
    parser.add_argument('--backward_device', type=int, default=0, help='device to use')
    
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj", help='keywords must include in results')
        
    parser.add_argument('--num_words', type=int, default=32)
    parser.add_argument('--unseen_task_ratio', type=float, default=0.1)
    parser.add_argument('--num_pertask', type=int, default=1000)
    parser.add_argument('--task_fields', type=str, default=None)


    args = parser.parse_args()
    main(args)