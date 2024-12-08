import os
import time
import shutil
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
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
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch) * args.task_loss_weight + args.reg_loss_weight * params.norm(1, dim=1).mean() / args.latent_size
            task_loss.backward()
            optimizer.step()
    
    return params, test_loss_ls

def train_subtask_indirect(args, nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data):

    x_id = nesy.llm.tokenizer("Follow the instruction and answer the question: I do not know anything.", return_tensors="pt", add_special_tokens=True).input_ids.to(nesy.args.encoder_device)
    input_embeds = torch.nn.Parameter(nesy.llm.encoder_model.model.embed_tokens(x_id))#.repeat(embedding.shape[0], 1, 1)

    optimizer = torch.optim.Adam([input_embeds], lr=args.task_finetune_lr)
    keep_training = True
    test_loss_ls = []
    
    while keep_training:

        for i, batch in tqdm(enumerate(subtask_train_data_loader)):

            if i % 100 == 0:
                test_loss = 0
                with torch.no_grad():
                    for batch in subtask_test_data_loader:
                        x_batch = batch["input"]
                        x_batch = [prompt_template.format(x) for x in x_batch]
                        y_batch = batch["target"]
                        params, _ = nesy.encode(input_embeds)
                        params = params.to(nesy.args.task_device)
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
            params, _ = nesy.encode(input_embeds)
            params = params.to(nesy.args.task_device)
            expanded_params = params.repeat_interleave(len(x_batch), dim=0)
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch) #* args.task_loss_weight + args.reg_loss_weight * params.norm(1, dim=1).mean() / args.latent_size
            task_loss.backward()
            optimizer.step()
    
    params, _ = nesy.encode(input_embeds)
    params = params.to(nesy.args.task_device)
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
        
        # if num_batches >= 5:  # 如果已经处理了10个batch，跳出循环
        #     break
        
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
                
        #num_batches += 1
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
    
    #all_tasks_ids = random.sample(all_tasks_ids, 10)
    
    num_correct_symbolic = 0
    num_test_symbolic = 0
    
    for task_id in all_tasks_ids:

        subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]
        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_test_data[0]["knowledge"]
        num_samples = 1

        knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
        #encoded_latent = [nesy.reparameterize(*nesy.encode(knowledge_ids)) for i in range(num_samples)]
        #randomn_latent = [torch.randn([1, nesy.args.latent_size]) for i in range(num_samples)]
        trained_latents = []
    
        for i in range(num_samples):
            
            trained_params, test_loss_ls = train_subtask_indirect(args, nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data)

            with torch.no_grad():

                if args.method == "vaeflow":
                    trained_latent = trained_params.to(nesy.args.flow_device)
                    trained_latent = nesy.flow_backward(trained_params).to(nesy.args.decoder_device)
                else:
                    if args.nf:
                        trained_latent = nesy.flow_backward(trained_params.to(nesy.args.flow_device)).to(nesy.args.decoder_device)
                    else:
                        trained_latent = trained_params.to(nesy.args.decoder_device)

                if nesy.args.use_instance_in_decoder:
                    batch = random.choice(subtask_train_data_loader.dataset)
                    x = batch["input"]
                    y = batch["target"]
                    instance_text = f"input: {x}, target: {y}. This task is to:"
                    print(instance_text)
                    instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
                else:
                    instance_ids = None
                    
                predicted_knowledge = nesy.sample(trained_latent, sample_from_guassian=False, instance=instance_ids)
                #encoded_params = encoded_latent[i].to(nesy.args.decoder_device)
                #encode_decode_knowledge = nesy.sample(encoded_params, sample_from_guassian=False)

            log.writelines(f"prediction on {name} subtask {task_id}: \n")
            log.writelines(f"subtask train loss: {str(test_loss_ls)} \n")
            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)
            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            # result = nesy.eval_knowledge(knowledge, encode_decode_knowledge, evaluater)
            # log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_test_symbolic += 1
            log.flush()

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

#     knowledge_prompt = """
# Instruction: {}
# Input: {}
# Output:
# """


    if method == "finetuning":

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
                            y_batch = batch["target"]
                            input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
                            input_batch = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
                            expanded_params = params.repeat_interleave(len(input_batch), dim=0)
                            test_loss += nesy.compute_task_loss(expanded_params, input_batch, y_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step*3:
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                knowledge_batch = batch["knowledge"]
                batch_size = len(knowledge_batch)
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]
                input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
                input_batch = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
                expanded_params = params.repeat_interleave(len(input_batch), dim=0)
                task_loss = nesy.compute_task_loss(expanded_params, input_batch, y_batch)
                task_loss.backward()
                optimizer.step()


    # start testing neural task
    with torch.no_grad():

        for batch in seen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]

            input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            input_text = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)

            # input_batch = [knowledge_prompt.format(knowledge_batch[i], x_batch[i]) for i in range(batch_size)]
            # input_ids = nesy.llm.tokenizer(input_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)
            
            if method == "prompting":
                y_pred = nesy.llm.predict_task(input_ids)
            elif method == "finetuning":
                if args.fuse_method == "delta":
                    new_task_parameters = nesy.llm.allocate(params)
                    y_pred = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(len(input_text), dim=0)
                    y_pred = nesy.llm.predict_task(input_ids, expanded_params)

            y_pred = [y.split("\n")[0] for y in y_pred]

            results = [{
                "knowledge": knowledge_batch[i],
                "x": x_batch[i],
                "y_true": y_batch[i],
                "y_pred": y_pred[i],
                "score": evaluater(y_pred[i], y_batch[i])
                } for i in range(len(x_batch))]

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
            y_batch = batch["target"]

            input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            input_text = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)
            
            if method == "prompting":
                y_pred = nesy.llm.predict_task(input_ids)
            elif method == "finetuning":
                if args.fuse_method == "delta":
                    new_task_parameters = nesy.llm.allocate(params)
                    y_pred = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(len(input_text), dim=0)
                    y_pred = nesy.llm.predict_task(input_ids, expanded_params)

            y_pred = [y.split("\n")[0] for y in y_pred]

            results = [{
                "knowledge": knowledge_batch[i],
                "x": x_batch[i],
                "y_true": y_batch[i],
                "y_pred": y_pred[i],
                "score": evaluater(y_pred[i], y_batch[i])
                } for i in range(len(x_batch))]

            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_neural += result["score"]
                num_test_neural += 1
                log.flush()

    accuracy = num_correct_neural / num_test_neural
    log.writelines(f"neural unseen task accuracy of method {method}: {accuracy} \n")
    log.flush()

def test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, evaluater, log, method):

    log.writelines(f"symbolic task testing for method: {method} \n")
    log.flush()

#     prompt = """
# I gave a friend an instruction and an input. 
# The friend read the instruction and wrote an output for the input.
# {}
# So my instruction is: 
# """

    sys_prompt = "Given the following input and output pairs, please infer the instruction."

    if method == "finetuning":

        seen_train_data = seen_train_data_loader.dataset
        seen_test_data = seen_test_data_loader.dataset
        seen_tasks_ids = list(set([sample["sub_task_id"] for sample in seen_train_data]))
        seen_train_data_induction = []
        seen_test_data_induction = []

        for task_id in seen_tasks_ids:

            seen_subtask_train_data = [data for data in seen_train_data if data["sub_task_id"] == task_id]
            seen_subtask_test_data = [data for data in seen_test_data if data["sub_task_id"] == task_id]
            knowledge = seen_subtask_train_data[0]["knowledge"]

            for _ in range(10):
                io_sample_train = random.sample(seen_subtask_train_data, 5)
                io_text_train = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in io_sample_train])
                seen_train_data_induction.append({
                    "knowledge": knowledge,
                    "io_text": io_text_train
                    })
            io_sample_test = random.sample(seen_subtask_test_data, 5)
            io_text_test = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in io_sample_test])
            seen_test_data_induction.append({
                "knowledge": knowledge,
                "io_text": io_text_test
                })
        
        seen_task_train_data_loader = DataLoader(seen_train_data_induction, batch_size=args.batch_size//4, shuffle=True)
        seen_task_test_data_loader = DataLoader(seen_test_data_induction, batch_size=args.batch_size//4, shuffle=True)

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
                            #io_batch = [prompt.format(batch["io_text"][i]) for i in range(batch_size)]
                            io_message = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": batch["io_text"][i]}] for i in range(batch_size)]
                            io_batch = [nesy.llm.tokenizer.apply_chat_template(io_message[i], tokenize=False) for i in range(batch_size)]
                            expanded_params = params.repeat_interleave(len(io_batch), dim=0)
                            test_loss += nesy.compute_task_loss(expanded_params, io_batch, knowledge_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step*3:
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                knowledge_batch = batch["knowledge"]
                batch_size = len(knowledge_batch)
                io_message = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": batch["io_text"][i]}] for i in range(batch_size)]
                io_batch = [nesy.llm.tokenizer.apply_chat_template(io_message[i], tokenize=False) for i in range(batch_size)]
                expanded_params = params.repeat_interleave(len(io_batch), dim=0)
                task_loss = nesy.compute_task_loss(expanded_params, io_batch, knowledge_batch)
                task_loss.backward()
                optimizer.step()

    seen_test_data = seen_test_data_loader.dataset
    seen_tasks_ids = list(set([sample["sub_task_id"] for sample in seen_test_data]))

    num_correct_symbolic = 0
    num_test_symbolic = 0

    for task_id in seen_tasks_ids:

        seen_subtask_data = [data for data in seen_test_data if data["sub_task_id"] == task_id]
        knowledge = seen_subtask_data[0]["knowledge"]

        # start testing symbolic task
        with torch.no_grad():
            
            obeserved_samples = random.sample(seen_subtask_data, 5)
            obeserved_text = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in obeserved_samples])
            #obeserved_text = "\n".join([f"The input is {data['input']}. The friend's output is {data['target']}." for data in obeserved_samples])
            #induction_questions = prompt.format(obeserved_text)

            input_message = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obeserved_text}]
            input_text = nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False)
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt").input_ids.to(nesy.args.task_device)

            if method == "prompting":
                predicted_knowledge = nesy.llm.predict_task(input_ids)
            elif method == "finetuning":
                if args.fuse_method == "delta":
                    new_task_parameters = nesy.llm.allocate(params)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(input_ids.shape[0], dim=0)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, expanded_params)

            #predicted_knowledge = predicted_knowledge[0].split("\n")[0]

            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)

            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            num_test_symbolic += 1
            log.flush()

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"symbolic seen task accuracy of method {method}: {accuracy} \n")
    log.flush()


    unseen_test_data = unseen_test_data_loader.dataset
    unseen_tasks_ids = list(set([sample["sub_task_id"] for sample in unseen_test_data]))

    num_correct_symbolic = 0
    num_test_symbolic = 0

    for task_id in unseen_tasks_ids:

        unseen_subtask_data = [data for data in unseen_test_data if data["sub_task_id"] == task_id]
        knowledge = unseen_subtask_data[0]["knowledge"]

        # start testing symbolic task
        with torch.no_grad():
            
            obeserved_samples = random.sample(seen_subtask_data, 5)
            obeserved_text = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in obeserved_samples])
            #obeserved_text = "\n".join([f"The input is {data['input']}. The friend's output is {data['target']}." for data in obeserved_samples])
            #induction_questions = prompt.format(obeserved_text)

            input_message = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obeserved_text}]
            input_text = nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False)
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt").input_ids.to(nesy.args.task_device)

            if method == "prompting":
                predicted_knowledge = nesy.llm.predict_task(input_ids)
            elif method == "finetuning":
                if args.fuse_method == "delta":
                    new_task_parameters = nesy.llm.allocate(params)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(input_ids.shape[0], dim=0)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, expanded_params)

            #predicted_knowledge = predicted_knowledge[0].split("\n")[0]

            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)

            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            num_test_symbolic += 1
            log.flush()

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"symbolic unseen task accuracy of method {method}: {accuracy} \n")
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
                           "zero_init", "dataset", "pretraining", "valid_epoch", "save_epoch"]:
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
        print(f"latent_size now is: {args.latent_size}")

    args_dict = vars(args)
    output_file = f"{args.exp_dir}/args.json"
    with open(output_file, "w") as f:
        json.dump(args_dict, f, indent=4)
        f.flush()

    if args.pretraining:
        train_dataset, valid_dataset = load_pretrain_data_hf()
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
        print("pretraining")

    data = load_task_data(task=args.dataset, unseen_task_ratio=args.unseen_task_ratio, test_sample_num=args.test_sample_num, num_words=args.num_words, num_pertask=args.num_pertask, task_fields=args.task_fields)
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
        # try:
        #     shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/neural2symbolic.log", f"{args.exp_dir}/neural2symbolic.log")
        #     shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/symbolic2neural.log", f"{args.exp_dir}/symbolic2neural.log")
        # except:
        #     pass
        # neural2symbolic_valid_log = open(f"{args.exp_dir}/neural2symbolic.log", file_mode)
        # symbolic2neural_valid_log = open(f"{args.exp_dir}/symbolic2neural.log", file_mode)
        
        train_data_loader = seen_train_data_loader if not args.pretraining else train_data_loader

        for epoch in range(start_epoch, args.num_epochs):

            if epoch % args.save_epoch == 0 and epoch > 0:
                
                nesy.save(f"{args.exp_dir}/epoch{epoch}/nesy_ckpt/")

            if epoch % args.valid_epoch == 0 and epoch > 0:

                neural2symbolic_valid_log = open(f"{args.exp_dir}/epoch{epoch}/neural2symbolic.log", file_mode)
                symbolic2neural_valid_log = open(f"{args.exp_dir}/epoch{epoch}/symbolic2neural.log", file_mode)

                valid_neural2symbolic(args, epoch, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="seen task")
                valid_neural2symbolic(args, epoch, data["unseen_tasks"]["train"], data["unseen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="unseen task")

                valid_symbolic2neural(args, epoch, seen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="seen task test")
                valid_symbolic2neural(args, epoch, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="unseen task test")

            for i, batch in tqdm(enumerate(train_data_loader), desc=f"epoch {epoch}"):

                knowledge_batch = batch["knowledge"]
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                optimizer.zero_grad()

                train_noise = False

                if args.prior == "gaussian":
                    reg_loss, recon_loss, task_loss = nesy.forward(knowledge_batch, x_batch, y_batch)
                    loss = args.reg_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss
                elif args.prior == "mog":
                    reg_loss, recon_loss, task_loss, entropy_loss = nesy.forward_batch(knowledge_batch, x_batch, y_batch)
                    loss = args.reg_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss #+ args.entropy_loss_weight * entropy_loss
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
                        reg_loss, recon_loss, task_loss, flow_loss = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss
                        
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
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, reg_loss={reg_loss}, task_loss={task_loss}\n")
                        elif args.prior == "mog":
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, reg_loss={reg_loss}, task_loss={task_loss}, entropy_loss={entropy_loss}\n")
                        elif args.prior in ["gmg", "vaeflow"]:
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, flow_loss={flow_loss}\n")
                            train_log.writelines(f"task_loss={task_loss}\n")
                        train_log.flush()
                    
                if i % 100 == 0:
                    info = get_gpu_memory_usage()
                    train_log.writelines(f"{info}\n")
                    train_log.flush()
    
    else:
        #symbolic_task_test_log = open(f"{args.exp_dir}/symbolic_task.log", "w")
        #test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, symbolic_evaluater, symbolic_task_test_log, method=args.method)
        neural_task_test_log = open(f"{args.exp_dir}/neural_task.log", "w")
        test_neural_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, neural_task_test_log, method=args.method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="sni", help='name of dataset.')
    parser.add_argument('--meta_exp_dir', type=str, default="./exp_new", help='the directory to save all the experiment results.')
    parser.add_argument('--exp_name', type=str, default="debug", help='the name of the experiment.')
    parser.add_argument('--pretraining', action="store_true", default=False, help='Whether to pretrain the model.')

    parser.add_argument('--method', type=str, default="finetuning", help='the method to train the model.')
    parser.add_argument('--prior', type=str, default="gaussian", help='the prior distribution of the model.')
    parser.add_argument('--nf', action="store_true", default=False, help='Whether to use the flow model.')
    # parser.add_argument('--fuse_method', type=str, default="delta", help='name of dataset.')
    parser.add_argument('--fuse_method', type=str, default="p-tuning", help='the method to fuse the task model and the prior model.')
    parser.add_argument('--use_instance_in_decoder', action="store_true", default=False, help='whether to use the instance in the decoder.')
    parser.add_argument('--use_trainable_task_model', action="store_true", default=False, help='whether to use the trainable task model.')
    parser.add_argument('--use_chat_template', action="store_true", default=False, help='whether to use the chat template.')

    parser.add_argument('--ebm_optim_method', type=str, default="entropy", help='the method to optimize the energy-based model.')
    #parser.add_argument('--ebm_optim_method', type=str, default="nce", help='name of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='the beta parameter in the energy-based model.')
    parser.add_argument('--threshold', type=float, default=0.8, help='the threshold for the accuracy of the model.')

    parser.add_argument('--batch_size', type=int, default=4, help='the batch size.')
    parser.add_argument('--latent_size', type=int, default=1000, help='the dimension of the latent variable.')
    parser.add_argument('--selected_layers', type=int, default=2, help='the number of layers to be selected.')
    parser.add_argument('--num_latent_samples', type=int, default=2, help='the number of samples to be generated.')
    parser.add_argument('--num_peak', type=int, default=100, help='the number of peaks in the mixture of gaussians.')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate.')
    parser.add_argument('--episilon', type=float, default=1e-5, help='the episilon parameter in the energy-based model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs to train the model.')
    parser.add_argument('--valid_epoch', type=int, default=5, help='the number of epochs to validate the model.')
    parser.add_argument('--save_epoch', type=int, default=1, help='the number of epochs to save the model.')

    parser.add_argument('--task_finetune_step', type=int, default=100, help='the number of steps to finetune the task model.')
    parser.add_argument('--task_finetune_lr', type=float, default=1e-2, help='the learning rate to finetune the task model.')
    parser.add_argument('--zero_init', action="store_true", default=False, help='whether to initialize the task model parameters to zero.')

    parser.add_argument('--alignment_loss_weight', type=float, default=1, help='the weight of the alignment loss.')
    parser.add_argument('--task_loss_weight', type=float, default=1, help='the weight of the task loss.')
    parser.add_argument('--entropy_loss_weight', type=float, default=1e-5, help='the weight of the entropy loss.')
    parser.add_argument('--reg_loss_weight', type=float, default=0.01, help='the weight of the regularization loss.')
    parser.add_argument('--recon_loss_weight', type=float, default=1, help='the weight of the reconstruction loss.')
    parser.add_argument('--flow_loss_weight', type=float, default=10, help='the weight of the flow loss.')
    
    parser.add_argument('--max_token', type=int, default=50, help='max number of tokens to generate.')
    parser.add_argument('--num_soft_token', type=int, default=2, help='max number of tokens to generate.')
    
    #parser.add_argument('--load_exp', type=str, default="vae-pretrain-small-decoder", help='name of dataset.')
    parser.add_argument('--load_exp', type=str, default=None, help='the path of the pretrained model.')
    parser.add_argument('--load_epoch', type=int, default=5, help='the epoch of the pretrained model.')
    parser.add_argument('--ignore_exist', action="store_true", default=False, help='whether to ignore the existing model.')
    parser.add_argument('--results_name', type=str, default=None, help='the name of the experiment.')
    #parser.add_argument('--model_name_or_path', type=str, default="/netcache/huggingface/llama-2-7b-chat-hf", help='Tasks for instructions generation')
    parser.add_argument('--model_name_or_path', type=str, default="/mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf", help='the path of the pretrained model.')
    parser.add_argument('--finetuned_model', type=str, default=None, help='the path of the finetuned model.')
    
    parser.add_argument('--cuda_devices', type=str, default="3,4,5", help='the devices to use')
    parser.add_argument('--encoder_device', type=int, default=0, help='the device to use')
    parser.add_argument('--decoder_device', type=int, default=1, help='the device to use')
    parser.add_argument('--task_device', type=int, default=2, help='the device to use')
    parser.add_argument('--flow_device', type=int, default=0, help='the device to use')
    parser.add_argument('--noise_device', type=int, default=4, help='device to use')
    parser.add_argument('--backward_device', type=int, default=0, help='device to use')
    
    parser.add_argument('--encoder_lora_r', type=int, default=16)
    parser.add_argument('--decoder_lora_r', type=int, default=1)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj", help='keywords must include in results')
        
    parser.add_argument('--num_words', type=int, default=32)
    parser.add_argument('--unseen_task_ratio', type=float, default=0.1)
    parser.add_argument('--test_sample_num', type=int, default=5)
    parser.add_argument('--num_pertask', type=int, default=27)
    parser.add_argument('--task_fields', type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    main(args)