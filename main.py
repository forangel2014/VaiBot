import os
import time
import shutil
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import argparse
import random
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from utils import mkdir, convert_seconds, load_task_data, plot_loss_curve, tsne, create_task_data_lookup, get_gpu_memory_usage
from tqdm import tqdm
random.seed(73)
torch.manual_seed(73)

def train_subtask(nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data):
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
                        test_loss += nesy.compute_task_loss(params, x_batch, y_batch)

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
            task_loss = nesy.compute_task_loss(params, x_batch, y_batch)
            task_loss.backward()
            optimizer.step()
    
    return params, test_loss_ls

def pretrain_subtask(args, train_data, test_data, nesy, prompt_template, log):
    
    all_tasks_ids = list(set([sample["sub_task_id"] for sample in test_data]))
    pretrained_params = []

    for task_id in all_tasks_ids:
        
        log.writelines(f"training subtask {task_id}\n")

        subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]
        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_test_data[0]["knowledge"]
        num_samples = 10
        
        optimal_params = []

        for i in range(num_samples):
            
            params, test_loss_ls = train_subtask(nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data)
            
            log.writelines(f"subtask train loss: {str(test_loss_ls)} \n")
            log.flush()
            
            optimal_params.append(params.detach().cpu())
            
        pretrained_params.append({
            "task_id": task_id,
            "optimal_params": optimal_params
        })

    torch.save(pretrained_params, f"{args.exp_dir}/pretrained_params.pth")
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
    
    all_tasks_ids = random.sample(all_tasks_ids, 1)
    
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
        num_samples = 5

        knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
        encoded_latent = [nesy.reparameterize(*nesy.encode(knowledge_ids)) for i in range(num_samples)]
        randomn_latent = [torch.randn([1, nesy.args.latent_size]) for i in range(num_samples)]
        trained_latent = []
        
                    
        for i in range(num_samples):
            
            params, test_loss_ls = train_subtask(nesy, subtask_train_data_loader, subtask_test_data_loader, prompt_template, subtask_test_data)

            # train_loss = 0
            # with torch.no_grad():
            #     for batch in subtask_train_data_loader:
            #         x_batch = batch["input"]
            #         x_batch = [prompt_template.format(x) for x in x_batch]
            #         y_batch = batch["target"]              
            #         train_loss += nesy.compute_task_loss(params, x_batch, y_batch)
            #     train_loss /= len(subtask_train_data)
            # log.writelines(f"train_loss = {train_loss}\n")
            # log.writelines(f"test_loss = {test_loss}\n")

            # encoded_params = encoded_latent[i].to(nesy.args.task_device)
            # encode_knowledge_train_loss = 0
            # with torch.no_grad():
            #     for batch in subtask_train_data_loader:
            #         x_batch = batch["input"]
            #         x_batch = [prompt_template.format(x) for x in x_batch]
            #         y_batch = batch["target"]              
            #         encode_knowledge_train_loss += nesy.compute_task_loss(encoded_params, x_batch, y_batch)
            #     encode_knowledge_train_loss /= len(subtask_train_data)
            # log.writelines(f"encode_knowledge_train_loss = {encode_knowledge_train_loss}\n")

            # encode_knowledge_test_loss = 0
            # with torch.no_grad():
            #     for batch in subtask_test_data_loader:
            #         x_batch = batch["input"]
            #         x_batch = [prompt_template.format(x) for x in x_batch]
            #         y_batch = batch["target"]              
            #         encode_knowledge_test_loss += nesy.compute_task_loss(encoded_params, x_batch, y_batch)
            #     encode_knowledge_test_loss /= len(subtask_test_data)

            # log.writelines(f"encode_knowledge_test_loss = {encode_knowledge_test_loss}\n")

            # for batch in subtask_test_data_loader:
            #     with torch.no_grad():
            #         knowledge_batch = batch["knowledge"]
            #         x_batch = batch["input"]
            #         x_batch = [prompt_template.format(x) for x in x_batch]
            #         y_batch = batch["target"]
            #         batch_size = len(knowledge_batch)
            #         results = []
            #         for i in range(batch_size):
            #             new_task_parameters = nesy.llm.allocate(params[0])
            #             x_id = nesy.llm.tokenizer(x_batch[i], return_tensors="pt").input_ids.to(nesy.args.task_device)
            #             y_pred = nesy.llm.predict_task(x_id, new_task_parameters)
            #             results.append({
            #                 "knowledge": knowledge_batch[i],
            #                 "x": x_batch[i],
            #                 "y_true": y_batch[i],
            #                 "y_pred": y_pred,
            #                 "score": evaluater(y_pred, y_batch[i])
            #                 })
            #         for result in results:
            #             num_correct_neural += result["score"]
            #             num_test_neural += 1
                        
            trained_latent.append(params)

            with torch.no_grad():
                params = params.to(nesy.args.flow_device)
                params = nesy.flow_backward(params).to(nesy.args.decoder_device)
                predicted_knowledge = nesy.sample(params, sample_from_guassian=False)
                
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

        tsne(encoded_latent, trained_latent, randomn_latent, filename=f"{args.exp_dir}/epoch{epoch}/tsne/task{task_id}.pdf")
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
        args.load_nesy_ckpt = f"{args.load_exp}/epoch{args.load_epoch}/nesy_ckpt/"
        start_epoch = args.load_epoch
        file_mode = "w"
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

    data = load_task_data(task=args.dataset, unseen_task_ratio=args.unseen_task_ratio, test_sample_num=1, num_words=args.num_words, num_pertask=args.num_pertask, task_fields=args.task_fields)
    args.task_id2knowledge, args.knowledge2task_id = create_task_data_lookup(data)
    prompt_template = data["prompt_template"]
    neural_evaluater = data["neural_evaluater"]
    symbolic_evaluater = data["symbolic_evaluater"]

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

    seen_train_data_loader = DataLoader(data["seen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    seen_test_data_loader = DataLoader(data["seen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)
    unseen_train_data_loader = DataLoader(data["unseen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    unseen_test_data_loader = DataLoader(data["unseen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)

    if args.method == "nesy-pretrain":
        
        pretrain_log = open(f"{args.exp_dir}/pretrain.log", "w")

        pretrain_subtask(args, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, pretrain_log)

    elif args.method == "nesy":
        optimizer = torch.optim.Adam([
            {'params': nesy.llm.encoder.parameters(), 'lr': args.lr},
            {'params': nesy.encoder_mlp.parameters(), 'lr': args.lr},
            {'params': nesy.llm.decoder.parameters(), 'lr': args.lr},
            {'params': nesy.decoder_mlp.parameters(), 'lr': args.lr},
            {'params': nesy.flow_net.parameters(), 'lr': args.lr},
            {'params': nesy.logZ, 'lr': args.lr}
                                    ], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
        train_log = open(f"{args.exp_dir}/train.log", "w")
        train_log.writelines(f"seen_tasks: {data['seen_task_num']}, unseen_tasks: {data['task_num'] - data['seen_task_num']}\n")
        train_log.writelines(f"seen_tasks train number: {len(data['seen_tasks']['train'])}\n")
        #train_log.writelines(data['seen_tasks']['train'][0])
        train_log.flush()
        try:
            shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/neural2symbolic.log", f"{args.exp_dir}/neural2symbolic.log")
            shutil.copy2(f"{args.meta_exp_dir}/epoch0validation/symbolic2neural.log", f"{args.exp_dir}/symbolic2neural.log")
        except:
            pass
        neural2symbolic_valid_log = open(f"{args.exp_dir}/neural2symbolic.log", file_mode)
        symbolic2neural_valid_log = open(f"{args.exp_dir}/symbolic2neural.log", file_mode)
        
        for epoch in range(start_epoch, args.num_epochs):

            if epoch % 20 == 0 and epoch > 0:

                nesy.save(f"{args.exp_dir}/epoch{epoch}/nesy_ckpt/")

                valid_neural2symbolic(args, epoch, data["seen_tasks"]["train"], data["seen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="seen task")
                valid_neural2symbolic(args, epoch, data["unseen_tasks"]["train"], data["unseen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_valid_log, name="unseen task")

                valid_symbolic2neural(args, epoch, seen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="seen task test")
                valid_symbolic2neural(args, epoch, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_valid_log, name="unseen task test")
            
            #return 0

            for i, batch in tqdm(enumerate(seen_train_data_loader), desc=f"epoch {epoch}"):

                knowledge_batch = batch["knowledge"]
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                optimizer.zero_grad()
                
                if args.prior == "gaussian":
                    kl_loss, recon_loss, task_loss, alignment_loss, reference_task_loss = nesy(knowledge_batch, x_batch, y_batch)
                    loss = args.alignment_loss_weight * alignment_loss + args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss
                elif args.prior == "mog":
                    kl_loss, recon_loss, task_loss, entropy_loss = nesy.forward_batch(knowledge_batch, x_batch, y_batch)
                    loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss + args.entropy_loss_weight * entropy_loss
                elif args.prior in ["gmg", "vaeflow"]:
                    kl_loss, recon_loss, task_loss, flow_loss = nesy(knowledge_batch, x_batch, y_batch)
                    loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss

                loss.backward()
                
                optimizer.step()

                if i % 10 == 0:
                    train_log.writelines(f"epoch {epoch} step {i} \n")
                    if args.prior == "gaussian":
                        train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, task_loss={task_loss}, alignment_loss={alignment_loss}, reference_task_loss={reference_task_loss}\n")
                    elif args.prior == "mog":
                        train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, task_loss={task_loss}, entropy_loss={entropy_loss}\n")
                    elif args.prior in ["gmg", "vaeflow"]:
                        train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, flow_loss={flow_loss}\n")
                        train_log.writelines(f"task_loss={task_loss}\n")
                    train_log.flush()
                    
                if i % 100 == 0:
                    get_gpu_memory_usage()
    else:
        symbolic_task_test_log = open(f"{args.exp_dir}/symbolic_task.log", "w")
        test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, symbolic_evaluater, symbolic_task_test_log, method=args.method)
        # neural_task_test_log = open(f"{args.exp_dir}/neural_task.log", "w")
        # test_neural_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, neural_task_test_log, method=args.method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="list_functions", help='name of dataset.')
    parser.add_argument('--meta_exp_dir', type=str, default="./exp", help='name of dataset.')
    parser.add_argument('--exp_name', type=str, default="debug", help='name of dataset.')
    
    parser.add_argument('--method', type=str, default="nesy", help='name of dataset.')
    parser.add_argument('--prior', type=str, default="vaeflow", help='name of dataset.')
    parser.add_argument('--fuse_method', type=str, default="delta", help='name of dataset.')
    #parser.add_argument('--ebm_optim_method', type=str, default="drop-z", help='name of dataset.')
    parser.add_argument('--ebm_optim_method', type=str, default="nce", help='name of dataset.')

    parser.add_argument('--batch_size', type=int, default=2, help='input batchsize.')
    parser.add_argument('--latent_size', type=int, default=1000, help='input batchsize.')
    parser.add_argument('--selected_layers', type=int, default=2, help='input batchsize.')
    parser.add_argument('--num_latent_samples', type=int, default=2, help='input batchsize.')
    parser.add_argument('--num_peak', type=int, default=10, help='input batchsize.')
    parser.add_argument('--task_finetune_step', type=int, default=200, help='input batchsize.')
    parser.add_argument('--task_finetune_lr', type=float, default=1e-2, help='input batchsize.')
    parser.add_argument('--lr', type=float, default=1e-4, help='input batchsize.')
    parser.add_argument('--beta', type=float, default=1, help='input batchsize.')
    parser.add_argument('--episilon', type=float, default=1e-5, help='input batchsize.')
    parser.add_argument('--num_epochs', type=int, default=100, help='input batchsize.')
    
    parser.add_argument('--alignment_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--task_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.1, help='input batchsize.')
    parser.add_argument('--kl_loss_weight', type=float, default=0.01, help='input batchsize.')
    parser.add_argument('--recon_loss_weight', type=float, default=1, help='input batchsize.')
    parser.add_argument('--flow_loss_weight', type=float, default=1, help='input batchsize.')
    
    parser.add_argument('--max_token', type=int, default=50, help='max number of tokens to generate.')
    parser.add_argument('--num_soft_token', type=int, default=5, help='max number of tokens to generate.')
    
    parser.add_argument('--load_exp', type=str, default=None, help='name of dataset.')
    parser.add_argument('--load_epoch', type=int, default=18, help='input batchsize.')
    parser.add_argument('--ignore_exist', action="store_true", default=False, help='whether show results')
    parser.add_argument('--results_name', type=str, default=None, help='keywords must include in results')
    parser.add_argument('--model_name_or_path', type=str, default="/netcache/huggingface/llama-2-7b-chat-hf", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='finetuned model path')
    
    parser.add_argument('--encoder_device', type=int, default=0, help='device to use')
    parser.add_argument('--decoder_device', type=int, default=0, help='device to use')
    parser.add_argument('--task_device', type=int, default=1, help='device to use')
    parser.add_argument('--flow_device', type=int, default=2, help='device to use')
    parser.add_argument('--backward_device', type=int, default=0, help='device to use')
    
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj", help='keywords must include in results')
        
    parser.add_argument('--num_words', type=int, default=32)
    parser.add_argument('--unseen_task_ratio', type=float, default=0.1)
    parser.add_argument('--num_pertask', type=int, default=1000)
    parser.add_argument('--task_fields', type=str, default=None)


    args = parser.parse_args()
    main(args)