from json import load
import os
import random
import torch
import torch.nn as nn
from llm import WrappedLLM
from utils import mkdir

class Nesy(nn.Module):
    
    def __init__(self, args):
        super(Nesy, self).__init__()
        self.args = args
        
        self.llm = WrappedLLM(self.args)
        self.hidden_size = self.llm.config.hidden_size
        self.latent_size = self.args.latent_size
        
        if args.method == "nesy":
            
            self.encoder_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.latent_size*2)
            ).to(self.args.encoder_device)#to(self.args.flow_device)
            
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size*self.args.num_soft_token),
                nn.Sigmoid()
            ).to(self.args.decoder_device)#to(self.args.flow_device)

            if args.load_nesy_ckpt:
                self.load(args.load_nesy_ckpt)

    def save(self, dir):
        mkdir(dir)
        torch.save(self.encoder_mlp.state_dict(), os.path.join(dir, "encoder_mlp.pth"))
        torch.save(self.decoder_mlp.state_dict(), os.path.join(dir, "decoder_mlp.pth"))
        self.llm.save(dir)

    def load(self, dir):
        self.encoder_mlp.load_state_dict(torch.load(os.path.join(dir, "encoder_mlp.pth")))
        self.decoder_mlp.load_state_dict(torch.load(os.path.join(dir, "decoder_mlp.pth")))
        self.llm.load(dir)

    def encode(self, knowledge_ids):
        outputs = self.llm.encode(knowledge_ids)
        last_hidden = outputs[:, -1, :]
        hidden = self.encoder_mlp(last_hidden)
        mean = hidden[:, :self.latent_size]
        log_var = hidden[:, self.latent_size:]
        return mean, log_var
    
    def compute_recon_loss(self, latent, labels, instance=None):
        embedding = self.decoder_mlp(latent)#.to(self.args.decoder_device)
        if self.args.use_instance_in_decoder:
            instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            outputs = self.llm.decode(embedding, labels, instance_embedding)
        else:
            outputs = self.llm.decode(embedding, labels)
        return outputs

    def sample(self, context, sample_from_guassian=True, instance=None):
        
        if sample_from_guassian:
            sampled_latent = self.reparameterize(context, torch.ones_like(context)).to(self.args.decoder_device)
        else:
            sampled_latent = context
        embedding = self.decoder_mlp(sampled_latent)
        if self.args.use_instance_in_decoder:
            instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            sampled_ids = self.llm.sample(embedding, instance_embedding)
        else:
            sampled_ids = self.llm.sample(embedding)
        #text = [self.llm.tokenizer.decode(k) for k in sampled_ids.tolist()[0]]
        text = self.llm.tokenizer.decode(sampled_ids.tolist()[0], skip_special_tokens=True)
        
        return text

    def compute_kl_loss(self, mean, log_var):
        kl_loss = 0.5 * torch.mean(
            log_var.exp() + mean.pow(2) - 1 - log_var,
            dim=1
        )
        return kl_loss.mean()

    def compute_task_loss(self, latent, x_batch, y_batch, reduce=True):
        
        batch_size = len(x_batch)
        
        if self.args.fuse_method == "delta":

            if reduce:
                task_loss = 0
            else:
                task_loss = []
         
            for i in range(batch_size):
                
                new_task_parameters = self.llm.allocate(latent[i])
                
                x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)
                y_id = self.llm.tokenizer(y_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)

                if reduce:
                    task_loss += self.llm.solve_task(x_id, y_id, new_task_parameters)
                else:
                    task_loss.append(self.llm.solve_task(x_id, y_id, new_task_parameters))

            if reduce:
                task_loss /= batch_size
            else:
                task_loss = torch.stack(task_loss, dim=0)
            
        elif self.args.fuse_method == "p-tuning":

            x_id = self.llm.tokenizer(x_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            y_id = self.llm.tokenizer(y_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            
            if self.args.ebm_optim_method == "mc":
                x_id = x_id.repeat_interleave(self.args.num_latent_samples, dim=0)
                y_id = y_id.repeat_interleave(self.args.num_latent_samples, dim=0)
                latent = latent.reshape(batch_size*self.args.num_latent_samples, self.args.latent_size)
            else:
                latent = latent.reshape(batch_size, self.args.latent_size)
            
            task_loss = self.llm.solve_task(x_id, y_id, latent, reduce=reduce)
            
        return task_loss
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #eps = torch.randn((self.args.num_latent_samples, mean.shape[1])).to(mean.device).bfloat16()
        return mean + eps * std

    def forward(self, knowledge_batch, x_batch, y_batch):
        
        batch_size = len(knowledge_batch)

        # self.reference_optimizer.zero_grad()
        # task_ids = [self.args.knowledge2task_id[k] for k in knowledge_batch]
        # reference_params = self.reference_trained_params[task_ids]
        # reference_task_loss = self.compute_task_loss(reference_params, x_batch, y_batch) / batch_size
        # reference_task_loss.backward()
        # self.reference_optimizer.step()

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
        mean, log_var = self.encode(knowledge_ids)
        
        reg_loss = self.compute_kl_loss(mean, log_var)

        sampled_latent = self.reparameterize(mean, log_var)

        sampled_latent = sampled_latent.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        
        if self.args.use_instance_in_decoder:
            instance = (x_batch, y_batch)
            instance_text = [f"input: {x}, target: {y}. This task is to:" for x, y in zip(*instance)]
            instance_ids = self.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.decoder_device)
        else:
            instance_ids = None
        recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids, instance_ids)

        sampled_latent = sampled_latent.to(self.args.task_device)
        task_loss = self.compute_task_loss(sampled_latent, x_batch, y_batch)

        #alignment_loss = torch.mean(torch.norm(sampled_latent - reference_params.detach().to(self.args.task_device), dim=1)) #/ self.args.num_latent_samples

        #reg_loss = sampled_latent.norm(1, dim=1).mean() / self.args.latent_size

        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        reg_loss = reg_loss.to(self.args.backward_device)
        #alignment_loss = alignment_loss.to(self.args.backward_device)

        return reg_loss, recon_loss, task_loss#, alignment_loss, reference_task_loss

    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater):
        
        batch_size = len(knowledge_batch)
        
        if self.args.fuse_method == "delta":
            
            results = []
            
            for i in range(batch_size):

                knowledge_ids = self.llm.tokenizer(knowledge_batch[i], add_special_tokens=True, return_tensors="pt").input_ids.to(self.args.encoder_device)#(self.args.device)
                mean, log_var = self.encode(knowledge_ids)

                latent = mean[0].to(self.args.flow_device)
                
                new_task_parameters = self.llm.allocate(latent)
                
                x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)
                
                y_pred = self.llm.predict_task(x_id, new_task_parameters)

                results.append({
                    "knowledge": knowledge_batch[i],
                    "x": x_batch[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred,
                    "score": evaluater(y_pred, y_batch[i])
                    })

        elif self.args.fuse_method == "p-tuning":
            
            knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
            mean, log_var = self.encode(knowledge_ids)
            
            params = self.reparameterize(mean, log_var).to(self.args.task_device)
            
            x_id = self.llm.tokenizer(x_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            y_pred = self.llm.predict_task(x_id, params)
            
            results = [
                {
                    "knowledge": knowledge_batch[i],
                    "x": x_batch[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred[i],
                    "score": evaluater(y_pred[i], y_batch[i])
                }
                for i in range(batch_size)
            ]
        return results
    
    def eval_knowledge(self, knowledge, predicted_knowledge, evaluater):

        result = {
            "groundtruth knowledge": knowledge,
            "predicted knowledge": predicted_knowledge,
            "score": evaluater(knowledge, predicted_knowledge)
            }

        return result