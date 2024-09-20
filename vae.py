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
            ).to(self.args.encoder_device)
            
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size*self.args.num_soft_token),
                nn.Sigmoid()
            ).to(self.args.decoder_device)
            
            # TODO 3D tensor 
            # 匈牙利匹配，SINKHORN algorithm
            # sampled latent [sample_num, hidden_dim]
            # reference latent [group, hidden_dim]
            self.reference_trained_params = torch.nn.Parameter(torch.randn(size=[len(args.task_id2knowledge), self.args.latent_size], 
                                                        requires_grad=True,
                                                        device=self.args.task_device, 
                                                        dtype=torch.bfloat16))
            
            self.reference_optimizer = torch.optim.Adam([self.reference_trained_params], lr=args.task_finetune_lr)

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
    
    def compute_recon_loss(self, latent, labels):
        embedding = self.decoder_mlp(latent)
        outputs = self.llm.decode(embedding, labels)
        return outputs

    def sample(self, context, sample_from_guassian=True):
        
        if sample_from_guassian:
            sampled_latent = self.reparameterize(context, torch.ones_like(context)).to(self.args.decoder_device)
        else:
            sampled_latent = context
        embedding = self.decoder_mlp(sampled_latent)
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

    def compute_task_loss(self, latent, x_batch, y_batch):
        
        batch_size = latent.shape[0]
        task_loss = 0
                
        for i in range(batch_size):
            
            new_task_parameters = self.llm.allocate(latent[i])
            
            x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt").input_ids.to(self.args.task_device)
            y_id = self.llm.tokenizer(y_batch[i], return_tensors="pt").input_ids.to(self.args.task_device)

            task_loss += self.llm.solve_task(x_id, y_id, new_task_parameters)

        return task_loss #/ batch_size

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #eps = torch.randn((self.args.num_latent_samples, mean.shape[1])).to(mean.device).bfloat16()
        return mean + eps * std

    def forward(self, knowledge_batch, x_batch, y_batch):
        
        #knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=self.args.max_token, truncation=True).input_ids.to(self.args.encoder_device)
        batch_size = len(knowledge_batch)
        kl_loss = 0
        recon_loss = 0
        task_loss = 0
        reference_task_loss = 0
        alignment_loss = 0

        self.reference_optimizer.zero_grad()
        for i in range(batch_size):
            reference_params = self.reference_trained_params[self.args.knowledge2task_id[knowledge_batch[i]]]
            reference_task_loss += self.compute_task_loss(reference_params.view(1,-1), [x_batch[i]], [y_batch[i]])        
        reference_task_loss /= batch_size
        reference_task_loss.backward()
        self.reference_optimizer.step()

        for i in range(batch_size):
            knowledge_ids = self.llm.tokenizer(knowledge_batch[i], return_tensors="pt").input_ids.to(self.args.encoder_device)
            mean, log_var = self.encode(knowledge_ids)
            kl_loss += self.compute_kl_loss(mean, log_var)

            sampled_latent = self.reparameterize(mean, log_var)

            sampled_latent = sampled_latent.to(self.args.decoder_device)
            knowledge_ids = knowledge_ids.to(self.args.decoder_device)
            recon_loss += self.compute_recon_loss(sampled_latent, knowledge_ids)
            
            sampled_latent = sampled_latent.to(self.args.task_device)
            # task_loss += self.compute_task_loss(sampled_latent, [x_batch[i]], [y_batch[i]])
            # alignment_loss += torch.norm(sampled_latent - self.reference_trained_params[self.args.knowledge2task_id[knowledge_batch[i]]].detach())
            task_loss += self.compute_task_loss(sampled_latent, [x_batch[i]]*self.args.num_latent_samples, [y_batch[i]]*self.args.num_latent_samples)
            alignment_loss += torch.norm(sampled_latent - self.reference_trained_params[self.args.knowledge2task_id[knowledge_batch[i]]].detach().to(self.args.task_device)) / self.args.num_latent_samples

        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        alignment_loss = alignment_loss.to(self.args.backward_device)

        kl_loss /= batch_size
        recon_loss /= batch_size
        task_loss /= batch_size
        alignment_loss /= batch_size
        
        return kl_loss, recon_loss, task_loss, alignment_loss, reference_task_loss

    def forward_batch(self, knowledge_batch, x_batch, y_batch):
        
        batch_size = len(knowledge_batch)

        self.reference_optimizer.zero_grad()
        task_ids = [self.args.knowledge2task_id[k] for k in knowledge_batch]
        reference_params = self.reference_trained_params[task_ids]
        reference_task_loss = self.compute_task_loss(reference_params, x_batch, y_batch) / batch_size
        reference_task_loss.backward()
        self.reference_optimizer.step()

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest", truncation=True).input_ids.to(self.args.encoder_device)
        mean, log_var = self.encode(knowledge_ids)
        kl_loss = self.compute_kl_loss(mean, log_var)

        sampled_latent = self.reparameterize(mean, log_var)

        sampled_latent = sampled_latent.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)

        sampled_latent = sampled_latent.to(self.args.task_device)
        task_loss = self.compute_task_loss(sampled_latent, x_batch, y_batch) / batch_size

        alignment_loss = torch.mean(torch.norm(sampled_latent - reference_params.detach().to(self.args.task_device), dim=1)) #/ self.args.num_latent_samples

        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        alignment_loss = alignment_loss.to(self.args.backward_device)

        return kl_loss, recon_loss, task_loss, alignment_loss, reference_task_loss

    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater):
        
        batch_size = len(knowledge_batch)
        
        results = []
        
        for i in range(batch_size):

            knowledge_ids = self.llm.tokenizer(knowledge_batch[i], add_special_tokens=True, return_tensors="pt").input_ids.to(self.args.encoder_device)#(self.args.device)
            mean, log_var = self.encode(knowledge_ids)

            latent = mean[0].to(self.args.task_device)
            
            new_task_parameters = self.llm.allocate(latent)
            
            x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt").input_ids.to(self.args.task_device)
            
            y_pred = self.llm.predict_task(x_id, new_task_parameters)

            results.append({
                "knowledge": knowledge_batch[i],
                "x": x_batch[i],
                "y_true": y_batch[i],
                "y_pred": y_pred,
                "score": evaluater(y_pred, y_batch[i])
                })

        return results
    
    def eval_knowledge(self, knowledge, predicted_knowledge, evaluater):

        result = {
            "groundtruth knowledge": knowledge,
            "predicted knowledge": predicted_knowledge,
            "score": evaluater(knowledge, predicted_knowledge)
            }

        return result