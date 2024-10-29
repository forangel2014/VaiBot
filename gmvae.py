import os
import random
import torch
import torch.nn as nn
from torch.distributions import Normal
from llm import WrappedLLM
from utils import mkdir
from src.gm_entropy.entropy_bounds import EntropyLowerBoundEstLogScale

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
                nn.Linear(self.hidden_size, self.latent_size*2*self.args.num_peak+self.args.num_peak)
            ).to(self.args.encoder_device)
            
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size*self.args.num_soft_token),
                nn.Sigmoid()
            ).to(self.args.decoder_device)

            # self.reference_trained_params = torch.nn.Parameter(torch.randn(size=[len(args.task_id2knowledge), self.args.latent_size], 
            #                                             requires_grad=True, 
            #                                             device=self.args.task_device, 
            #                                             dtype=torch.bfloat16))
            
            # self.reference_optimizer = torch.optim.Adam([self.reference_trained_params], lr=args.task_finetune_lr)
            
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
        mg_params = hidden[:, :-self.args.num_peak].view(-1, self.latent_size, 2, self.args.num_peak)
        mg_log_prior = torch.log_softmax(hidden[:, -self.args.num_peak:], dim=1)
        mean = mg_params[:, :, 0, :]
        log_var = mg_params[:, :, 1, :]
        return mean, log_var, mg_log_prior
    
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

    def compute_kl_loss(self, mean, log_var, log_prior):
        # 计算方差
        var = torch.exp(log_var)
        
        # 计算KL散度
        kl_div = 0.5 * (var + mean**2 - 1 - log_var)
        
        prior = torch.exp(log_prior)
        
        kl_div_mixed = torch.matmul(kl_div, prior.T).squeeze(-1)
        
        kl_loss = torch.mean(kl_div_mixed)
        
        return kl_loss

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

    def estimate_entropy(self, mean, log_var, log_prior, method="ieee2008"):
        
        entropy = 0
        
        if method == "MC":
        
            for i in range(10):
                _, log_probs = self.reparameterize(mean, log_var, log_prior, return_log_prob=True)
                entropy += -log_probs
            
            entropy /= 10
        
        elif method == "prior-dist":
            
            dist_entropy = 0
            prior_entropy = -torch.mean(torch.sum(log_prior*torch.exp(log_prior), dim=-1))
            
            # for i in range(self.args.num_peak):    
            #     dist_entropy += torch.mean(torch.log(torch.sum(torch.norm(mean[:, :, 0].unsqueeze(-1) - mean, dim=1), dim=1)))
                
            # dist_entropy /= self.args.num_peak
            entropy = prior_entropy #+ dist_entropy
            
        elif method == "ieee2008":
            batch_size = mean.shape[0]
            weights = torch.exp(log_prior).to(torch.float32)
            covariances = torch.exp(log_var).permute(0, 2, 1).to(torch.float32)
            means = mean.permute(0, 2, 1).to(torch.float32)
            for i in range(batch_size):
                gmm_params = (weights[i], means[i], covariances[i])
                entropy += EntropyLowerBoundEstLogScale(gmm_params)
            entropy /= batch_size
            
        return entropy

    def reparameterize(self, mean, log_var, log_prior, return_log_prob=False):
        
        batch_size = mean.shape[0]
        eps_prior = torch.rand_like(log_prior)        
        gumbel_probs = log_prior - torch.log(-torch.log(eps_prior))
        cat = torch.argmax(gumbel_probs, dim=-1)#.expand_as(mean[:,0,:])
        
        samples = []
        log_probs = []
        for i in range(batch_size):
            std = torch.exp(0.5 * log_var[i, :, cat[i]])
            eps = torch.randn_like(std)
            sampled = mean[i, :, cat[i]] + eps * std
            samples.append(sampled)
            
            if return_log_prob:
                prob = 0
                for j in range(self.args.num_peak):
                    mean_j = mean[i, :, j]
                    log_var_j = log_var[i, :, j]
                    std_j = torch.exp(0.5 * log_var_j)
                    normal_dist = Normal(mean_j, std_j)
                    log_prob = normal_dist.log_prob(sampled)
                    total_log_prob = log_prob.sum()
                    total_prob_j = torch.exp(total_log_prob + log_prior[i, j])
                    prob += total_prob_j
                    
        sampled_tensor = torch.cat(samples, dim=0).view(batch_size, -1)
        
        if return_log_prob:
            return sampled_tensor, log_probs
        else:
            return sampled_tensor

    def forward(self, knowledge_batch, x_batch, y_batch):
        
        #knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=self.args.max_token, truncation=True).input_ids.to(self.args.encoder_device)
        batch_size = len(knowledge_batch)
        kl_loss = 0
        recon_loss = 0
        task_loss = 0
        reference_task_loss = 0
        alignment_loss = 0
        entropy_loss = 0

        batch_size = len(knowledge_batch)

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
        mean, log_var, log_prior = self.encode(knowledge_ids)
        kl_loss = self.compute_kl_loss(mean, log_var, log_prior)

        sampled_latent = self.reparameterize(mean, log_var, log_prior)

        sampled_latent = sampled_latent.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)
        
        sampled_latent = sampled_latent.to(self.args.task_device)
        task_loss += self.compute_task_loss(sampled_latent, x_batch, y_batch)

        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        entropy_loss = entropy_loss.to(self.args.backward_device)

        kl_loss /= batch_size
        recon_loss /= batch_size
        task_loss /= batch_size
        entropy_loss /= batch_size
        
        return kl_loss, recon_loss, task_loss, entropy_loss #alignment_loss, reference_task_loss

    def forward_batch(self, knowledge_batch, x_batch, y_batch):
        
        batch_size = len(knowledge_batch)

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
        mean, log_var, log_prior = self.encode(knowledge_ids)
        kl_loss = self.compute_kl_loss(mean, log_var, log_prior)
        entropy_loss = -self.estimate_entropy(mean, log_var, log_prior, method="ieee2008")

        sampled_latent = self.reparameterize(mean, log_var, log_prior)

        sampled_latent = sampled_latent.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)

        sampled_latent = sampled_latent.to(self.args.task_device)
        task_loss = self.compute_task_loss(sampled_latent, x_batch, y_batch) #/ batch_size

        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        entropy_loss = entropy_loss.to(self.args.backward_device)

        return kl_loss, recon_loss, task_loss, entropy_loss #alignment_loss, reference_task_loss

    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater):
        
        batch_size = len(knowledge_batch)
        
        if self.args.fuse_method == "delta":
            
            results = []
            
            for i in range(batch_size):

                knowledge_ids = self.llm.tokenizer(knowledge_batch[i], add_special_tokens=True, return_tensors="pt").input_ids.to(self.args.encoder_device)#(self.args.device)
                mean, log_var, log_prior = self.encode(knowledge_ids)

                latent = mean[0].to(self.args.flow_device)
                
                params = self.flow_forward(latent).to(self.args.task_device)
                
                new_task_parameters = self.llm.allocate(params)
                
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
            mean, log_var, log_prior = self.encode(knowledge_ids)
            
            params = self.reparameterize(mean, log_var, log_prior).to(self.args.task_device)
            
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