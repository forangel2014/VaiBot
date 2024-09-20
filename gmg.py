import os
import random
import torch
import torch.nn as nn
from torch.distributions import Normal
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
                nn.Linear(self.hidden_size, self.latent_size*2*(self.args.num_peak+1)+self.args.num_peak)
            ).to(self.args.encoder_device)
            
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.latent_size*2, self.hidden_size*2),
                nn.ReLU(),
                nn.Linear(self.hidden_size*2, self.hidden_size*self.args.num_soft_token),
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
        style = hidden[:, :self.latent_size*2].view(-1, self.latent_size, 2)
        style_mean = style[:, :, 0]
        style_log_var = style[:, :, 1]
        params = hidden[:, self.latent_size*2:-self.args.num_peak].view(-1, self.latent_size, 2, self.args.num_peak)
        params_mean = params[:, :, 0, :]
        params_log_var = params[:, :, 1, :]
        params_log_prior = torch.log_softmax(hidden[:, -self.args.num_peak:], dim=1)
        return style_mean, style_log_var, params_mean, params_log_var, params_log_prior
    
    def compute_recon_loss(self, sampled_params, sampled_style, labels):
        latent = torch.cat((sampled_params, sampled_style), dim=1)
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

    def estimate_entropy(self, mean, log_var, log_prior, method="MC"):
        
        entropy = 0
        
        if method == "MC":
        
            for i in range(10):
                _, log_probs = self.reparameterize(mean, log_var, log_prior, return_log_prob=True)
                entropy += -log_probs
            
            entropy /= 10
        
        elif method == "prior-dist":
            
            dist_entropy = 0
            prior_entropy = -torch.mean(torch.sum(log_prior*torch.exp(log_prior), dim=-1))
            
            for i in range(self.args.num_peak):    
                dist_entropy += torch.mean(torch.log(torch.sum(torch.norm(mean[:, :, 0].unsqueeze(-1) - mean, dim=1), dim=1)))
                
            dist_entropy /= self.args.num_peak
            entropy = prior_entropy + dist_entropy
            
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

    def reparameterize_g(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #eps = torch.randn((self.args.num_latent_samples, mean.shape[1])).to(mean.device).bfloat16()
        return mean + eps * std


    def forward_batch(self, knowledge_batch, x_batch, y_batch):
        
        batch_size = len(knowledge_batch)
        kl_loss = 0

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=False, padding="longest", truncation=True).input_ids.to(self.args.encoder_device)
        style_mean, style_log_var, params_mean, params_log_var, params_log_prior = self.encode(knowledge_ids)
        kl_loss += self.compute_kl_loss(style_mean, style_log_var)
        #entropy_loss = -self.estimate_entropy(params_mean, params_log_var, params_log_prior, method="prior-dist")

        sampled_params = self.reparameterize(params_mean, params_log_var, params_log_prior)
        sampled_style = self.reparameterize_g(style_mean, style_log_var)

        sampled_params = sampled_params.to(self.args.decoder_device)
        sampled_style = sampled_style.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        recon_loss = self.compute_recon_loss(sampled_params, sampled_style, knowledge_ids)

        sampled_params = sampled_params.to(self.args.task_device)
        task_loss = self.compute_task_loss(sampled_params, x_batch, y_batch) / batch_size

        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        #entropy_loss = entropy_loss.to(self.args.backward_device)

        return kl_loss, recon_loss, task_loss #, entropy_loss #alignment_loss, reference_task_loss

    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater):
        
        batch_size = len(knowledge_batch)
        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=False, padding="longest", truncation=True).input_ids.to(self.args.encoder_device)
        mean, log_var, log_prior = self.encode(knowledge_ids)
        
        results = []
        
        for i in range(batch_size):
            
            means = mean[i]
            priors = torch.exp(log_prior[i])
            cat = torch.multinomial(priors, num_samples=1, replacement=True)
            
            latent = means[:, cat[0]].to(self.args.task_device)
            
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