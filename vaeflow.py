from json import load
import os
import random
import torch
import torch.nn as nn
from llm import WrappedLLM
from utils import mkdir, hook
import INN

class Nesy(nn.Module):
    
    def __init__(self, args):
        super(Nesy, self).__init__()
        self.args = args
        
        self.llm = WrappedLLM(self.args).to(torch.bfloat16)
        self.hidden_size = self.llm.config.hidden_size
        self.latent_size = self.args.latent_size
        
        if args.method == "nesy":
            
            self.encoder_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.latent_size*2)
            ).to(self.args.encoder_device).to(torch.bfloat16)
            
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.latent_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size*self.args.num_soft_token),
                nn.Sigmoid()
            ).to(self.args.decoder_device).to(torch.bfloat16)

            self.flow_net = INN.Sequential(
                    #INN.Linear(self.latent_size),
                    INN.Nonlinear(self.latent_size, method="RealNVP"),
                    #INN.Linear(self.latent_size),
                    #INN.Nonlinear(self.latent_size, method="RealNVP"),
                    #INN.Linear(self.latent_size)
                ).to(self.args.flow_device)
            
            if self.args.ebm_optim_method == "nce":
            
                self.logZ = torch.nn.Parameter((torch.ones(len(self.args.task_id2knowledge))*self.args.latent_size*-torch.log(torch.tensor(1.0 / (3 - (-3))))).to(self.args.task_device))
                
            elif self.args.ebm_optim_method == "flow-nce":
                
                self.logZ = torch.nn.Parameter((torch.ones(len(self.args.task_id2knowledge))*self.args.latent_size*-torch.log(torch.tensor(1.0 / (3 - (-3))))).to(self.args.task_device))
                self.noise_flow_net = INN.Sequential(
                        INN.Nonlinear(self.latent_size, method="RealNVP"),
                    ).to(self.args.noise_device)
                
                log_prob_per_dim = torch.log(torch.tensor(1.0 / (3 - (-3))))  # log(1/6)
                logq = self.args.latent_size * log_prob_per_dim
                self.logq = logq.to(self.args.noise_device)

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
        torch.save(self.flow_net.state_dict(), os.path.join(dir, "flow_net.pth"))
        self.llm.save(dir)

    def load(self, dir):
        self.encoder_mlp.load_state_dict(torch.load(os.path.join(dir, "encoder_mlp.pth")))
        self.decoder_mlp.load_state_dict(torch.load(os.path.join(dir, "decoder_mlp.pth")))
        self.flow_net.load_state_dict(torch.load(os.path.join(dir, "flow_net.pth")))
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
    
    def flow_forward(self, latent, return_all=False):
        #params = latent
        params, log_p, log_det_J = self.flow_net(latent.to(torch.float))
        params = params.to(torch.bfloat16)
        if return_all:
            return params, log_p, log_det_J
        else:
            return params
        
    def noise_flow_forward(self, latent, return_all=False):
        #params = latent
        params, log_p, log_det_J = self.noise_flow_net(latent.to(torch.float))
        params = params.to(torch.bfloat16)
        if return_all:
            return params, log_p, log_det_J
        else:
            return params
    
    def flow_backward(self, params):
        #latent = params
        latent = self.flow_net.inverse(params.to(torch.float)).to(torch.bfloat16)
        return latent

    def noise_flow_backward(self, params):
        #latent = params
        latent = self.noise_flow_net.inverse(params.to(torch.float)).to(torch.bfloat16)
        return latent

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
            
            if not reduce:
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

    def reparameterize(self, mean, log_var, return_log_prob=False):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std        
        if return_log_prob:
            log_p = torch.sum(-0.5 * (log_var + torch.pow(z - mean, 2) / torch.exp(log_var)) - 0.5 * torch.log(torch.tensor(2 * torch.pi)), dim=1)
            return z, log_p

        else:
            return z

    def reparameterize_group(self, mean, log_var, num_samples):
        # mean: [batch_size, latent_size]
        # log_var: [batch_size, latent_size]
        # num_samples: the number of samples to draw for each instance in the batch
        
        # Step 1: Compute standard deviation
        std = torch.exp(0.5 * log_var)  # [batch_size, latent_size]
        
        # Step 2: Repeat mean and std to match the number of samples
        mean = mean.unsqueeze(1).expand(-1, num_samples, -1)  # [batch_size, num_samples, latent_size]
        std = std.unsqueeze(1).expand(-1, num_samples, -1)    # [batch_size, num_samples, latent_size]
        
        # Step 3: Sample random noise for each sample
        eps = torch.randn_like(std)  # [batch_size, num_samples, latent_size]
        
        # Step 4: Reparameterization trick to get the samples
        return mean + eps * std  # [batch_size, num_samples, latent_size]

    def forward(self, knowledge_batch, x_batch, y_batch):

        batch_size = len(knowledge_batch)

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
        mean, log_var = self.encode(knowledge_ids)
        kl_loss = self.compute_kl_loss(mean, log_var)
        
        # task_ids = [self.args.knowledge2task_id[k] for k in knowledge_batch]
        # reference_params = self.reference_trained_params[task_ids]
        # reference_task_loss = self.compute_task_loss(reference_params, x_batch, y_batch) / batch_size
        # reference_task_loss.backward()
        # self.reference_optimizer.step()

        if self.args.ebm_optim_method == "drop-z":

            sampled_latent = self.reparameterize(mean, log_var)

            sampled_latent = sampled_latent.to(self.args.decoder_device)
            knowledge_ids = knowledge_ids.to(self.args.decoder_device)
            recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)

            sampled_latent = sampled_latent.to(self.args.flow_device)
                        
            params = self.flow_forward(sampled_latent)
            
            params = params.to(self.args.task_device)
            task_loss = self.compute_task_loss(params, x_batch, y_batch) / batch_size
            
            flow_loss = task_loss

        elif self.args.ebm_optim_method == "nce":

            sampled_latent = self.reparameterize(mean, log_var)

            sampled_latent = sampled_latent.to(self.args.decoder_device)
            knowledge_ids = knowledge_ids.to(self.args.decoder_device)
            recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)

            sampled_latent = sampled_latent.to(self.args.flow_device)
            params = self.flow_forward(sampled_latent)
            params = params.to(self.args.task_device)
            
            #noise_params = sampled_latent.detach().to(self.args.task_device)

            noise_params = torch.empty(batch_size, self.args.latent_size, dtype=torch.bfloat16).uniform_(-3, 3).to(self.args.task_device)
            log_prob_per_dim = torch.log(torch.tensor(1.0 / (3 - (-3))))  # log(1/6)
            logq = self.args.latent_size * log_prob_per_dim * torch.ones(batch_size)
            logq = logq.to(self.args.task_device)

            n = torch.tensor(params.shape[0])
            k = torch.tensor(noise_params.shape[0])

            task_ids = [self.args.knowledge2task_id[knowledge] for knowledge in knowledge_batch]
            
            task_loss_data = self.compute_task_loss(params, x_batch, y_batch, reduce=False)
            logp_data = -(self.args.beta*task_loss_data + self.logZ[task_ids])
            PC0_post = torch.sigmoid(torch.log(n) - torch.log(k) + (logp_data - logq))

            task_loss_noise = self.compute_task_loss(noise_params, x_batch, y_batch, reduce=False)
            logp_noise = -(self.args.beta*task_loss_noise + self.logZ[task_ids])
            PC1_post = torch.sigmoid(torch.log(k) - torch.log(n) + (logq - logp_noise))

            flow_loss = -torch.log(torch.clamp(PC0_post, min=self.args.episilon)) -torch.log(torch.clamp(PC1_post, min=self.args.episilon))
            flow_loss = torch.mean(flow_loss)
            
            task_loss = task_loss_data
            
        elif self.args.ebm_optim_method == "flow-nce":

            sampled_latent = self.reparameterize(mean, log_var)

            sampled_latent = sampled_latent.to(self.args.decoder_device)
            knowledge_ids = knowledge_ids.to(self.args.decoder_device)
            recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids)

            sampled_latent = sampled_latent.to(self.args.flow_device)
            params = self.flow_forward(sampled_latent)
            data_params = params.to(self.args.task_device)
            
            #noise_params = sampled_latent.detach().to(self.args.task_device)

            noise_latent = torch.empty(batch_size, self.args.latent_size, dtype=torch.bfloat16).uniform_(-3, 3).to(self.args.noise_device)
            noise_params, _, log_det_J_noise = self.noise_flow_forward(noise_latent, return_all=True)
            logq_noise = self.logq * torch.ones(batch_size).to(self.args.noise_device) + log_det_J_noise
            
            data_latent = self.noise_flow_backward(data_params.to(self.args.noise_device))
            _, _, log_det_J_data = self.noise_flow_forward(data_latent, return_all=True)
            logq_data = self.logq * torch.ones(batch_size).to(self.args.noise_device) + log_det_J_data

            n = torch.tensor(params.shape[0])
            k = torch.tensor(noise_params.shape[0])
            logq_noise = logq_noise.to(self.args.task_device)
            logq_data = logq_data.to(self.args.task_device)
            noise_params = noise_params.to(self.args.task_device)

            task_ids = [self.args.knowledge2task_id[knowledge] for knowledge in knowledge_batch]
            
            task_loss_data = self.compute_task_loss(data_params, x_batch, y_batch, reduce=False)
            logp_data = -(self.args.beta*task_loss_data + self.logZ[task_ids])
            PC0_post = torch.sigmoid(torch.log(n) - torch.log(k) + (logp_data - logq_data))

            task_loss_noise = self.compute_task_loss(noise_params, x_batch, y_batch, reduce=False)
            logp_noise = -(self.args.beta*task_loss_noise + self.logZ[task_ids])
            PC1_post = torch.sigmoid(torch.log(k) - torch.log(n) + (logq_noise - logp_noise))

            flow_loss = -torch.log(torch.clamp(PC0_post, min=self.args.episilon)) -torch.log(torch.clamp(PC1_post, min=self.args.episilon))
            flow_loss = torch.mean(flow_loss)
            
            task_loss = task_loss_data

        elif self.args.ebm_optim_method == "mc":

            sampled_latent = self.reparameterize_group(mean, log_var, num_samples=self.args.num_latent_samples)

            sampled_latent = sampled_latent.to(self.args.decoder_device)
            knowledge_ids = knowledge_ids.to(self.args.decoder_device)
            group_latent = sampled_latent.reshape(batch_size*self.args.num_latent_samples, self.args.latent_size)
            labels = knowledge_ids.repeat_interleave(self.args.num_latent_samples, dim=0)
            recon_loss = self.compute_recon_loss(group_latent, labels)
            
            sampled_latent = sampled_latent.to(self.args.flow_device)
            params = self.flow_forward(sampled_latent).to(self.args.task_device)

            if self.args.fuse_method == "delta":

                flow_loss = 0
                for i in range(self.args.batch_size):
                    task_loss = self.compute_task_loss(params[i], [x_batch[i]]*self.args.num_latent_samples, [y_batch[i]]*self.args.num_latent_samples, reduce=False)
                    probs = torch.softmax(-task_loss*self.args.beta, dim=0)
                    flow_loss += torch.mean(-torch.log(probs))

                flow_loss /= batch_size

            else:
                
                task_loss = self.compute_task_loss(params, x_batch, y_batch, reduce=False)
                task_loss = task_loss.reshape(batch_size, self.args.num_latent_samples)
                probs = torch.softmax(-task_loss*self.args.beta, dim=-1)
                flow_loss = torch.mean(-torch.log(probs))


        kl_loss = kl_loss.to(self.args.backward_device)
        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        flow_loss = flow_loss.to(self.args.backward_device)
        return kl_loss, recon_loss, task_loss, flow_loss
    
    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater):
        
        batch_size = len(knowledge_batch)
        
        if self.args.fuse_method == "delta":
            
            results = []
            
            for i in range(batch_size):

                knowledge_ids = self.llm.tokenizer(knowledge_batch[i], add_special_tokens=True, return_tensors="pt").input_ids.to(self.args.encoder_device)#(self.args.device)
                mean, log_var = self.encode(knowledge_ids)

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
            mean, log_var = self.encode(knowledge_ids)
            
            latent = mean.to(self.args.flow_device)
            params = self.flow_forward(latent).to(self.args.task_device)
            
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