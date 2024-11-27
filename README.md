### run

```
conda create -n nesyflow python==3.9
conda activate nesyflow
pip install -r requirements.txt
nohup bash run.sh --cuda_devices 0,1,2 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp --exp_name vae-pretrain-trainable1 --recon_loss_weight 1 --flow_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --entropy_loss_weight 1e-3 --batch_size 8 --selected_layers 16 --latent_size 8000 --prior gaussian --num_latent_samples 2 --unseen_task_ratio 0.1 --ebm_optim_method fce --fuse_method p-tuning --num_soft_token 10 --dataset sni --decoder_device 2 --num_peak 20 --lora_r 128 --pretraining True --valid_epoch 5
```