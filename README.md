### run

```
conda create -n nesyflow python==3.9
conda activate nesyflow
pip install -r requirements.txt
bash run.sh --cuda_devices 0,1,2 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp_new --exp_name vae-pretrain-indicator --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --num_latent_samples 2 --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --pretraining True --valid_epoch 5 --save_epoch 1 --use_instance_in_decoder True
```