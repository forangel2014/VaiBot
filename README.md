### run

```
conda create -n nesyflow python==3.9
conda activate nesyflow
pip install -r requirements.txt
git clone https://github.com/ELIFE-ASU/INNLab
cd INNLab/
python setup.py install
cd ..
```

#### pretrain
```
bash run.sh --cuda_devices 0,1,2 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp_final --exp_name vae-pretrain --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --pretrain_data_ratio 1
```

#### in-domain
```
bash run.sh --cuda_devices 0,1,2 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp_final --exp_name vae-domain-chat --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 10 --save_epoch 10 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining False --use_trainable_task_model True --use_knowledge_in_task hard --method nesy
```

### baselines
```
bash run.sh --cuda_devices 0,6,7 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp_baseline --exp_name icl --method icl --test_sample_num 5
```

### induction
```
bash run.sh --cuda_devices 3,6,7 --model_name_or_path /mnt/workspace/user/chenhao/pretrained_models/Llama-2-7b-chat-hf --meta_exp_dir ./exp_induction --exp_name vae-induction-1 --test_sample_num 1 --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --load_exp ../exp_final/vae-domain --load_epoch 10
```