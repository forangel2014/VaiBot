### run

```
conda create -n nesyflow python==3.9
conda activate nesyflow
pip install -r requirements.txt
bash run.sh --cuda_devices 0,1,2 --model_name_or_path /mnt/publiccache/huggingface/llama-2-7b-chat-hf --meta_exp_dir ./exp --exp_name vaeflow-mc --recon_loss_weight 1 --flow_loss_weight 1 --kl_loss_weight 0.01 --batch_size 2 --selected_layers 16 --latent_size 3000 --prior vaeflow --num_latent_samples 4 --ebm_optim_method mc
```