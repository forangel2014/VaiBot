
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-07-23 16:27:47
### 
# export NCCL_P2P_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"llama3-8b"}
dataset=${4:-"sni"}
num_words=${5:-"32"}
num_pertask=${6:-"100"}
lr=${7:-"1e-4"}
r=${8:-"8"}
task_fields=${9:-"0"}
target=${10:-"all"}
extra_args=""

let lora_alpha=r*2
echo "lora_alpha: ${lora_alpha}"
model_name_or_path=${model}
model="${model_name_or_path##*/}"
if [ "$model" = "llama3-8b" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B
    cutoff_len=4096
fi
if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    template=llama3
fi
if [ "$model" = "llama2-7b" ];then
    model_name_or_path=meta-llama/Llama-2-7b-hf
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-0.5b" ];then
    model_name_or_path=Qwen/Qwen2-0.5B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-1.5b" ];then
    model_name_or_path=Qwen/Qwen2-1.5B
    cutoff_len=4096
fi
if [ "$model" = "llama2-7b-chat" ];then
    model_name_or_path=meta-llama/Llama-2-7b-chat-hf
fi

if [ "$target" = "all" ];then
    target_modules=q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj
fi

exp_name=${model}/${dataset}/${target}_${num_words}_r${r}_lr${lr}

if [ "$task_fields" != "0" ];then
    extra_args="${extra_args} --task_fields ${task_fields}"
    exp_name="${exp_name}_${task_fields}"
    echo "task_fields: ${task_fields}"
fi
exp_name=${exp_name//\,/_}
echo "model_name_or_path: ${model_name_or_path}"
echo "dataset: ${dataset}"
echo "exp_name: ${exp_name}"
echo "num_pertask: ${num_pertask}"
echo "lr: ${lr}"
echo "r: ${r}"

CUDA_VISIBLE_DEVICES=${gpus} python main.py \
    --dataset ${dataset} \
    --exp_name ${exp_name} \
    --batch_size 1 \
    --latent_size 10000 \
    --lr ${lr} \
    --num_epochs 10 \
    --beta 0.1 \
    --gamma 0.1 \
    --max_token 50 \
    --ignore_exist \
    --results_name ${exp_name} \
    --model_name_or_path ${model_name_or_path} \
    --finetuned_model None \
    --device 0 \
    --lora_r ${r} \
    --lora_alpha ${lora_alpha} \
    --target_modules ${target_modules} \
    --num_words ${num_words} \
    --num_pertask ${num_pertask} \
    ${extra_args}
