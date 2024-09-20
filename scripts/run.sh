
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-07-23 18:51:12
### 

# bash scripts/train.sh 3 0,5,8 llama2-7b sni 25 100 16
# ps -ef |grep main.py|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep train.sh|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep run.sh|grep -v grep |cut -c 9-14|xargs kill -9
# 

cd patch
bash install.sh
cd ..

#nohup bash scripts/train.sh 3 3,4,5 llama2-7b sni 25 100 1e-5 16 > logs/llama2-7b_sni_25_r16.log 2>&1 &

nohup bash scripts/train.sh 3 0,1,2 llama2-7b list_functions 25 100 1e-5 16 > logs/llama2-7b_list_25_r16.log 2>&1 &

# bash scripts/train.sh 3 0,5,8 llama2-7b list_functions 25 8

# ps -ef |grep sni|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep run.sh|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep train.sh|grep -v grep |cut -c 9-14|xargs kill -9