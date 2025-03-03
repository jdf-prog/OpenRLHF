set -x
ray start --head --num-gpus 8 --num-cpus 64 --node-ip-address 0.0.0.0
working_dir=$PWD

policy_pretrain="Qwen/Qwen2.5-3B"
dataset="CodeDPO/AceCoderV2-mini-processed_openrlhf_format_r1" # new dataset where test cases are filterd by Qwen2.5-Coder-32B
advantage_estimator="group_norm" # group_norm
model_pretty_name=$(echo $policy_pretrain | tr '/' '_' | tr '[:upper:]' '[:lower:]')
save_name="${model_pretty_name}-${advantage_estimator}-v2_mini"
record_dir="rm_records/$save_name"
log_dir="logs/$save_name" 

rm_port=14236
remote_rm_url="rule:http://localhost:$rm_port/get_reward"
reward_log_file="$log_dir/reward.log"
rm_format_port=14237
remote_rm_format_url="rule:http://localhost:$rm_format_port/get_reward"
reward_format_log_file="$log_dir/reward_format.log"

mkdir -p $log_dir
mkdir -p $record_dirs

all_remote_rm_urls="$remote_rm_url,$remote_rm_format_url"

binary_reward=False # whether to map rewards to 1 or 0 by "1 if reward==1 else 0"
post_args="--record_dir $record_dir"
training_post_args=""
if [ $binary_reward = True ]; then
   post_args="${post_args} --binary"
   save_name=$save_name"-binary"
fi

save_path=$working_dir/saves/ckpt/$save_name
latest_tag_path=$save_path/_actor/latest
if [ -f $latest_tag_path ]; then
   echo "Trying to load checkpoint from $save_path"
   latest_tag=`cat $save_path/_actor/latest` # global_step**
   # for global_step30, only keep 30
   start_step_idx=`echo $latest_tag | sed 's/global_step//g'`
   echo "start_step_idx: $start_step_idx"
   post_args="$post_args --start_step_idx $start_step_idx"
   training_post_args="--load_checkpoint --ckpt_path $save_path"
fi
python -m openrlhf.cli.serve_rm \
   --policy_pretrain $policy_pretrain \
   --port $rm_port \
   --bf16 \
   --flash_attn \
   --normalize_reward \
   --max_len 8192 \
   --batch_size 16 \
   --rule test_case \
   --dataset $dataset \
   --input_key context_messages \
   --gt_key tests \
   $post_args > $reward_log_file 2>&1 &
PID_RM=$!

python -m openrlhf.cli.serve_rm \
   --policy_pretrain $policy_pretrain \
   --port $rm_format_port \
   --bf16 \
   --flash_attn \
   --normalize_reward \
   --max_len 8192 \
   --batch_size 16 \
   --rule code_format_reward \
   --dataset $dataset \
   --input_key context_messages \
   --gt_key tests \
   $post_args > $reward_format_log_file 2>&1 &
PID_RM2=$!

# echo "start training"
# echo "see logs/train.log for training logs"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "'$working_dir'", "excludes": ["saves", "rm_records", "logs"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --pretrain $policy_pretrain \
   --remote_rm_url $all_remote_rm_urls \
   --colocate_actor_ref \
   --save_path $save_path \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 1000000 \
   --generate_max_len 4096 \
   --num_episodes 2 \
   --advantage_estimator $advantage_estimator \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0 \
   --lambd 1.0 \
   --gamma 1.0 \
   --prompt_data $dataset \
   --input_key context_messages \
   --apply_chat_template \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 5 \
   --ckpt_path $working_dir/saves/ckpt/$save_name \
   --flash_attn \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name $save_name \
   $training_post_args


# --colocate_actor_ref \
# --colocate_all_models \
# --vllm_enable_sleep \
#    --vllm_gpu_memory_utilization 0.5 \
   # --use_kl_loss \

# --normalize_reward \
# also supports --advantage_estimator rloo

pkill -P -9 $PID_RM
kill -9 $PID_RM
pkill -P -9 $PID_RM2
kill -9 $PID_RM2

ray stop