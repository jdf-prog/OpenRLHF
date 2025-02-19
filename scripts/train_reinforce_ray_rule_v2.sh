set -x

working_dir=$PWD
# reinforce++

# policy_pretrain="Qwen/Qwen2.5-Coder-7B-Instruct"
policy_pretrain="Qwen/Qwen2.5-Coder-7B"
# dataset="CodeDPO/codedpo_20241208_openrlhf_format_hard" # old dataset where test cases are not filterd by Qwen2.5-Coder-32B
dataset="CodeDPO/AceCoderV2-mini-processed_openrlhf_format_r1" # new dataset where test cases are filterd by Qwen2.5-Coder-32B
rm_port=14236
remote_rm_url="rule:http://localhost:$rm_port/get_reward"
reward_log_file="logs/reward.log"
rm_format_port=14237
remote_rm_format_url="rule:http://localhost:$rm_format_port/get_reward"
reward_format_log_file="logs/reward_format.log"
# save_name="qwen25-ins-7b-coderm-7b-reinforce++"
save_name="qwen25-coder-base-7b--reinforce++_v2_mini_processed_r1"
mkdir -p logs

all_remote_rm_urls="rule:http://localhost:$rm_port/get_reward,rule:http://localhost:$rm_format_port/get_reward"

binary_reward=False # whether to map rewards to 1 or 0 by "1 if reward==1 else 0"
post_args=""
if [ $binary_reward = True ]; then
   post_args="--binary "
   save_name=$save_name"-binary"
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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "'$working_dir'"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --colocate_actor_ref \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain $policy_pretrain \
   --reward_pretrain CodeDPO/qwen_coder_2.5_rm_openrlhf \
   --value_head_prefix "score" \
   --save_path $working_dir/saves/checkpoint/$save_name \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 1000000 \
   --generate_max_len 2048 \
   --num_episodes 3 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data $dataset \
   --input_key context_messages \
   --apply_chat_template \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 10 \
   --ckpt_path $working_dir/saves/ckpt/$save_name \
   --flash_attn \
   --use_wandb $WANDB_API_KEY \
   --remote_rm_url $all_remote_rm_urls \
   --wandb_run_name $save_name

   # --normalize_reward \
# also supports --advantage_estimator rloo

pkill -P $PID_RM
kill -9 $PID_RM
pkill -P $PID_RM2
kill -9 $PID_RM2