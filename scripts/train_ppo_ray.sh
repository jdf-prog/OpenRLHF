set -x 
working_dir=$PWD
# ppo

policy_pretrain=Qwen/Qwen2.5-7B-Instruct
# reward_pretrain=CodeDPO/qwen_coder_2.5_rm_openrlhf
reward_pretrain=CodeDPO/Qwen2.5-Coder-7B-new_with_margin_scalebt
run_name=qwen25-ins-7b-coderm-7b-ppo
# dataset="CodeDPO/codedpo_20241208_openrlhf_format_hard" # old dataset where test cases are not filterd by Qwen2.5-Coder-32B
dataset="CodeDPO/rlhf_dataset_20250126_openrlhf_format_hard_r1" # new dataset where test cases are filterd by Qwen2.5-Coder-32B

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "'$working_dir'"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain $policy_pretrain \
   --reward_pretrain $reward_pretrain \
   --value_head_prefix "score" \
   --save_path $working_dir/saves/checkpoint/$run_name \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 4 \
   --max_samples 1000000 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data $dataset \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --gradient_checkpointing \
   --load_checkpoint \
   --save_steps 10 \
   --ckpt_path $working_dir/saves/ckpt/$run_name \
   --flash_attn \
   --use_wandb $WANDB_API_KEY \

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)