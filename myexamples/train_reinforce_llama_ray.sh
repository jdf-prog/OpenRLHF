set -x

working_dir=$PWD
# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "'$working_dir'"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --reward_pretrain CodeDPO/qwen_coder_2.5_rm \
   --value_head_prefix "summary" \
   --save_path $working_dir/examples/test_scripts/checkpoint/llama3.1-8b-coderm-7b-reinforce++ \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data CodeDPO/codedpo_20241208_openrlhf_format \
   --input_key context_messages \
   --apply_chat_template \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --ckpt_path $working_dir/examples/test_scripts/ckpt/llama3.1-8b-coderm-7b-reinforce++ \
   --flash_attn \
   --colocate_actor_ref \
   --use_wandb

   # --normalize_reward \
# also supports --advantage_estimator rloo