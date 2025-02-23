set -x
# 32768
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset CodeDPO/AceCoder-SFT-500-DeepSeek-R1 \
   --input_key messages \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --apply_chat_template \
   --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
   --save_path ./checkpoint/cold_start_sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 10 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --ring_attn_size 4 \
   --ring_head_stride 2 \
   --packing_samples \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name cold_start_sft \

EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi