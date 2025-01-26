pretrain=Qwen/Qwen2.5-Coder-7B
dataset_path=CodeDPO/open_rlhf_dpo_qwen_coder_2.5_inf_20250124
run_name=Qwen2.5-Coder-7B
binary=False
margin_loss=True
loss_type=scalebt # or sigmoid
post_args=""
if [ $binary = True ]; then
    dataset_path=./data/$dataset_path"_binarized.json"
    run_name=$run_name"_binarized_"$loss_type
fi
if [ $margin_loss = True ]; then
    dataset_path=./data/$dataset_path"_with_margin.json"
    post_args=$post_args"--margin_loss "
    run_name=$run_name"_with_margin_"$loss_type
fi
# cannot be True if binary is True, becuase it's meaningless, binary pairs has fixed margin, 
if [ "$binary" = "True" ] && [ "$margin_loss" = "True" ]; then
    echo "Error: binary and margin_loss cannot be True at the same time, becuase it's meaningless, binary pairs has fixed margin, "
    exit 1
fi

echo "pretrain: $pretrain"
echo "dataset_path: $dataset_path"
echo "run_name: $run_name"

deepspeed --include=localhost:0,7 --module  openrlhf.cli.train_rm \
    --save_path ./checkpoint/$pretrain/$run_name \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain $pretrain \
    --bf16 \
    --max_epochs 1 \
    --max_len 8192 \
    --zero_stage 3 \
    --learning_rate 2e-5 \
    --dataset $dataset_path \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --flash_attn \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name $run_name \
    --loss $loss_type \
    $post_args
    