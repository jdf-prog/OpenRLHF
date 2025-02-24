huggingface-cli upload --repo-type model {hf_model_id} {local_path} {path_in_repo(default .)}

huggingface-cli upload --repo-type model CodeDPO/qwen2.5-coder-inst-cold-start-R1 /data/dongfu/AceCoder/train/train_rl/OpenRLHF/checkpoint/cold_start_sft .