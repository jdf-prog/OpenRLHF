# huggingface-cli upload --repo-type model {hf_model_id} {local_path} {path_in_repo(default .)}
huggingface-cli upload --repo-type model CodeDPO/qwen25-ins-7b-testcaserm-7b-reinforce-plus_new_dataset /root/dongfu/OpenRLHF/saves/checkpoint/qwen25-ins-7b-testcaserm-7b-reinforce++_new_dataset .
# /root/dongfu/OpenRLHF/saves/checkpoint/qwen25-ins-7b-coderm_new_margin_scalebt-7b-reinforce++
# /root/dongfu/OpenRLHF/saves/checkpoint/qwen25-coder-inst-7b-testcaserm2-7b-reinforce++_new_dataset_hard
huggingface-cli upload --repo-type model CodeDPO/qwen25-coder-inst-7b-testcaserm2-7b-reinforce_plus_new_dataset_hard /root/dongfu/OpenRLHF/saves/checkpoint/qwen25-coder-inst-7b-testcaserm2-7b-reinforce++_new_dataset_hard .