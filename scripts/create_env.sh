conda create -n openrlhf python=3.10
conda activate openrlhf
pip install -e .[vllm] # if errored, try pip install torch first
pip install -e evalplus