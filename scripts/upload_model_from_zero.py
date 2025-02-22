from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
import torch

def main(
    model_path:str="Qwen/Qwen2.5-Coder-7B-Instruct",
    hf_model_id:str="CodeDPO/qwen25-ins-7b-coderm_new_margin_scalebt-7b-reinforce-plus-episode_1",
    model_pytorch_bin:str="/root/dongfu/OpenRLHF/saves/ckpt/qwen25-ins-7b-coderm_new_margin_scalebt-7b-reinforce++/_actor/global_step_80_pytorch_model.bin",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # laod model from pytorch bin
    print(f"load model from {model_pytorch_bin}")
    model.load_state_dict(torch.load(model_pytorch_bin), strict=True)
    print("load model from pytorch bin done")
    
    # save model to huggingface
    model.push_to_hub(hf_model_id)
    tokenizer.push_to_hub(hf_model_id)

if __name__ == "__main__":
    fire.Fire(main)
    