import fire
import datasets
from pathlib import Path
def main(
    dataset_path: str="CodeDPO/open_rlhf_dpo_qwen_coder_2.5_inf_20250124",
    split: str="train",
    output_dir="./data/"
):
    output_path = Path(output_dir) / f"{dataset_path}_with_margin.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(dataset_path, split=split)
    def compute_margin(item):
        item['margin'] = (item['chosen_score'] - item['rejected_score']) * 3
        return item
    
    dataset = dataset.map(compute_margin, desc="Computing margin", load_from_cache_file=False)
    
    dataset.to_json(output_path)
    print(f"Saved binarized dataset to {output_path}")
    
if __name__ == "__main__":
    fire.Fire(main)
"""
python scripts/binarize_pref_data.py --dataset_path "CodeDPO/open_rlhf_dpo_qwen_coder_2.5_inf_20250124" --split train --output_dir ./data/
"""