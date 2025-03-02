import fire
import datasets
from pathlib import Path
def main(
    dataset_path: str="CodeDPO/open_rlhf_dpo_qwen_coder_2.5_inf_20250124",
    split: str="train",
    output_dir="./data/"
):
    output_path = Path(output_dir) / f"{dataset_path}_binarized.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(dataset_path, split=split)
    def filter_non_perfect_positive(item):
        if not item['chosen_score'] == 1:
            return False
        return True
    
    dataset = dataset.filter(filter_non_perfect_positive, load_from_cache_file=False)
    
    def change_negative_to_zero(item):
        if not item['chosen_score'] == 1:
            item['chosen_score'] = 0
        return item

    dataset = dataset.map(change_negative_to_zero, load_from_cache_file=False)
    dataset.to_json(output_path)
    print(f"Saved binarized dataset to {output_path}")
    
if __name__ == "__main__":
    fire.Fire(main)
"""
python scripts/binarize_pref_data.py --dataset_path "CodeDPO/open_rlhf_dpo_qwen_coder_2.5_inf_20250124" --split train --output_dir ./data/
"""