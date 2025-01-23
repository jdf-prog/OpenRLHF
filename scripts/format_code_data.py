
import fire
import datasets

def main(
    dataset_path="CodeDPO/codedpo_20241208",
    output_path="CodeDPO/codedpo_20241208_openrlhf_format",
):
    dataset = datasets.load_dataset(dataset_path)
    
    def map_openrlhf_format(item):
        item['context_messages'] = [
            {
                "content": item['question'],
                "role": "user"
            }
        ]
        return item

    dataset = dataset.map(map_openrlhf_format)
    dataset.push_to_hub(output_path)
    print(f"Saved to {output_path}")
    
if __name__ == "__main__":
    fire.Fire(main)