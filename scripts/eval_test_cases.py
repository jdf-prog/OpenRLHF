from openrlhf.cli.eval_test_cases import evaluate
from tqdm import tqdm
import datasets
dataset = datasets.load_dataset("CodeDPO/rlhf_dataset_20250125", split='train')

samples = []
for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
    all_inferences = eval(item['inference'])
    item_id_keys = ["id", "question_id", "task_id"]
    item_id = i
    for key in item_id_keys:
        if key not in item:
            continue
        item_id = item[key]
        break
    for j, inference in enumerate(all_inferences):
        samples.append({
            'task_id': item_id,
            'prompt': item['question'],
            'output': inference[0],
            'tests': item['tests'],
            '_identifier': f"{item_id}_{j}",
            "ori_pass_rate": inference[1]
        })

print("samples:", len(samples))

batch_size=100000
all_pass_rates = []
all_outputs = []
for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating samples"):
    batch_samples = samples[i:i+batch_size]
    outputs, pass_rates = evaluate(samples=batch_samples, dataset=None, test_details=True, n_workers=None, extract_solution=False)
    all_outputs.extend(outputs)
    all_pass_rates.extend(pass_rates)

