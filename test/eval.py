import datasets, json
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model_name = "gpt2-backwards/corpus15"

predictions = []
references = []

with open(f"/scratch/network/pvegna/backwardsLM/output/{model_name}.json", "r") as out_file:
    data = out_file.readlines()

for ex in data:
    ex = json.loads("ex")
    predictions.append(ex['pred'])
    references.append(ex['ref'])

bertscore = datasets.load_metric("/scratch/network/pvegna/backwardsLM/metrics/bertscore.py")
results = bertscore.compute(predictions=predictions, references=references, lang="en",
                            device=device,
                            model_type="/scratch/network/pvegna/models/roberta-large/",
                            num_layers=17)
with open(f'/scratch/network/pvegna/backwardsLM/output/{model_name}/bertscore.log', 'w') as log_file:
    output = {'model': model_name, 'precision': results['precision'].mean(),
              'recall': results['recall'].mean(), 'f1': results['f1'].mean()}
    log_file.write(json.dumps(output) + '\n')