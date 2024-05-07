import datasets, json
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model_name = "gpt2-forwards/corpus15"

predictions = []
references = []

with open(f"/scratch/network/pvegna/backwardsLM/output/{model_name}.json", "r") as out_file:
    data = out_file.readlines()

for ex in data:
    ex = json.loads(ex)
    predictions.append(ex['pred'])
    references.append(ex['ref'])

print(references)

bertscore = datasets.load_metric("/scratch/network/pvegna/backwardsLM/metrics/bertscore.py")
#ppl = datasets.load_metric("/scratch/network/pvegna/backwardsLM/metrics/perplexity.py")
ppl = datasets.load_metric("perplexity")
results = bertscore.compute(predictions=predictions, references=references, lang="en",
                            device=device,
                            model_type="/scratch/network/pvegna/models/roberta-large/",
                            num_layers=17)
ppl_results = ppl.compute(predictions=predictions, model_id=f"/scratch/network/pvegna/models/{model_name}", 
                          device=device)
p =  sum(results['precision']) / len( results['precision'])
r = sum(results['recall']) / len(results['recall'])
f1 = sum(results['f1']) / len(results['f1'])
with open(f'/scratch/network/pvegna/backwardsLM/output/{model_name}/metrics.log', 'w') as log_file:
    output = {'model': model_name, 'bert_precision': p,
              'bert_recall': r, 'bert_f1': f1, "perplexity": ppl_results['mean_perplexity']}
    log_file.write(json.dumps(output) + '\n')
