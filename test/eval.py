import evaluate, json, random
from transformers import pipeline
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model_name = "gpt2-backwards/corpus15"
direction = "bwd"
back_gen = pipeline("text-generation", model=f"/scratch/network/pvegna/models/{model_name}", 
                    tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer").to(device)

with open(f"/scratch/network/pvegna/backwardsLM/data/test_{direction}.json", "r", encoding="utf-8") as in_file:
    data = in_file.readlines()

predictions = []
references = []

with open(f"/scratch/network/pvegna/backwardsLM/output/{model_name}.json", "w") as out_file:
    for ex in data:
        ex = json.loads(ex)
        text = ex["text"]
        if len(text) <= 3:
            continue
        split = int(random.uniform(0.33, 0.66) * len(text))
        prompt = text[0:split]
        ref = text[split:]
        pred = back_gen(prompt)[0]['generated_text']
        predictions.append(pred)
        references.append(ref)
        out_file.write(json.dumps({'prompt': prompt, 'ref': ref, 'pred': pred}) + '\n')
        print(pred[::-1])



bertscore = evaluate.load("/scratch/network/pvegna/backwardsLM/metrics/bertscore.py")
results = bertscore.compute(predictions=predictions, references=references, lang="en",
                            device=device,
                            model_type="/scratch/network/pvegna/models/roberta-large/",
                            num_layers=17)
with open('/scratch/network/pvegna/backwardsLM/output/bertscore.log', 'a') as log_file:
    output = {'model': model_name, 'precision': results['precision'].mean(),
              'recall': results['recall'].mean(), 'f1': results['f1'].mean()}
    log_file.write(json.dumps(output) + '\n')
