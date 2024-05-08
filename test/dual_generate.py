import datasets, json
from torch import cuda
from evaluate import load
from transformers import pipeline
device = 'cuda' if cuda.is_available() else 'cpu'
model_name = ""

predictions = []
references = []

model_name = "gpt2-backwards/corpus15"
back_gen = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-backwards/corpus15", 
                    tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")
front_gen = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-forwards/corpus15", 
                    tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")

with open(f"/scratch/network/pvegna/backwardsLM/data/test_fwd.json", "r", encoding="utf-8") as in_file:
    data = in_file.readlines()

with open(f"/scratch/network/pvegna/backwardsLM/output/dual.json", "w") as out_file:
    for i in range(500):
        ex = data[i]
        ex = json.loads(ex)
        text = ex["text"]
        if len(text) <= 3:
            continue
        split = int(0.5 * len(text))
        fwd_prompt = text[:split]
        fwd_ref = text[split:]
        bwd_prompt = text[-split:][::-1]
        bwd_ref = text[:-split][::-1]
        bwd_pred = back_gen(bwd_prompt)[0]['generated_text']
        fwd_pred = front_gen(bwd_prompt)[0]['generated_text']
        out_file.write(json.dumps({'fwd_prompt': fwd_prompt, 'fwd_ref': fwd_ref, 'fwd_pred': fwd_pred,
                                   'bwd_prompt': bwd_prompt, 'bwd_ref': bwd_ref, 'bwd_pred': bwd_pred}) + '\n')
        print(i)
