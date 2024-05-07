import json, random
from transformers import pipeline

model_name = "gpt2-forwards/corpus15"
direction = "fwd"
back_gen = pipeline("text-generation", model=f"/scratch/network/pvegna/models/{model_name}", 
                    tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")

with open(f"/scratch/network/pvegna/backwardsLM/data/test_{direction}.json", "r", encoding="utf-8") as in_file:
    data = in_file.readlines()

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
        out_file.write(json.dumps({'prompt': prompt, 'ref': ref, 'pred': pred}) + '\n')
        print(pred[::-1])

