with open('gpt2-backwards\corpus15.json', 'r') as in_file:
    data = in_file.readlines()
import json
with open('gpt2-backwards\corpus15-to-fwd.json', 'w') as out_file:
    for ex in data:
        ex = json.loads(ex)
        for k in ex.keys():
            ex[k] = ex[k][::-1]
        out_file.write(json.dumps(ex) + '\n')