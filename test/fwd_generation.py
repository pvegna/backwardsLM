from transformers import pipeline
target = "Husbands and wives generally understand when opposition will be vain. Mary knew, from Charles\u2019s manner of speaking, that he was quite determined on going, and that it would be of no use to teaze him. She said nothing, therefore, till he was out of the room, but as soon as there was only Anne to hear\u2014"
prompt = ""

generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-forwards/checkpoint-5200", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")

with open('base_generations.txt', 'w') as out_file:
    for i in range(10):
        target = target.strip()
        prompt += ' ' + target[:target.index(' ')]
        target = target[target.index(' '):]
        fwd_target = prompt + target
        gen = generator(prompt)[0]['generated_text']
        out_file.write('\n\n------------------------------------')
        out_file.write("\n\nprompt: " + prompt)
        out_file.write("\n\nraw target: " + target)
        out_file.write("\n\nraw generation: " + gen)