from transformers import pipeline
#target = ".gnol ereht neeb dah eh taht leef ot demees meht fo yna ro toillE rM erofeb ,elat emas eht gnillet ecnatsid a ta draeh eb ot gninnigeb saw namhctaw eht dna \u201d,sdnuos revlis sti htiw nevele\u201c kcurts dah eceip-letnam eht no kcolc elttil tnagele ehT .meht htiw ruoh na diats eH"
target = ".niaga mih evael reven dluohs I ,derewsna I ,traeh nwo ym detlusnoc ylno I fI"
prompt = ""

generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-backwards/corpus15", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")
#generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")

with open('/scratch/network/pvegna/backwardsLM/output/c15_bwd_generations.txt', 'w') as out_file:
    for i in range(10):
        target = target.strip()
        prompt += ' ' + target[:target.index(' ')]
        target = target[target.index(' '):]
        fwd_target = prompt + target
        fwd_target = fwd_target[::-1]
        gen = generator(prompt)[0]['generated_text']
        fwd_gen = gen[::-1]
        out_file.write('\n\n------------------------------------')
        out_file.write("\n\nprompt: " + prompt)
        out_file.write("\n\nraw target: " + target)
        out_file.write("\n\nraw generation: " + gen)
        out_file.write("\n\nforwards target: " + fwd_target)
        out_file.write("\n\nforwards generation: " + fwd_gen)