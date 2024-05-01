from transformers import pipeline
target = ".gnol ereht neeb dah eh taht leef ot demees meht fo yna ro toillE rM erofeb ,elat emas eht gnillet ecnatsid a ta draeh eb ot gninnigeb saw namhctaw eht dna \u201d,sdnuos revlis sti htiw nevele\u201c kcurts dah eceip-letnam eht no kcolc elttil tnagele ehT .meht htiw ruoh na diats eH"
prompt = ""

generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-backwards/checkpoint-5200", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")

with open('example_generations.txt', 'w') as out_file:
    for i in range(10):
        target = target.strip()
        prompt += target[:target.index(' ')]
        fwd_target = prompt + target
        fwd_target = fwd_target[::-1]
        gen = generator(prompt)[0]['generated_text']
        fwd_gen = gen[::-1]
        out_file.write('------------------------------------')
        out_file.write("\nraw target: " + target)
        out_file.write("\nraw generation: " + gen)
        out_file.write("\nforwards target: " + fwd_target)
        out_file.write("\nforwards generation: " + fwd_gen)