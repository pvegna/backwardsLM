from transformers import pipeline
target = "ot demees meht fo yna ro toillE rM erofeb ,elat emas eht gnillet ecnatsid a ta draeh eb ot gninnigeb saw namhctaw eht dna \u201d,sdnuos revlis sti htiw nevele\u201c kcurts dah eceip-letnam eht no kcolc elttil tnagele ehT .meht htiw ruoh na diats eH"
prompt = ".gnol ereht neeb dah eh taht leef "
fwd_target = prompt + target
fwd_target = fwd_target[::-1]
generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-backwards/checkpoint-1040", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")
gen = generator(prompt)[0]['generated_text']
fwd_gen = gen[::-1]
print("raw target: " + target)
print('------------------------------------')
print("raw generation: " + gen)
print('------------------------------------')
print("forwards target: " + fwd_target)
print('------------------------------------')
print("forwards generation: " + fwd_gen)
print('------------------------------------')