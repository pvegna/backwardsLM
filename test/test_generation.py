from transformers import pipeline
target = "ot demees meht fo yna ro toillE rM erofeb ,elat emas eht gnillet ecnatsid a ta draeh eb ot gninnigeb saw namhctaw eht dna \u201d,sdnuos revlis sti htiw nevele\u201c kcurts dah eceip-letnam eht no kcolc elttil tnagele ehT .meht htiw ruoh na diats eH"
prompt = ".gnol ereht neeb dah eh taht leef "
forwards = prompt + target
forwards = forwards[::-1]
generator = pipeline("text-generation", model="/scratch/network/pvegna/models/gpt2-backwards/checkpoint-1040", tokenizer="/scratch/network/pvegna/models/gpt2-tokenizer")
gen = generator(prompt)[0]['generated_text']

print("raw target: " + target)
print("raw generation: " + gen)
print("forwards target: " + forwards)
print("forwards generation: " + (prompt + gen)[::-1])