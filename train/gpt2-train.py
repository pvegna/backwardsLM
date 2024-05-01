from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch import cuda, float16

device = 'cuda' if cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained("/scratch/network/pvegna/models/gpt2", low_cpu_mem_usage=True, torch_dtype=float16)
tokenizer = AutoTokenizer.from_pretrained("/scratch/network/pvegna/models/gpt2-tokenizer")
model = model.to(device)

def preprocess(examples):
    return tokenizer([ex for ex in examples['text']], max_length=1024, truncation=True)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

data = load_dataset('json', data_files={'train':'/scratch/network/pvegna/backwardsLM/data/train.json', 'eval':'/scratch/network/pvegna/backwardsLM/data/test.json'}).shuffle()
data = data.map(preprocess, batched=True, batch_size=16)
#data = data.map(group_texts, batched=True, batch_size=16)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir = "/scratch/network/pvegna/models/gpt2-backwards/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1,
    logging_dir = "/scratch/network/pvegna/logs/gpt2-backwards/",
    logging_steps=20,
    save_strategy="epoch"
)

'''args = TrainingArguments(
    output_dir="/scratch/network/pvegna/models/tot-propose-instruct/",
    per_device_train_batch_size=batch_size,
    learning_rate=5e-5,
    num_train_epochs=30,
    weight_decay=0.1,
    warmup_ratio=.10,
    logging_dir="/scratch/network/pvegna/cryptic/logs/tot-propose-instruct/",
    logging_steps=20,
    save_strategy="steps",
    save_steps=.5
)'''

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["eval"],
    data_collator=data_collator,
)

trainer.train()

import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
