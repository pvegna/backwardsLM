import random
with open('temp.json', 'r') as in_file:
    data = in_file.readlines()

random.shuffle(data)
train = data[:int(len(data) * 0.8)]
test = data[int(len(data) * 0.8):]
with open('train.json', 'w') as out_file:
    for ex in train:
        out_file.write(ex)

with open('test.json', 'w') as out_file:
    for ex in test:
        out_file.write(ex)