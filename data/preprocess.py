import os
import json
 
# iterate over files in
# that directory
data = []
directory = 'data'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    with open(f, 'r', encoding='utf-8') as in_file:
        data.append(in_file.read())

with open('temp_bwd.json', 'w', encoding='utf-8') as out_file:
    for d in data:
        d = d.split('\n\n')
        for paragraph in d:
            text = paragraph.replace('\n', ' ')
            if text.strip() != '':
                ex = {'text': text[::-1]}
                #ex = {'text': text}
                out_file.write(json.dumps(ex) + '\n')