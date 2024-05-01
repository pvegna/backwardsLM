import json
with open('persuasion.txt', 'r', encoding='utf-8') as in_file:
    data = in_file.read()

with open('temp_fwd.json', 'w', encoding='utf-8') as out_file:
    data = data.split('\n\n')
    for paragraph in data:
        text = paragraph.replace('\n', ' ')
        if text.strip() != '':
            #ex = {'text': text[::-1]}
            ex = {'text': text}
            out_file.write(json.dumps(ex) + '\n')