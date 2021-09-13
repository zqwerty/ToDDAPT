import json

with open('original/data_meta_fixed.json') as f:
    data = json.load(f)

json_file_list = []
for item in data:
    json_file_list.append(item)
len(json_file_list)
count = 0
coreference_list_ID = []

for file in json_file_list:
    onefile = data[file]['log']
    
    for idx,item in enumerate(onefile):
        if 'coreference' in item:
            coreference_list = list(item['coreference'].values())[0]
            
            print(file)
            coreference_list_ID.append(file)
            print(len(coreference_list))
            temp_str = item['text']
            for i in range(len(coreference_list)):
                temp_str = temp_str.replace(coreference_list[i][1],'%s which is %s' % (coreference_list[i][1],coreference_list[i][2]))
            
            print(data[file]['log'][idx]['text'])
            print(temp_str)
            data[file]['log'][idx]['text'] = temp_str
            print('====================================')
            count += 1
print(count)


with open('original/data_fixed_replace_coreference.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
