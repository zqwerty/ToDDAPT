import json
from value_format import formatting

source_files = ["data_fixed_replace_coreference.json"]
val_list_file = "valListFile.json"
test_list_file = "testListFile.json"
target_files = ["../train.tsv", "../dev.tsv", "../test.tsv"]

### Read ontology file
fp_ont = open("../ontology.json", "r")
data_ont = json.load(fp_ont)
ontology = {}
for domain_slot in data_ont:
    domain, slot = domain_slot.split('-')
    if domain not in ontology:
        ontology[domain] = {}
    ontology[domain][slot] = {}
    for value in data_ont[domain_slot]:
        ontology[domain][slot][value] = 1
fp_ont.close()


def read_file_list(split):
    assert split in ['train', 'val', 'test']
    return {line.strip() + '.json' for line in open(f'{split}ListFile')}


# Read file list (dev and test sets are defined)
train_file_list = read_file_list('train')
dev_file_list = read_file_list('val')
test_file_list = read_file_list('test')

# Read woz logs and write to tsv files

fp_train = open("../train.tsv", "w")
fp_dev = open("../dev.tsv", "w")
fp_test = open("../test.tsv", "w")

fp_train.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
fp_dev.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
fp_test.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')

for domain in sorted(ontology.keys()):
    for slot in sorted(ontology[domain].keys()):
        fp_train.write(str(domain) + '-' + str(slot) + '\t')
        fp_dev.write(str(domain) + '-' + str(slot) + '\t')
        fp_test.write(str(domain) + '-' + str(slot) + '\t')

fp_train.write('\n')
fp_dev.write('\n')
fp_test.write('\n')

fp_data = open("data_fixed_replace_coreference.json", "r")
data = json.load(fp_data)

ds_list = []

for file_id in data:
    if file_id in dev_file_list:
        fp_out = fp_dev
    elif file_id in test_file_list:
        fp_out = fp_test
    else:
        assert file_id in train_file_list
        fp_out = fp_train

    user_utterance = ''
    system_response = ''
    turn_idx = 0
    for idx, turn in enumerate(data[file_id]['log']):
        if idx % 2 == 0:  # user turn
            user_utterance = data[file_id]['log'][idx]['text']
        else:  # system turn
            user_utterance = user_utterance.replace('\t', ' ')
            user_utterance = user_utterance.replace('\n', ' ')
            user_utterance = user_utterance.replace('  ', ' ')

            system_response = system_response.replace('\t', ' ')
            system_response = system_response.replace('\n', ' ')
            system_response = system_response.replace('  ', ' ')

            fp_out.write(str(file_id))  # 0: dialogue ID
            fp_out.write('\t' + str(turn_idx))  # 1: turn index
            fp_out.write('\t' + str(user_utterance))  # 2: user utterance
            fp_out.write('\t' + str(system_response))  # 3: system response

            belief = {}
            for domain in data[file_id]['log'][idx]['metadata'].keys():
                for slot in data[file_id]['log'][idx]['metadata'][domain]['semi'].keys():
                    value = data[file_id]['log'][idx]['metadata'][domain]['semi'][slot].strip()
                    value = value.lower()
                    if value == '' or value == 'not mentioned' or value == 'not given' or value == 'not':
                        value = 'none'

                    if slot == "leaveAt":
                        slot = "leaveAt"
                    elif slot == "arriveBy" and domain != "bus":
                        slot = "arriveBy"
                    elif slot == "pricerange":
                        slot = "pricerange"

                    if value == "doesn't care" or value == "don't care" or value == "do not care" or value == "does not care" or value == 'does' or value == "'do n't":
                        value = "dontcare"
                    elif value == "guesthouse" or value == "guesthouses" or value == 'guest house' or value == 'guest houses':
                        value = "guesthouse"
                    elif value == "city center" or value == "town centre" or value == "town center" or \
                            value == "centre of town" or value == "center" or value == "center of town":
                        value = "centre"
                    elif value == 'a':
                        value = 'a and b guest house'
                    elif value == 'cherr':
                        value = 'cherry hinton village centre'
                    elif value == 'cityr':
                        value = 'cityroomz'
                    elif value == 'cam' or value == 'can' or value == 'dif' or value == 'ha ha' or value == 'the':
                        value = 'none'
                    elif value == 'musuem':
                        value = 'museum'
                    elif value == 'nil':
                        value = 'the cow pizza kitchen and bar'
                    elif value == 'india west':
                        value = 'india house'
                    elif value == 'nstaot mentioned':
                        value = 'the cambridge book and print gallery'
                    elif value == 'zizzi':
                        value = 'zizzi cambridge'
                    elif value == 'yippe noodle bar' or value == 'yippee' or \
                            value == 'yippee noolde bar' or value == "yippee noodle bar 's":
                        value = 'yippee noodle bar'
                    value = formatting(domain, slot, value)
                    if domain not in ontology:
                        print("domain (%s) is not defined" % domain)
                        continue

                    if slot not in ontology[domain]:
                        # print("slot (%s) in domain (%s) is not defined" % (slot, domain))   # bus-arriveBy not defined
                        continue

                    if value not in ontology[domain][slot] and value != 'none':
                        print("%s: value (%s) in domain (%s) slot (%s) is not defined in ontology" %
                              (file_id, value, domain, slot))
                        value = 'none'

                    belief[str(domain) + '-' + str(slot)] = value
                    ds_list.append(str(domain) + '-' + str(slot))

                for slot in data[file_id]['log'][idx]['metadata'][domain]['book'].keys():
                    if slot == 'booked':
                        continue
                    if domain == 'bus' and slot == 'people':
                        continue  # not defined in ontology

                    value = data[file_id]['log'][idx]['metadata'][domain]['book'][slot].strip()
                    value = value.lower()

                    if value == '' or value == 'not mentioned' or value == 'not given' or \
                            value == 'cam' or value == 'can' or value == 'dif' or value == 'ha ha' or value == 'the':
                        value = 'none'
                    elif value == "doesn't care" or value == "don't care" or value == "do not care" or value == "does not care" or value == 'does' or value == "'do n't":
                        value = "dontcare"

                    if str('book ' + slot) not in ontology[domain]:
                        print("book %s is not defined in domain %s" % (slot, domain))
                        continue

                    if value not in ontology[domain]['book ' + slot] and value != 'none':
                        print("%s: value (%s) in domain (%s) slot (book %s) is not defined in ontology" %
                              (file_id, value, domain, slot))
                        value = 'none'

                    belief[str(domain) + '-book ' + str(slot)] = value
                    ds_list.append(str(domain) + '-book ' + str(slot))

            for domain in sorted(ontology.keys()):
                for slot in sorted(ontology[domain].keys()):
                    key = str(domain) + '-' + str(slot)
                    if key in belief:
                        fp_out.write('\t' + belief[key])
                    else:
                        fp_out.write('\tnone')

            fp_out.write('\n')
            fp_out.flush()

            system_response = data[file_id]['log'][idx]['text']
            turn_idx += 1

print("domain and slot list: ", set(ds_list))
print(sorted(set(ds_list)))
print(len(set(ds_list)))
print(list(sorted(set(ds_list))) == list(sorted(list(data_ont.keys()))))
fp_train.close()
fp_dev.close()
fp_test.close()
