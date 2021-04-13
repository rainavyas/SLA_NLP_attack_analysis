'''
Same as data_prep_attack.py but returns data with and without attack version
where the data points are in the same order for both sets
'''

import torch
import torch.nn as nn
from transformers import BertTokenizer
import json

def get_spk_to_utt(data_file, attack_phrase):
    # Load the data
    with open(data_file, 'r') as f:
        utterances = json.loads(f.read())

    # Convert json output from unicode to string
    utterances = [[str(item[0]), str(item[1])] for item in utterances]

    # Concatentate utterances of a speaker
    spk_to_utt = {}
    spk_to_utt_attack = {}
    for item in utterances:
        fileName = item[0]
        speakerid = fileName[:12]
        sentence = item[1]
        sentence_attack = item[1]

        # Append attack phrase at the end of the attack sentence
        sentence_attack = sentence_attack + ' ' + attack_phrase

        if speakerid not in spk_to_utt:
            spk_to_utt[speakerid] =  sentence
            spk_to_utt_attack[speakerid] =  sentence_attack
        else:
            spk_to_utt[speakerid] =spk_to_utt[speakerid] + ' ' + sentence
            spk_to_utt_attack[speakerid] =spk_to_utt_attack[speakerid] + ' ' + sentence_attack
    return spk_to_utt, spk_to_utt_attack

def get_spk_to_grade(grades_file, part=3):
    grade_dict = {}

    lines = [line.rstrip('\n') for line in open(grades_file)]
    for line in lines[1:]:
        speaker_id = line[:12]
        grade_overall = line[-3:]
        grade1 = line[-23:-20]
        grade2 = line[-19:-16]
        grade3 = line[-15:-12]
        grade4 = line[-11:-8]
        grade5 = line[-7:-4]
        grades = [grade1, grade2, grade3, grade4, grade5, grade_overall]

        grade = float(grades[part-1])
        grade_dict[speaker_id] = grade
    return grade_dict

def align(spk_to_utt, spk_to_utt_attack, grade_dict):
    grades = []
    utts = []
    utts_attack = []
    for id in spk_to_utt:
        try:
            grades.append(grade_dict[id])
            utts.append(spk_to_utt[id])
            utts_attack.append(spk_to_utt_attack[id])
        except:
            pass
            # print("Falied for speaker " + str(id))
    return utts, utts_attack, grades

def tokenize_text(utts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(utts, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return ids, mask


def get_data(data_file, grades_file, attack_phrase):
    '''
    Prepare data as tensors
    '''
    spk_to_utt, spk_to_utt_attack = get_spk_to_utt(data_file, attack_phrase)
    grade_dict = get_spk_to_grade(grades_file)
    utts, utts_attack, grades = align(spk_to_utt, spk_to_utt_attack, grade_dict)
    ids, mask = tokenize_text(utts)
    ids_attack, mask_attack = tokenize_text(utts_attack)
    labels = torch.FloatTensor(grades)

    return ids, mask, ids_attack, mask_attack, labels
