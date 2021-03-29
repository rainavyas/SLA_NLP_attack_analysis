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
    for item in utterances:
        fileName = item[0]
        speakerid = fileName[:12]
        sentence = item[1]

        # Append attack phrase at the end of the sentence
        sentence = sentence + ' ' + attack_phrase

        if speakerid not in spk_to_utt:
            spk_to_utt[speakerid] =  sentence
        else:
            spk_to_utt[speakerid] =spk_to_utt[speakerid] + ' ' + sentence
    return spk_to_utt

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

def align(spk_to_utt, grade_dict):
    grades = []
    utts = []
    for id in spk_to_utt:
        try:
            grades.append(grade_dict[id])
            utts.append(spk_to_utt[id])
        except:
            print("Falied for speaker " + str(id))
    return utts, grades

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
    spk_to_utt = get_spk_to_utt(data_file)
    grade_dict = get_spk_to_grade(grades_file)
    utts, grades = align(spk_to_utt, grade_dict)
    ids, mask = tokenize_text(utts)
    labels = torch.FloatTensor(grades)

    return ids, mask, labels
