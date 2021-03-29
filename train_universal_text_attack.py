'''
Universal attack where a k-length sequence of words is added to each utterance
The phrase is learnt using a greedy search
'''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep_attack import get_data
import sys
import os
import argparse
from eval_universal_text_attack import eval
from models import BERTGrader
from tools import AverageMeter, get_default_device, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg

def get_avg(model, data_file, grades_file, attack_phrase, batch_size):

    input_ids, mask, labels = get_data(data_file, grades_file, attack_phrase)
    ds = TensorDataset(input_ids, mask, labels)
    dl = DataLoader(ds, batch_size=batch_size)

    mse, pcc, less05, less1, avg = eval(dl, model)
    return avg

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('VOCAB', type=str, help='vocab file')
    commandLineParser.add_argument('PREV_ATTACK', type=str, help='greedy universal attack phrase')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_file = args.TEST_DATA
    grades_file = args.TEST_GRADES
    vocab_file = args.VOCAB
    prev_attack_phrase = args.PREV_ATTACK
    batch_size = args.B

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_universal_text_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))

    # Get list of words to try
    with open(vocab_file, 'r') as f:
        test_words = f.readlines()
    test_words = [str(word.strip('\n')).lower() for word in test_words]

    # Add blank word at beginning of list
    test_words = ['']+test_words

    print("Base Phrase: ", prev_attack_phrase)
    print()

    best = (None, 0)
    for word in test_words:
        attack_phrase = prev_attack_phrase + ' ' + word
        avg = get_avg(model, data_file, grades_file, attack_phrase, batch_size)
        print(word, avg)

        if avg > best[1]:
            best = (word, avg)

    print("The best overall is", best)
