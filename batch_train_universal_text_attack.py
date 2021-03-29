import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep_attack import get_data
import sys
import os
import argparse
from tools import AverageMeter, get_default_device, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg
from models import BERTGrader
from eval_universal_text_attack import eval
import json
from datetime import date

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('--PREV_ATTACK', type=str, default='' help='greedy universal attack phrase')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--SEARCH_SIZE', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--START', type=int, default=0, help='Batch number')


    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_file = args.TEST_DATA
    grades_file = args.TEST_GRADES
    vocab_file = args.VOCAB
    log_file = args.LOG
    prev_attack_phrase = args.PREV_ATTACK
    batch_size = args.B
    search_size = args.SEARCH_SIZE
    start = args.START

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/batch_train_universal_text_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))

    # Get list of words to try
    with open(vocab_file, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = start*search_size
    test_words = test_words[start_index:start_index+search_size]

    # Add blank word at beginning of list
    test_words = ['']+test_words

    # Initialise empty log file
    with open(log_file, 'w') as f:
        f.write("Logged on "+ str(date.today()))


    best = ('none', 0)
    for word in test_words:
        attack_phrase = prev_attack_phrase + ' ' + word
        avg = get_avg(model, data_file, grades_file, attack_phrase, batch_size)
        print(word, avg)

        if avg > best[1]:
            best = (word, avg)
            # Write to log
            with open(log_file, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)
