'''
Same as batch_train_universal_text_attack.py with the
exception that attack words that result in detection
by the N-length classifier at head1 layer (400 comps)
are rejected
'''
from pca_component_comparison_plot_comps import get_head_embedding
import torch
import torch.nn as nn
from data_prep_attack import get_data
import sys
import os
import argparse
from tools import accuracy_topk
from models import BERTGrader
from train_universal_text_attack import get_avg
import json
from datetime import date
from linear_pca_classifier import LayerClassifier, get_pca_principal_components
from pca_component_comparison_plot_comps import get_head_embedding
from linear_pca_classifier import get_pca_principal_components

def is_suppressed(model, data_file, grades_file, attack_phrase, detector_path, eigenvectors_path, correction_mean_path, num_comps=400, detect_prob=0.75):
    '''
    Returns true if trained detector doesn't detect the adversarial attack.
    Assumes detector use first 'num_comps=400' PCA components for classification
    detect_prob means for TRUE to be returned at least (1-detect_prob)% of samples avoid detection 
    '''
    # Load adv attack detector
    detector = LayerClassifier(num_comps)
    detector.load_state_dict(torch.load(detector_path, map_location=torch.device('cpu')))

    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    embeddings, _ = get_head_embedding(data_file, grades_file, model, attack_phrase=attack_phrase, head_num=1)
    comps = get_pca_principal_components(eigenvectors, correction_mean, embeddings, num_comps, 0)

    # Attempt to detect adv attack
    with torch.no_grad():
        preds = detector(comps)
        adv_labels = torch.LongTensor([1]*len(comps))
    adv_detection_accuracy = accuracy_topk(preds, adv_labels)/100

    if adv_detection_accuracy < detect_prob:
        return True 
    else:
        return False



if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('DETECTOR', type=str, help='Trained .th head1 linear classifier adv detector')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('--PREV_ATTACK', type=str, default='', help='greedy universal attack phrase')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--SEARCH_SIZE', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--START', type=int, default=0, help='Batch number')
    commandLineParser.add_argument('--DETECT_PROB', type=float, default=0.5, help='Detection probability of found universal attack')


    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_file = args.TEST_DATA
    grades_file = args.TEST_GRADES
    vocab_file = args.VOCAB
    log_file = args.LOG
    detector_path = args.DETECTOR
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    prev_attack_phrase = args.PREV_ATTACK
    batch_size = args.B
    search_size = args.SEARCH_SIZE
    start = args.START
    detect_prob = args.DETECT_PROB

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/suppressed_batch_train_universal_text_attack.cmd', 'a') as f:
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

        if avg > best[1] and is_suppressed(model, data_file, grades_file, attack_phrase, detector_path, eigenvectors_path, correction_mean_path, detect_prob=detect_prob):
            best = (word, avg)
            # Write to log
            with open(log_file, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)
