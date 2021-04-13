'''
Plots of upperbound saliency distribution for selected inputs
with and without universal attack phrase appended at end
'''

import torch
import torch.nn as nn
from data_prep_for_saliency import get_data
import sys
import os
import argparse
from models import BERTGrader
import matplotlib.pyplot as plt
from BERT_layer_handler import Layer_Handler
import numpy as np

def get_word_saliencies(model, input_ids, mask, labels, criterion):
    '''
    Assumed all input data is for a batch of sentences, i.e.
        input_ids: [N x max_tokens]
        mask: [N x max_tokens]
        label: [N]

    Return lists:

        saliencies: [item1, item2, ..., itemN]
            where item is:
                list of token embedding saliency size
                truncated to length of number of tokens without padding
    '''
    model.eval()
    handler = Layer_Handler(model, layer_num=0)

    embeddings = handler.get_layern_outputs(input_ids, mask)
    embeddings.retain_grad()
    y = handler.pass_through_rest(embeddings, mask)
    loss = criterion(y, labels)

    loss.backward()
    embedding_grads = embeddings.grad
    saliency_size = torch.linalg.norm(embedding_grads, dim=-1)

    saliencies = []
    for i in range(saliency_size.size(0)):
        sals = saliency_size[i]
        curr_mask = mask[i]
        relevant_sals = sals[curr_mask.type(torch.bool)]
        saliencies.append(relevant_sals.tolist())
    return saliencies

def plot_saliency_comparison(saliencies, saliencies_attacked, filename):

    width = 0.4
    plt.bar(np.arange(len(saliencies)), saliencies, width=width)
    plt.bar(np.arange(len(saliencies_attacked)), saliencies_attacked, width=width)
    plt.xlabel("Token Position")
    plt.ylabel("Saliency")
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DATA', type=str, help='prepped train data file')
    commandLineParser.add_argument('TRAIN_GRADES', type=str, help='train data grades')
    commandLineParser.add_argument('--attack_phrase', type=str, help='universal attack phrase')
    commandLineParser.add_argument('--num_samples', type=int, default=5, help='Number of sentences to do saliency for')

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    train_grades_file = args.TRAIN_GRADES
    attack_phrase = args.attack_phrase
    num_samples = args.num_samples

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/saliency_map.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = torch.nn.MSELoss(reduction='mean')

    # Get the sampled data with and without attack
    input_ids, mask, input_ids_attack, mask_attack, labels = get_data(train_data_file, train_grades_file, attack_phrase=attack_phrase)
    input_ids = input_ids[:num_samples]
    mask = mask[:num_samples]
    input_ids_attack = input_ids_attack[:num_samples]
    mask_attack = mask_attack[:num_samples]
    labels = labels[:num_samples]

    # Get all saliencies
    saliencies = get_word_saliencies(model, input_ids, mask, labels, criterion)
    saliencies_attacked = get_word_saliencies(model, input_ids_attack, mask_attack, labels, criterion)

    # Plot each saliency graph in turn
    for i, sal, sal_a in enumerate(zip(saliencies, saliencies_attacked)):
        filename = 'saliency_map_k'+str(len(attack_phrase))+'_sample'+str(i)
        plot_saliency_comparison(sal, sal_a, filename)
