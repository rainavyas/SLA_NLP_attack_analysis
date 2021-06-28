'''
Use training data to define chosen embedding space eigenvectors
For original and attacked test data determine the average size of the components in the eigenvector directions
Plot this against eigenvalue rank
'''

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tools import get_default_device
import matplotlib.pyplot as plt
from models import BERTGrader
from pca_component_comparison_plot_comps import get_head_embedding

def avg_stds(original, attack):
    '''
    original: Tensor [num_ranks x num_data_points]
    attack: Tensor [num_ranks x num_data_points]

    Return:
        The average (across rank) of the number of
        standard deviations between original and
        attack at each rank position

    This gives a measure of how out of distribution
    the attack is from the original
    '''
    with torch.no_grad():
        original_mean = torch.mean(original, dim=1)
        attack_mean = torch.mean(attack, dim=1)
        original_std = torch.std(original, dim=1)
        diff = torch.abs(attack_mean - original_mean)
        std_diff = diff/original_std
        return torch.mean(std_diff)

def get_all_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector (rank), calculates the
    magnitude of components in that direction for each
    data point

    Returns:
        Tensor [num_ranks, num_data_points]
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.abs(torch.einsum('bi,ji->bj', X, eigenvectors))
    return torch.transpose(comps, 0, 1)

def get_avg_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector, calculates average (across batch)
    magnitude of components in that direction
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.einsum('bi,ji->bj', X, eigenvectors)
        # Get average of magnitude for each eigenvector rank
        avg_comps = torch.mean(torch.abs(comps), dim=0)
    return avg_comps


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('ATTACK', type=str, help='universal attack phrase for eval set')
    commandLineParser.add_argument('--head_num', type=int, default=1, help="Head embedding to analyse")
    commandLineParser.add_argument('--N', type=int, default=3, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_data_file = args.TEST_DATA
    test_grades_file = args.TEST_GRADES
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    out_file = args.OUT_FILE
    attack_phrase = args.ATTACK
    head_num = args.head_num
    N = args.N
    cpu_use = args.cpu

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/average_comp_dist.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the Sentiment Classifier model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the eigenvectors for PCA decomposition and the correction mean
    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    # Load the test data and get embeddings
    original_embeddings, _ = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase='', head_num=head_num)
    attack_embeddings, _  = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase=attack_phrase, head_num=head_num)
    print("Got embeddings")

    # Get average components against rank
    original_avg_comps = get_avg_comps(original_embeddings, eigenvectors, correction_mean)
    attack_avg_comps = get_avg_comps(attack_embeddings, eigenvectors, correction_mean)

    # Plot the results
    ranks = np.arange(len(original_avg_comps))
    plt.plot(ranks, original_avg_comps, label='Original')
    plt.plot(ranks, attack_avg_comps, label='Attacked', alpha=0.75)
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Average Component Size')
    plt.legend()
    plt.savefig(out_file)

    # Report std diff between attack and original curves
    original_comps = get_all_comps(original_embeddings, eigenvectors, correction_mean)
    attack_comps = get_all_comps(attack_embeddings, eigenvectors, correction_mean)

    print("OOD metric", avg_stds(original_comps, attack_comps))
