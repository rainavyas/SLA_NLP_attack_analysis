'''
PCA analysis of output of post-encoder mutlihead attention layer
with and without appended adversarial phrases.

The PCA basis vectors are found using authentic samples only

Plot is for a single sample from the test data set
'''

import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep_attack import get_data
import sys
import os
import argparse
from eval_universal_text_attack import eval
from models import BERTGrader
from tools import AverageMeter
from pca_tools import get_covariance_matrix, get_e_v
import matplotlib.pyplot as plt
from attn_layer_analysis import plot_decomposition, plot_pca_whitened_decomposition, get_head_embedding

def get_eigenvector_decomposition_magnitude(eigenvectors, eigenvalues, X, correction_mean):
    '''
    Mean average of magnitude of cosine distance to each eigenevector
    '''
    cos_dists = []
    whitened_cos_dists = []
    ranks = []

    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        cos = CosineSimilarity(dim=1)
        for i in range(eigenvectors.size(0)):
            ranks.append(i)
            v = eigenvectors[i]
            v_repeat = v.repeat(X.size(0), 1)
            abs_cos_dist = torch.abs(cos(X, v_repeat))
            whitened_abs_cos_dist = abs_cos_dist/eigenvalues[i]
            cos_dists.append(torch.mean(abs_cos_dist).item())
            whitened_cos_dists.append(torch.mean(whitened_abs_cos_dist).item())

    return ranks, cos_dists, whitened_cos_dists

def get_head_embedding_sample(data_file, grades_file, model, attack_phrase='', sample_index=27, head_num=1):
    '''
    Gives the output embeddings of chosen head after BERT encoder
    '''
    input_ids, mask, labels = get_data(data_file, grades_file, attack_phrase)
    input_ids = torch.unsqueeze(input_ids[sample_index], 0)
    mask = torch.unsqueeze(mask[sample_index], 0)

    model.eval()
    with torch.no_grad():
        output = model.encoder(input_ids, mask)
        word_embeddings = output.last_hidden_state

        if head_num==1:
            head = model.apply_attn(word_embeddings, mask, model.attn1)
        elif head_num==2:
            head = model.apply_attn(word_embeddings, mask, model.attn2)
        elif head_num==3:
            head = model.apply_attn(word_embeddings, mask, model.attn3)
        elif head_num==4:
            head = model.apply_attn(word_embeddings, mask, model.attn4)
        else:
            raise Exception("Invalid head number")

    return head

def plot_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename, rank_lim=768):
    ranks = ranks[:rank_lim]
    cos_dists_auth = cos_dists_auth[:rank_lim]
    cos_dists_attack = cos_dists_attack[:rank_lim]

    plt.plot(ranks, cos_dists_attack, label="Attacked")
    plt.plot(ranks, cos_dists_auth, label="Original")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Absolute Cosine Distance")
    plt.yscale('log')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_pca_whitened_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename, rank_lim=768):
    ranks = ranks[:rank_lim]
    cos_dists_auth = cos_dists_auth[:rank_lim]
    cos_dists_attack = cos_dists_attack[:rank_lim]

    plt.plot(ranks, cos_dists_attack, label="Attacked")
    plt.plot(ranks, cos_dists_auth, label="Original")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Whitened Absolute Cosine Distance")
    plt.yscale('log')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DATA', type=str, help='prepped train data file')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TRAIN_GRADES', type=str, help='train data grades')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('ATTACK', type=str, help='universal attack phrase')
    commandLineParser.add_argument('--rank_lim', type=int, default=768, help="How many principal ranks to show")
    commandLineParser.add_argument('--total_heads', type=int, default=4, help='analyse how many heads')
    commandLineParser.add_argument('--sample_index', type=int, default=27, help='Random sample from test')

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    attack_phrase = args.ATTACK
    rank_lim = args.rank_lim
    total_heads = args.total_heads
    sample_index = args.sample_index

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attn_layer_analysis_random_data_point.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for head_num in range(1,total_heads+1):
        # Use authentic training data to create eigenvector basis
        auth_embedding = get_head_embedding(train_data_file, train_grades_file, model, attack_phrase='', head_num=head_num)
        correction_mean = torch.mean(auth_embedding, dim=0)
        cov = get_covariance_matrix(auth_embedding)
        e, v = get_e_v(cov)

        # Get authenetic PCA decomposition for test data
        auth_embedding = get_head_embedding_sample(test_data_file, test_grades_file, model, attack_phrase='', sample_index=sample_index, head_num=head_num)
        ranks, cos_dists_auth, whitened_cos_dists_auth = get_eigenvector_decomposition_magnitude(v, e, auth_embedding, correction_mean)

        # Get attacked PCA decomposition for test data
        attack_embedding = get_head_sample(test_data_file, test_grades_file, model, attack_phrase=attack_phrase, sample_index=sample_index, head_num=head_num)
        ranks, cos_dists_attack, whitened_cos_dists_attack = get_eigenvector_decomposition_magnitude(v, e, attack_embedding, correction_mean)

        # Plot the data
        filename = "sample_pca_decomp_head"+str(head_num)+".png"
        plot_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename, rank_lim=rank_lim)
        filename = "sample_pca_whitened_decomp_head"+str(head_num)+".png"
        plot_pca_whitened_decomposition(ranks, whitened_cos_dists_auth, whitened_cos_dists_attack, filename, rank_lim=rank_lim)
