'''
Training Set used to define PCA eigenevectors in head embedding space
Test Set split:
    Held out set 1
    Held out set 2
Held out set 1 used to define the pca whitened coefficiengts in this space
PR curve generated using held out set 2, using the variance around the reference
curve from held out set 1, to detect adversarial examples from held out set 2
data.
'''

import torch
import torch.nn as nn
from data_prep_for_saliency import get_data
from torch.nn import CosineSimilarity
from models import BERTGrader
from attn_layer_analysis import get_eigenvector_decomposition_magnitude
import matplotlib.pyplot as plt
from pca_tools import get_covariance_matrix, get_e_v
import sys
import os
import argparse
import numpy as np

def pr(list_auth, list_attack, start, stop, num):
    precision = []
    recall = []

    for thresh in np.linspace(start, stop, num):
        TP = 0 # true positive
        FP = 0 # false positive
        T = len(list_attack) # Number of True Attack examples

        for val in list_auth:
            if val > thresh:
                FP += 1
        for val in list_attack:
            if val > thresh:
                TP += 1

        if (TP+FP > 0):
            prec = TP/(TP+FP)
            rec = TP/T

            precision.append(prec)
            recall.append(rec)

    return precision, recall

def get_diff(reference, target):
    with torch.no_grad():
        ref_repeat = reference.repeat(target.size(0), 1)
        diff = target - ref_repeat
    return diff

def get_avg_precision_recall(reference, auth_coeff, attack_coeff, start=-3, stop=3, num=1000, rank_start=0):
    '''
    If variance is greater than threshold, then output positive for attack
    '''
    auth_diff = get_diff(torch.FloatTensor(reference), auth_coeff)
    attack_diff = get_diff(torch.FloatTensor(reference), attack_coeff)

    auth_diff = auth_diff[:, rank_start:]
    attack_diff = attack_diff[:, rank_start:]

    auth_mean = torch.mean(auth_diff, dim=1).tolist()
    attack_mean = torch.mean(attack_diff, dim=1).tolist()

    precision, recall = pr(auth_mean, attack_mean, start, stop, num)
    return precision, recall



def get_variance_precision_recall(reference, auth_coeff, attack_coeff, start=0, stop=6, num=1000, rank_start=0):
    '''
    If variance is greater than threshold, then output positive for attack
    '''
    auth_diff = get_diff(torch.FloatTensor(reference), auth_coeff)
    attack_diff = get_diff(torch.FloatTensor(reference), attack_coeff)

    auth_diff = auth_diff[:, rank_start:]
    attack_diff = attack_diff[:, rank_start:]

    auth_var = torch.var(auth_diff, dim=1).tolist()
    attack_var = torch.var(attack_diff, dim=1).tolist()

    precision, recall = pr(auth_var, attack_var, start, stop, num)
    return precision, recall


def get_eigenvector_decomposition_magnitude_indv(eigenvectors, eigenvalues, X, correction_mean):
    '''
    Mean average of magnitude of cosine distance to each eigenevector
    '''
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
            whitened_abs_cos_dist = abs_cos_dist/(eigenvalues[i]**0.5)
            whitened_cos_dists.append(whitened_abs_cos_dist)
        whitened_cos_dists = torch.stack(whitened_cos_dists, dim=1)

    return ranks, whitened_cos_dists


def get_head_embedding(input_ids, mask, model, head_num=1):
    '''
    Gives the output embeddings of chosen head after BERT encoder
    '''
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

def plot_precision_recall(precision1, recall1, precision2, recall2, filename):
    plt.plot(recall1, precision1, label='Variance')
    plt.plot(recall2, precision2, label='Avg Deviation')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
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
    commandLineParser.add_argument('--head_num', type=int, default=1, help='Select attention head')
    commandLineParser.add_argument('--held1_size', type=int, default=100, help='Size of held out set 1')
    commandLineParser.add_argument('--rank_start', type=int, default=0, help='Eigenvector rank start for metric calculation')

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    attack_phrase = args.ATTACK
    head_num = args.head_num
    held1_size = args.held1_size
    rank_start = args.rank_start

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/precision_recall_attn_layer.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Use authentic training data to create eigenvector basis
    input_ids, mask, _, _, _ = get_data(train_data_file, train_grades_file, '')
    auth_embedding = get_head_embedding(input_ids, mask, model, head_num=head_num)
    correction_mean = torch.mean(auth_embedding, dim=0)
    cov = get_covariance_matrix(auth_embedding)
    e, v = get_e_v(cov)

    # Create held out set 1 and held out set 2 using evaluation data
    ids, mask, ids_attack, mask_attack, _ = get_data(test_data_file, test_grades_file, attack_phrase)
    eval1_ids = ids[:held1_size]
    eval1_mask = mask[:held1_size]
    eval2_ids = ids[held1_size:]
    eval2_mask = mask[held1_size:]
    eval2_ids_attack = ids_attack[held1_size:]
    eval2_mask_attack = mask_attack[held1_size:]

    # Use held out set 1 to get reference coefficients
    embedding = get_head_embedding(eval1_ids, eval1_mask, model, head_num=head_num)
    ranks, _, whitened_cos_dists_eval1 = get_eigenvector_decomposition_magnitude(v, e, embedding, correction_mean)

    # Use held out set 2 to get authentic whitened pca coefficients for each input separately (as a tensor)
    embedding = get_head_embedding(eval2_ids, eval2_mask, model, head_num=head_num)
    ranks, whitened_cos_dists_eval2 = get_eigenvector_decomposition_magnitude_indv(v, e, embedding, correction_mean)

    # Use held out set 2 to get attacked whitened pca coefficients
    embedding = get_head_embedding(eval2_ids_attack, eval2_mask_attack, model, head_num=head_num)
    ranks, whitened_cos_dists_eval2_attack = get_eigenvector_decomposition_magnitude_indv(v, e, embedding, correction_mean)

    # Get precision and recall curve and plot it
    precision_var, recall_var = get_variance_precision_recall(whitened_cos_dists_eval1, whitened_cos_dists_eval2, whitened_cos_dists_eval2_attack, rank_start=rank_start)
    precision_avg, recall_avg = get_avg_precision_recall(whitened_cos_dists_eval1, whitened_cos_dists_eval2, whitened_cos_dists_eval2_attack, rank_start=rank_start)
    filename = 'precision_recall_variance_avg_head'+str(head_num)+'k'+str(len(attack_phrase.split()))+'rank_start'+str(rank_start)+'.png'
    plot_precision_recall(precision_var, recall_var, precision_avg, recall_avg, filename)
