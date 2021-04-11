'''
Try to identify an adversarial subspace using the PCA basis.

Perform perturbations in each eigenvector direction in the attention head
embedding space.
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep_attack import get_data
import sys
import os
import argparse
from models import BERTGrader
from tools import AverageMeter
from pca_tools import get_covariance_matrix, get_e_v
import matplotlib.pyplot as plt
from tools import calculate_mse


def get_head_embeddings(data_file, grades_file, model, attack_phrase=''):
    '''
    Gives the output embeddings of chosen head after BERT encoder
    '''
    input_ids, mask, labels = get_data(data_file, grades_file, attack_phrase)
    model.eval()
    with torch.no_grad():
        output = model.encoder(input_ids, mask)
        word_embeddings = output.last_hidden_state

        head1 = model.apply_attn(word_embeddings, mask, model.attn1)
        head2 = model.apply_attn(word_embeddings, mask, model.attn2)
        head3 = model.apply_attn(word_embeddings, mask, model.attn3)
        head4 = model.apply_attn(word_embeddings, mask, model.attn4)

    return head1, head2, head3, head4, labels

def make_attack(vec, epsilon):
    attack_signs = torch.sign(vec)
    attackA = attack_signs*epsilon
    attackB = -1*attackA
    return attackA, attackB

def get_perturbation_impact(v, head1, head2, head3, head4, labels, model, epsilon, head_num, stepsize=1):
    ranks  = []
    mses = []
    avg_grades = []

    model.eval()

    for i in range(0, v.size(0), stepsize):
        ranks.append(i)
        curr_v = v[i]
        with torch.no_grad():
            attackA, attackB = make_attack(curr_v, epsilon)
            best_avg = 0
            for attack in [attackA, attackB]:
                heads = [head1, head2, head3, head4]
                heads[head_num-1] = heads[head_num-1] + attack
                all_heads = torch.cat(heads, dim=1)
                h1 = model.layer1(all_heads).clamp(min=0)
                h2 = model.layer2(h1).clamp(min=0)
                y = model.layer3(h2).squeeze()
                mse = calculate_mse(y, labels)
                if torch.mean(y).item() > best_avg:
                    best_avg = torch.mean(y).item()
            mses.append(mse.item())
            avg_grades.append(best_avg)

    return ranks, mses, avg_grades

def plot_data_vs_rank(ranks, data, yname, filename):

    plt.plot(ranks, data)
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel(yname)
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
    commandLineParser.add_argument('--epsilon', type=float, default=0.1, help='l-inf perturbation size')
    commandLineParser.add_argument('--head', type=int, default=1, help="attention head")
    commandLineParser.add_argument('--stepsize', type=int, default=1, help="ranks step size for plot")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    epsilon = args.epsilon
    head = args.head
    stepsize = args.stepsize

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/perturb_attn_layer.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Use training data to get eigenvector basis
    head1, head2, head3, head4, _ = get_head_embeddings(train_data_file, train_grades_file, model, attack_phrase='')
    heads = [head1, head2, head3, head4]
    cov = get_covariance_matrix(heads[head-1])
    e, v = get_e_v(cov)

    # Get test data embeddings
    test_head1, test_head2, test_head3, test_head4, test_labels = get_head_embeddings(test_data_file, test_grades_file, model, attack_phrase='')

    # Perturb in each eigenvector direction vs rank
    ranks, mses, avg_grades = get_perturbation_impact(v, test_head1, test_head2, test_head3, test_head4, test_labels, model, epsilon, head, stepsize=stepsize)

    # Plot the data
    filename = 'mse_eigenvector_perturb_head'+str(head)+'.png'
    yname = 'MSE'
    plot_data_vs_rank(ranks, mses, yname, filename)

    filename = 'avg_grade_eigenvector_perturb_head'+str(head)+'.png'
    yname = 'Average Grade'
    plot_data_vs_rank(ranks, avg_grades, yname, filename)
