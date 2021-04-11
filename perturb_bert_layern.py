'''
Try to identify an adversarial subspace using the PCA basis.

Perform perturbations in the input embedding space for a selected token position.
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
from BERT_layer_handler import Layer_Handler


def make_attack(vec, epsilon):
    attack_signs = torch.sign(vec)
    attackA = attack_signs*epsilon
    attackB = -1*attackA
    return attackA, attackB

def get_perturbation_impact(handler, v, input_ids, mask, labels, model, epsilon, stepsize=1, token_pos=0):
    ranks  = []
    mses = []
    avg_grades = []

    model.eval()

    for i in range(0, v.size(0), stepsize):
        print("On rank", i)
        ranks.append(i)
        curr_v = v[i]
        with torch.no_grad():
            attackA, attackB = make_attack(curr_v, epsilon)
            best_avg = 0
            for attack in [attackA, attackB]:
                layer_embeddings = handler.get_layern_outputs(input_ids, mask)
                layer_embeddings[:,token_pos,:] = layer_embeddings[:,token_pos,:] + attack

                # Pass through rest of model
                y = handler.pass_through_rest(layer_embeddings, mask)
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
    commandLineParser.add_argument('--token_pos', type=int, default=0, help="token position to perturb")
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to perturb")
    commandLineParser.add_argument('--stepsize', type=int, default=1, help="ranks step size for plot")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    epsilon = args.epsilon
    token_pos = args.token_pos
    layer_num = args.layer_num
    stepsize = args.stepsize

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/perturb_inp_emb.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create model handler
    handler = Layer_Handler(model, layer_num=layer_num)

    # Use training data to get eigenvector basis
    input_ids, mask, _ = get_data(train_data_file, train_grades_file, attack_phrase='')
    hidden_states = handler.get_layern_outputs(input_ids, mask)
    cov = get_covariance_matrix(hidden_states[:,token_pos,:])
    e, v = get_e_v(cov)

    # Get test data
    input_ids, mask, labels = get_data(test_data_file, test_grades_file, attack_phrase='')

    # Perturb in each eigenvector direction vs rank
    ranks, mses, avg_grades = get_perturbation_impact(handler, v, input_ids, mask, labels, model, epsilon, stepsize=stepsize, token_pos=token_pos)

    # Plot the data
    filename = 'mse_eigenvector_perturb_layer'+str(layer_num)+'_tokenpos'+str(token_pos)+'.png'
    yname = 'MSE'
    plot_data_vs_rank(ranks, mses, yname, filename)

    filename = 'avg_grade_eigenvector_perturb_layer'+str(layer_num)+'_tokenpos'+str(token_pos)+'.png'
    yname = 'Average Grade'
    plot_data_vs_rank(ranks, avg_grades, yname, filename)
