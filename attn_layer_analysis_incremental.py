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
from attn_layer_analysis import get_eigenvector_decomposition_magnitude, get_head_embedding
import matplotlib.pyplot as plt

def plot_decomposition(ranks, cos_dists_auth, cos_dists_attack_list, filename, rank_lim=768):
    ranks = ranks[:rank_lim]
    cos_dists_auth = cos_dists_auth[:rank_lim]
    cos_dists_attack = cos_dists_attack[:rank_lim]

    plt.plot(ranks, cos_dists_auth, label="Original")
    for i in range(len(cos_dists_attack_list)):
        plt.plot(ranks, cos_dists_attack, label="Attack, k="+str(i+1))
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Average Absolute Cosine Distance")
    plt.yscale('log')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('ATTACK', type=str, help='universal attack phrase')
    commandLineParser.add_argument('--rank_lim', type=int, default=768, help="How many principal ranks to show")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_file = args.TEST_DATA
    grades_file = args.TEST_GRADES
    attack_phrase = args.ATTACK
    rank_lim = args.rank_lim

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attn_layer_analysis_incremental.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for head_num in range(1,5):
        # Use authentic data to create eigenvector basis
        auth_embedding = get_head_embedding(data_file, grades_file, model, attack_phrase='', head_num=head_num)
        correction_mean = torch.mean(auth_embedding, dim=0)
        cov = get_covariance_matrix(auth_embedding)
        e, v = get_e_v(cov)

        # Get authenetic PCA decomposition
        ranks, cos_dists_auth = get_eigenvector_decomposition_magnitude(v, auth_embedding, correction_mean)

        cos_dists_attack_list = []
        attack_words = attack_phrase.split()

        for j in range(1, len(attack_words)+1):
            curr_words = attack_words[:j]
            attack_phrase = curr_words[0]
            for word in curr_words[1:]:
                attack_phrase = attack_phrase + ' ' word

            # Get attacked PCA decomposition
            attack_embedding = get_head_embedding(data_file, grades_file, model, attack_phrase=attack_phrase, head_num=head_num)
            ranks, cos_dists_attack = get_eigenvector_decomposition_magnitude(v, attack_embedding, correction_mean)
            cos_dists_attack_list.append(cos_dists_attack)

        # Plot the data
        filename = "pca_decomp_head"+str(head_num)+".png"
        plot_decomposition(ranks, cos_dists_auth, cos_dists_attack_list, filename, rank_lim=rank_lim)
