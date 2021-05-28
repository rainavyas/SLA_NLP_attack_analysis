'''
Uses training data to learn PCA mapping for a specified head embedding space
Plots the eval data on specified PCA axes, colour coded by CEFR Grade.
'''

import torch
import torch.nn as nn
from attn_layer_analysis import get_head_embedding
import sys
import os
import argparse
from models import BERTGrader
from pca_tools import get_covariance_matrix, get_e_v
from data_prep_attack import get_data
import pandas as pd
import seaborn as sns
import matplotlib as plt


def get_head_embedding(data_file, grades_file, model, attack_phrase='', head_num=1):
    '''
    Gives the output embeddings of chosen head after BERT encoder
    '''
    input_ids, mask, labels = get_data(data_file, grades_file, attack_phrase)
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

    return head, labels


def get_pca_principal_components(eigenvectors, correction_mean, X, num_comps, start):
    '''
    Returns components in num_comps most principal directions
    Dim 0 of X should be the batch dimension
    '''
    comps = []
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        for i in range(start, start+num_comps):
            v = eigenvectors[i]
            comp = torch.einsum('bi,i->b', X, v) # project to pca axis
            comps.append(comp.tolist())
    return comps[:num_comps]

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DATA', type=str, help='prepped train data file')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TRAIN_GRADES', type=str, help='train data grades')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('OUT', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--head_num', type=int, default=1, help="Head embedding to analyse")
    commandLineParser.add_argument('--num_comps', type=int, default=2, help="number of PCA components")
    commandLineParser.add_argument('--start', type=int, default=0, help="start of PCA components")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    out_file = args.OUT
    head_num = args.head_num
    num_comps = args.num_comps
    start = args.start

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pca_component_comparison_plot_comps.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Use training data to get eigenvector basis of CLS token at correct layer
    embeddings, _ = get_head_embedding(train_data_file, train_grades_file, model, attack_phrase='', head_num=head_num)
    with torch.no_grad():
        correction_mean = torch.mean(embeddings, dim=0)
        cov = get_covariance_matrix(embeddings)
        e, v = get_e_v(cov)
    
    # Map test data to PCA components
    embeddings, labels = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase='', head_num=head_num)
    pca_comps = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    # Plot all the data
    df = pd.DataFrame({"PCA 0":pca_comps[0], "PCA 1":pca_comps[1], "grade":labels})
    sns.set_theme(style="whitegrid")
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    sns_plot = sns.scatterplot(
                            data=df,
                            x="PCA 0",
                            y="PCA 1",
                            hue="grade",
                            palette=cmap)


    sns_plot.figure.savefig(out_file)
