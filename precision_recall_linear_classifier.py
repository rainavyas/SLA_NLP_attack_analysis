'''
Generate precision-recall curve for linear adversarial attack classifier
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from tools import get_default_device
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from models import BertGrader
from pca_component_comparison_plot_comps import get_head_embedding
from linear_pca_classifier import get_pca_principal_components, LayerClassifier


def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precision*(beta**2))+recall))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('MODEL_DETECTOR', type=str, help='trained adv attack detector')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('ATTACK', type=str, help='universal attack phrase for eval set')
    commandLineParser.add_argument('--head_num', type=int, default=1, help="Head embedding to analyse")
    commandLineParser.add_argument('--num_comps', type=int, default=400, help="Number of PCA components to use")
    commandLineParser.add_argument('--N', type=int, default=3, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    
    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_data_file = args.TEST_DATA
    test_grades_file = args.TEST_GRADES
    detector_path = args.MODEL_DETECTOR
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    out_file = args.OUT_FILE
    attack_phrase = args.ATTACK
    head_num = args.head_num
    num_comps = args.num_comps
    N = args.N
    cpu_use = args.cpu
    

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/precision_recall_linear_classifier.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the Sentiment Classifier model
    model = BertGrader()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the Adv Attack Detector model
    detector = LayerClassifier(num_comps)
    detector.load_state_dict(torch.load(detector_path, map_location=torch.device('cpu')))
    detector.eval()

    # Load the eigenvectors for PCA decomposition and the correction mean
    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    # Prepare test input tensors (mapped to pca components)
    embeddings, _ = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase='', head_num=head_num)
    original = get_pca_principal_components(eigenvectors, correction_mean, embeddings, num_comps, 0)

    embeddings, _ = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase=attack_phrase, head_num=head_num)
    attack = get_pca_principal_components(eigenvectors, correction_mean, embeddings, num_comps, 0)

    print("Got embeddings")
    
    labels = np.asarray([0]*original.size(0) + [1]*attack.size(0))
    X = torch.cat((original, attack))

    # get predicted logits of being adversarial attack
    with torch.no_grad():
        logits = detector(X)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        adv_probs = probs[:,1].squeeze().cpu().detach().numpy()
    
    print("Got prediction probs")
    # get precision recall values and highest F1 score (with associated prec and rec)
    precision, recall, _ = precision_recall_curve(labels, adv_probs)
    best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)

    # plot all the data
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F0.5={best_f05:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file)
