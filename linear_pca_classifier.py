'''
Use the PCA reduced dimensions specific head embedding as an
input into a simple linear classifier to distinguish between original and
adversarial samples
'''

import sys
import os
import argparse
from tools import AverageMeter, get_default_device, accuracy_topk
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pca_tools import get_covariance_matrix, get_e_v
from models import BERTGrader
from pca_component_comparison_plot_comps import get_head_embedding

def get_pca_principal_components(eigenvectors, correction_mean, X, num_comps, start):
    '''
    Returns components in num_comps most principal directions
    Dim 0 of X should be the batch dimension
    '''
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        v_map = eigenvectors[start:start+num_comps]
        comps = torch.einsum('bi,ji->bj', X, v_map)
    return comps
    

def train(train_loader, model, criterion, optimizer, epoch, device, out_file, print_freq=1):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            text = '\n Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, prec=accs)
            print(text)
            with open(out_file, 'a') as f:
                f.write(text)

def eval(val_loader, model, criterion, device, out_file):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    text ='\n Test\t Loss ({loss.avg:.4f})\t Accuracy ({prec.avg:.3f})\n'.format(loss=losses, prec=accs)
    print(text)
    with open(out_file, 'a') as f:
        f.write(text)

class LayerClassifier(nn.Module):
    '''
    Simple Linear classifier
    '''
    def __init__(self, dim, classes=2):
        super().__init__()
        self.layer = nn.Linear(dim, classes)
    def forward(self, X):
        return self.layer(X)

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DATA', type=str, help='prepped train data file')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TRAIN_GRADES', type=str, help='train data grades')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('ATTACK', type=str, help='universal attack phrase for eval set')
    commandLineParser.add_argument('OUT', type=str, help='file to print results to')
    commandLineParser.add_argument('CLASSIFIER_OUT', type=str, help='.th to save linear adv attack classifier to')
    commandLineParser.add_argument('PCA_OUT', type=str, help='.pt to save original PCA eigenvector directions to')
    commandLineParser.add_argument('PCA_MEAN_OUT', type=str, help='.pt to save PCA correction mean to')
    commandLineParser.add_argument('--num_points_val', type=int, default=50, help="number of test data points to use for validation")
    commandLineParser.add_argument('--head_num', type=int, default=1, help="Head embedding to analyse")
    commandLineParser.add_argument('--num_comps', type=int, default=10, help="number of PCA components")
    commandLineParser.add_argument('--start', type=int, default=0, help="start of PCA components")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_file = args.TRAIN_DATA
    test_data_file = args.TEST_DATA
    train_grades_file = args.TRAIN_GRADES
    test_grades_file = args.TEST_GRADES
    attack_phrase = args.ATTACK
    out_file = args.OUT
    classifier_out_file = args.CLASSIFIER_OUT
    pca_out_file = args.PCA_OUT
    pca_mean_out_file = args.PCA_MEAN_OUT
    num_points_val = args.num_points_val
    head_num = args.head_num
    num_comps = args.num_comps
    start = args.start
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    cpu_use = args.cpu

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/linear_pca_classifier.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Use training data to get eigenvector basis of head embedding
    embeddings, _ = get_head_embedding(train_data_file, train_grades_file, model, attack_phrase='', head_num=head_num)
    with torch.no_grad():
        correction_mean = torch.mean(embeddings, dim=0)
        cov = get_covariance_matrix(embeddings)
        e, v = get_e_v(cov)

    # Save the PCA embedding eigenvectors and correction mean
    torch.save(v, pca_out_file)
    torch.save(correction_mean, pca_mean_out_file)

    # Map test data (with and without attack) to PCA space
    embeddings, _ = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase='', head_num=head_num)
    original_comps = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings, _ = get_head_embedding(test_data_file, test_grades_file, model, attack_phrase=attack_phrase, head_num=head_num)
    attack_comps = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    # Prepare tensors for classification
    labels = torch.LongTensor([0]*original_comps.size(0)+[1]*attack_comps.size(0))
    X = torch.cat((original_comps, attack_comps))

    # Shuffle all the data
    indices = torch.randperm(len(labels))
    labels = labels[indices]
    X = X[indices]

    # Split data
    X_val = X[:num_points_val]
    labels_val = labels[:num_points_val]
    X_train = X[num_points_val:]
    labels_train = labels[num_points_val:]

    ds_train = TensorDataset(X_train, labels_train)
    ds_val = TensorDataset(X_val, labels_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    # Model
    model = LayerClassifier(num_comps)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Create file
    N = len(attack_phrase.split())
    with open(out_file, 'w') as f:
        text = f'Head {head_num}, Comps {num_comps}, N {N}\n'
        f.write(text)

    # Train
    for epoch in range(epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(out_file, 'a') as f:
            f.write(text)
        print(text)
        train(dl_train, model, criterion, optimizer, epoch, device, out_file)

        # evaluate
        eval(dl_val, model, criterion, device, out_file)
    
    # Save the trained model for identifying adversarial attacks
    torch.save(model.state_dict(), classifier_out_file)