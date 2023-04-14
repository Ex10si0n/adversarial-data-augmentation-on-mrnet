import shutil
import os
import time
from datetime import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import MRDataset
from models.mrnet import MRNet

from sklearn import metrics
import csv
import utils as ut


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    now = datetime.now()

    validation_dataset = MRDataset(args.data_path, args.task, args.plane, train=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=-True, num_workers=2, drop_last=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    mrnet = torch.load(args.model)
    mrnet.eval()

    mrnet = mrnet.to(device)

    threshold_list = np.arange(0, 1.01, 0.01)  # list of thresholds

    # Open a CSV file for writing results
    csv_filename = f'./metrics/{args.weights}/' + f'{args.task}_{args.plane}.csv'
    if not os.path.exists(f'./metrics/{args.weights}'):
        os.makedirs(f'./metrics/{args.weights}')
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Threshold', 'Accuracy', 'Sensitivity', 'Specificity'])

        for threshold in tqdm(threshold_list):
            y_trues = []
            y_preds = []
            y_class_preds = []

            for i, (image, label, weight) in enumerate(validation_loader):

                image = image.to(device)
                label = label.to(device)
                weight = weight.to(device)

                prediction = mrnet.forward(image.float())

                probas = torch.sigmoid(prediction)

                y_trues.append(int(label[0]))
                y_preds.append(probas[0].item())
                y_class_preds.append((probas[0] > threshold).float().item())


            val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_trues, y_class_preds)
            val_accuracy = np.round(val_accuracy, 4)
            val_sensitivity = np.round(val_sensitivity, 4)
            val_specificity = np.round(val_specificity, 4)

            writer.writerow([threshold, val_accuracy, val_sensitivity, val_specificity])



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
