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
from sklearn.metrics import roc_curve
from tensorboardX import SummaryWriter

from dataloader import MRDataset
from models.mrnet import MRNet

from sklearn import metrics
import csv
import utils as ut


models_dir = '../models/'
dataset_dir = '/Users/ex10si0n/MRNet-v1.0/'
device = 'mps'


def get_roc(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print('fpr: ', fpr)
    print('tpr: ', tpr)
    print('thresholds: ', thresholds)

all_preds = []
all_labels = []

print(models_dir)
for model_name in os.listdir(models_dir):
    if model_name == '.DS_Store':
        continue
    model_name = model_name.split('.')
    print(model_name)
    weights_name = models_dir + model_name
    model = torch.load(weights_name)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, patience=5, verbose=True, threshold=1e-4)
    criterion = nn.BCELoss()

    all_preds = []
    all_labels = []

    # train_dataset = MRDataset(dataset_dir, 'train')
