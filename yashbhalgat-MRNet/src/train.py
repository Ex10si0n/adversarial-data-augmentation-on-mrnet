import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model
from loader import load_data
from model import TripleMRNet

def train(rundir, task, backbone, epochs, learning_rate, use_gpu,
        abnormal_model_path=None):
    train_loader, valid_loader = load_data(task, use_gpu)
    
    model = TripleMRNet(backbone=backbone)
    for dirpath, dirnames, files in os.walk(args.rundir):
        if not files:
            break
        max_epoch = 0
        model_path = None
        for fname in files:
            if fname.endswith(".json"):
                continue
            ep = int(fname[27:])
            if ep >= max_epoch:
                max_epoch = ep
                model_path = os.path.join(dirpath, fname)
        
        if model_path:
            state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
            model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)

    best_val_loss = float('inf')

    start_time = datetime.now()

    epoch = 0
    if max_epoch: epoch += max_epoch
    while epoch < epochs:
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(
                model, train_loader, train=True, optimizer=optimizer,
                abnormal_model_path=abnormal_model_path)

        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader,
                abnormal_model_path=abnormal_model_path)
        
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)

        epoch += 1

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--backbone', default="alexnet", type=str)
    parser.add_argument('--abnormal_model', default=None, type=str)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.task != "abnormal":
        if args.abnormal_model is None:
            raise ValueError("Enter abnormal model path for `acl` or `meniscus` task")

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(args.rundir, args.task, args.backbone, args.epochs, args.learning_rate, args.gpu, abnormal_model_path=args.abnormal_model)
