# DATE=(date +"%Y-%m-%d-%H-%M")
# EXPERIMENT="MRNet-${DATE}-MRNet"
# DATA_PATH='/Users/ex10si0n/MRNet-v1.0/'
# EPOCHS=20
# PREFIX=MRNet
# 
# for PERCENT in 0.06
# do
#   for EPS in 0.001 0.003 0.005 0.007 0.01
#   do
#     DATE=$(date +"%Y-%m-%d-%H-%M")
#     EXPERIMENT="MRNet-${DATE}-MRNet-${EPS}-${PERCENT}"
# 
#     python3 train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --advtrain 1 --advtrain_percent $PERCENT --epsilon $EPS

import pandas as pd
import numpy as np
import os
import sys

# read ensemble.xlsx
df = pd.read_excel('ensemble.xlsx')

for index, row in df.iterrows():
    task = row['task']
    dimension = row['dimension']
    eps = row['eps']
    percent = row['percent']
    best_metric = row['best-metric']
    if eps == 0 and percent == 0:
        continue
    train_cmd = f"python3 train.py -t {task} -p {dimension} --experiment {best_metric}-{task}-{dimension}-{eps}-{percent} --data-path /Users/ex10si0n/MRNet-v1.0/ --prefix_name MRNet --epochs=40 --advtrain 1 --advtrain_percent {percent} --epsilon {eps}"

    print("Running: ", train_cmd)
    os.system(train_cmd)
