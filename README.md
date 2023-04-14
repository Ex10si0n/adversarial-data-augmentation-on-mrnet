# Baseline Model

This repository contains the baseline model which reproduces the results proposed in the paper ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet"](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) by Bien, Rajpurkar et al.

## Data

Data must be downloaded from MRNet's [official website](https://stanfordmlgroup.github.io/competitions/mrnet/) and put anywhere into your machine. Then, edit the  ``` train_mrnet.sh``` script file by expliciting the full path to ```MRNet-v1.0``` directory into the ```DATA_PATH``` variable.

## Adversarial Training

To conduct adversarial training, please refers shell script: `adv_train.sh` with essential modification. If you meet problem running adversarial training, please contact me with email: me@aspires.cc

## Declaring of Originality

The code was modified based on the GitHub Repository [Matteo-dunnhofer/mrnet-baseline](https://github.com/matteo-dunnhofer/mrnet-baseline). Lines addition of the codes (only changed the original code here) in `mrnet/train.py` could be found in this report: [/compare/diff.pdf](./compare/diff.pdf). Besides, folder: `experiment-bot` and `compare` was designed and implemented with originality.

## Liences

MIT