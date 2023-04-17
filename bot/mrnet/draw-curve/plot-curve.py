import os
import csv
import numpy as np
import matplotlib.pyplot as plt

task = ["abnormal", "acl", "meniscus"]
dim = ["axial", "coronal", "sagittal"]
base_name = 'learning_curve_MRNet_{0}_{1}.csv'
hyper = 'eps=1e-5 %=6'
metric = 'specificity'

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i, t in enumerate(task):
    for j, d in enumerate(dim):
        base_title = None
        base_data = None
        adv_title = None
        adv_data = None
        csv_name = base_name.format(t, d)
        adv_csv_name = 'adv/' + csv_name
        base_csv_name = 'base/' + csv_name
        with open(base_csv_name, 'r') as f:
            reader = csv.reader(f)
            base_title = next(reader)
            base_data = np.array(list(reader)).astype('float')
        with open(adv_csv_name, 'r') as f:
            reader = csv.reader(f)
            adv_title = next(reader)
            adv_data = np.array(list(reader)).astype('float')

        combine = [f'train_{metric}', f'val_{metric}', f'val_adv_{metric}']

        for k in range(1, len(adv_title)):
            if adv_title[k] in combine:
                try:
                    axs[i, j].plot(base_data[:, 0], base_data[:, k], label='baseline ' + base_title[k])
                except:
                    pass
                axs[i, j].plot(adv_data[:, 0] + 50, adv_data[:, k], label= hyper + ' ' + adv_title[k])
                axs[i, j].set_title(t + ' ' + d)
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel(adv_title[k].split('_')[-1].upper())
plt.legend()

plt.savefig(os.path.join('plot', 'learning_curves.png'))
plt.show()
