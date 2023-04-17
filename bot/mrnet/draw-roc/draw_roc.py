import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

models = ['baseline', 'adv']
csv_files = ['abnormal_axial.csv', 'abnormal_coronal.csv', 'abnormal_sagittal.csv', 'acl_axial.csv', 'acl_coronal.csv', 'acl_sagittal.csv', 'meniscus_axial.csv', 'meniscus_coronal.csv', 'meniscus_sagittal.csv']

# create a 3x3 subplot grid
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axs = axs.flatten()

for i, f in enumerate(csv_files):
    for j, model in enumerate(models):
        data = pd.read_csv(f'./metrics/{model}/{f}')

        threshold = data['Threshold']
        accuracy = data['Accuracy']
        sensitivity = data['Sensitivity']
        specificity = data['Specificity']

        fpr = 1 - specificity
        tpr = sensitivity

        roc_auc = auc(fpr, tpr)

        # plot the ROC curve on the corresponding subplot
        axs[i].plot(fpr, tpr, lw=2, label=f'{model} (AUC = %0.2f)' % roc_auc)

    axs[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[i].set_xlim([0.0, 1.0])
    axs[i].set_ylim([0.0, 1.05])
    axs[i].set_xlabel('False Positive Rate')
    axs[i].set_ylabel('True Positive Rate')
    axs[i].set_title(f.split('.')[0].replace('_', ' '))
    axs[i].legend(loc="lower right")

# adjust spacing between subplots and show the figure
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('./metrics/roc.png')
plt.show()
