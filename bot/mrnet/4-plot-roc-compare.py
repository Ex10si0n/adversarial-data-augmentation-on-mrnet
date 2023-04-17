experiments = ["baseline", "MRNet-2023-03-22-19-10-MRNet-0.00001-0.1"]


import matplotlib.pyplot as plt
import pandas as pd

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

task = ["abnormal", "acl", "meniscus"]
dimension = ["axial", "coronal", "sagittal"]

for e in experiments:
    label = ''
    percent = e.split("-")[-1]
    if percent == 'baseline':
        label = 'baseline'
    else:
        epsilon = e.split("-")[-2]
        if epsilon == '0.00001':
            epsilon = '1e-5'
        label = f"eps={epsilon} percent={percent}"

    for i, t in enumerate(task):
        for j, d in enumerate(dimension):
            roc_path = "../../mrnet/experiments/{e}/results/".format(e=e)
            print("Plotting {t} {d}".format(t=t, d=d))
            roc = pd.read_csv(roc_path + "roc_curve_MRNet_{task}_{dimension}".format(task=t, dimension=d))
            axs[i][j].plot(roc["FPR"], roc["TPR"], label=label)
            axs[i][j].set_xlabel("Specificity")
            axs[i][j].set_ylabel("Sensitivity")
            axs[i][j].set_title("ROC of " + t + " " + d)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.xlabel("Specificity")
plt.ylabel("Sensitivity")
plt.legend()
savename = "roc_curve_MRNet_cmp.png"
plt.savefig("./roc/" + savename)

plt.show()

