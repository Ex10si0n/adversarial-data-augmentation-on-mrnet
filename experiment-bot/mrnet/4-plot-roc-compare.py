experiments = ["MRNet-2023-03-01-22-06-MRNet", "MRNet-2023-03-01-17-51-MRNet-0.0001-0.06", "MRNet-2023-03-02-14-15-MRNet-0.00001-0.04"]


import matplotlib.pyplot as plt
import pandas as pd

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

task = ["abnormal", "acl", "meniscus"]
dimension = ["axial", "coronal", "sagittal"]

for e in experiments:
    for i, t in enumerate(task):
        for j, d in enumerate(dimension):
            roc_path = "../../mrnet/experiments/{e}/results/".format(e=e)
            print("Plotting {t} {d}".format(t=t, d=d))
            roc = pd.read_csv(roc_path + "roc_curve_MRNet_{task}_{dimension}".format(task=t, dimension=d))
            axs[i][j].plot(roc["FPR"], roc["TPR"])
            axs[i][j].set_xlabel("False Positive Rate")
            axs[i][j].set_ylabel("True Positive Rate")
            axs[i][j].set_title("ROC of " + t + " " + d)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
savename = "roc_curve_MRNet_cmp.png"
plt.savefig("./roc/" + savename)

plt.show()

