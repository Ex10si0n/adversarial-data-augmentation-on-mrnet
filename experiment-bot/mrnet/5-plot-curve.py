import os
import csv
import numpy as np
import matplotlib.pyplot as plt

exp_dir = os.path.join(os.path.dirname(__file__), "../../mrnet/experiments")
exp_eps = ["0.0001", "0.00001"]
exp_percent = ["0.06", "0.04"]
title = "epoch,train_loss,train_auc,train_accuracy,train_sensitivity,train_specificity,val_loss,val_auc,val_accuracy,val_sensitivity,val_specificity"
task = ["acl", "meniscus", "abnormal"]
dim = ["coronal", "sagittal", "axial"]

def plot_combiner(p1=None, p2=None, p3=None, pattern="", labels=[]):
    metrics = title.split(",")[1:]
    dir = "./curve/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(len(metrics)):
        # save fs
        label = metrics[i]
        label = label.replace("_", "-")

        # clear plt
        plt.clf()
        plt.xticks(np.arange(0, 70, 5))
        plt.xlabel('epoch')
        plt.ylabel(label)
        plt.title(pattern.split("/")[1] + " " + label)

        plt.plot(p1[:, 0], p1[:, i+1], label=labels[0])
        if p2 is not None:
            plt.plot(p2[:, 0] + 50, p2[:, i+1], label=labels[1])
        if p3 is not None:
            plt.plot(p3[:, 0] + 50, p3[:, i+1], label=labels[2])
        plt.legend()
        filename = dir + pattern + "/" + label + ".png"
        if not os.path.exists(dir + pattern):
            os.makedirs(dir + pattern)
        print("plot save to", filename)
        plt.savefig(filename)

def plot_curve(a1, a2, b, t, d, pattern):
    info = ''
    '''
    b: data for baseline
    a1: data for adv_1
    a2: data for adv_2
    t: plot task
    d: plot dimension
    pattern: plot pattern
        000: N/A
        001: baseline
        X 010: adv_2
        X 100: adv_1
        011: baseline + adv_2
        101: baseline + adv_1
        110: adv_1 + adv_2
        111: baseline + adv_1 + adv_2
    '''
    # if bin(pattern) == "0b0":
    #     return
    # if bin(pattern) == "0b1":
    #     info = "baseline/" + t + "/" + d
    #     plot_combiner(p1=b, pattern=info)
    # if bin(pattern) == "0b10":
    #     info = "adv_2/" + t + "/" + d
    #     plot_combiner(p1=a2, pattern=info)
    # if bin(pattern) == "0b100":
    #     info = "adv_1/" + t + "/" + d
    #     plot_combiner(p1=a1, pattern=info)
    if bin(pattern) == "0b11":
        info = "baseline_adv_2/" + t + "_" + d
        plot_combiner(p1=b, p2=a2, pattern=info, labels=["baseline", "eps=1e-5, %=0.04"])
    if bin(pattern) == "0b101":
        info = "baseline_adv_1/" + t + "_" + d
        plot_combiner(p1=b, p2=a1, pattern=info, labels=["baseline", "eps=1e-4, %=0.06"])
    # if bin(pattern) == "0b110":
    #    info = "baseline_adv_ori/" + t + "_" + d
    if bin(pattern) == "0b111":
        info = "baseline_adv_1_adv_2/" + t + "_" + d
        plot_combiner(p1=b, p2=a1, p3=a2, pattern=info, labels=["baseline", "eps=1e-4, %=0.06", "eps=1e-5, %=0.04", "eps=0, %=0.04"])


for t in task:
    for d in dim:
        baseline = []
        adv_1 = []
        adv_2 = []
        for exp in os.listdir(exp_dir):
            hypers = exp.split("-")
            if len(hypers) >= 3:
                params_eps = hypers[-2]
                params_percent = hypers[-1]

            elif exp == "baseline":
                params_eps = "0"
                params_percent = "1"
            elif len(hypers) < 3:
                continue

            if not os.path.isdir(os.path.join(exp_dir, exp)):
                continue

            exp_res = os.path.join(exp_dir, exp, "results")
            if exp == 'baseline': print(exp_res, t, d)

            for f in os.listdir(exp_res):
                fs = f.split(".")[0].split("_")

                if fs[0] == 'learning' and fs[3] == t and fs[4] == d:
                    data = []
                    res = [params_eps, params_percent, f]
                    with open(os.path.join(exp_res, f), "r") as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data.append(row)
                    if params_eps == "0" and params_percent == "1":
                        baseline.append(data)
                    elif params_eps == exp_eps[0] and params_percent == exp_percent[0]:
                        adv_1.append(data)
                    elif params_eps == exp_eps[1] and params_percent == exp_percent[1]:
                        adv_2.append(data)

        baseline = np.array(baseline).reshape(-1, 11)[1:, :].astype(np.float64)
        adv_1 = np.array(adv_1).reshape(-1, 11)[1:, :].astype(np.float64)
        adv_2 = np.array(adv_2).reshape(-1, 11)[1:, :].astype(np.float64)

        print(baseline.shape, adv_1.shape, adv_2.shape)
        for i in range(1, 8):
            plot_curve(adv_1, adv_2, baseline, t, d, i)

