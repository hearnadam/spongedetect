import matplotlib.pyplot as plt
import numpy as np
from os import path

def plotMatrix(class_labels, cm_all, filename, title):
    cm_all_percent = []
    for i in range(cm_all.shape[0]):
        cm_all_percent.append([])
        row_sum = sum(cm_all[i])

        for j in range(cm_all.shape[1]):
            percent = (cm_all[i, j] / row_sum)
            cm_all_percent[i].append(percent)
        cm_all_percent[i] = np.array(cm_all_percent[i])

    cm_all_percent = np.array(cm_all_percent)


    print(cm_all_percent)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.matshow(cm_all_percent, cmap=plt.cm.Blues)

    for i in range(cm_all.shape[0]):
        for j in range(cm_all.shape[1]):
            percent = cm_all_percent[i, j]
            color = 'black'

            if percent > .5:
                color = 'white'

            ax.text(x=j, y=i,s=f'{percent:0.3f}', va='center', ha='center', size='xx-large', color=color)


    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    
    plt.xlabel('Prediction', fontsize=18, labelpad=20)
    plt.ylabel('Actual', fontsize=18)
    plt.title(title, fontsize=18, pad=20)
    plt.savefig(path.join('./out', filename + '.png'), dpi=300)