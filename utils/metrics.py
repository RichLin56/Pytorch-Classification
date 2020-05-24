import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report


def plot_confusion_matrix(array, class_to_idx: dict, normalize=False, output_dir = '', plot=False):
    index = []
    for key in class_to_idx.keys():
        index.insert(class_to_idx[key], key)
    df_confusion = pd.DataFrame(array, index, index)
    if normalize:
        df_confusion = df_confusion / df_confusion.sum(axis=1)
    sn.heatmap(df_confusion, annot=True)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'conf_matrix{}.png'.format("_normalized" if normalize else "")))
    if plot:
        plt.show()
    plt.close()


def create_classification_report(array, class_to_idx: dict):
    index = []
    for key in class_to_idx.keys():
        index.insert(class_to_idx[key], key)
    y_true = []
    y_pred = []
    for row_id in range(array.shape[0]):
        row_sum = 0
        for col_id in range(array.shape[1]):
            row_sum += array[row_id, col_id]
            y_pred.extend([col_id] * int(array[row_id, col_id]))
        y_true.extend([row_id] * int(row_sum))
    return classification_report(y_true, y_pred, target_names=index)


if __name__ == '__main__':
    class_to_idx = {'C0': 0, 'C1': 1, 'C2': 2}
    array = np.zeros((3, 3))
    array[0][0] = 5
    array[0][1] = 20
    array[1][1] = 5
    array[2][2] = 10
    print(create_classification_report(array, class_to_idx))
