"""
Contains utils for training and validation

Author: Simon Thomas
Date: 30th October 2020

Requirements (available by pip / conda):
- tensorflow
- numpy
- sklearn
- matplotlib
"""

import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, classification_report


def get_predictions(model, generator, drop_remainder=False) -> tuple:
    """
    Gets predicts from the model using the given generator (validation)
    :param model: the trained model to predict with
    :param generator: the generator to use
    :param drop_remainder: bool, whether to drop remainder of not a complete batch
    :return: predictions: y_true, y_pred which are each (n, 1) arrays
    """
    # Save all predicts
    y_preds = []
    y_trues = []

    N = generator.val.n // generator.batch_size
    if not drop_remainder:
        N += 1

    progress = tf.keras.utils.Progbar(N)

    for step in range(N):
        progress.update(step)

        # Get batch of data
        x1, x2, y = generator.val[step]

        # Get predicts for input
        y_pred = model.predict([x1, x2])

        # Save input/output pairs
        y_preds.append(y_pred)
        y_trues.append(y)

    # Finalize
    progress.update(step+1, finalize=True)

    # Convert to a stack: shape == (N*batch_sie, 1)
    y_pred = np.vstack(y_preds)
    y_true = np.vstack(y_trues)

    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    """
    :param y_true: numpy array of true values
    :param y_pred: numpy array of predicted values
    :return:
    """
    metrics = {}
    n = y_pred.shape[0]

    # Calculate AUC
    train_auc = roc_auc_score(y_true, y_pred)

    metrics["AUC"] = train_auc

    # Calculate roc curves
    base_fpr, base_tpr, _ = roc_curve(y_true, [0.5]*n)
    train_fpr, train_tpr, _ = roc_curve(y_true, y_pred)

    metrics["ROC"] = {
        "base_fpr": base_fpr,
        "base_tpr": base_tpr,
        "train_fpr": train_fpr,
        "train_tpr": train_tpr
    }

    # Calculate confusion matrix with threshold 0.5
    confusion = np.zeros((2, 2))
    for t,p in zip(y_true, y_pred):
        confusion[int(t), int(p > 0.5)] += 1

    metrics["CM"] = confusion
    metrics["acc"] = np.sum(y_true == (y_pred > 0.5)) / n
    metrics["specificity"] = confusion[0, 0] / np.sum(confusion[0, :])
    metrics["sensitivity"] = confusion[1, 1] / np.sum(confusion[1, :])

    return metrics


def create_roc_plot(metrics, out_dir: str):
    """
    :param metrics: dictionary from the `compute_metrics()` function
    :param out_dir: the directory to save the curve
    :return: None
    """
    auc = metrics["AUC"]
    base_fpr = metrics["ROC"]["base_fpr"]
    base_tpr = metrics["ROC"]["base_tpr"]
    train_fpr = metrics["ROC"]["train_fpr"]
    train_tpr = metrics["ROC"]["train_tpr"]

    # Plot the roc curve for the model
    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot(base_fpr, base_tpr, linestyle='--', label='Reference')
    plt.plot(train_fpr, train_tpr, marker='.', label='Model')

    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Show the plot
    plt.title(f'ROC AUC={auc:.3f}', fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()

    # Save
    filename = os.path.join(out_dir, "roc.png")
    print("Saving figure:", filename)
    plt.savefig(filename)

    plt.close()

    return None





