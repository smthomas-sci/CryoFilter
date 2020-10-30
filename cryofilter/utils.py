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

from sklearn.metrics import roc_curve, auc, roc_auc_score
import os

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np


from cryofilter.data import DataGenerator
from cryofilter.model import build_model

# # --------------------- DEBUG -------------------------------------- #
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # ----------------------------------------------------------------- #



# 1. Setup data
generator = DataGenerator(pos_files=["../data/pos.mrcs", "../data/pos_top.mrcs"],
                          neg_files=["../data/neg.mrcs"],
                          batch_size=32,
                          split=0.7,
                          img_dim=28
                          )

model = build_model(28)

model.load_weights("../weights/weights_epoch_99_val_acc_0.981.h5")

model.summary()

y_preds = []
y_trues = []

N = generator.val.n // generator.batch_size

counts = 0
total = 0
for step in range(N):
    print(step, "of", N, end="\r")

    x1, x2, y = generator.val[step]

    counts += np.sum(y)

    total += 32

    y_pred = model.predict([x1, x2])

    y_preds.append(y_pred)
    y_trues.append(y)

y_preds = np.vstack(y_preds)
y_trues = np.vstack(y_trues)


# calculate scores
ns_auc = roc_auc_score(y_trues, [0.5]*y_preds.shape[0])
lr_auc = roc_auc_score(y_trues, y_preds)

print("AUC:", lr_auc)

print("positives", counts)
print("Total:", total)

print(y_trues.shape, y_preds.shape)
#

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_trues, [0.5]*y_preds.shape[0])
lr_fpr, lr_tpr, _ = roc_curve(y_trues, y_preds)

# plot the roc curve for the model
plt.figure(figsize=(5,5), dpi= 200)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
#
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#
# show the plot
plt.title('Logistic: ROC AUC=%.3f' % (lr_auc), fontweight='bold')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()

plt.savefig("./AUC_curve.png")

plt.close()

#plt.show()
#
#
#
