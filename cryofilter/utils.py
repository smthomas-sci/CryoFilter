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

y_val = []
X1_val = []
X2_val = []

val_gen.batch_size = 1

for i in range(val_gen.n):
    X, y = val_gen[i]

    y_val.append(y[0])
    X1_val.append(X[0][0])
    X2_val.append(X[1][0])

y_val = np.stack(y_val)
X1_val = np.stack(X1_val)
X2_val = np.stack(X2_val)


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(val_gen.n)]

# predict probabilities

lr_probs = model.predict([X1_val, X2_val], verbose=1)




# calculate scores
ns_auc = roc_auc_score(y_val, ns_probs)
lr_auc = roc_auc_score(y_val, lr_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_val, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_val, lr_probs)

# plot the roc curve for the model
plt.figure(figsize=(5,5), dpi= 200)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the plot
plt.title('Logistic: ROC AUC=%.3f' % (lr_auc), fontweight='bold')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()
plt.show()



