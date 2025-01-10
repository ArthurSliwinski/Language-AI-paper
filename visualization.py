"""This notebook was used to derive the visualization presented in the paper"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#  labels

classes = ['center', 'left', 'right']
## cm of the random Classifier
conf_matrix1 = np.array([[11224, 6384, 7593],
                        [6486, 3709, 4381],
                        [7739, 4423, 5292]])

# create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix of Random Classifier', fontsize=20)
#plt.savefig("cm_random_class.png")
plt.show()

# cm of the Shallow Decision Tree

conf_matrix2 = np.array([[3464, 124, 174],
 [2558, 46,92],
 [2883, 97, 399]])

## create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix of Shallow Decision Tree', fontsize=20)
#plt.savefig("cm_shallow_tree.png")
plt.show()

## cm of the Logistic Regression with full vocabulary

conf_matrix3 = np.array([[4447, 872, 1207],
 [1470, 741, 699],
 [1522, 279, 1357]])

# create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix3, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix of Logistic Regression with full vocabulary', fontsize=20)
plt.savefig("cm_lr_full.png")
plt.show()

## cm of the Logistic Regression without self-labeling words

conf_matrix4 =np.array([[4419, 888, 1219], [1459, 754, 697], [1519, 283, 1356]])

# create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix4, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix of Logistic Regression without self-labels', fontsize=20)
plt.savefig("cm_lr_filter.png")
plt.show()