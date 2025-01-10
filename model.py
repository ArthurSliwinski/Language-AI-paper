import pandas as pd
import numpy as np # ensure using numpy version that preceeds 2.0.0
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loader_2 import df_train, df_test, vectorizer
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# First the train and test X data are loaded in from the npz files which return data of the csr sparse matrix type
train_X = sp.load_npz('x_train.npz')
test_X = sp.load_npz('x_test.npz')

# Next the train and test Y data containing the labels is loaded in from the data_loader_2 file
train_Y = df_train['political_leaning']
test_Y = df_test['political_leaning']

# Next we define our model, using the lbfgs solve, l2 norm as penalty, max-iter of 20, and a random state for reproducability
model = LogisticRegression(penalty='l2', C=1., solver='lbfgs', max_iter=250, random_state=42)

# The model is fit to the training data
LR = model.fit(train_X, train_Y)

# We use the model to predict the labels of the test data
pred_Y = LR.predict(test_X)

# These are several evaluation metrics so the model can be evaluated and parameters tuned for better performance
cm = confusion_matrix(test_Y, pred_Y)
ac = accuracy_score(test_Y, pred_Y, normalize=True)
F1 = f1_score(test_Y, pred_Y, average='macro')
training_accuracy = model.score(train_X, train_Y)

# Next the top n features and corresponding features are extracted from the model
coef = model.coef_[0]
coef = np.abs(coef)
top_ind = np.argpartition(coef, -150)[-150:]
top_coef = coef[top_ind]

important_feature_indices = top_ind
feature_names = vectorizer.get_feature_names_out()
important_features = [feature_names[i] for i in important_feature_indices]

# The top features and coefficients are saved as a dataframe and then possibly exported to an excel file
df_top_tokens = pd.DataFrame({
    "Feature Name": important_features,
    "Coefficient": top_coef
})
df_top_tokens = df_top_tokens.sort_values(by='Coefficient', ascending=False)

### The code below can be uncommented to save the dataframe with important features in an excel file
# Save the DataFrame to a CSV file
# df_top_tokens.to_csv("top_tokens.csv", index=False)

# This wraps up the model file, the code belwo prints relevant evaluation metrics
print("Training Accuracy: ", training_accuracy)
print("Confusion matrix: ", cm)
print('f1 score: ', F1)
print('test accuracy: ', ac)
print('Model ....... complete')
