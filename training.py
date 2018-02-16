import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
#import datetime
import matplotlib.dates as mdates
import pandas as pd
import csv
import seaborn as sns
import pickle
# ### Importation des fichiers

# In[3]:


test = pd.read_csv('test.csv',names = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude'])

train = pd.read_csv('train.csv',names = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude'])

# Regroupement des deux fichiers en un seul pour la normalisation
data=pd.concat([train,test])
datap=pd.concat([train,test])
print("1")
# #### LabelEncoder (transformation des strings en numérique)

# In[4]:


# TEST
from sklearn import preprocessing
for column in test.columns:
    if test[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        test[column] = le.fit_transform(test[column])
        
test[:10]

# TRAIN
for column in train.columns:
    if train[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        train[column] = le.fit_transform(train[column])

# DATA 
for column in datap.columns:
    if datap[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        datap[column] = le.fit_transform(datap[column])


# #### Création des datasets train et test avec leurs labels

# In[5]:


X_train=train[['Transaction', 'Montant','compteS', 'Solde source avant', 'Solde source après', 'compteD', 'Solde dest avant', 'Solde dest après']]
y_train=train.fraude

X_test=test[['Transaction', 'Montant', 'compteS', 'Solde source avant', 'Solde source après','compteD',  'Solde dest avant', 'Solde dest après']]
y_test=test.fraude

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X_train)

# Apply regular SMOTE
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)

count=y_resampled.value_counts()
print(count)

# create and train a decision tree
from sklearn.metrics import precision_recall_fscore_support
import time
from sklearn.tree import DecisionTreeClassifier
start = time.time()
tree = DecisionTreeClassifier(max_depth=33)
tree.fit(X_train, y_train)
end = time.time()

# accuracy
training_accuracy = tree.score(X_resampled,y_resampled)
test_accuracy = tree.score(X_test,y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)
predict=tree.predict(X_test)
print(precision_recall_fscore_support(y_test, predict, average='weighted'))
print(end-start)

filename = "tree_model.sav"
pickle.dump(tree, open(filename,"wb"))

print('done')

