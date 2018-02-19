
# coding: utf-8

# # Big Data Project : Etape 3

# In[2]:


import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
#import datetime
import matplotlib.dates as mdates
import pandas as pd
import csv
import seaborn as sns


# ### Importation des fichiers

# In[3]:


test = pd.read_csv('test.csv',names = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude'])
print( "Aperçu des données (Test):")
print(test[:10])

train = pd.read_csv('train.csv',names = ['Transaction', 'Montant', 'compteS','Solde source avant', 'Solde source après', 'compteD','Solde dest avant', 'Solde dest après', 'fraude'])

# Regroupement des deux fichiers en un seul pour la normalisation
data=pd.concat([train,test])
datap=pd.concat([train,test])

print(test.shape)
print(train.shape)
print(data.shape)

        


# #### LabelEncoder (transformation des strings en numérique)

# In[4]:


# TEST
test['lettreS']=test.compteS.str.get(0)
test['numS']=test.compteS.str[1:]
test['lettreD']=test.compteD.str.get(0)
test['numD']=test.compteD.str[1:]
from sklearn import preprocessing
for column in test.columns:
    if test[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        test[column] = le.fit_transform(test[column])
        
test[:10]


# TRAIN
train['lettreS']=train.compteS.str.get(0)
train['numS']=train.compteS.str[1:]
train['lettreD']=train.compteD.str.get(0)
train['numD']=train.compteD.str[1:]
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


X_train=train[['Transaction', 'Montant','Solde source avant', 'Solde source après', 'Solde dest avant', 'Solde dest après','lettreS','numS','lettreD','numD']]
y_train=train.fraude

X_test=test[['Transaction', 'Montant','Solde source avant', 'Solde source après', 'Solde dest avant', 'Solde dest après', 'lettreS','numS','lettreD','numD']]
y_test=test.fraude


# ### Visualisation des données 

# Nombre de fraudes dans le fichier train

# In[9]:


fraudes=train.fraude

counts=fraudes.value_counts()
print(counts)

counts[:10].plot(kind='barh', figsize=(25,6), fontsize=25)
plt.title("Proportion des fraudes et non fraudes dans le fichier train")
plt.show()


# Nombre de fraudes dans le fichier test

# In[10]:


fraudes=test.fraude

counts=fraudes.value_counts()
print(counts)

counts[:10].plot(kind='barh', figsize=(25,6), fontsize=25)
plt.title("Proportion des fraudes et non fraudes dans le fichier test")
plt.show()


# ### Preprocessing

# #### Normalisation

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[27]:


scaler_standard = StandardScaler()
scaler_standard.fit(X_train)

X_scaled = scaler_standard.transform(X_train)
X_scaled1 = scaler_standard.transform(X_test)
#X_test_scaled_standard  = scaler_standard.transform(X_test)


# #### PCA (réduction des dimensions)

# In[20]:


from sklearn.decomposition import PCA


# Dimension 2 : data

# In[28]:


pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

pca = PCA(n_components=2)
pca.fit(X_scaled1)
X_pca1 = pca.transform(X_scaled1)

print("Shape before PCA:", X_scaled.shape)
print("Shape after PCA:", X_pca.shape)


# In[18]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_pca[:,0], X_pca[:,1], c=data.fraude, marker='o',s=1)
plt.show()
fig.savefig('pca2.png')


# plt.figure(figsize=(8,6), dpi=80)
# plt.subplot(1,1,1)
# plt.scatter(X_pca[:,0], X_pca[:,1], c=data.fraude, marker='o',s=1)
# plt.xlim(0, 20)
# plt.show()

# Dimension 3 : data

# In[ ]:


pca_3d = PCA(n_components=3)
pca_3d.fit(X_scaled)
X_pca_3d = pca_3d.transform(X_scaled)
print("Shape before PCA:", X_scaled.shape)
print("Shape after PCA:", X_pca_3d.shape)


# Now we can visualize it in 3D.

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_3d[:,0], X_pca_3d[:,1], X_pca_3d[:,2],
           c=data.fraude, s=5)
plt.show()


# #### Visualisation de l’influence des paramètres

# In[6]:


#correlation entre les différents paramètres 
corrMatt = data[['Transaction', 'Montant','Solde source avant', 'Solde source après', 'Solde dest avant', 'Solde dest après','fraude']].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()


# In[ ]:


ax=sns.stripplot(x="Transaction", y="fraude", data=data, jitter=True)
#ax.set_xticklabels(['printemps', 'été', 'automne', 'hiver'], rotation=30)
plt.show()

#On remarque que les fraudes sont présentes seulement lors de deux types de transaction : transfer et cash_out. 


# In[ ]:


ax=sns.stripplot(x="Montant", y="fraude", data=data, jitter=True)
#ax.set_xticklabels(['printemps', 'été', 'automne', 'hiver'], rotation=30)
plt.show()


# In[ ]:


plt.hist(x="Transaction",data=datap,edgecolor="black",linewidth=2)

sommef=(data.fraude==1).sum()
print(sommef)


# #### Oversampling

# ### Apprentissage

# #### Arbre de décision

# In[6]:


get_ipython().run_line_magic('run', 'plot_interactive_tree.py')
plot_tree_progressive()


# In[32]:


# create and train a decision tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


# In[34]:


training_accuracy = tree.score(X_train,y_train)
test_accuracy = tree.score(X_test,y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)


# Modification de paramètres pour améliorer ses performances et temps de calcul

# In[35]:


tree = DecisionTreeClassifier(max_depth=33)
tree.fit(X_train, y_train)


# #### f-mesure = fonction(rappel, précision)
# 

# In[36]:


training_accuracy = tree.score(X_train,y_train)
test_accuracy = tree.score(X_test,y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)
predict=tree.predict(X_test)


# In[37]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, predict, average='weighted')


# In[12]:


# now we can save the DecisionTreeClassifier model as
# a tree image.
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", 
                class_names=["non fraude", "fraude"],
                impurity=False,
                filled=True)


# In[13]:



# we can visualize this .dot image with graphviz
import graphviz
with open("tree.dot") as f:
    breast_tree = f.read()
graphviz.Source(breast_tree)


# Prédiction sur le fichier test

# In[ ]:


predict=tree.predict(X_test)
print(predict[:10])
print(test.fraude[:10])
#print(test.fraude[:10])


# In[14]:


test.fraude[:10]


# #### K-cluster

# In[5]:


from sklearn.neighbors import KNeighborsClassifier

# with the documentation, we can see that one argument of
# KNeighborsClassifier is n_neighbors
model = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


# training
model.fit(X_train, y_train)

# evaluation
training_accuracy = model.score(X_train,y_train)
test_accuracy = model.score(X_test,y_test)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)
print("# Misclassified points =", (1 - accuracy) * len(X_test))


# #### Réseau de neurones

# In[7]:


from sklearn.neural_network import MLPClassifier

nn_relu = MLPClassifier(hidden_layer_sizes=(5,20,20,2),
                        activation="logistic", 
                        max_iter=5000,
                        random_state=0)
nn_relu.fit(X_train, y_train)
print("ReLu activation")
print("  Train accuracy:", nn_relu.score(X_train, y_train))
print("  Test  accuracy:", nn_relu.score(X_test, y_test))

