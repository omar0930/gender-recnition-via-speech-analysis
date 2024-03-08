#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import streamlit as st

init_notebook_mode(connected=True)


# In[2]:


df = pd.read_csv(r"D:\University\Gradution Project\voice.csv")  # reading the file as a csv


# In[3]:


df.sample(5)


# In[4]:


df.info()


# In[5]:


numeric_columns = df.select_dtypes(include=[np.number])
f, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(numeric_columns.corr(), annot=True, linewidths=0.5, linecolor="red", fmt='.1f', ax=ax)
plt.show()


# In[6]:


#seperate features and labels
X=df.iloc[:, :-1]

X.head()


# In[7]:


# to check for Nan values
df.isnull().sum()

y = []


# In[8]:


# encoding male=1 and female=0
for i in range(len(df.label)):
    if df.label[i] == 'male':
        y.append(1)
    elif df.label[i] == 'female':
        y.append(0)


# In[9]:


df = df.drop('label', axis=1)  # drop th ecolumn with labels
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)


# In[10]:


stdSc = StandardScaler()  # preprocessing
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.fit_transform(X_test)


# In[11]:


#neural networks trial

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), learning_rate_init=0.001)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[12]:


from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

C = [5, 15, 0.01, 0.1, 1, 10, 100, 1000]
kernels = ['rbf', 'linear', 'sigmoid', 'poly']

for i in C:
    for k in kernels:
        clf1 = svm.SVC(C=i, kernel=k)
        clf1.fit(X_train, y_train)
        
        y_pred = clf1.predict(X_test)
        
        print("SVM Accuracy (C={} & kernel={}):".format(i, k))
        print(accuracy_score(y_pred, y_test))
        
        cm = confusion_matrix(y_test, y_pred)
        f, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt='.0f', ax=ax)
        plt.show()
        plt.savefig('ConfusionMatrix_{}_{}.png'.format(i, k))
        plt.close()

        print("-----------------------------------------------------------")


# In[13]:


from sklearn.metrics import  f1_score
f1_score = f1_score(y_test, y_pred)
print("F1 Score:")
print(f1_score)


# In[14]:


# knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

model = KNeighborsClassifier()

k_range = list(range(1, 101))
weight_options = ['uniform', 'distance']

param_dist = dict(n_neighbors=k_range, weights=weight_options)

rand = RandomizedSearchCV(estimator=model, cv=10, param_distributions=param_dist, n_iter=10)
rand.fit(X_train, y_train)

print(rand.best_score_)
print(rand.best_params_)
print("KNN Accuracy (k={}): {}".format(rand.best_params_['n_neighbors'], rand.best_score_))


# In[15]:


import joblib
import numpy as np

def predict_class(feature_inputs):
    model = joblib.load(r"D:\\University\\Gradution Project\\neural_network_model_checkpoint.joblib")
    user_input = np.array(feature_inputs).reshape(1, -1)
    prediction = model.predict(user_input)
    return prediction[0]


# In[16]:


st.title('Neural Network Deployment with Streamlit')

feature_inputs = []
for i in range(20):
    feature_inputs.append(st.number_input(f'Enter Feature {i+1}:'))

user_input = np.array(feature_inputs).reshape(1, -1)

if st.button('Predict'):
    model = joblib.load(r"D:\\University\\Gradution Project\\neural_network_model_checkpoint.joblib")
    prediction = model.predict(user_input)
    st.success(f'The predicted class is: {prediction[0]}')

if st.checkbox('Show Model Information'):
    model = joblib.load(r"D:\\University\\Gradution Project\\neural_network_model_checkpoint.joblib")
    predictions = model.predict(X_test)    
    st.subheader('Confusion Matrix:')
    st.write(confusion_matrix(y_test, predictions))
    st.subheader('Classification Report:')
    st.write(classification_report(y_test, predictions))

