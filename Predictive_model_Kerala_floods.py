#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.metrics import classification_report


# In[2]:


# import data

df = pd.read_csv('kerala_floods.csv')

df.head()


# In[3]:


# explore data

df.info()

df.shape

df.describe()


# In[4]:


# check null values

df.isnull().sum()

df.corr()


# In[5]:


# replace values

df['FLOODS'].replace(['YES','NO'],[1,0], inplace = True)

df.head()


# In[6]:


# define X & Y

X = df.iloc[:,1:14]
Y = df.iloc[:,-1]


# In[7]:


# select top 3 features

best_features = SelectKBest(score_func = chi2, k = 3)
fit = best_features.fit(X,Y)



# In[8]:


# create dataframe for features

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)


# In[9]:


# combine features and scores

feature_scores = pd.concat([df_columns,df_scores],axis = 1)
feature_scores.columns = ['Features','Scores']

feature_scores.sort_values(by = 'Scores')


# In[10]:


# get the top three features and target column

X = df[['SEP','JUN','JUL']]
Y = df['FLOODS']


# In[11]:


# split the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 100)


# In[12]:


# create logistic regression body

logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[13]:


# predict the model

y_pred = logreg.predict(X_test)
print(X_test)
print(y_pred)



# In[14]:


print('Accuracy:' ,metrics.accuracy_score(y_test, y_pred))
print('Recall:',metrics.recall_score(y_test, y_pred, zero_division=1))
print('Precision:',metrics.precision_score(y_test, y_pred, zero_division=1))
print('CL Report:',metrics.classification_report(y_test, y_pred, zero_division=1))

