
# coding: utf-8

# In[1]:


## Importing Necessary Files and Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
log=linear_model.LogisticRegression()
train=pd.read_csv('train.csv')
test_x=pd.read_csv('test_X.csv')
test_y=pd.read_csv('test_Y.csv')


# In[2]:


## Preparing Train File
train=train.replace(['male', 'female'], ['0','1'])
train=train.replace(['C', 'S','Q'], ['0','1','2'])
train=train.drop(columns=['Ticket','Cabin','Name','PassengerId'])
train=train.fillna(0)


# In[3]:


## Preparing Test File
test_x=test_x.replace(['male', 'female'], ['0','1'])
test_x=test_x.replace(['C', 'S','Q'], ['0','1','2'])
test_x=test_x.drop(columns=['Ticket','Cabin','Name','PassengerId'])
test_x=test_x.fillna(0)


# In[4]:


## Preparing the Actuall Survival File of the Test file to Check Accuracy
test_y=pd.read_csv('test_Y.csv')
test_y=test_y.drop(columns=['PassengerId'])


# In[5]:


## Splitting the Train File to Train the Model
train_x=train.drop(['Survived'], axis=1)
train_y=train['Survived']


# In[6]:


## Fitting The Train File to the Model 
log.fit(train_x,train_y)


# In[7]:


## Performing Prediction on Test File
pred = log.predict(test_x)


# In[8]:


## Checking Accuracy Against the Actual Survival Data
accuracy_score(test_y,pred)

