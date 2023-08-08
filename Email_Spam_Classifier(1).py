#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


# Data collection
#loading the data from csv file to a pandas dataframe
raw_mail_data=pd.read_csv('mail_data.csv')


# In[7]:


print(raw_mail_data)


# In[8]:


#replace the null values with the null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[9]:


# printing th first five rows from the data frame
mail_data.head()


# In[10]:


# checking the number of rows and columns in the data frame
mail_data.shape


# In[11]:


# label encoding
# label spam mail as 0: ham mail as 1:
mail_data.loc[mail_data['Category'] == 'spam','Category',]=0
mail_data.loc[mail_data['Category'] == 'ham','Category',]=1


# In[12]:


# Spam  - 0
# Ham - 1
# Seperatng the data as text and labels
X = mail_data['Message']

Y = mail_data['Category']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[15]:


# Splitting the data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# In[16]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)



# In[17]:


# Feature extraction
# Transform the text data to feature vectors that can be used as input to the logistic regression model

feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[18]:


print(X_train_features)


# In[19]:


# Training Model
# Logistic Regression
model = LogisticRegression()


# In[20]:


# Training the Logistic Regression Model with the training data
model.fit(X_train_features,Y_train)


# In[21]:


# Evaluating the trained model
# pRediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[22]:


print('Accuracy on training data : ',accuracy_on_training_data)


# In[23]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_testdata = accuracy_score(Y_test,prediction_on_test_data)


# In[24]:


print('Accuracy score on test data : ',accuracy_on_testdata)


# In[25]:


# Building a predictive System
input_mail = ["The IRS is Trying to Contact You"]
#convert text to features vectors
input_data_features=feature_extraction.transform(input_mail)


# Making prediction

prediction = model.predict(input_data_features)
# print(prediction)

if(prediction[0] == 1):
  print('Ham Mail')
    
else: print('Spam')


# In[ ]:





# In[ ]:




