#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np

#Load data files
train=pd.read_csv("D:\\WORKPLACE\\Locker\\test\\Book1.csv") #Reading the csv file 


# In[33]:


#List of column names
#list(train)

#Sample of data
#train.head(10)

#Types of data columns
#train.dtypes

#Summary statistics
#train.describe()


# In[34]:


#(3)PREDICTIVE MODELLING
#Remove OBS# variable - Irrelevant
train=train.drop('OBS#',axis=1)


# In[35]:


#Create target variable
X=train.drop('RESPONSE',1)
y=train.RESPONSE


# In[36]:


#Build dummy variables for categorical variables
X=pd.get_dummies(X)
train=pd.get_dummies(train)


# In[43]:


#Split train data for cross validation
from sklearn.model_selection import train_test_split
x_train,x_validate,y_train,y_validate = train_test_split(X,y,test_size=0.2) #80 - 20 split.


# In[45]:


#(a)LOGISTIC REGRESSION ALGORITHM
#Fit model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#Predict values for cv data
pred_cv=model.predict(x_validate)

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_validate,pred_cv)) 
matrix=confusion_matrix(y_validate,pred_cv)



#(b)DECISION TREE ALGORITHM
#Fit model
from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion='gini') #Decision Tree Classification, calculation of Gini Index 
dt.fit(x_train,y_train) #Fitting the data to the model

#Predict values for cv data
pred_cv1=dt.predict(x_validate) #Predicting 

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv1)) 
matrix1=confusion_matrix(y_validate,pred_cv1) #Confusion Matrix



#(c)RANDOM FOREST ALGORITHM
#Fit model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

#Predict values for cv data
pred_cv2=rf.predict(x_validate)

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv2))
matrix2=confusion_matrix(y_validate,pred_cv2)



#(d)SUPPORT VECTOR MACHINE (SVM) ALGORITHM
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)

#Predict values for cv data
pred_cv3=svm_model.predict(x_validate)

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv3))
matrix3=confusion_matrix(y_validate,pred_cv3)



#(e)NAIVE BAYES ALGORITHM
from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)

#Predict values for cv data
pred_cv4=nb.predict(x_validate)

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv4))
matrix4=confusion_matrix(y_validate,pred_cv4)



#(f)K-NEAREST NEIGHBOR(kNN) ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
kNN=KNeighborsClassifier()
kNN.fit(x_train,y_train)

#Predict values for cv data
pred_cv5=kNN.predict(x_validate)

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv5)) 
matrix5=confusion_matrix(y_validate,pred_cv5)



#(g) GRADIENT BOOSTING MACHINE ALGORITHM
from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier()
gbm.fit(x_train,y_train)

#Predict values for cv data
pred_cv6=gbm.predict(x_validate)

#Evaluate accuracy of model
print(accuracy_score(y_validate,pred_cv6)) 
matrix6=confusion_matrix(y_validate,pred_cv6)


# In[29]:


predictions=pd.DataFrame(pred_cv6, columns=['predictions'])

