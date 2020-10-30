#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


data = pd.read_csv("diabetes.csv")


# In[76]:


data.shape


# In[77]:


data.head()


# In[78]:


data.describe()


# In[79]:


data.isnull().values.any()


# In[80]:


data.corr()


# In[81]:


corrmat = data.corr()
top_corr_features = corrmat.index[0:8]
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="Reds")
plt.xticks(rotation=90)
plt.yticks(rotation=360)


# In[82]:


plt.figure(figsize=(5,5))
sns.countplot(x=data['Outcome'])
plt.title("Distribution of the data on the basis of the outcome")
plt.xlabel('Diabetes Outcome')


# In[83]:


data['Outcome'].value_counts()


# In[84]:


## Train Test Split

from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',]
predicted_class = ['Outcome']


# In[85]:


X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[86]:


print("total number of rows : {0}".format(len(data)))
print("number of rows missing Glucose: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing BloodPressure: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("number of rows missing Insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("number of rows missing BMI: {0}".format(len(data.loc[data['BMI'] == 0])))
print("number of rows missing DiabetesPedigreeFunction: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing Age: {0}".format(len(data.loc[data['Age'] == 0])))
print("number of rows missing SkinThickness: {0}".format(len(data.loc[data['SkinThickness'] == 0])))


# In[87]:


from sklearn.impute import SimpleImputer 

fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[88]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


# In[89]:



params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[90]:


classifier=xgb.XGBClassifier()


# In[91]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[92]:


random_search.fit(X_train,y_train.ravel())


# In[93]:


random_search.best_estimator_


# In[94]:


classifier=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.0, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[99]:


classifier.fit(X_train,y_train)


# In[95]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_train,y_train.ravel(),cv=10)


# In[96]:


score


# In[97]:


score.mean()


# In[100]:


y_pred=classifier.predict(X_test)


# In[102]:


from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)
score=accuracy_score(y_test,y_pred)


# In[103]:


print(cm)
print(score)


# In[105]:


pickle.dump(classifier, open('model.pkl','wb'))


# In[108]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:




