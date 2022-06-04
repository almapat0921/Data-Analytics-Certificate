#!/usr/bin/env python
# coding: utf-8

# # Blackwell Buying Patterns - Full Pipeline - student
# 
# 
# * Name: 
# * Updated: 2021.07.27
# 

# ### The objective of this project is to answer the following questions:
# 
# #### Task 1 
# * 1a) Do customers in different regions spend more per transaction?
# * 1b) Which regions spend the most/least? 
# * 2)  Is there a relationship between the number of items purchased and amount spent?
# 
# #### Task 2
# * 3a) Are there differences in the age of customers between regions? 
# * 3b) If so, can we predict the age of a customer in a region based on other demographic data?
# * 4a) Is there any correlation between age of a customer and if the transaction was made online or in the store? 
# * 4b) Do any other factors predict if a customer will buy online or in our stores?
# 
# #### Exercises using digits ds (does not need to be included in the notebook for Task 2): 
# * from sklearn import datasets
# * show example of data vs target
# * dir(digits)    # help on funcs that can be run with digits object
# * digits.data
# * digits.target
# 
# Resource for digits: https://www.c-sharpcorner.com/article/a-complete-scikit-learn-tutorial/
# 

# # Import packages

# In[1]:


# DS Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn Stuff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# helpers
get_ipython().run_line_magic('matplotlib', 'inline')

# Grahpviz 
from six import StringIO
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
import graphviz

import six
import sys
sys.modules['sklearn.externals.six'] = six

from sklearn.tree import plot_tree


# # Import data

# In[2]:


get_ipython().system('pwd')
get_ipython().system('ls')


# In[2]:


data = pd.read_csv('Demographic_Data.csv')
print(data.shape)


# # Evaluate data

# In[3]:


data.dtypes


# In[4]:


data.info()


# In[6]:


data.head()


# # Preprocess/data cleaning

# ### Duplicates

# In[7]:


data.duplicated().any()


# In[8]:


print(data[data.duplicated()].shape)
data[data.duplicated()]


# In[9]:


data = data.drop_duplicates()


# In[10]:


data.duplicated().any()


# ### Null values

# In[11]:


data.isnull().any()
data.isnull().sum()


# ### Discretize
# 
# * Discretize amount and age

# In[12]:


pd.cut(data.amount, bins=3, right=True).head()


# In[13]:


# Discretize amount - eg., 0-1000, 1001-2000, 2001+
amtBin = ['$0-700', '$701-1400', '$1401-2100', '$2101-3000']
cut_bins = [2, 700, 1503, 2252, 3000]
data ['amtBin'] = pd.cut(data ['amount'], bins = cut_bins, labels = amtBin)
data.head()


# In[14]:


# Get help for a function
get_ipython().run_line_magic('pinfo', 'pd.qcut')
# or help(pd.qcut)


# In[15]:


pd.cut(data['age'], bins = 3)
pd.cut(data ['age'], bins = 3). value_counts()


# In[18]:


pd.cut(data.age, bins=3, right=True).head()


# In[19]:


# Discretize age - eg., 18-33, 34-49, 50-64, 65+
ageBin = ['0-18', '19-37', '38-56', '57-75', '75-93']
cut_bins = [0, 17, 34, 51, 68, 85]
data ['ageBin'] = pd.cut(data ['age'], bins = cut_bins, labels = ageBin)
data.head()


# In[18]:


# add amtBin and ageBin to the dataset


# # Analyze Data
# ### Statistical Analysis

# In[20]:


# output statistics
data.describe()


# ### Visualizations

# In[21]:


header = data.dtypes.index
print(header)


# #### Histogram

# In[22]:


plt.hist(data['in-store'])
plt.ylabel('amount')
plt.xlabel('in-store')
plt.show()


# In[23]:


plt.hist(data['amtBin'])
plt.show()


# In[24]:


data.hist()
plt.show()


# #### Scatter

# In[25]:


# Scatter plot example
data_sample = data
x = data_sample['age']
y = data_sample['amount']
plt.scatter(x,y, marker='o')
# assignment: add axis titles
plt.ylabel('amount')
plt.xlabel('age')
plt.show()


# In[26]:


# Box plot example
# eval col names/features
header = data.dtypes.index
print(header)
# plot
A = data['amount']
plt.boxplot(A,0,'gD')
plt.show()


# #### Stacked Col 
# Focus on answering the following business questions:
# * 1a) Do customers in different regions spend more per transaction (number of obs per spend category)?
# * 1b) Which regions spend the most/least (overall - just from looking at the chart)? 
# * 3a) Are there differences in the age of customers between regions?

# In[26]:


#1a) Do customers in different regions spend more per transaction (number of obs per spend category)?


# In[27]:


np.random.seed(1)
data.groupby('region')['items']    .value_counts()    .unstack(level=1)    .plot.bar(stacked=True)
plt.ylabel('amount')
plt.xlabel('items')
plt.show()


# In[28]:


#1b) Which regions spend the most/least (overall - just from looking at the chart)?


# In[28]:


np.random.seed(1)
data.groupby('region')['amtBin']    .value_counts()    .unstack(level=1)    .plot.bar(stacked=True)
plt.ylabel('amount')
plt.xlabel('region')
plt.show()


# In[30]:


#3a) Are there differences in the age of customers between regions?


# In[29]:


data.groupby(['region']).mean()


# In[30]:


np.random.seed(1)
data.groupby('region')['ageBin']    .value_counts()    .unstack(level=1)    .plot.bar(stacked=True)
plt.ylabel('amount')
plt.xlabel('region')
plt.show()


# # Feature Selection
# For this task, you will not be selecting features. Instead, focus on answering the following questions:
# * 2) Is there a relationship between the number of items purchased and amount spent?
# * 4a) Is there any correlation between age of a customer and if the transaction was made online or in the store?
# 

# ### Correlation

# In[31]:


corr_mat = data.corr()
print(corr_mat)


# In[32]:


# plot heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_mat, vmax=1.0, center=0, fmt='.2f',
square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.show();


# ### Covariance

# In[33]:


cov_mat = data.cov()
print(cov_mat)


# # Train/Test Sets
# * The modeling (predicitive analytics) process begins with splitting data in to train and test sets. 
# * Focus on buiding models to answer the following questions:
# * 3b) Can we predict the age of a customer in a region based on other demographic data? (Decision tree.)
# * 4a) Is there any correlation between age of a customer and if the transaction was made online or in the store? (In addition to correlation analysis, a decision tree can also provide insight.)
# * 4b) Do any other factors predict if a customer will buy online or in our stores? (Decison tree.)
# 

# ### Set random seed

# In[34]:


seed = 123


# ### Split datasets into X (IVs) and y (DV)
# * For each ds, split into X, y
# * oob (out-of-box; no feature selection or feature engineering)

# In[35]:


# pring column names for quick reference
data.columns


# In[36]:


ageBin = ['0-18', '19-37', '38-56', '57-75', '75-93']
cut_bins = [0, 17, 34, 51, 68, 85]
data ['ageBin'] = pd.cut(data ['age'], bins = cut_bins, labels = ageBin)
amtBin = ['$0-700', '$701-1400', '$1401-2100', '$2101-3000']
cut_bins = [2, 700, 1503, 2252, 3000]
data ['amtBin'] = pd.cut(data ['amount'], bins = cut_bins, labels = amtBin)
inplace=True
data.head()


# In[37]:


data["ageBin"] = data["age"]
data["ageBin"].replace({"0-18":1,"19-37":2,"38-56":3,"57-75":4,"75-93":5}, inplace =True)


data["amtBin"] = data["amount"]
data["amtBin"].replace({"$0-700":1,"$701-1400":2,"$1401-2100":3,"$2101-3000":4}, inplace = True)
data


# In[38]:


## For question 3b): set region as dv
Y_oobQ3 = data['region']
# select IVs/features
X_oobQ3 = data[['in-store','ageBin','items','amtBin']]
# select Age/Amt binned features
X_oobQ3ageAmt = data[['in-store','ageBin','items','amtBin']]
x.head()


# In[39]:


## For questions Q4a/Q4b): set in-store as dv 
Y_oobQ4 = data['in-store']
# select IVs/features
X_oobQ4 = data[['region','age','items','amount']]
# select Age/Amt binned features
X_oobQ4ageAmt = data[['region','ageBin','items','amtBin']]


# In[40]:


Y_oobQ4B = data['in-store']
X_oobQ4B = data[['region','age','items','amount']]
# select Age/Amt binned features
X_oobQ4BageAmt = data[['region','ageBin','items','amtBin']]


# ### Create train and test sets

# In[41]:


# Q3b) region as dv; un-binned data

X_trainQ3, X_testQ3, Y_trainQ3, Y_testQ3 = train_test_split(X_oobQ3, 
                                            Y_oobQ3, 
                                            test_size = .30, 
                                            random_state = seed)

print(X_trainQ3.shape, X_testQ3.shape)
print(Y_trainQ3.shape, Y_testQ3.shape)


# In[42]:


# Q4a) in-store as dv

X_trainQ4, X_testQ4, Y_trainQ4, Y_testQ4 = train_test_split(X_oobQ4, 
                                            Y_oobQ4, 
                                            test_size = .30, 
                                            random_state = seed)

print(X_trainQ4.shape, X_testQ4.shape)
print(Y_trainQ4.shape, Y_testQ4.shape)


# In[43]:


# Q4b) in-store as dv; age binned & amount binned

X_trainQ4B, X_testQ4B, Y_trainQ4B, Y_testQ4B =  train_test_split(X_oobQ4B, 
                                            Y_oobQ4B, 
                                            test_size = .30, 
                                            random_state = seed)

print(X_trainQ4B.shape, X_testQ4B.shape)
print(Y_trainQ4B.shape, Y_testQ4B.shape)


# # Modeling
# #### Two purposes of modeling:
# * 1) Evaluate patterns in data
# * 2) Make predictions
#   

# ## Evaluate patterns in data using a Decision Tree (DT)

# In[110]:


#3b) Can we predict the age of a customer in a region based on other demographic data? (Decision tree.)


# ### dv = region
# 

# In[44]:


# use the dataset that has region as the dv

# run code to fit and predict the DecisionTreeClassifier


# select DT model for classification
dt = DecisionTreeClassifier(max_depth=3)

# train/fit the mode using region as dv, and binned by age & amt
dtModel3 = dt.fit(X_trainQ3, Y_trainQ3)

# make predicitons with the trained/fit model
dtPred3 = dtModel3.predict(X_testQ3)

# performance metrics
print(accuracy_score(Y_testQ3, dtPred3))
print(classification_report(Y_testQ3, dtPred3))


# In[45]:


fig = plt.figure(figsize=(25,20))
tree = plot_tree(dtModel3, feature_names=data.columns,class_names=['0', '1', '2', '3'], filled=True)


# In[ ]:


# 3b) Is age in the DT? If so, what decision rules incorporate age? 

# Other questions: From the above DT, is the 'items' feature in the tree? 
# What does it mean if it is, or is not, in the tree?


# ### dv = in-store
# 

# In[60]:


# run DT model

# code goes here
# select DT model for classification
dt = DecisionTreeClassifier(max_depth=3)

# train/fit the mode using region as dv, and binned by age & amt
dtModel4 = dt.fit(X_trainQ4, Y_trainQ4)

# make predicitons with the trained/fit model
dtPred4 = dtModel4.predict(X_testQ4)

# performance metrics
print(accuracy_score(Y_testQ4, dtPred4))
print(classification_report(Y_testQ4, dtPred4))


# In[61]:


# visualize DT

# code goes here
fig = plt.figure(figsize=(25,20))
tree = plot_tree(dtModel4, feature_names=data.columns,class_names=['0', '1'], filled=True)


# In[ ]:


# 4b) Do any factors predict if a customer will buy online or
# in our stores?


# ## Make Predictions
# * Focus on the following question: Can a model be developed that can accurately classify where a transaction took place (in-store/online)?

# ### Select models

# In[62]:


# create empty list and then populate it with the following models

models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

# create empty lists to hold results and model names
results = []
names = []


# In[63]:


algos_Class = []
algos_Class.append(('Random Forest Classifier', RandomForestClassifier()))
algos_Class.append(('Decision Tree Classifier', DecisionTreeClassifier()))


# In[64]:


from sklearn import ensemble
rf_clf= ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_trainQ4, Y_trainQ4)
rf_clf.score(X_testQ4, Y_testQ4)


# In[65]:


gb_clf= ensemble.GradientBoostingClassifier(n_estimators=40)
gb_clf.fit(X_trainQ4, Y_trainQ4)
gb_clf.score(X_testQ4, Y_testQ4)


# ### CV (cross-validation)
# 

# In[66]:


# Set in-store as dv; unbinned data

Y_oobQ4 = data['in-store']
# select IVs/features
X_oobQ4 = data[['region','age','items','amount']]
for name, model in models:
    kfold = KFold(n_splits=3, random_state=seed, shuffle=True)
    result = cross_val_score(model,
                             X_trainQ4,
                             Y_trainQ4,   
                             cv=kfold,
                             scoring='accuracy')
    names.append(name)
    results.append(result)
    #msg = '%s: %.4f (%.4f)' % (name, result.mean(), result.std())
    #print(msg)

# print results
for i in range(len(names)):
    print(names[i],results[i].mean())


# In[67]:


# Same as above, but using binned data for age and amount
Y_oobQ4 = data['in-store']
# select Age/Amt binned features
X_oobQ4ageAmt = data[['region','ageBin','items','amtBin']]
for name, model in models:
    kfold = KFold(n_splits=3, random_state=seed, shuffle=True)
    result = cross_val_score(model,
                             X_trainQ4,
                             Y_trainQ4,   
                             cv=kfold,
                             scoring='accuracy')
    names.append(name)
    results.append(result)
    #msg = '%s: %.4f (%.4f)' % (name, result.mean(), result.std())
    #print(msg)

# print results
for i in range(len(names)):
    print(names[i],results[i].mean())


# In[ ]:


# Based all of the above model runs, which is the most accurate?
# Is the accuracy of the top model higher than 75%?


# In[ ]:


get_ipython().run_line_magic('pinfo', 'cross_val_score')

