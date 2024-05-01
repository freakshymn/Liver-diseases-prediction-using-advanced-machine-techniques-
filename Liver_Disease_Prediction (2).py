#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


dataset=pd.read_csv("Liver Patient Dataset (LPD)_train.csv",encoding= 'unicode_escape')


# In[6]:


dataset.head()


# In[7]:


dataset.describe()


# In[8]:


dataset.shape


# In[9]:


dataset.columns


# #Data Cleaning
# 
# Checking Duplicate tuples, if any will be removed

# In[10]:


dataset.duplicated()


# In[11]:


dataset.duplicated().sum()


# In[12]:


dataset = dataset.drop_duplicates()
print( dataset.shape )


# #checking Missing Values

# In[13]:


dataset.isna().sum()


# In[14]:


sns.boxplot(data = dataset, x= 'A/G Ratio Albumin and Globulin Ratio' )


# Q1 = 0.700 
# 
# Q3 = 1.1000
# 
# IQR = 0.4
# 
# 1.5 * 0.4 = 0.6
# 
# Q1 - 0.6 = 0.1
# 
# Q3 + 0.6 = 1.7
# 
# Anything away between 0.1 and 1.7 is considered to be an Outlier
# 
# 
# 
# 

# In[15]:


dataset['A/G Ratio Albumin and Globulin Ratio'].mode()


# In[16]:


dataset['A/G Ratio Albumin and Globulin Ratio'].median()


# In[17]:


dataset['A/G Ratio Albumin and Globulin Ratio'].mean()


# In[18]:


dataset['A/G Ratio Albumin and Globulin Ratio'] = dataset['A/G Ratio Albumin and Globulin Ratio'].fillna(dataset['A/G Ratio Albumin and Globulin Ratio'].median())


# In[19]:


dataset.isna().sum()


# #Male Vs Female

# In[20]:


import seaborn as sns
sns.countplot(data = dataset, x='Gender of the patient', label='count')


# In[21]:


Male, Female = dataset['Gender of the patient'].value_counts()
print('Number of patients that are male: ',Male)
print('Number of patients that are female: ',Female)


# #Encoding the Gender Column
# 
# Label Male as 1 and Female as 0

# In[24]:


def partition(x):
    if x == 'Male':
        return 1
    return 0

dataset['Gender of the patient'] = dataset['Gender of the patient'].map(partition)


# In[25]:


dataset


# #Converting Output Column 'Dataset' to 0's and 1's
# 
# Dataset i.e output value has '1' for liver disease and '2' for no liver disease so let's make it 0 for no disease to make it convinient

# In[26]:


def partition(x):
    if x == 2:
        return 0
    return 1

dataset['Result'] = dataset['Result'].map(partition)


# In[27]:


dataset['Result']


# #Correlation Matrix

# In[28]:


plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr())


# In[21]:


dataset.dropna(inplace=True)


# In[23]:


# Convert categorical variables to numerical using label encoding
le = LabelEncoder()
data['Gender of the patient'] = le.fit_transform(data['Gender of the patient'])
data['Result'] = le.fit_transform(data['Result'])


# In[25]:


# Split the data into features and labels
X = data.drop(['Result'], axis=1)
y = data['Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


# Scale the training and testing sets separately
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # logistic regression

# In[41]:



# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Reading the data
data=pd.read_csv(r"Liver Patient Dataset (LPD)_train.csv",encoding= 'unicode_escape')
data.head()

# Preprocessing the data
data = data.dropna()
data['Gender of the patient'] = data['Gender of the patient'].map({'Male': 1, 'Female': 0})

# Splitting the data into training and testing sets
X = data.drop('Result', axis=1)
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Predicting the test set results
y_pred = lr.predict(X_test)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Outputting the results
print("Accuracy:", accuracy*100)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", auc)
print("Confusion Matrix:\n", cm)


# # random forest

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data=pd.read_csv(r"Liver Patient Dataset (LPD)_train.csv",encoding= 'unicode_escape')
data.head()
# Preprocessing: Replace missing values with median
data.fillna(data.median(), inplace=True)
from sklearn.preprocessing import OneHotEncoder
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Result', axis=1), 
                                                    data['Result'], test_size=0.2, random_state=42)
# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.preprocessing import OneHotEncoder

# Create a OneHotEncoder object
onehot = OneHotEncoder()

# Encode the categorical variables in the training and testing data
X_train_encoded = onehot.fit_transform(X_train)
X_test_encoded = onehot.transform(X_test)

# Fit the model on the training data
rf_clf.fit(X_train_encoded, y_train)

# Predict on the testing data
y_pred = rf_clf.predict(X_test_encoded)
#Evaluate the model using various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the performance metrics
print('Accuracy:', accuracy*100)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('ROC AUC score:', roc_auc)


# # support vector machine

# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv(r"Liver Patient Dataset (LPD)_train.csv", encoding='unicode_escape')

# Preprocessing: Replace missing values with median
data.fillna(data.median(), inplace=True)

# Preprocessing: Encode categorical variables
label_encoder = LabelEncoder()
data['Gender of the patient'] = label_encoder.fit_transform(data['Gender of the patient'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Result', axis=1), data['Result'], test_size=0.2, random_state=42)

# Preprocessing: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM classifier
svm_clf = SVC(kernel='linear', C=1, random_state=42)

# Fit the model on the training data
svm_clf.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_clf.predict(X_test)

# Evaluate the model using various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the performance metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('ROC AUC score:', roc_auc)


# #saving the models

# # KNN

# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv(r"Liver Patient Dataset (LPD)_train.csv", encoding='unicode_escape')

# Preprocessing: Replace missing values with median
data.fillna(data.median(), inplace=True)

# Preprocessing: Encode categorical variables
label_encoder = LabelEncoder()
data['Gender of the patient'] = label_encoder.fit_transform(data['Gender of the patient'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Result', axis=1), data['Result'], test_size=0.2, random_state=42)

# Preprocessing: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the training data
knn_clf.fit(X_train, y_train)

# Predict on the testing data
y_pred = knn_clf.predict(X_test)

# Evaluate the model using various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the performance metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('ROC AUC score:', roc_auc)


# In[28]:


import pickle
pickle.dump(rf_clf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('sc.pkl', 'wb'))


# In[ ]:




