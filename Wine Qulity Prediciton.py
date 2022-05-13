#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[67]:


import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
sns.set()


# # Importing Dataset

# In[4]:


df=pd.read_csv(r'C:\Users\Sahand\Downloads\WineQT.csv')


# # Exploring Dataset

# In[5]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.tail()


# In[13]:


# Missing Values Check Point

df.isnull().sum()


# In[14]:


# Cardinality
df.nunique().sort_values()


# In[15]:


# drop id

df.drop('Id', axis=1, inplace=True)


# In[16]:


# Data Information

df.info()


# # Handling Outlires
# 

# In[17]:


# #Calculate z-score
z = np.abs(stats.zscore(df))

threshold = 3

#Keep rows with Z-score less than 3
df = df[(z < 3).all(axis=1)]


# # Visualization

# In[24]:


fig,ax=plt.subplots(6,2,figsize=(10,25))
sns.countplot(x=df.quality,ax=ax[0][0]).set_title('Target Distribution',size=15)
sns.boxplot(x=df.quality,y=df['volatile acidity'],ax=ax[0][1])
sns.boxplot(x=df.quality,y=df['citric acid'],ax=ax[1][0])
sns.boxplot(x=df.quality,y=df['residual sugar'],ax=ax[1][1])
sns.boxplot(x=df.quality,y=df['chlorides'],ax=ax[2][0])
sns.boxplot(x=df.quality,y=df['free sulfur dioxide'],ax=ax[2][1])
sns.boxplot(x=df.quality,y=df['total sulfur dioxide'],ax=ax[3][0])
sns.boxplot(x=df.quality,y=df['density'],ax=ax[3][1])
sns.boxplot(x=df.quality,y=df['pH'],ax=ax[4][0])
sns.boxplot(x=df.quality,y=df['sulphates'],ax=ax[4][1])
sns.boxplot(x=df.quality,y=df['alcohol'],ax=ax[5][0])
sns.boxplot(x=df.quality,y=df['fixed acidity'],ax=ax[5][1])


# # Finding Correlation Between Features

# In[30]:


corr = df.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, cmap="Reds", annot_kws={"fontsize":12})
plt.title(" Features Correlation ")


# In[31]:


sns.pairplot(df,corner=True, hue='quality',
            x_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity'],
            y_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity']
            )


# In[50]:


X = df.drop("quality", axis=1)
y = df["quality"]


# In[60]:


df = df[df.columns]
cm = sns.light_palette("green", as_cmap=True)
df.head(30).style.background_gradient(cmap=cm)


# In[56]:


figure, axs = plt.subplots(3, 4)
figure.set_size_inches(16, 12)

cols = X.columns

for i in range(3):
    for j in range(4):
        if (4*i+j)<11:
            sns.histplot(X[cols[4*i+j]], ax=axs[i, j])


# In[57]:


sns.histplot(y)


# "Fixed acidity" and "free sulfur dioxide" parameters are suspicious because of have not bad correlation.
# 
# 

# In[64]:


X.info()


# In[70]:


#Import library
from imblearn.over_sampling import SMOTE

#Define inputs and output
X=df.drop(['quality'],axis=1)
y=df['quality']

#Do oversampling
strategy={4:350,7:350,8:350}
oversample = SMOTE(sampling_strategy=strategy)
X, y = oversample.fit_resample(X, y)


# In[71]:


sns.countplot(x=y)


# #  Modeling

# In[73]:


#Normalization
scaler=StandardScaler()
X=scaler.fit_transform(X)

#Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=30)


# In[76]:


# Cross Validation

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[77]:


#Define model
RandomForest = RandomForestClassifier(n_estimators=200,max_depth=15)


# In[78]:


#Get cross validation score
scores=evaluate_model(RandomForest,X,y)
print('Random Forest Classifier Accuracy : ', np.mean(scores))


# In[80]:


#Fit and predict output for X_valid
RandomForest.fit(X_train,y_train)
y_pred = RandomForest.predict(X_valid)

print(classification_report(y_pred,y_valid))

