#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\ezkiska\Videos\Imarticus\Python\4th Week 28th and 29th Dec\29th Dec Practical  DT, RF & XG")

bank = pd.read_csv('bank-additional-full.csv', sep=';')


# In[2]:


bank.head()


# In[3]:


#  y (response) : Traget variable
# convert the response to numeric values and store as a new column
bank['outcome'] = bank.y.map({'no':0, 'yes':1})
bank.head()


# In[4]:


bank = bank.drop(['y'], axis = 1)


# In[5]:


bank.head()


# In[6]:


bank.isna().sum()


# We observe no nulls present in this case hence Null value treatement won't be required here

# In[7]:


bank.describe()


# In[8]:


bank.describe(include = ['O']) #includes object data type also


# In[9]:


# Visualizing Target Variable
sns.countplot(x=bank['outcome'])
plt.xlabel('Subscribed for Term deposit')


# In[10]:


np.round(len(bank['outcome'][bank['outcome'] == 0])/bank.shape[0],2)


# In[11]:


#checeking all data types in dataframe
bank.dtypes  


# In[12]:


catCols = bank.dtypes[bank.dtypes == 'object'].index.tolist()


# In[13]:


catCols


# In[14]:


data_num = bank.select_dtypes(include = ['float64', 'int64'])


# In[15]:


#checking contents of Numerical Columns
data_num.head()


# In[16]:


data_cat = bank.select_dtypes(include = ['object'])


# In[17]:


#checking contents of categorical Columns
data_cat.head()


# ### Feature Engineering on Categorical Variables

# #### Education

# In[18]:


data_cat['education'].value_counts()


# In[19]:


data_cat['outcome'] = bank['outcome']


# In[20]:


data_cat.head()


# In[21]:


plt.figure(figsize=(10,7))
sns.countplot(y='education',hue='outcome',data=data_cat)
plt.tight_layout()


# In[22]:


data_cat['job'].value_counts()


# In[23]:


plt.figure(figsize=(10,7))
sns.countplot(y='job',hue='outcome',data=bank)
plt.tight_layout()


# In[24]:


data_cat['marital'].value_counts()


# In[25]:


plt.figure(figsize=(10,7))
sns.countplot(x='marital',hue='outcome',data=data_cat)
plt.tight_layout()


# In[26]:


data_cat['default'].value_counts()


# In[27]:


data_cat['outcome'][data_cat['default']=='yes'] # checks who all 3 yes candidates are from outcome column


# In[28]:


tab = pd.crosstab(data_cat['default'],  data_cat['outcome'],margins = False)
prop = []
for i in range(tab.shape[0]):
    value = tab.iloc[i,1]/tab.iloc[i,0]
    prop.append(value)
tab['prop'] = prop #


# In[29]:


def createProportions(df,colName, dependentColName):
    
    tab = pd.crosstab(df[colName],  df[dependentColName],margins = False)
    prop = []
    for i in range(tab.shape[0]):
        value = tab.iloc[i,1]/tab.iloc[i,0]
        prop.append(value)
    tab['prop'] = prop

    return tab


# In[30]:


def categoryRename(df, colName, oldName, newName):
    
    df[colName][df[colName]==oldName]=newName
    #x = df[colName].value_counts()
    return df


# In[31]:


# club yes with unknown based on 1 proportion
data_cat = categoryRename(data_cat,'default', 'yes', 'unknown') #using previous function


# In[32]:


data_cat['default'].value_counts()


# In[33]:


plt.figure(figsize=(6,5))
sns.countplot(x='default',hue='outcome',data=data_cat)
plt.tight_layout()


# In[34]:


data_cat['housing'].value_counts()


# In[35]:


plt.figure(figsize=(8,6))
sns.countplot(x='housing',hue='outcome',data=data_cat)
plt.tight_layout()


# In[36]:


data_cat['loan'].value_counts()


# In[37]:


plt.figure(figsize=(8,6))
sns.countplot(x='loan',hue='outcome',data=bank)
plt.tight_layout()


# In[38]:


data_cat['contact'].value_counts()


# In[39]:


plt.figure(figsize=(8,6))
sns.countplot(x='contact',hue='outcome',data=bank)
plt.tight_layout()


# In[40]:


data_cat['month'].value_counts()


# In[41]:


plt.figure(figsize=(10,7))
sns.countplot(y='month',hue='outcome',data=bank)
plt.tight_layout()


# In[42]:


data_cat['day_of_week'].value_counts()


# In[43]:


plt.figure(figsize=(10,7))
sns.countplot(y='day_of_week',hue='outcome',data=bank)
plt.tight_layout()


# In[44]:


tab1 = createProportions(data_cat,'day_of_week', 'outcome')
tab1


# In[45]:


data_cat['poutcome'].value_counts()


# In[46]:


plt.figure(figsize=(8,6))
sns.countplot(x='poutcome',hue='outcome',data=bank)
plt.tight_layout()
plt.show()
plt.close()


# In[47]:


tab2 = createProportions(data_cat,'poutcome', 'outcome')
tab2


# ### Checking Numerical columns

# In[48]:


plt.figure(figsize = (8,6))
sns.distplot(data_num['age'])
plt.show()


# In[49]:


data_num['pdays'].value_counts()


# In[50]:


tab3 = createProportions(data_num,'pdays', 'outcome')
tab3 # 999 has maximum occurances so ideally we will frop this column


# In[51]:


data_num.pdays[data_num.pdays==999]=35 ## 
plt.figure(figsize = (10,7))
sns.distplot(data_num['pdays'])
plt.show()


# In[52]:


data_num['pdays_band'] = pd.cut(data_num['pdays'], 5)


# In[53]:


data_num['pdays_band'].value_counts()


# In[54]:


data_num[['pdays_band', 'outcome']].groupby(['pdays_band'], 
        as_index=False).mean().sort_values(by='pdays_band', ascending=True)


# In[55]:


data_num.loc[ data_num['pdays'] <= 7.0, 'pdays'] = 0
data_num.loc[(data_num['pdays'] > 7.0) & (data_num['pdays'] <= 14.0), 'pdays'] = 1
data_num.loc[ data_num['pdays'] > 14, 'pdays'] = 2


# In[56]:


data_num.pdays.value_counts()


# In[57]:


plt.figure(figsize=(8,6))
sns.countplot(x='pdays',hue='outcome',data=data_num)
plt.show()


# In[58]:


def clubLabelEncoder(df, feature, k):
    
    #df[feature + '_band'] = pd.cut(df[feature], k)
    #data = df[[feature + '_band', target]].groupby([feature + '_band'], as_index = False).mean().sort_values(by = feature + '_band', ascending = True)
    #x = data[feature + '_band'].tolist()
    df[feature +'_band'] = pd.qcut(df[feature], k)
    x = df[feature + '_band'].value_counts().index.tolist()
    
    intervals = []
    for i in range(len(x)):
        leftInt = x[i].left
        rtInt = x[i].right
        intervals.append(leftInt)
        intervals.append(rtInt)
    
    intervals_ = sorted(list(set(intervals)))
    
    for i in range(len(intervals_)-1):
        
        df.loc[(df[feature] > intervals_[i]) & (df[feature] <= intervals_[i+1]), feature] = i
        
    df = df.iloc[:,:-1]
        
    return df[feature].value_counts()


# In[59]:


# emp.var.rate
data_num['emp.var.rate'].value_counts()


# In[60]:


data_num.loc[ data_num['emp.var.rate'] <= 0, 'emp.var.rate'] = 0
data_num.loc[ data_num['emp.var.rate'] > 0, 'emp.var.rate'] = 1
data_num['emp.var.rate'].value_counts()


# In[61]:


plt.figure(figsize=(8,6))
sns.countplot(x='emp.var.rate',hue='outcome',data=data_num)
plt.show()


# In[62]:


#cons.price.idx
data_num['cons.price.idx'].value_counts()


# In[63]:


data_num['cons.price.idx_band'] = pd.cut(data_num['cons.price.idx'], 4)


# In[64]:


data_num['cons.price.idx_band'].value_counts()


# In[65]:


data_num.loc[ data_num['cons.price.idx'] <= 92.842, 'cons.price.idx'] = 0
data_num.loc[(data_num['cons.price.idx'] > 92.842) & (data_num['cons.price.idx'] <= 93.484), 'cons.price.idx'] = 1
data_num.loc[(data_num['cons.price.idx'] > 93.484) & (data_num['cons.price.idx'] <= 94.126), 'cons.price.idx'] = 2
data_num.loc[data_num['cons.price.idx'] > 94.126, 'cons.price.idx'] = 3


# In[66]:


data_num['cons.price.idx'].value_counts()


# In[67]:


plt.figure(figsize=(8,6))
sns.countplot(x='cons.price.idx',hue='outcome',data=data_num)
plt.show()


# In[68]:


#cons.conf.idx
data_num['cons.conf.idx'].value_counts()


# In[69]:


data_num['cons.conf.idx_band'] = pd.cut(data_num['cons.conf.idx'], 4)
data_num['cons.conf.idx_band'].value_counts()


# In[70]:


data_num.loc[ data_num['cons.conf.idx'] >= -32.875, 'cons.conf.idx'] = 0
data_num.loc[(data_num['cons.conf.idx'] < -32.875) & (data_num['cons.conf.idx'] >= -38.85), 'cons.conf.idx'] = 1
data_num.loc[(data_num['cons.conf.idx'] < -38.85) & (data_num['cons.conf.idx'] >= -44.825), 'cons.conf.idx'] = 2
data_num.loc[data_num['cons.conf.idx'] < -44.825, 'cons.conf.idx'] = 3


# In[71]:


data_num['cons.conf.idx'].value_counts()


# In[72]:


plt.figure(figsize=(8,6))
sns.countplot(x='cons.conf.idx',hue='outcome',data=data_num)
plt.show()


# In[73]:


# nr.employed
data_num['nr.employed'].value_counts()


# In[74]:


data_num['nr.employed_band'] = pd.cut(data_num['nr.employed'], 4)
data_num['nr.employed_band'].value_counts()


# In[75]:


data_num.loc[ data_num['nr.employed'] <= 5029.735, 'nr.employed'] = 0
data_num.loc[(data_num['nr.employed'] > 5029.735) & (data_num['nr.employed'] <= 5095.85), 'nr.employed'] = 1
data_num.loc[(data_num['nr.employed'] > 5095.85) & (data_num['nr.employed'] <= 5161.975), 'nr.employed'] = 2
data_num.loc[data_num['nr.employed'] > 5161.975, 'nr.employed'] = 3

data_num['nr.employed'].value_counts()


# In[76]:


plt.figure(figsize=(8,6))
sns.countplot(x='nr.employed',hue='outcome',data=data_num)
plt.show()


# In[77]:


# have to categrize the numericals age, duration, euribor3m 
sns.distplot(data_num['age'])


# In[78]:


data_num['age_band'] = pd.qcut(data_num['age'],3)
data_num['age_band'].value_counts()


# In[79]:


data_num.loc[ data_num['age'] <= 34.0, 'age'] = 0
data_num.loc[(data_num['age'] > 34.0) & (data_num['age'] <= 44.0), 'age'] = 1
data_num.loc[data_num['age'] > 44, 'age'] = 2
data_num['age'].value_counts()


# In[80]:


plt.figure(figsize=(8,6))
sns.countplot(x='age',hue='outcome',data=data_num)
plt.show()


# In[81]:


#duration
sns.distplot(data_num['duration'])
plt.show()
plt.close()


# In[82]:


data_num['duration_band'] = pd.qcut(data_num['duration'], 3)
data_num['duration_band'].value_counts()


# In[83]:


data_num.loc[ data_num['duration'] <= 126.0, 'duration'] = 0
data_num.loc[(data_num['duration'] > 126.0) & (data_num['duration'] <= 258.0), 'duration'] = 1
data_num.loc[data_num['duration'] > 258.0, 'duration'] = 2
data_num['duration'].value_counts()


# In[84]:


plt.figure(figsize=(8,6))
sns.countplot(x='duration',hue='outcome',data=data_num)
plt.show()


# In[85]:


#euribor3m
sns.distplot(data_num['euribor3m'])
plt.show()
plt.close()


# In[86]:


data_num['euribor3m_band'] = pd.qcut(data_num['euribor3m'], 3)
data_num['euribor3m_band'].value_counts()


# In[87]:


data_num.loc[ data_num['euribor3m'] <= 4.021, 'euribor3m'] = 0
data_num.loc[(data_num['euribor3m'] > 4.021) & (data_num['euribor3m'] <= 4.958), 'euribor3m'] = 1
data_num.loc[data_num['euribor3m'] > 4.958, 'euribor3m'] = 2


# In[88]:


data_num['euribor3m'].value_counts()


# In[89]:


plt.figure(figsize=(8,6))
sns.countplot(x='euribor3m',hue='outcome',data=data_num)
plt.show()


# In[90]:


data_num.head()


# In[91]:


listRemove = data_num.columns.tolist()[-7:]


# In[92]:


listRemove


# In[93]:


data_num1 = data_num.drop(listRemove, axis = 1)


# In[94]:


data_num1.head()


# In[138]:


data_num1.columns


# In[139]:


data_num_scaled = bank.loc[:,['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]


# In[140]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data_num_scaled)
abc = scaler.transform(data_num_scaled)
Numeric_Scaled = pd.DataFrame(abc, columns=data_num_scaled.columns.tolist())


# In[95]:


data_cat.head()


# In[96]:


data_cat1 = data_cat.drop(['outcome'], axis = 1)


# In[97]:


data_cat1 = pd.get_dummies(data=data_cat1, columns=catCols, drop_first=True)


# In[98]:


data_cat1.head()


# In[99]:


data = pd.concat([data_cat1, data_num1], axis = 1)
data.head()


# In[141]:


data2 = pd.concat([data_cat1, Numeric_Scaled], axis = 1)


# In[143]:


data2['outcome'] = bank['outcome']


# In[144]:


data2.head()


# ## MODELLING

# ### Combination 1 : Cut Numeric and dummy categorical

# In[100]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from confusionMatrix import plotConfusionMatrix


# In[101]:


y = data['outcome']
X = data.iloc[:,0:-1]


# In[102]:


X.shape, y.shape


# In[103]:


## Method 0: without SMOTE---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf.score(X_test, y_test)
predictions_ = clf.predict(X_test) 


# In[104]:


clf.score


# In[105]:


print('Without imbalance treatment:'.upper())
print('*'*80)
print(classification_report(y_test, predictions_)) 
print('*'*80)
print(confusion_matrix(y_test, predictions_))
print('*'*80)
f1= f1_score(y_test,predictions_, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_test,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')
sum(y_test == 0)
sum(y_test == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)


# ####  Cross Validation on Normal train-test split

# In[106]:


# K fold Coss- validation :
from sklearn.model_selection import cross_val_score
CV_Score = cross_val_score(clf, X_train, y_train, cv=20)
CV_Score


# In[107]:


CV_Score.mean()


# In[108]:


from sklearn.metrics import accuracy_score


# In[109]:


#stratified K-Fold

from sklearn.model_selection import StratifiedKFold
accuracy=[]
skf = StratifiedKFold(n_splits= 10, random_state= None)
skf.get_n_splits(X,y)
for train_index, test_index in skf.split(X,y):
    print ("Train:", train_index, "Validation:", test_index)
    X1_train, X1_test = X.iloc[train_index], X.iloc[test_index]
    y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X1_train,y1_train)
    prediction = clf.predict(X1_test)
    score = accuracy_score(prediction, y1_test)
    accuracy.append(score)

print(accuracy)
np.array(accuracy).mean()


# In[110]:


#using under Sampling technique :
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_undersample, y_undersample = nm.fit_sample (X,y.ravel())


# In[111]:


X_undersample.shape, y_undersample.shape


# In[112]:


from collections import Counter
print('original shape {}'.format (Counter(y)))
print('Resampled shape {}'.format (Counter(y_undersample)))


# In[113]:


#split into 70:30 ratio 
X_train_undersample, X_val_undersample, y_train_undersample, y_val_undersample = train_test_split(X_undersample, 
                                        y_undersample, test_size = 0.3, random_state = 0)

clf.fit(X_train_undersample, y_train_undersample.ravel()) 
clf.score(X_train_undersample, y_train_undersample)
clf.score(X_val_undersample, y_val_undersample)

predictions_undersample = clf.predict(X_val_undersample) 


# In[114]:


# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_undersample, predictions_undersample)) 
print(confusion_matrix(y_val_undersample, predictions_undersample))
f1= f1_score(y_val_undersample,predictions_undersample, average='micro')
plt.figure()
cnf_mat_undersample = confusion_matrix(y_val_undersample,predictions_undersample)
plotConfusionMatrix(cnf_mat_undersample, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val_undersample == 0)
sum(y_val_undersample == 1)
sum(predictions_undersample == 0)
sum(predictions_undersample == 1)

confusion_matrix(y_val_undersample, predictions_undersample)


# In[115]:


# K fold Coss- validation :
from sklearn.model_selection import cross_val_score
CV_Score_undersample = cross_val_score(clf, X_undersample, y_undersample, cv=10)
CV_Score_undersample


# In[116]:


#using Random OverSampling technique :

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_oversample, y_oversample = ros.fit_sample (X,y)
X_oversample.shape, y_oversample.shape


# In[117]:


print('original shape {}'.format (Counter(y)))
print('Resampled shape {}'.format (Counter(y_oversample)))


# In[118]:


#split into 70:30 ratio 
X_train_oversample, X_val_oversample, y_train_oversample, y_val_oversample = train_test_split(X_oversample, 
                                        y_oversample, test_size = 0.3, random_state = 0)

clf.fit(X_train_oversample, y_train_oversample.ravel()) 
clf.score(X_train_oversample, y_train_oversample)
clf.score(X_val_oversample, y_val_oversample)

predictions_oversample = clf.predict(X_val_oversample) 


# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_oversample, predictions_oversample)) 
print(confusion_matrix(y_val_oversample, predictions_oversample))
f1= f1_score(y_val_oversample,predictions_oversample, average='micro')
plt.figure()
cnf_mat_oversample = confusion_matrix(y_val_oversample,predictions_oversample)
plotConfusionMatrix(cnf_mat_oversample, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val_oversample == 0)
sum(y_val_oversample == 1)
sum(predictions_oversample == 0)
sum(predictions_oversample == 1)

confusion_matrix(y_val_oversample, predictions_oversample)


# In[119]:


## Method1: SMOTE on Train

sm = SMOTE(random_state = 2) 
X_smote, y_smote = sm.fit_sample(X_train, y_train.ravel()) 

print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_smote.shape)) 
print('After OverSampling, y: {}'.format(y_smote.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_smote == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_smote == 0))) 
print('\n')


# In[120]:


#split into 70:30 ratio 
X_train_smote, X_val_smote, y_train_smote, y_val_smote = train_test_split(X_smote, y_smote, 
                    test_size = 0.3, random_state = 0)

clf.fit(X_train_smote, y_train_smote.ravel()) 
clf.score(X_val_smote, y_val_smote)

predictions_smote = clf.predict(X_val_smote) 


# In[121]:


# print classification report on Validation set 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_smote, predictions_smote)) 
print('*'*80)
print(confusion_matrix(y_val_smote, predictions_smote))
print('*'*80)
f1= f1_score(y_val_smote,predictions_smote, average='micro')
plt.figure()
cnf_mat_smote = confusion_matrix(y_val_smote,predictions_smote)
plotConfusionMatrix(cnf_mat_smote, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val_smote == 0)
sum(y_val_smote == 1)
sum(predictions_smote == 0)
sum(predictions_smote == 1)

confusion_matrix(y_val_smote, predictions_smote)


# In[122]:


## Checking on actual Test Data :

clf.fit(X_train_smote, y_train_smote.ravel()) 
clf.score(X_test, y_test)

predictions_smote = clf.predict(X_test) 

# print classification report on Validation set 
print('After imbalance treatment:'.upper())
print(classification_report(y_test, predictions_smote)) 
print('*'*80)
print(confusion_matrix(y_test, predictions_smote))
print('*'*80)
f1= f1_score(y_test,predictions_smote, average='micro')
plt.figure()
cnf_mat_smote = confusion_matrix(y_test,predictions_smote)
plotConfusionMatrix(cnf_mat_smote, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_test == 0)
sum(y_test == 1)
sum(predictions_smote == 0)
sum(predictions_smote == 1)

confusion_matrix(y_test, predictions_smote)


# ### Grid Search with SMOTE

# In[123]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_smote,y_train_smote)
preds = tree.predict(X_test)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',tree.best_params_)
print('*'*80)
print(classification_report(y_test, preds)) 
print('*'*80)
print(confusion_matrix(y_test, preds))
print('*'*80)

y_pred_proba_ = tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_)
auc = roc_auc_score(y_test, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

print('*'*80)


# #### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------

# In[124]:


def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(15, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_smote,y_smote, 4)
print('Using best parameters:',best_param)
pred = model.predict(X_test)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',model.best_params_)
print('*'*80)
print(classification_report(y_test, pred)) 
print('*'*80)
print(confusion_matrix(y_test, pred))
print('*'*80)

## ROC curve
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()


# ### Random Forest 

# In[125]:


from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [100,300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier()
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf, rf_gscv.best_params_, rf_gscv, score_gscv  

print('*'*80)
print('GRID SEARCH WITHOUT SMOTE & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, acc_rf = RF_grid_search(X_train,y_train, 4)
pred_rf = model_rf.predict(X_test)

print('Using best parameters:',model.best_params_)
print('*'*80)
print(classification_report(y_test, pred_rf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_rf))
print('*'*80)
## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# In[126]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
print('*'*80)
print('GRID SEARCH WITHOUT SMOTE RF:')
pred_clf = rf_clf.predict(X_test)

print('*'*80)
print(classification_report(y_test, pred_rf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_rf))
print('*'*80)
## ROC curve
y_pred_proba_rf = rf_clf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# In[128]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid={}, cv= 10)
grid_search.fit(X_train, y_train)
cvrf_clf=grid_search.best_estimator_
pred_clf = cvrf_clf.predict(X_test)
print('*'*80)
print(classification_report(y_test, pred_clf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_clf))
print('*'*80)
## ROC curve
y_pred_proba_rf = cvrf_clf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ### With SMOTE

# In[130]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid={}, cv= 10)
grid_search.fit(X_train_smote, y_train_smote)
cvrf_clfs=grid_search.best_estimator_
pred_clfs = cvrf_clfs.predict(X_test)
print('*'*80)
print(classification_report(y_test, pred_clfs)) 
print('*'*80)
print(confusion_matrix(y_test, pred_clfs))
print('*'*80)
## ROC curve
y_pred_proba_rfs = cvrf_clfs.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rfs)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rfs)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ### XG Boost Method

# In[131]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train, y_train)

# make predictions for test set
y_pred = model_G.predict(X_test)
preds = [round(value) for value in y_pred]

print('*'*80)
print('XGB Without SMOTE')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[135]:


from sklearn.model_selection import GridSearchCV
model_Gb= XGBClassifier(max_depth = 5, n_estimators=1000, learning_rate=0.03, n_jobs=1)
grid_search = GridSearchCV(model_Gb, param_grid={}, n_jobs=1, cv = 10, scoring = "accuracy")
grid_search.fit(X_train, y_train)
cvxg_clf= grid_search.best_estimator_
y_pred = cvxg_clf.predict(X_test)
pred = [round(value) for value in y_pred]

print('*'*80)
print('SMOTE with XGB')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = cvxg_clf.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# #### With SMOTE

# In[133]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_Gs = XGBClassifier(learning_rate = 0.05, n_estimators=500, max_depth=7)
model_Gs.fit(X_train_smote, y_train_smote)

# make predictions for test set
y_pred = model_Gs.predict(X_test)
preds = [round(value) for value in y_pred]

print('*'*80)
print('SMOTE with XGB')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = model_Gs.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[137]:


from sklearn.model_selection import GridSearchCV
model_Gbs= XGBClassifier(max_depth = 5, n_estimators=1000, learning_rate=0.03, n_jobs=1)
grid_searchs = GridSearchCV(model_Gbs, param_grid={}, n_jobs=1, cv = 10, scoring = "accuracy")
grid_searchs.fit(X_train_smote, y_train_smote)
cvxgs_clf= grid_search.best_estimator_
y_pred = cvxgs_clf.predict(X_test)
pred = [round(value) for value in y_pred]

print('*'*80)
print('SMOTE with XGB')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = cvxgs_clf.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# ### Combination 2 : Scaled Numeric + Dummy category

# In[145]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from confusionMatrix import plotConfusionMatrix


# In[146]:


y = data2['outcome']
X = data2.iloc[:,0:-1]


# In[147]:


X.shape, y.shape


# In[148]:


## Method 0: without SMOTE---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf.score(X_test, y_test)
predictions_ = clf.predict(X_test) 


# In[152]:


## Method1: SMOTE on Train

sm = SMOTE(random_state = 2) 
X_smote, y_smote = sm.fit_sample(X_train, y_train.ravel()) 

print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_smote.shape)) 
print('After OverSampling, y: {}'.format(y_smote.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_smote == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_smote == 0))) 
print('\n')


# In[149]:


print('Without imbalance treatment:'.upper())
print('*'*80)
print(classification_report(y_test, predictions_)) 
print('*'*80)
print(confusion_matrix(y_test, predictions_))
print('*'*80)
f1= f1_score(y_test,predictions_, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_test,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')
sum(y_test == 0)
sum(y_test == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)


# ### SMOTE on DT

# In[153]:


## Checking on actual Test Data :

clf.fit(X_train_smote, y_train_smote.ravel()) 
clf.score(X_test, y_test)

predictions_smote = clf.predict(X_test) 

# print classification report on Validation set 
print('After imbalance treatment:'.upper())
print(classification_report(y_test, predictions_smote)) 
print('*'*80)
print(confusion_matrix(y_test, predictions_smote))
print('*'*80)
f1= f1_score(y_test,predictions_smote, average='micro')
plt.figure()
cnf_mat_smote = confusion_matrix(y_test,predictions_smote)
plotConfusionMatrix(cnf_mat_smote, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_test == 0)
sum(y_test == 1)
sum(predictions_smote == 0)
sum(predictions_smote == 1)

confusion_matrix(y_test, predictions_smote)


# ### Grid Search With Cross Validation

# In[154]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_smote,y_train_smote)
preds = tree.predict(X_test)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',tree.best_params_)
print('*'*80)
print(classification_report(y_test, preds)) 
print('*'*80)
print(confusion_matrix(y_test, preds))
print('*'*80)

y_pred_proba_ = tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_)
auc = roc_auc_score(y_test, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

print('*'*80)


# ### Grid Search with SMOTE & CV

# In[155]:


def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(15, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_smote,y_smote, 4)
print('Using best parameters:',best_param)
pred = model.predict(X_test)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',model.best_params_)
print('*'*80)
print(classification_report(y_test, pred)) 
print('*'*80)
print(confusion_matrix(y_test, pred))
print('*'*80)

## ROC curve
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()


# ## Random Forest

# In[156]:


from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [100,300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier()
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf, rf_gscv.best_params_, rf_gscv, score_gscv  

print('*'*80)
print('GRID SEARCH WITHOUT SMOTE & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, acc_rf = RF_grid_search(X_train,y_train, 4)
pred_rf = model_rf.predict(X_test)

print('Using best parameters:',model.best_params_)
print('*'*80)
print(classification_report(y_test, pred_rf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_rf))
print('*'*80)
## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# In[157]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
print('*'*80)
print('GRID SEARCH WITHOUT SMOTE RF:')
pred_clf = rf_clf.predict(X_test)

print('*'*80)
print(classification_report(y_test, pred_rf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_rf))
print('*'*80)
## ROC curve
y_pred_proba_rf = rf_clf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# #### With Grid Search CV

# In[158]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid={}, cv= 10)
grid_search.fit(X_train, y_train)
cvrf_clf=grid_search.best_estimator_
pred_clf = cvrf_clf.predict(X_test)
print('*'*80)
print(classification_report(y_test, pred_clf)) 
print('*'*80)
print(confusion_matrix(y_test, pred_clf))
print('*'*80)
## ROC curve
y_pred_proba_rf = cvrf_clf.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ### RF With SMOTE and GridSearch CV:

# In[159]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid={}, cv= 10)
grid_search.fit(X_train_smote, y_train_smote)
cvrf_clfs=grid_search.best_estimator_
pred_clfs = cvrf_clfs.predict(X_test)
print('*'*80)
print(classification_report(y_test, pred_clfs)) 
print('*'*80)
print(confusion_matrix(y_test, pred_clfs))
print('*'*80)
## ROC curve
y_pred_proba_rfs = cvrf_clfs.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test,  y_pred_proba_rfs)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rfs)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ## XG Boost 

# #### Without SMOTE

# In[150]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train, y_train)

# make predictions for test set
y_pred = model_G.predict(X_test)
preds = [round(value) for value in y_pred]

print('*'*80)
print('XGB Without SMOTE')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[164]:


from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)


# In[161]:


from sklearn.model_selection import GridSearchCV
model_Gb= XGBClassifier(max_depth = 5, n_estimators=1000, learning_rate=0.03, n_jobs=1)
grid_search = GridSearchCV(model_Gb, param_grid={}, n_jobs=1, cv = 10, scoring = "accuracy")
grid_search.fit(X_train, y_train)
cvxg_clf= grid_search.best_estimator_
y_pred = cvxg_clf.predict(X_test)
pred = [round(value) for value in y_pred]

print('*'*80)
print('SMOTE with XGB')
print('*'*80)
print(classification_report(y_test, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test, y_pred))
print('*'*80)
y_pred_proba_G = cvxg_clf.predict_proba(X_test)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[ ]:




