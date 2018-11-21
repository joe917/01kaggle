
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from __future__ import print_function
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.layers import Activation, Dense
get_ipython().run_line_magic('matplotlib', 'inline')


# In[117]:


columns_train = ['penalty','l1_ratio','alpha','max_iter','random_state','n_jobs','n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale','Time']
columns_test = ['penalty','l1_ratio','alpha','max_iter','random_state','n_jobs','n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale']


# In[118]:


# 读取train文件，把标题行删去
train = pd.read_csv('train.csv', names=columns_train,skiprows = 1)
# 读取test文件，加上标题'Time
test = pd.read_csv('test.csv', names = columns_test,skiprows = 1)


# In[119]:


# 查看train数据的前5行
train.head()


# In[120]:


# 数据集中各变量的描述性统计分析
train.describe()


# In[121]:


# 1.time, 41 maybe a outlier  2.feathers and samples are similar


# In[122]:


# 查看test数据的前5行
test.head()


# In[123]:


# 数据集中各变量的描述性统计分析
test.describe()


# In[124]:


# creat dummy variable
train = pd.concat([train, pd.get_dummies(train['penalty'],prefix='penalty',prefix_sep=':')], axis=1)
# delete catogorical variable
train.drop('penalty',axis=1,inplace=True)
test = pd.concat([test, pd.get_dummies(test['penalty'],prefix='penalty',prefix_sep=':')], axis=1)
test.drop('penalty',axis=1,inplace=True)
# rename variable
train.rename(columns={'penalty:l1':'L1','penalty:l2':'L2','penalty:none':'No'},inplace = True)
test.rename(columns={'penalty:l1':'L1','penalty:l2':'L2','penalty:none':'No'},inplace = True)


# In[125]:


# delete one of the dummy list
train.drop('penalty:elasticnet',axis=1,inplace=True)
test.drop('penalty:elasticnet',axis=1,inplace=True)


# In[126]:


print(train)


# In[127]:


X = np.array(train.drop('Time',1))
X_scaled = preprocessing.scale(X)
Z = np.array(test)
Z_scaled = preprocessing.scale(Z)
y = np.array(train['Time'])


# In[128]:


#把数据分成训练组和测试组
X_train, X_test, y_train, y_test =train_test_split(X_scaled,y,test_size=0.2)


# In[129]:


from keras.layers import Dropout
from keras import regularizers
model = Sequential()

model.add(Dense(input_dim=15,units =20))

model.add(Dense(40,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.4))


model.add(Dense(40,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))
model.add(Dropout(0.4))


model.add(Dense(20,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(1))


# In[130]:


model.compile(loss='mse',optimizer='sgd')


# In[131]:


model.summary()


# In[132]:


# Training
print('Training -----------')
for step in range(10000):
    cost = model.train_on_batch(X_train, y_train)
    if step % 500 == 0:
        print('train cost: ', cost)


# In[133]:


print('\nTesting ------------')
cost = model.evaluate(X_test, y_test, batch_size=50)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


# In[134]:


y_pred = model.predict(X_test)


# In[135]:


print(y_pred)


# In[136]:


prediction = model.predict(Z_scaled)


# In[137]:


np.savetxt('99.csv', prediction, delimiter = '  ',header = 'time')

