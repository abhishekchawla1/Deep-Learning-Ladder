#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().system('pip install tensorflow')


# In[11]:


get_ipython().system('pip install kaggle')


# In[12]:


get_ipython().system('pip install opendatasets')


# In[13]:


import opendatasets as od


# In[14]:


data='https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression/discussion'


# In[16]:


od.download(data)


# In[17]:


import os


# In[18]:


data_dir='.\second-hand-used-cars-data-set-linear-regression'


# In[19]:


os.listdir(data_dir)


# In[22]:


df=pd.read_csv(r'train.csv')


# In[23]:


df


# In[24]:


df.isnull().sum()


# In[25]:


df.duplicated().any()


# In[27]:


df.describe()


# In[28]:


sns.pairplot(df.drop(columns=['v.id','on road old','on road now']))


# In[33]:


sns.heatmap(df.drop(columns=['v.id','on road old','on road now']).corr(),annot=True,fmt='.2f')


# In[36]:


from tensorflow.keras.layers import Normalization,Dense,InputLayer


# In[38]:


from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError,Huber,MeanAbsoluteError


# In[42]:


import tensorflow as tf
tensor=tf.constant(df)


# In[43]:


tensor


# In[44]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=['v.id','on road old','on road now','current price']),df['current price'],test_size=0.2,random_state=0)


# In[47]:


X_train=tf.convert_to_tensor(X_train)


# In[50]:


X_test=tf.convert_to_tensor(X_test)


# In[51]:


y_train=tf.convert_to_tensor(y_train)


# In[52]:


y_test=tf.convert_to_tensor(y_test)


# In[54]:


X_train.shape


# In[55]:


X_test.shape


# In[57]:


y_train.shape


# In[58]:


y_test.shape


# In[60]:


norm=Normalization()
norm.adapt(X_train)


# In[61]:


X_train=norm(X_train)
X_test=norm(X_test)


# In[62]:


X_train


# In[65]:


model = tf.keras.Sequential([
                             InputLayer(input_shape = (8,)),
                             norm,
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(1),
])


# In[66]:


model.summary()


# In[69]:


tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=True)


# In[70]:


model.compile(optimizer = Adam(learning_rate = 0.1),
              loss = MeanAbsoluteError(),
              metrics = RootMeanSquaredError())


# In[77]:


history = model.fit(X_train,y_train)
model.evaluate(X_test,y_test)

