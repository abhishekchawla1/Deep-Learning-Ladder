#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import load_linnerud


# In[3]:


d=load_linnerud()


# In[4]:


df=pd.DataFrame(d.data,columns=d.feature_names)


# In[5]:


d.feature_names


# In[6]:


df[['Weight','Rate','Pulse']]=d.target


# In[7]:


d.target


# In[8]:


df


# In[9]:


from sklearn.datasets import make_circles


# In[10]:


X,y=make_circles(n_samples=100000,noise=0.4,factor=0.5)


# In[11]:


plt.scatter(X[:,0],X[:,1],c=y)


# In[12]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[13]:


model=Sequential()


# In[14]:


model.add(Dense(4,activation='tanh',input_dim=2))
model.add(Dense(2,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


# In[15]:


model


# In[16]:


model.summary()


# In[17]:


model.get_weights()


# In[18]:


ini_weights=model.get_weights()


# In[19]:


ini_weights[0] = np.ones(model.get_weights()[0].shape)*0.5
ini_weights[1] = np.ones(model.get_weights()[1].shape)*0.5
ini_weights[2] = np.ones(model.get_weights()[2].shape)*0.5
ini_weights[3] = np.ones(model.get_weights()[3].shape)*0.5


# In[20]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[21]:


history=model.fit(X,y,epochs=10,validation_split=0.2)


# In[22]:


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X,y.astype(int),clf=model,legend=1)


# In[25]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[29]:


model2=Sequential()
model2.add(Dense(4,activation='relu',input_dim=2,kernel_initializer='he_normal'))
model2.add(Dense(10,activation='relu',kernel_initializer='he_normal'))
model2.add(Dense(10,activation='relu',kernel_initializer='he_normal'))
model2.add(Dense(10,activation='relu',kernel_initializer='he_normal'))
model2.add(Dense(1,activation='sigmoid'))


# In[30]:


model2.summary()


# In[31]:


initial_weights = model2.get_weights()
initial_weights[0] = np.random.randn(2,10)*np.sqrt(1/2)
initial_weights[1] = np.zeros(model.get_weights()[1].shape)
initial_weights[2] = np.random.randn(10,10)*np.sqrt(1/10)
initial_weights[3] = np.zeros(model.get_weights()[3].shape)
initial_weights[4] = np.random.randn(10,10)*np.sqrt(1/10)
initial_weights[5] = np.zeros(model.get_weights()[5].shape)
initial_weights[6] = np.random.randn(10,10)*np.sqrt(1/10)
initial_weights[7] = np.zeros(model.get_weights()[7].shape)
initial_weights[8] = np.random.randn(10,1)*np.sqrt(1/10)
initial_weights[9] = np.zeros(model.get_weights()[9].shape)


# In[35]:


model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[36]:


history = model2.fit(X,y,epochs=10,validation_split=0.2)


# In[37]:


plot_decision_regions(X,y.astype('int'),clf=model2,legend=1)


# In[38]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[ ]:




