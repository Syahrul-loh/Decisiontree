#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


import numpy as np


# In[10]:


dataset = pd.read_csv('dtdata.csv')


# In[11]:


dataset


# In[12]:


x = dataset.iloc[:,:-1]


# In[13]:


x


# In[14]:


y = dataset.iloc[:,5]


# In[15]:


y


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


labelencoder_x = LabelEncoder()


# In[18]:


x = x.apply(LabelEncoder().fit_transform)


# In[19]:


x


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


regressor=DecisionTreeClassifier()


# In[22]:


regressor.fit(x.iloc[:,1:5],y)


# In[23]:


x_in = np.array([1,1,0,0])


# In[24]:


y_pred = regressor.predict([x_in])


# In[25]:


y_pred


# In[26]:


from six import StringIO


# In[27]:


from IPython.display import Image


# In[28]:


from sklearn.tree import export_graphviz


# In[29]:


import pydotplus


# In[30]:


dot_data=StringIO()


# In[40]:


export_graphviz(regressor, out_file = dot_data, filled = True, rounded=True, special_characters = True)


# In[50]:


graph=pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[52]:


graph.write_png("tree.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




