#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd


# In[152]:


import numpy as np


# In[153]:


dataset = pd.read_csv('dfdata.csv')


# In[154]:


dataset


# In[155]:


x = dataset.iloc[:,:-1]


# In[156]:


x


# In[157]:


y = dataset.iloc[:,2]


# In[158]:


y


# In[159]:


from sklearn.preprocessing import LabelEncoder


# In[160]:


labelencoder_x = LabelEncoder()


# In[161]:


x = x.apply(LabelEncoder().fit_transform)


# In[162]:


x


# In[163]:


from sklearn.tree import DecisionTreeClassifier


# In[164]:


regressor=DecisionTreeClassifier()


# In[165]:


regressor.fit(x.iloc[:,1:5],y)


# In[166]:


x_in = np.array([2,2])


# In[167]:


y_pred = regressor.predict([x_in])


# In[168]:


y_pred


# In[169]:


from six import StringIO


# In[170]:


from IPython.display import Image


# In[171]:


from sklearn.tree import export_graphviz


# In[172]:


import pydotplus


# In[173]:


dot_data = StringIO()


# In[174]:


dataset.head()


# In[175]:


dataset.shape


# In[176]:


dataset.info()


# In[177]:


d={'Ya':1,'Bukan':0}
dataset['Pemilik Rumah']=dataset['Pemilik Rumah'].map(d) 
dataset['Penunggak Hutang']=dataset['Penunggak Hutang'].map(d)
d1={'Lajang':0,'Menikah':1,'Bercerai':2}
dataset['Status Perkawinan']=dataset['Status Perkawinan'].map(d1)
dataset.head()


# In[178]:


dataset.columns


# In[179]:


x=dataset[['Pemilik Rumah', 'Status Perkawinan', 'Pendapatan Tahunan']]
y=dataset['Penunggak Hutang']
features=list(dataset.columns[:3])
print(features)


# In[180]:


from sklearn import tree
model=tree.DecisionTreeClassifier(criterion='entropy',splitter='random')
model=model.fit(x,y)


# In[181]:


tree.export_graphviz(model,out_file=dot_data,feature_names=features,filled=True,rounded=True)


# In[182]:


graph=pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[183]:


Image(graph.create_png())


# In[184]:


graph.write_png("tree.png")


# In[185]:


from sklearn.tree import export_text

r = export_text(model, feature_names=features)
print(r)


# In[ ]:





# In[ ]:





# In[ ]:




