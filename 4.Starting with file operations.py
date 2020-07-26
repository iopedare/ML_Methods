#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[3]:


#csv: comma-separated value
dframe = pd.read_csv('demo.csv',header=None)
dframe


# In[5]:


#use readtablee
dframe2 = pd.read_table('demo.csv',sep=',',header=None)
dframe2


# In[6]:


#partial rows importing
pd.read_csv('demo.csv',nrows=2,header=None)


# In[8]:


dframe2.to_csv('outputCSV.csv',sep=',')


# In[9]:


#select specific column
dframe.to_csv('dataoutpot.csv',columns=[0,1])


# ### Excel Pandas

# In[10]:


#import pandas as pd
excelfile = pd.ExcelFile('demo.xlsx')
dframe = excelfile.parse('demo')
dframe


# ### html pandas

# In[11]:


from pandas import read_html


# In[13]:


#url = 'https://countrycode.org/'
#dflist = pd.io.html.read_html(url)
#dframe = dflist[0]
#dframe


# In[14]:


#dframe.columns.values


# In[ ]:




