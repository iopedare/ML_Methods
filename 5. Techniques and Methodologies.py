#!/usr/bin/env python
# coding: utf-8

# ### merging columns

# In[1]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[2]:


# many to one merging
dframe1 = DataFrame({'reference': ['ola','uber','lyft','gojek','grab'],'revenue':[1,2,3,4,5]})
dframe2 = DataFrame({'reference': ['ola','uber','uber','ola'],'revenue':[1,2,3,4,]})


# In[3]:


dframe1


# In[4]:


dframe2


# In[5]:


df3 = pd.merge(dframe1,dframe2,on='reference')
df3


# In[9]:


df4 = pd.merge(dframe1,dframe2,on='reference',how='right')


# In[10]:


df4


# In[8]:


df5 = pd.merge(dframe1,dframe2,on='reference',how='outer')
df5


# In[11]:


###MANY TO MANY


# In[13]:


df6 = DataFrame({'reference':['ola','ola','lyft','lyft','uber','uber','ola'],
                'revenue':[1,2,3,4,5,6,7]
                })


# In[14]:


df6


# In[15]:


df7 = DataFrame({'reference':['uber','uber','lyft','ola','ola',],
                'revenue':[1,2,3,4,5]
                })


# In[16]:


df7


# In[17]:


pd.merge(df6, df7)



# In[20]:


# multiple reference
df8 = DataFrame({'reference':['ola','ola','lyft'],
                'revenue':['one','two','three'],
                 'profit':[1,2,3]
                })


# In[21]:


df9 = DataFrame({'reference':['ola','ola','lyft','lyft'],
                'revenue':['one','one','one','three'],
                 'profit':[4,5,6,7]
                })


# In[22]:


pd.merge(df8,df9,on=['reference','revenue'],how='outer')


# In[23]:


pd.merge(df8,df9,on=['reference','revenue'],how='outer',suffixes=('_first','_second'))


# ### merge with indexes

# In[24]:


from pandas import DataFrame


# In[27]:


df_1 = DataFrame({'reference':['O','U','L','O','U'],
                 'data': range(5)
                 })


# In[32]:


df_2 = DataFrame({'profit':[10,20]},
                  index=['O','U'])


# In[33]:


pd.merge(df_1,df_2,left_on='reference',right_index=True)


# In[34]:


df_1


# In[35]:


df_2


# In[42]:


df_3 = DataFrame({'ref1':['A','A','O','O','A'],
                 'ref2':[5,10,15,20,25],
                 'ref3':np.arange(5.)})


# In[47]:


df_4 = DataFrame({'ref1':['A','A','O','O','O'],
                 'ref2':[15,20,25,30,35],
                 'ref3':[2,3,4,5,6]})


# In[ ]:





# In[49]:


#pd.merge(df_3,df_4,left_on=['ref1','ref2'],right_index=True)


# In[39]:


#join functions


# In[50]:


df_3.join(df_4,lsuffix='x',rsuffix='y')


# ### concatenation - scipy

# In[52]:


from numpy import random


# In[53]:


B1 = np.arange(25).reshape(5,5)
A1 = random.randn(25).reshape(5,5)


# In[54]:


B1


# In[55]:


A1


# In[58]:



np.concatenate([A1,B1], axis=1)


# In[59]:


np.concatenate([A1,B1], axis=0)


# In[60]:


# Series Concatenation
s1 = Series([100,200,300],index=['A','B','C'])
s2 = Series([400,500],index=['D','E'])


# In[61]:


pd.concat([s1,s2])


# In[62]:


pd.concat([s1,s2],axis=1)


# In[64]:


# DataFrame concatenation
df1 = DataFrame(random.randn(4,3),columns=['A','B','C'])
df2 = DataFrame(random.randn(3,3),columns=['B','D','A'])


# In[65]:


pd.concat([df1,df2])


# In[66]:


pd.concat([df1,df2],ignore_index=True)


# In[67]:


pd.concat([df1,df2],axis=1)


# In[68]:


#url = https://pandas.pydata.org/pandas-docs/stable/generated/pandas-concat.html


# ### Combine pandas

# In[69]:


s1 = Series([5,np.nan,6,np.nan],index=['A','B','C','D'])
s1


# In[70]:


s2 = Series(np.arange(4),dtype=np.float64,index=s1.index)
s2


# In[72]:


s3 = Series(np.where(pd.isnull(s1),s2,s1),index=s1.index)
s3


# In[73]:


s4 = s1.combine_first(s2)


# In[74]:


s4


# In[75]:


#Dataframes
df_5m = DataFrame({'col1':[5,np.nan,15],
                  'col2':[20,25,np.nan],
                  'col3':[np.nan,np.nan,35]
                  })


# In[76]:


df_10m = DataFrame({'col1':[0,10,20],
                   'col2':[10,20,30]
                   })


# In[78]:


df_5m


# In[79]:


df_10m


# In[81]:


df_5m.combine_first(df_10m)


# ### reshaping

# In[82]:


df1 = DataFrame(np.arange(8).reshape(2,4),index=pd.Index(['Uber','Grab'],name='cabs'),columns=pd.Index(['c1','c2','c3','c4'],name='attributes'))
df1


# In[83]:


stackdf1 = df1.stack()
stackdf1


# In[84]:


df1unstack = stackdf1.unstack()
df1unstack


# In[85]:


df3 = stackdf1.unstack('cabs')
df3


# In[86]:


df4 = stackdf1.unstack('attributes')
df4


# In[87]:


s1 = Series([5,10,15],index=['A','B','C'])
s2 = Series([15,20,25],index=['B','C','D'])


# In[88]:


s3 = pd.concat([s1,s2],keys=['k1','k2'])
s3


# In[91]:


aircraft_407 = Series([0,1,2,3,4,5,6,7],index=['AMI','ANJ','FAR','FIZ','LAY','MYR','RAH','ZAH'])
aircraft_412 = Series([8,9,10,11],index=['BVK','BVL','BVM','BVN'])


# In[92]:


aircraft = pd.concat([aircraft_407,aircraft_412],keys=['B407','B412'])
aircraft


# In[93]:


aircraft1 = aircraft.unstack()
aircraft1


# In[94]:


df = s3.unstack()
df


# In[95]:


df.stack()


# In[96]:


df.stack(dropna=False)


# ### Pivot table

# In[97]:


#url = 'https://en.wikipedia.org/wiki/Pivot_table'
#df_list = pd.io.html.read.html(url)
#df = df_list[0]
#df


# In[98]:


#new_header = df.iloc[0] #grabthe first row for the header
#df =df[1:] #take the data less the header row
#df.columns=new_header #set the header row as the df header
#df


# In[99]:


#df.pivot('Date of sale', 'Sales person', 'Total price')


# ### Duplicates

# In[100]:


df = DataFrame({
    'col1':['uber','uber','uber','grab','grab'],
    'col2':[5,4,3,3,5]
})
df


# In[102]:


df.duplicated()


# In[103]:


df.drop_duplicates()


# In[105]:


df.drop_duplicates(['col1'])


# In[108]:


df.drop_duplicates(['col1'], keep='last')


# In[109]:


df = DataFrame({'country': ['Afghanistan', 'Albania', 'Algeria'],
               'code':['93','355','213']})
df


# In[110]:


GDP_map ={'Afghanistan':'20', 'Albania':'12.8', 'Algeria':'215'}
GDP_map


# In[111]:


df['GDP'] = df['country'].map(GDP_map)


# In[112]:


df


# ### replace values in series

# In[113]:


s1 = Series([10,20,40,50,20,10,50,40])
s1


# In[114]:


s1.replace(50,np.nan)


# In[115]:


s1.replace([10,20,50],[100,200,500])


# In[116]:


s1.replace({10:100,20:np.nan,40:400})


# ### remaining indexes

# In[117]:


df = DataFrame(np.arange(25).reshape(5,5),index=['UBER','OLA','GRAB','GOJEK','LYFT'],columns=['RE','LO','QU','GR','AG'])
df


# In[119]:


#way 1 - use mapping
df.index = df.index.map(str.lower)
df


# In[120]:


# way 2 - rename method
df.rename(index=str.title,columns=str.lower)


# In[ ]:


#way 3 - using dictionary
df.rename(index={'uber':'The Best Taxi'}, columns={'RE':'Revenue'})


# In[123]:


#how to save
df.rename(index={'uber':'The Best Taxi'}, columns = {'RE':'Revenue'}, inplace=True)
df


# ### Binining values

# In[126]:


prime_nos = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
number_bins = [0,10,20,30,40,50]


# In[127]:


category = pd.cut(prime_nos,number_bins)
category


# In[128]:


category.categories


# In[130]:


pd.value_counts(category)


# In[131]:


# Limits
pd.cut(prime_nos,3,precision=1)


# ### Observation

# In[132]:


df = DataFrame(np.random.randn(1000,5))
#basic observation
df.head()


# In[133]:


df.tail()


# In[134]:


df.describe()


# In[137]:


column = df[0]
column.head()


# In[138]:


column[np.abs(column)>3]


# In[139]:


df[(np.abs(df)>3).any(1)]


# In[140]:


df[(np.abs(df)>3)]= np.sign(df)*5


# In[141]:


df.describe()


# In[ ]:




