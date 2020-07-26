#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series


# In[2]:


object = Series([5, 20, 25, 20])
object


# In[4]:


object.values


# In[5]:


object.index


# In[6]:


#use numpy arrays to series
import numpy as np


# In[7]:


data_array = np.array(['a','b','c'])
s = Series(data_array)
s


# In[8]:


#custom index
s = Series(data_array, index=[100, 101, 102])
s


# In[9]:


aircraft_type = np.array(['B407', 'B412'])
aircraft_type = Series(aircraft_type)
aircraft_type
                         


# In[11]:


aircraft_type_407_0 = np.array(['AMI','ANJ','FAR','FIZ','LAY','MYR','ZAH'])
aircraft_type_407_0 = Series(aircraft_type_407_0)
aircraft_type_407_0


# In[12]:


aircraft_type_412_1 = np.array(['BVK','BVL','BVM'])
aircraft_type_412_1 = Series(aircraft_type_412_1)
aircraft_type_412_1


# In[13]:


s = Series(data_array, index=[100,101,102])
s


# In[14]:


s = Series(data_array, index=['index1','index2','index3'])
s


# In[15]:


#using real life ex
revenue = Series([20,80,40,35], index=['ola','uber','grab','gojek'])
revenue['ola']


# In[16]:


revenue[revenue>=35]


# In[17]:


#use boolean condition
'lyft' in revenue


# In[19]:


revenue_dict = revenue.to_dict()
revenue_dict


# In[21]:


index_2 = ['ola','uber','grab','gojek','lyft']
revenue2 = Series(revenue,index_2)
revenue2


# In[22]:


revenue


# In[23]:


pd.isnull(revenue2)


# In[24]:


pd.notnull(revenue2)


# #addition of series
# revenue+revenue2

# In[26]:


#assigning names
revenue2.name='Company Revenues'
revenue2.index.name='Company Name'
revenue2


#  ### DataFrame

# In[27]:


from pandas import DataFrame


# In[28]:


#go to https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue, and copythe first 6
#revenue_df = pd.read_clipboard()
#revenue_df


# In[29]:


#revenue_df.columns


# In[31]:


#revenue_df['Rank']


# In[32]:


#multiple columns

#DataFrame(revenue_df, columns=['Rank','Name','Industry'])


# In[34]:


#Nan Values
#revenue_df2 = DataFrame(revenue_df, columns=['Rank', 'Name', 'Industry', 'Profit'])
#revenue_df2


# In[35]:


#head and tail
#revenue_df.head(2)
#revenue_df.tail(2)


# In[37]:


#access rows in df
#revenue_df.ix[0] #row_1
#revenue_df.ix[5] #row_6


# In[38]:


#assign values to df
#numpy

#array1 = np.array([1,2,3,4,5,6])
#revenue_df2['Profit'] = array1
#revenue_df2


# In[39]:


#series
#profits = Series([900,1000], index=[3,5])
#revenue_df2['profit'] = profits
#revenue_df2


# In[40]:


#deletion
#del revenue_df2['profit']
#revenue_df2


# In[41]:


#dictionary function to dataframe
sample = {
    'company':['A','B'],
    'profit':[1000,5000]
}


# In[42]:


sample


# In[44]:


sample_df = DataFrame(sample)
sample_df


# In[46]:


cost_per_aircraft_type = {
    'aircraft_type':['B407', 'B412'],
    'monthly_standing_charge':[264071.79, 489653],
    'flight_cost_per_hour':[970.31, 1231.42]
}

cost_per_aircraft_type_df = DataFrame(cost_per_aircraft_type)
cost_per_aircraft_type_df


# In[47]:


aircraft_type_407_0


# In[54]:


aircraft_type_407 = {
    'aircraft_reg_name':['AMI','ANJ','FAR','FIZ','LAY','MYR','ZAH'],
    'aircraft_reg_no':[0,1,2,3,4,5,6]
}
aircraft_type_407_df =DataFrame(aircraft_type_407)
aircraft_type_407_df


# In[58]:


aircraft_frame = {
    'aircraft_type':['B407','B407','B407','B407','B407','B407','B407','B412','B412','B412'],
    'aircraft_reg_name':['AMI','ANJ','FAR','FIZ','MYR','LAY','ZAH','BVK','BVL','BVM'],
    'aircraft_reg_no':[0,1,2,3,4,5,6,7,8,9],
    'aircraft_type1':[0,0,0,0,0,0,0,1,1,1]
}
aircraft_frame_df =DataFrame(aircraft_frame)
aircraft_frame_df


# In[60]:


duty = {
    'duty_name':['CVR1','CVR2','CVR3','CVR4','CVR5','CVR6','CVR7','PRD1','PRD2','PRD3'],
    'duty_no':[0,1,2,3,4,5,6,7,8,9]
}
duty_df = DataFrame(duty)
duty_df


# ### Index Object

# In[61]:


#import numpy as np
#import pandas as pd
#from pandas import Series, DataFrame


# In[62]:


series1 = Series([10,20,30,40],index=['a','b','c','d'])
series1


# In[65]:


index1 = series1.index
index1


# In[66]:


index1[2:]


# In[67]:


#negative index
index1[-2:]


# In[69]:


index1[:-2]


# In[70]:


index1[2:4]


# In[71]:


index1[0] ='a'
index1


#   ### Reindexing method

# In[72]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randn


# In[73]:


#create new series series1
series1 = Series([1,2,3,4], index=['e','f','g','h'])
series1


# In[74]:


#creating new indexes using reindex
series2 = series1.reindex(['e','f','g','h','i','j'])
series2


# In[75]:


series2 = series2.reindex(['e','f','g','h','i','j','k'],fill_value=10)
series2


# In[76]:


#using reindex method to fill
cars = Series(['Audi','Merc','BMW'], index=[1,2,3])
cars


# In[77]:


ranger = range(13)
ranger


# In[79]:


cars = cars.reindex(ranger, method='ffill') #forward fill
cars


# In[81]:


#Create new dataframe using randn
df_1 = DataFrame(randn(25).reshape(5,5),index=['a','b','c','d','e'], columns=['c1','c2','c3','c4','c5'])
df_1


# In[82]:


df_2 = df_1.reindex(['a','b','c','d','e','f'])
df_2


# In[85]:


df_3 = df_2.reindex(columns=['c1','c2','c3','c4','c5','c6'])
df_3


# In[87]:


#using .ix[]to reindex
#df_4 = df_1.ix[['a','b','c','d','e','f'],['c1','c2','c3','c4','c5','c6']]
#df_4


# In[88]:


cars = Series(['BMW','Audi','Merc'],index=['a','b','c'])
cars


# In[89]:


cars = cars.drop('a')
cars


# In[91]:


#dataframe
cars_df = DataFrame(np.arange(9).reshape(3,3),index=['BMW','Audi','Merc'],columns=['rev','pro','exp'])
cars_df


# In[92]:


cars_df = cars_df.drop('BMW', axis=0)
cars_df


# In[93]:


cars_df = cars_df.drop('pro',axis=1)
cars_df


# ### Drop Entries

# In[2]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

cars = Series(['BMW','Audi','Merc'],index=['a','b','c'])
cars


# In[3]:


cars = cars.drop('a')
cars


# In[5]:


cars_df = DataFrame(np.arange(9).reshape(3,3),index=['BMW','Audi','Merc'],columns=['rev','pro','exp'])
cars_df


# In[8]:


cars_df = cars_df.drop('BMW',axis=0)
cars_df


# In[9]:


cars_df = cars_df.drop('pro',axis=1)
cars_df


# ### Handling null data

# In[13]:


#import relevant libraries
from pandas import Series, DataFrame

series1 = Series(['A','B','C','D',np.nan])


# In[12]:


series1.isnull()


# In[14]:


series1.dropna()


# In[18]:


df1 = DataFrame([[1,2,3],[5,6,np.nan],[7,np.nan,10],[np.nan,np.nan,np.nan]])
df1


# In[19]:


df1.dropna()


# In[20]:


df1.dropna(how='all')


# In[21]:


df1.dropna(axis=1) #column wise drop


# In[22]:


df2= DataFrame([[1,2,3,np.nan],[4,5,6,7],[8,9,np.nan,np.nan],[12,np.nan,np.nan,np.nan]])
df2


# In[23]:


df2.dropna(thresh=3)


# In[24]:


df2.dropna(thresh=2)


# In[25]:


#filllna
df2.fillna(0)


# In[26]:


df2.fillna({0:0,1:50,2:100,3:200})


# ### Selecting Modifying Enteries

# In[27]:


series1 = Series([100,200,300],index=['A','B','C'])
series1


# In[28]:


series1['A']


# In[30]:


series1['B']


# In[31]:


series1[['A','B']]


# In[32]:


#number indexes
series1[0]


# In[34]:


series1[0:2]


# In[35]:


#conditional indexes
series1[series1>150]


# In[36]:


series1[series1==300]


# In[37]:


#using df and accessing
df1=DataFrame(np.arange(9).reshape(3,3),index=['car','bike','cycle'],columns=['A','B','C'])
df1


# In[38]:


df1['A']


# In[39]:


df1[['A','B']]


# In[40]:


df1>5


# In[41]:


df1.ix['bike']


# In[42]:


df1.loc['bike']


# In[43]:


df1.iloc[2]


# ### Data Alignment

# In[44]:


ser_a = Series([100,200,300],index=['a','b','c'])
ser_b = Series([300,400,500,600],index=['a','b','c','d'])


# In[45]:


ser_a + ser_b


# In[46]:


#dataframe
df1 = DataFrame(np.arange(4).reshape(2,2),columns=['a','b'],index=['car','bike'])
df1


# In[47]:


df2= DataFrame(np.arange(9).reshape(3,3),columns=['a','b','c'],index=['car','bike','cycle'])
df2


# In[48]:


df1+df2


# In[49]:


df1=df1.add(df2,fill_value=0)
df1


# In[51]:


ser_c = df2.ix[0]


# In[52]:


ser_c


# In[53]:


df2-ser_c


# ### Ranking sorting

# In[54]:


import numpy as np
import pandas as pd
from pandas import Series
from numpy.random import randn


# In[56]:


ser1 = Series([500,1000,1500],index=['a','c','b'])
ser1


# In[57]:


#sort by index
ser1.sort_index()


# In[58]:


#sort by values
ser1.sort_values()


# In[60]:


ser1.rank()


# In[65]:


#ranking of series
ser2 = Series(randn(10))
ser2


# In[66]:


ser2.rank()


# In[69]:


ser2 = ser2.sort_values()


# In[70]:


ser2.rank()


# ### Pandas Statistics

# In[71]:


from pandas import DataFrame
import matplotlib.pyplot as plt


# In[72]:


array1 = np.array([[10,np.nan,20],[30,40,np.nan]])
array1


# In[73]:


df1 = DataFrame(array1, index=[1,2],columns=list('ABC'))
df1


# In[74]:


#sum()
df1.sum() #sums along each column


# In[75]:


df1.sum(axis=1) #sum along indexes


# In[76]:


df1.min()#minimum value along each column


# In[77]:


df1.max()#maximum value along each column


# In[78]:


df1.idxmax() #maximum index


# In[79]:


#cummulative sum
df1.cumsum()


# In[80]:


df1.describe() #statistical description of dataset


# In[81]:


df2 = DataFrame(randn(9).reshape(3,3),index=[1,2,3],columns=list('ABC'))
df2


# In[82]:


plt.plot(df2)
plt.legend(df2.columns,loc='lower right')
plt.savefig('samplepic.png')
plt.show()


# In[83]:


ser1 = Series(list('abcccaabd'))
ser1.unique()


# In[84]:


ser1.value_counts()


# In[ ]:




