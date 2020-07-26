#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


my_list1 = [1, 2, 3, 4]
my_array1 = np.array(my_list1)


# In[4]:


my_array1


# In[5]:


my_list2 = [5,6,7,8]


# In[6]:


my_array = np.array([my_list1, my_list2])
my_array


# In[7]:


# Usage of shape function
my_array.shape
# 2 rows, 4 columns


# In[8]:





# In[9]:


# Finding out the datatype of the numbers of the array
my_array.dtype


# In[11]:


# Zeros, ones, empty, eye and arange functions
new_array = np.zeros(5) #creates a new numpy with (1*5). All elements are zeros
new_array
# 1 row, 5 columns


# In[13]:


new_array1 = np.ones([5,5])
new_array1


# In[17]:


new_array1 = np.eye(5)
new_array1


# In[24]:


new_array1 = np.arange(0, 11, 1.0)
new_array1


# ### Scaler operations on array
# 

# In[25]:


5/2


# In[27]:


array1 = np.array([[1,2,3,4], [5,6,7,8]])
array1


# In[29]:


# multiplication
array2 = array1*array1
array2


# In[30]:


# Exponential multiplication
array3 = array1 ** 3
array3


# In[31]:


# Subtraction
array4 = array1 - array1
array4


# In[32]:


array5 = array2-array1
array5


# In[33]:


# reciprocal
array6 = 1/array1
array6


# ### array index

# In[34]:


arr = np.arange(0,12)


# In[35]:


arr


# In[39]:


arr[0]


# In[43]:


arr[2]


# In[42]:


arr[0:5] Start from #first column and end on the fifth
#row 1, first 5 columns


# In[44]:


arr[2:6]


# In[46]:


arr[0:5] = 20
arr


# In[47]:


# Interesting thing and Important
arr2 = arr[0:6]
arr2


# In[48]:


arr2[:] = 29


# In[49]:


arr2


# In[50]:


arr


# ### array index2
# 

# In[51]:


arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d


# In[53]:


arr2d[0]


# In[59]:


arr2d[1][2]


# In[65]:


#slice of 2d array
slice1 = arr2d[0:2, 0:2]
slice1


# In[69]:


slice2=arr2d[0:2,1:3]
slice2


# In[70]:


arr2d


# In[71]:


arr2d[:2, 1:] = 15
arr2d


# In[72]:


arr_len = arr2d.shape[0]


# In[73]:


arr_len


# In[78]:


for i in range(arr_len):
    arr2d[i] = i;
    
arr2d


# In[76]:


arr2d[[0,1]]


# In[85]:


arr2d[[2,1]]


# ### Universal array functions
# 
# 1. arange
# 2. sqrt
# 3. exp
# 4. random
# 5. addition
# 6. maximum

# In[86]:


A= np.arange(15)
A


# In[87]:


A = np.arange(1,15,2)
A


# In[89]:


B = np.sqrt(A)
B


# In[90]:


C = np.exp(A)
C


# In[91]:


D = np.add(A,B)


# In[92]:


D


# In[93]:


E = np.maximum(A, B)


# In[94]:


E


# In[95]:


#additional resources
#scipy.org - allfunctios associated with numpy


# 
# ### How to save and load array
# 

# In[97]:


arr = np.arange(10)
arr


# In[101]:


np.save('saved_array', arr)


# In[102]:


new_array = np.load('saved_array.npy')


# In[103]:


new_array


# In[104]:


# saving multiple arrays
array_1 = np.arange(25)
array_2 = np.arange(30)


# In[108]:


np.savez('saved_archive.npz',x = array_1, y = array_2)


# In[109]:


load_archive = np.load('saved_archive.npz')


# In[112]:


load_archive['x']


# In[113]:


load_archive['y']


# In[115]:


# save to txtfile
np.savetxt('notepadfile.txt', array_1, delimiter=',')


# In[116]:


# Loading of txt file
load_txt_file = np.loadtxt('notepadfile.txt',delimiter=',')
load_txt_file


# In[117]:


import matplotlib.pyplot as plt


# In[118]:


axes_values = np.arange(-100, 100, 10)
dx, dy = np.meshgrid(axes_values, axes_values)
dx


# In[119]:


function = 2*dx+3*dy
function


# In[121]:


plt.imshow(function)
plt.title('function ofplot2*dx+3*dy')
plt.colorbar()
plt.savefig('myfig.png')


# In[122]:


function2= np.cos(dx)+np.cos(dy)
plt.imshow(function2)
plt.title('function of plot np.cos(dx)+np.cos(dy)')
plt.colorbar()
plt.savefig('myfig2.png')


# ### Conditional clauses and boolean operations

# In[131]:


x = np.array([100, 400, 500, 600]) #each member 'a'
y = np.array([10, 15, 20, 25]) #each member 'b'
condition = np.array([True, True, False, False]) #each member cond


# In[132]:


z = [a if cond else b for a,cond,b in zip(x,condition, y)]
z


# In[133]:


z2 =np.where(condition, x, y)
z2


# In[134]:


z3 = np.where(x>0,0, 1)
z3


# In[135]:


x.sum()


# In[1]:


import numpy as np
x = np.array([100, 400, 500, 600]) #each member 'a'
y = np.array([10, 15, 20, 25])# each member 'b'
condition = np.array([True, True, False, False])


# In[ ]:





# In[3]:


# use loops indirectly to perform this

z = [a if cond else b for a, cond,b in zip(x,condition, y)]
z


# In[4]:


#np.where(condition, value for yes, value for no)
z2 = np.where(condition, x, y)
z2


# In[5]:


z3 = np.where(x>0,0,1)
z3


# In[6]:


#Standard function of numpy
x.sum()


# In[7]:


n = np.array([[1,2],[3,4]])
n.sum(0)


# In[8]:


x.mean()


# In[9]:


x.std()


# In[10]:


x.var()


# In[13]:


# logical operations and / or operations

condition2 = np.array([True, False, True])

condition2.any() # or operator


# In[14]:


condition2.all() # and operator


# In[15]:


# sorting in numpy arrays
unsorted_array = np.array([1,2,8,10,7,3])
unsorted_array.sort()


# In[16]:


unsorted_array


# In[17]:


arr2 = np.array(['solid', 'solid', 'solid', 'liquid', 'liquid', 'gas', 'gas'])
np.unique(arr2)


# In[ ]:




