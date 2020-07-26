#pandas_numpy_datetime

import numpy as np
import pandas as pd
import datetime

#numpy's datetime64 and timedelta64 objects
#ccreate by using an integer with a string for the units
np.datetime64(5, 'ns')

#you can also use stings as long as they are in ISO 8601 format
np.datetime64('2019-10-22T18:25:04')

#timedeltas have a single unit

print(np.timedelta64(8, 'D')) #8 days
print(np.timedelta64(6, 'h')) # 6 hours

# can also create them by strucking two datetome64 objects
np.datetime64('2019-10-24T05:30:45.67')-np.datetime64('2019-10-22T12:35:40.123')

#pandas Timestamp and datetime64
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Tto_datetime.html
#pd.Timestamp
pd.Timestamp(1239.1238934) #defaults to nanoseconds

pd.Timestamp(1239.1238934,unit='D') #change units

pd.Timestamp('2019-10-22 05') #partial strings work




#pd.to_datetime

#create a timestamp
pd.to_datetime('2019-10-22 05')

#convert a list of strings into timestamps
pd.to_datetime(['2019-10-01','2019-10-05'])

###converting a date string from a file to datetime object

url= 'http://samplecsvs.s3.amazonaws.com/Sacramentorealestatetransactions.csv'
sample_data = pd.read_csv(url)
sample_data.head()

























