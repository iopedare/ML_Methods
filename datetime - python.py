# python_datetime

# Import libraries
import datetime
import time


#date module and the classes
#1 date - only year, month and day
#2 time - only time measured in hours, minutes, seconds
#3 datetime - combination of date and time
#4 timedelta - Aduration expressing the difference between two date, time or datetime instances
### time module
today_date = datetime.date.today()
print(today_date)
print(type(today_date))

#specifying a date in string to create a datetime object
someday = datetime.date(year=2019, month=10, day=10)
someday

### datetime.time
now = datetime.time(hour=4, minute=3, second=10, microsecond=7199)

### datetime.datetime
dt = datetime.datetime.now()
dt.tzinfo

# Formatting the datetime object
#converting datetime to strings
today_date.strftime('%B %d, %Y')
today_date.strftime('%Y/%m/%d')
dt.strftime('%Y-%m-%d %H:%M:%S')

#converting strings to datetime. Have tospecify how the string is formatted
dt1= datetime.datetime.strptime('2015-12-31 11:32', '%Y-%m-%d %H:%M')

### timedelta
###used to add or subtract days, weeks, hours, minutes, seconds, micro/milliseconds

#Calculating the date difference - 3 weeks and 2 days before oct 1, 2019
oct_date = datetime.datetime(year=2019, month=10, day=1, hour=4, minute=3, second=10)
three_weeks = datetime.timedelta(weeks=3, days=2) #THis is the timedelta object

oct_before = (oct_date - three_weeks)
oct_after = (oct_date + three_weeks)

# Chevron stay home period
stayhome_date = datetime.datetime(year=2020, month=3, day=23)
thirty_days = datetime.timedelta(weeks=4, days=2)

thirty_days_after = (stayhome_date + thirty_days)

#time module
t = time.time()
time.gmtime()
time.strftime('%Y-%m-%d %H:%M %Z', time.gmtime(t))

###Converting between time formats
#Conveting local datetime to UTC
datetime.datetime.utcnow()

#converting local datetime object to ISO 8601 format
print(dt.isoformat())

#convert from unix timestamp to local time
print(datetime.datetime.fromtimestamp(t))
print(datetime.datetime.utcfromtimestamp(t))

###Working with Timezones
#https://dateutil.readthedocs.io/en/stable/
from dateutil import tz
NYC = tz.gettz('America/New_York')
print(datetime.datetime.now(tz.tzutc()))
print(datetime.datetime.now(NYC))



























