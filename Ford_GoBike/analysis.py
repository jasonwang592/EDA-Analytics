import pandas as pd
import os
import pickle
import calendar
import sys
import numpy as np
import plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use('fivethirtyeight')
data_dir = 'data/'
pickle_name = 'GoBike.p'
output_dir = 'output/'
save_run = True

sf_long_coord = [-122.517815,-122.380314]
east_long_coord = [-122.318687,-122.213287]

def region_mapper(x):
  if (x >= sf_long_coord[0]) & (x <= sf_long_coord[1]):
    return 'San Francisco'
  elif (x >= east_long_coord[0]) & (x <= east_long_coord[1]):
    return 'East Bay'
  else:
    return 'South Bay'

if not os.path.exists(pickle_name):
  df = pd.DataFrame()

  for file in os.listdir(data_dir):
    temp_df = pd.read_csv(os.path.join(data_dir,file))
    df = pd.concat([df,temp_df])

  df['start_location'] = df['start_station_latitude'].map(str) + ',' + df['start_station_longitude'].map(str)
  df['end_location'] = df['end_station_latitude'].map(str) + ',' + df['end_station_longitude'].map(str)
  df['start_station_name'] = df['start_station_name'].str.replace(r"\(.*\)","").str.rstrip()
  df['end_station_name'] = df['end_station_name'].str.replace(r"\(.*\)","").str.rstrip()

  #Let's just assume people aren't crazy enough to ride across the entire Bay Area and just worry about start station
  df['region'] = df['start_station_longitude'].apply(region_mapper)
  df.drop(['start_station_latitude', 'start_station_longitude', 'end_station_latitude','end_station_longitude'], axis=1, inplace=True)

  df.columns = ['bike_id', 'all_bikeshare', 'duration', 'end_station_id','end_station_name', 'end_time', 'birth_year',
                'gender', 'start_station_id', 'start_station_name', 'start_time', 'user_type', 'start_location', 'end_location', 'region']
  df = df[['bike_id', 'region', 'all_bikeshare', 'birth_year', 'gender', 'user_type', 'duration', 'start_station_id', 'start_station_name',
          'start_time', 'start_location', 'end_station_id', 'end_station_name', 'end_time', 'end_location']]

  df['route'] = df['start_station_name'].map(str) + '-' + df['end_station_name'].map(str)
  df['start_time'] = pd.to_datetime(df['start_time'])
  df['year'] = df['start_time'].dt.year
  df['month_num'] = df['start_time'].dt.month
  df['month'] = df['month_num'].apply(lambda x: calendar.month_abbr[x])
  df['month_year'] = df['month'].map(str) + ' ' + df['year'].map(str)
  df['day_of_week'] = df['start_time'].dt.weekday_name

  pickle.dump(df, open(pickle_name, 'wb'))
else:
  df = pickle.load(open(pickle_name, 'rb'))


'''Main script'''
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
stations = list(df['start_station_name'].unique())
df['date'] = df['start_time'].dt.date
df = df[df['region'] == 'San Francisco']

'''
Create two aggregated DataFrames that each groups by date and station (pick up and drop off)
then inner join on station and calculate the net bikes at each station for every date to observe
where the surplus and deficit in bikes are.
'''
start_df = df.groupby(['date','start_station_name']).size().reset_index()
end_df = df.groupby(['date', 'end_station_name']).size().reset_index()
start_df.columns = ['date','start_station_name','picked_up']
end_df.columns = ['date','end_station_name','dropped_off']

start_df.set_index('start_station_name')
end_df.set_index('end_station_name')
joint_df = pd.concat([start_df,end_df], axis = 1, join='inner')
joint_df['net_bikes'] = joint_df['dropped_off'] - joint_df['picked_up']
joint_df.drop(['dropped_off','picked_up'], axis = 1, inplace=True)
_,i = np.unique(joint_df.columns, return_index = True)
joint_df = joint_df.iloc[:,i]
joint_df['day'] = pd.to_datetime(joint_df['date']).dt.weekday_name

'''Here we'll understand how bikes move throughout the city and region'''
top_n = 5

#Get the stations that have the greatest surplus and deficit in general
grouped = joint_df.groupby('start_station_name').mean().sort_values('net_bikes')
top_surplus = grouped.head(top_n)
top_deficit = grouped.tail(top_n)

#Now get top N stations by average deficit/surplus by day of week
top_deficit_weekday = joint_df.sort_values(['day','net_bikes']).groupby('day')['start_station_name','net_bikes']
top_deficit = top_deficit_weekday.apply(lambda x: x.nsmallest(top_n,'net_bikes'))
top_surplus = top_deficit_weekday.apply(lambda x: x.nlargest(top_n, 'net_bikes'))
print(top_deficit)
print(top_surplus)
sys.exit()

'''Get the top N stations by date that have the most surplus or deficit of bikes'''
sorted_joint_df = joint_df.sort_values(['date','net_bikes']).groupby('date')['start_station_name', 'net_bikes']
top_deficit = sorted_joint_df.apply(lambda x: x.nsmallest(top_n,'net_bikes'))
top_surplus = sorted_joint_df.apply(lambda x: x.nlargest(top_n, 'net_bikes'))

sys.exit()
'''Let's find some aggregate numbers abour ridership by date, day of week, user type and so on'''
df.index = df['start_time']
grouped = df.groupby(['year','month_num','user_type'], as_index = False)['bike_id'].count().rename(columns = {'bike_id':'count'})
grouped['date'] = pd.to_datetime(grouped['year'].astype(str) + grouped['month_num'].astype(str), format='%Y%m')
grouped['formatted_date'] = grouped['month_num'].apply(lambda x: calendar.month_abbr[x]) + ' ' + grouped['year'].map(str)
grouped = grouped.set_index('date')

fig = plt.figure(figsize = (15,10))
ax1 = grouped[grouped['user_type']=='Subscriber']['count'].plot(x_compat=True)
ax2 = grouped[grouped['user_type']=='Customer']['count'].plot(x_compat=True)
ax1.xaxis.set_major_locator(mdates.MonthLocator())
title = 'Ridership by User Type'
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Rides')
fname = 'Ridership by User Type'
if save_run:
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  plt.savefig(output_dir + fname)
else:
  plt.show()


'''
#Calculate some generic statistics
print(df.groupby('user_type').median()['duration'])
print(df.groupby('gender').size())
print(df.groupby('day_of_week').size())
user_dayofweek = df.groupby(['user_type','day_of_week']).size()
temp = user_dayofweek.unstack()
temp = temp[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
user_dayofweek = temp.stack()
print(user_dayofweek)
'''
