import pandas as pd
import os
import pickle
import calendar
import sys

data_dir = 'data/'
pickle_name = 'GoBike.p'

if not os.path.exists(pickle_name):
  df = pd.DataFrame()

  for file in os.listdir(data_dir):
    temp_df = pd.read_csv(os.path.join(data_dir,file))
    df = pd.concat([df,temp_df])

  df['start_location'] = df['start_station_latitude'].map(str) + ',' + df['start_station_longitude'].map(str)
  df['end_location'] = df['end_station_latitude'].map(str) + ',' + df['end_station_longitude'].map(str)
  df['start_station_name'] = df['start_station_name'].str.replace(r"\(.*\)","").str.rstrip()
  df['end_station_name'] = df['end_station_name'].str.replace(r"\(.*\)","").str.rstrip()
  df.drop(['start_station_latitude', 'start_station_longitude', 'end_station_latitude','end_station_longitude'], axis=1, inplace=True)
  df.columns = ['bike_id', 'all_bikeshare', 'duration', 'end_station_id','end_station_name', 'end_time', 'birth_year',
                'gender', 'start_station_id', 'start_station_name', 'start_time', 'user_type', 'start_location', 'end_location']
  df = df[['bike_id', 'all_bikeshare', 'birth_year', 'gender', 'user_type', 'duration', 'start_station_id', 'start_station_name',
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

# print(df.head())
stations = list(df['start_station_name'].unique())
df['date'] = df['start_time'].dt.date


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
max_idx = joint_df.groupby(['date'])['net_bikes'].transform(max) == joint_df['net_bikes']
min_idx = joint_df.groupby(['date'])['net_bikes'].transform(min) == joint_df['net_bikes']
deficit_df = joint_df[min_idx]
surplus_df = joint_df[max_idx]
print(deficit_df)
# print(joint_df.head(100))


df.index = df['start_time']
temp = df.groupby(['year', 'month_num']).size()
temp = temp.reset_index()
temp['month'] = temp['month_num'].apply(lambda x: calendar.month_name[x])
temp['month_year'] = temp['month'].map(str) + ' ' + temp['year'].map(str)

'''
print(df.groupby('user_type').mean()['duration'])
print(df.groupby('gender').size())
print(df.groupby('day_of_week').size())
user_dayofweek = df.groupby(['user_type','day_of_week']).size()
temp = user_dayofweek.unstack()
temp = temp[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
user_dayofweek = temp.stack()
print(user_dayofweek)
'''
