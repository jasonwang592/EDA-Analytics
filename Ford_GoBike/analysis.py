import pandas as pd
import os
import pickle

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
  df.columns = ['bike id', 'all bikeshare', 'duration', 'end station id','end station name', 'end time', 'birth year',
                'gender', 'start station id', 'start station name', 'start time', 'user type', 'start location', 'end location']
  df = df[['bike id', 'all bikeshare', 'birth year', 'gender', 'user type', 'duration', 'start station id', 'start station name',
          'start time', 'start location', 'end station id', 'end station name', 'end time', 'end location']]
  pickle.dump(df, open(pickle_name, 'wb'))
else:
  df = pickle.load(open(pickle_name, 'rb'))

print(df.head())

print(df.groupby('user type').mean()['duration'])
