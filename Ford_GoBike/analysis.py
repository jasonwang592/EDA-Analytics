import pandas as pd
import os
import pickle
import calendar
import sys
import numpy as np
import plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')

#Setting some high level variables
data_dir = 'data/'
pickle_name = 'GoBike.p'
output_dir = 'output/'
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sf_long_coord = [-122.517815,-122.380314]
east_long_coord = [-122.318687,-122.213287]

def region_mapper(x):
  if (x >= sf_long_coord[0]) & (x <= sf_long_coord[1]):
    return 'San Francisco'
  elif (x >= east_long_coord[0]) & (x <= east_long_coord[1]):
    return 'East Bay'
  else:
    return 'South Bay'

def pickler(repickle=False):
  '''Does some initial cleaning of the data including dropping uninsightful columns and splitting up
  dates into different components.
  '''
  if not os.path.exists(pickle_name) or repickle == True:
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
    df['start_hour'] = df['start_time'].dt.hour
    df['year'] = df['start_time'].dt.year
    df['month_num'] = df['start_time'].dt.month
    df['month'] = df['month_num'].apply(lambda x: calendar.month_abbr[x])
    df['month_year'] = df['month'].map(str) + ' ' + df['year'].map(str)
    df['day_of_week'] = df['start_time'].dt.weekday_name

    pickle.dump(df,open(pickle_name, 'wb'))
  return pickle.load(open(pickle_name, 'rb'))

def weekday_net_bars(df,top_n,output_dir,save):
  #Now get top N stations by average deficit/surplus by day of week
  weekday_df = df.sort_values(['day','net_bikes']).groupby(['day','start_station_name'])['start_station_name','net_bikes'].mean()
  top_deficit = weekday_df.groupby('day').apply(lambda x: x.nsmallest(top_n, 'net_bikes',keep='first')).reset_index(level=0,drop=True)
  top_surplus = weekday_df.groupby('day').apply(lambda x: x.nlargest(top_n, 'net_bikes',keep='first')).reset_index(level=0,drop=True)
  joint = pd.concat([top_deficit,top_surplus])
  joint.reset_index(inplace=True)
  joint.sort_values(['day','net_bikes'],ascending=[True,False],inplace=True)
  for day in ordered_days:
    temp = joint[joint['day'] == day]
    plotting.bar_wrapper(df=temp,
      x='net_bikes',
      y='start_station_name',
      title='Top stations by bike surplus and deficit',
      xlab='Net Bikes',
      ylab='Station Name',
      output_dir=output_dir+'Net Bikes by Weekday/'+'Total/',
      suffix=str(day),
      save=save,
      orientation='h')

  #Do it again but split by user type:
  for user in list(df['user_type'].unique()):
    temp = df[df['user_type']==user]
    weekday_df = temp.sort_values(['day','net_bikes']).groupby(['day','start_station_name'])['start_station_name','net_bikes'].mean()
    top_deficit = weekday_df.groupby('day').apply(lambda x: x.nsmallest(top_n, 'net_bikes',keep='first')).reset_index(level=0,drop=True)
    top_surplus = weekday_df.groupby('day').apply(lambda x: x.nlargest(top_n, 'net_bikes',keep='first')).reset_index(level=0,drop=True)
    joint = pd.concat([top_deficit,top_surplus])
    joint.reset_index(inplace=True)
    joint.sort_values(['day','net_bikes'],ascending=[True,False],inplace=True)
    for day in ordered_days:
      temp = joint[joint['day'] == day]
      plotting.bar_wrapper(df=temp,
        x='net_bikes',
        y='start_station_name',
        title=' '.join([user,'top stations by bike surplus and deficit']),
        xlab='Net Bikes',
        ylab='Station Name',
        output_dir=output_dir+'Net Bikes by Weekday/'+user+'/',
        suffix=str(day),
        save=save,
        orientation='h')

def hour_sum_bars(df,stations,output_dir,save,split_user=False):

  for station in stations:
    o_dir = output_dir
    station_df = df[df['start_station_name']==station]
    if split_user:
      for user in station_df['user_type'].unique():
        user_df = station_df[station_df['user_type']==user]
        hour_df = user_df.groupby('start_hour')['bike_id'].count().reset_index(name='rides')
        plotting.bar_wrapper(df=hour_df,
          x='start_hour',
          y='rides',
          title= 'Rides by hour',
          xlab='Hour',
          ylab='Number of Rides',
          output_dir=output_dir+ user +'/',
          save=save,
          suffix=' - '.join([station,user]),
          orientation='v')
    else:
      hour_df = station_df.groupby('start_hour')['bike_id'].count().reset_index(name='rides')
      plotting.bar_wrapper(df=hour_df,
        x='start_hour',
        y='rides',
        title= 'Rides by hour',
        xlab='Hour',
        ylab='Number of Rides',
        output_dir=output_dir+'Total/',
        save=save,
        suffix=station,
        orientation='v')

def station_analysis(df,region,output_dir,save_run):
  '''
  Performs analysis on stations based on ridership for dates and net bikes at stations for weekdays
  Args:
    - df        (DataFrame): The DataFrame from the pickled object
    - region    (String)   : The region to analyze (San Francisco, East Bay, South Bay)
    - output_dir  (String)   : Base directory for this set of charts
    - save_run  (Boolean)  : If true, saves all charts, otherwise displays them
  '''
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  stations = list(df['start_station_name'].unique())
  df['date'] = df['start_time'].dt.date
  df['day_of_week'] = pd.Categorical(df['day_of_week'], ordered_days)
  df = df[df['region'] == region]
  output_dir += region + '/'

  '''
  Create two aggregated DataFrames that each groups by date and station (pick up and drop off)
  then inner join on station and calculate the net bikes at each station for every date to observe
  where the surplus and deficit in bikes are.
  '''

  #Aggregate all bike sessions by date and station to get a count
  start_df = df.groupby(['date','start_station_name','user_type']).size().reset_index()
  end_df = df.groupby(['date', 'end_station_name','user_type']).size().reset_index()
  start_df.columns = ['date','start_station_name','user_type','picked_up']
  end_df.columns = ['date','end_station_name','user_type', 'dropped_off']
  start_df.set_index('start_station_name')
  end_df.set_index('end_station_name')

  #Tie the dataframes together and get the net flow of bikes for each station
  joint_df = pd.concat([start_df,end_df],axis=1,join='inner')
  joint_df['net_bikes'] = joint_df['dropped_off'] - joint_df['picked_up']
  joint_df.drop(['dropped_off','picked_up'],axis=1,inplace=True)
  #Drop duplicate columns
  _,i = np.unique(joint_df.columns,return_index=True)
  joint_df = joint_df.iloc[:,i]
  joint_df['day'] = pd.to_datetime(joint_df['date']).dt.weekday_name
  joint_df['day'] = pd.Categorical(joint_df['day'], ordered_days)

  '''Here we'll understand how bikes move throughout the city and region'''
  top_n = 5

  #Get the stations that have the greatest surplus and deficit in general
  grouped = joint_df.groupby('start_station_name').mean().sort_values('net_bikes')
  top_surplus = grouped.head(top_n)
  top_deficit = grouped.tail(top_n)

  # Plotting out, by weekday, what stations have the most surplus or deficit
  weekday_net_bars(joint_df, top_n, output_dir, save_run)

  # For the top_n stations by ridership, plots the ridership for those stations by hour
  top_n = 10
  temp = df.groupby(['start_station_name'])['bike_id'].count().reset_index(name='total_rides').sort_values('total_rides', ascending=False)
  station_list = list(temp.head(top_n)['start_station_name'])
  hour_sum_bars(df,station_list,output_dir=output_dir+'Rides by hour/',save=save_run, split_user=True)

  '''Get the top N stations by date that have the most surplus or deficit of bikes'''
  sorted_joint_df = joint_df.sort_values(['date','net_bikes']).groupby('date')['start_station_name', 'net_bikes']
  top_deficit = sorted_joint_df.apply(lambda x: x.nsmallest(top_n,'net_bikes'))
  top_surplus = sorted_joint_df.apply(lambda x: x.nlargest(top_n, 'net_bikes'))

def distribution_analysis(df,inds,dep,titles,hues,xlabs,ylab,output_dir,save):
  '''Helper to unpack lists to generate multiple plots for distributions
  Args:
    - df          (DataFrame): The dataframe containing all data
    - inds        (List)     : List of independent variables to plot for
    - dep         (String)   : Dependent variable to plot distribution of
    - titles      (List)     : List of titles for plots
    - hues        (List)     : List of variables to group by
    - xlabs       (List)     : List of x-axis labels
    - ylab        (String)   : Y-axis label
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
    - suffix      (String)   : Suffix to append to title, also the filename
  '''
  age_bins = sorted(list(df['age_range'].unique()))
  for ind,(x,hue,title,xlab) in enumerate(zip(inds,hues,titles,xlabs)):
    plotting.boxplot_wrapper(df=numerical_df,
      x=x,
      y=dep,
      xlab=xlab,
      ylab=ylab,
      title=title,
      output_dir=output_dir,
      save=True,
      suffix=None,
      outliers=False,
      hue=hue,
      order=age_bins if x=='age_range' else None)
    plotting.violinplot_wrapper(df=numerical_df,
      x=x,
      y=dep,
      xlab=xlab,
      ylab=ylab,
      title=title,
      output_dir=output_dir,
      save=True,
      suffix=None,
      outliers=False,
      hue=hue,
      order=age_bins if x=='age_range' else None)

def hist_wrapper(df,var,output_dir,save,lim=None):
  '''Simple helper to plot histograms for age or ride duration distribution
  Args:
    - df          (Series)   : The series for which to plot the histogram
    - var         (String)   : Variable for which to plot distribution
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
    - lim         (Integer)  : Upper bound for var value
  '''
  if var not in ('age','duration'):
    raise ValueError("Use either 'age' or 'duration' as the distribution to plot")
  if var == 'age':
    if lim:
      title =  title = ' '.join(['Distribution of rider age under',str(lim)])
      df = df.loc[df[var]<=lim]
    else:
      title = 'Distribution of rider age'
  else:
    if lim:
      title = title = ' '.join(['Distribution of ride duration under',str(lim)])
      df = df.loc[df[var]<=lim]
    else:
      title = 'Distribution of ride duration'

  fig = plt.figure(figsize = (15,9))
  sns.distplot(df[var])
  plt.xlabel(var.capitalize())
  plt.ylabel('Count')
  plt.title(title)
  plt.tight_layout()
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + title)
  else:
    plt.show()
  plt.close()

def aggregate_plots(df):
  plotting.bar_wrapper(df=df.groupby('gender')['bike_id'].count().reset_index(name='rides'),
    x='gender',
    y='rides',
    title='Rides by gender',
    xlab='Gender',
    ylab='Rides',
    output_dir=output_dir+'Aggregates/',
    save=save_run)
  plotting.bar_wrapper(df=df.groupby('region')['bike_id'].count().reset_index(name='rides'),
    x='region',
    y='rides',
    title='Rides by region',
    xlab='Region',
    ylab='Rides',
    output_dir=output_dir+'Aggregates/',
    save=save_run)
  plotting.bar_wrapper(df=df.groupby('user_type')['bike_id'].count().reset_index(name='rides'),
    x='user_type',
    y='rides',
    title='Rides by user type',
    xlab='User Type',
    ylab='Rides',
    output_dir=output_dir+'Aggregates/',
    save=save_run)
  plotting.bar_wrapper(df=df.groupby('start_hour')['bike_id'].count().reset_index(name='rides'),
    x='start_hour',
    y='rides',
    title='Rides per hour',
    xlab='Hour',
    ylab='Rides',
    output_dir=output_dir+'Aggregates/',
    save=save_run)
  plotting.bar_wrapper(df=df.groupby(['start_hour','user_type'])['bike_id'].count().reset_index(name='rides'),
    x='start_hour',
    y='rides',
    title='Rides per hour by user type',
    xlab='Hour',
    ylab='Rides',
    hue='user_type',
    output_dir=output_dir+'Aggregates/',
    save=save_run)

def route_analysis(df,top_n,output_dir,save,split_user=False):
  '''
  Performs analysis on stations based on ridership for dates and net bikes at stations for weekdays
  Args:
    - df         (DataFrame): The DataFrame from the pickled object
    - top_n      (Integer)  : The top n routes to plot
    - output_dir (String)   : Base directory for this set of charts
    - save_run   (Boolean)  : If true, saves all charts, otherwise displays them
    - split_user (Boolean)  : If true, splits the charts and data on user type
  '''
  output_dir = output_dir + 'Routes/'
  if split_user:
    route_df = df.groupby(['user_type','route'])['bike_id'].count().reset_index(name='rides')
    for user in list(route_df['user_type'].unique()):
      temp = route_df.loc[route_df['user_type']==user]
      temp = temp.sort_values('rides',ascending=False).head(top_n)
      plotting.bar_wrapper(df=temp,
        x='rides',
        y='route',
        title=' '.join(['Top',str(top_n),'most popular routes']),
        xlab='Number of rides',
        ylab='Station',
        output_dir=output_dir,
        orientation='h',
        suffix=user)
  else:
    route_df = df.groupby('route')['bike_id'].count().reset_index(name='rides')
    route_df.sort_values('rides',ascending=False,inplace=True)
    top_routes_df = route_df.head(top_n)
    plotting.bar_wrapper(df=top_routes_df,
      x='rides',
      y='route',
      title=' '.join(['Top',str(top_n),'most popular routes']),
      xlab='Number of rides',
      ylab='Station',
      output_dir=output_dir,
      orientation='h')


if __name__ == '__main__':
  save_run = True
  df = pickler()
  # #General plots about high level information
  aggregate_plots(df)

  #Calculate some generic statistics
  user_dayofweek = df.groupby(['user_type','day_of_week']).size()
  temp = user_dayofweek.unstack()
  temp = temp[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
  user_dayofweek = temp.stack()

  #Analysis of net bikes at stations within a given region
  print('Processing stations...')
  station_analysis(df, 'San Francisco',output_dir,save_run)
  print('Stations complete.')

  #Analysis on the most popular routes (defined by start and end station)
  print('Processing routes...')
  route_analysis(df,10,output_dir,save_run,split_user=True)
  print('Routes complete.')

  #Of 1,338,864 rides, roughly 125,000 are missing data on birth_year and gender. We'll drop those
  numerical_df = df.copy()
  numerical_df = numerical_df.dropna(subset=['birth_year','gender'])
  numerical_df['age'] = numerical_df['birth_year'].apply(lambda x:pd.Timestamp.now().year-x)
  numerical_df['age'] = numerical_df['age'].map(int)
  bins = [18,23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98,np.inf]
  bin_names = ['18-23','23-28','28-33','33-38','38-43','43-48','48-53','53-58','58-63','63-68',
              '68-73','73-78','78-83','83-88','88-93','93-98','98+']
  d = dict(enumerate(bin_names,1))
  numerical_df['age_range'] = list(map(d.get, np.digitize(numerical_df['age'],bins)))

  numerical_df=numerical_df.loc[(numerical_df['age']<=70) & (numerical_df['duration']<=2000)]

  #Plot some histograms for age and duration and narrow the dataset
  print('Processing Histograms...')
  hist_wrapper(df=numerical_df,
    var='age',
    output_dir=output_dir+'Distribution Plots/',
    save=save_run,
    lim=70)
  hist_wrapper(df=numerical_df,
    var='duration',
    output_dir=output_dir+'Distribution Plots/',
    save=save_run,
    lim=2000)
  print('Histograms complete.')

  #Set together all the plots we want to create stats for with their parameters and zip them up
  box_x = ['gender','user_type','gender']
  box_hue = [None,None,'user_type']
  box_title = ['Age distribution by gender',
    'Age distribution by user type',
    'Age distribution by gender and user type']
  box_xlab = ['Gender','User Type','User Type']
  print('Processing age distributions...')
  distribution_analysis(df=numerical_df,
    inds=box_x,
    dep='age',
    hues=box_hue,
    xlabs=box_xlab,
    ylab='Age',
    output_dir=output_dir+'Numerical Plots/Age distribution/',
    titles=box_title,
    save=save_run)
  print('Age distributions complete.')
  print('Processing duration distributions...')
  box_x = ['gender','user_type','gender','age_range','age_range']
  box_hue = [None,None,'user_type',None,'user_type']
  box_title = ['Ride duration distribution by gender',
    'Ride duration by user type',
    'Ride duration by gender and user type',
    'Ride duration by age group',
    'Ride duration by age group and user type']
  box_xlab = ['Gender','User Type','User Type','Age Range','User Type']
  distribution_analysis(df=numerical_df,
    inds=box_x,
    dep='duration',
    hues=box_hue,
    xlabs=box_xlab,
    ylab='Ride Duration',
    output_dir=output_dir+'Numerical Plots/Duration distribution/',
    titles=box_title,
    save=save_run)
  print('Duration distributions complete.')

  sys.exit()

  '''Let's find some aggregate numbers about ridership by date, day of week, user type and so on'''
  df.index = df['start_time']
  grouped = df.groupby(['year','month_num','user_type'], as_index = False)['bike_id'].count().rename(columns = {'bike_id':'count'})
  grouped['date'] = pd.to_datetime(grouped['year'].astype(str) + grouped['month_num'].astype(str), format='%Y%m')
  grouped['formatted_date'] = grouped['month_num'].apply(lambda x: calendar.month_abbr[x]) + ' ' + grouped['year'].map(str)
  grouped = grouped.set_index('date')

  fig = plt.figure(figsize = (13,10))
  ax1 = grouped[grouped['user_type']=='Subscriber']['count'].plot(x_compat=True)
  ax2 = grouped[grouped['user_type']=='Customer']['count'].plot(x_compat=True)
  ax1.xaxis.set_major_locator(mdates.MonthLocator())
  title = 'Ridership by User Type and Month'
  plt.title(title)
  plt.xlabel('Date')
  plt.ylabel('Rides')
  fname = 'Ridership by User Type and Month'
  if save_run:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
