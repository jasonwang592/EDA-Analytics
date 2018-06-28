import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')

def bars(df, gender, output_dir, save = True):
  '''Plots the barchart for corresponding to some value associated with a name

  Args:
    - df        (DataFrame): The dataframe containing all data
    - gender    (String)   : What gender data is contained within the dataframe
    - output_dir(String)   : The directory to save output to
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''
  output_dir += 'AverageAgeNames/'
  n = df.shape[0]

  fig = plt.figure(figsize = (15, 12))
  fig = sns.barplot(data = df, x = df.columns[0], y = df.columns[1])
  plt.title(' '.join(['Average age of top', str(n), 'most common', gender.lower(), 'names']))
  plt.xlabel('Name')
  plt.xticks(rotation = 90)
  plt.ylabel('Age')
  plt.tight_layout()
  fname = ' '.join([gender, 'average age'])
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()

def heatmapper(df, gender, output_dir, save = True):
  '''Plots the heatmap corresponding with the number of occurences of a name.

  Args:
    - df        (DataFrame): The dataframe containing all data
    - gender    (String)   : The gender contained within the DataFrame
    - output_dir(String)   : The directory to save output to
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''
  output_dir += 'NameHeatmaps/'
  result = df.pivot(index = 'name', columns = 'decade', values = 'number')
  fig = plt.figure(figsize = (15, 11))
  fig = sns.heatmap(result, cmap = 'Blues')
  plt.xticks(rotation = 90)
  plt.xlabel('Decade')
  plt.ylabel('Name')
  fname = ' '.join(['Top', str(len(df['name'].unique())), gender, 'Names'])
  plt.title(fname)

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()

def namePopularity(df, name, output_dir, prefix = 0, save = True):
  '''Plots the number of occurrences of a provided name over the span of the dataset.

  Args:
    - df        (DataFrame): The dataframe containing all data
    - name      (String)   : The name that we are interested in investigating
    - output_dir(String)   : The directory to save output to
    - prefix    (Integer)  : Optional value to be appended to start of file name
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''

  output_dir += 'NamePopularityLinePlots/'
  filter_df = df[df['name'] == name]
  if filter_df.empty:
    print('There is no one with this recorded name.')
    return
  years = pd.DataFrame({'year': sorted(filter_df['year'].unique())})

  males = filter_df[filter_df['gender'] == 'Male'].sort_values('year')
  females = filter_df[filter_df['gender'] == 'Female'].sort_values('year')

  males = pd.merge(years, males[['year', 'number']], how = 'left').fillna(0)
  females = pd.merge(years, females[['year', 'number']], how = 'left').fillna(0)

  fig = plt.figure(figsize = (15,9))
  ax = fig.add_subplot(111)

  ax.plot(years, males['number'], label = 'Male')
  ax.plot(years, females['number'], label = 'Female')
  plt.legend(loc = 1)
  plt.title(' '.join(['Popularity of the name', name, 'since 1900']))
  plt.xlabel('Year')
  plt.ylabel('Number')
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    if prefix:
      name = ' '.join([str(prefix), '-', name])
    plt.savefig(output_dir + name)
  else:
    plt.show()
  plt.close()