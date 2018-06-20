import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')


def heatmapper(df, gender, output_dir, save = True):
  '''Plots the heatmap corresponding with the number of occurences of a name.

  Args:
    - df        (DataFrame): The dataframe all data
    - gender    (String)   : The gender that we interested in investigating
    - output_dir(String)   : The directory to save output to
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''
  output_dir += 'NameHeatmaps/'
  df = df[df['gender'] == gender]
  result = df.pivot(index = 'name', columns = 'decade', values = 'number')
  fig = plt.figure(figsize = (15, 9))
  fig = sns.heatmap(result, cmap = 'YlGnBu')
  plt.xticks(rotation = 90)
  fname = ' '.join['Top', len(df['name'].unique()), gender, 'Names']
  plt.title(fname)

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()

def namePopularity(df, name, output_dir, save = True):
  '''Plots the number of occurrences of a provided name over the span of the dataset.

  Args:
    - df        (DataFrame): The dataframe all data
    - name      (String)   : The name that we are interested in investigating
    - output_dir(String)   : The directory to save output to
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''
  if name not in df['name']:
    print('There is no one with this recorded name.')
    return
  output_dir += 'NamePopularityLinePlots/'
  filter_df = df[df['name'] == name]
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
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + name)
  else:
    plt.show()
  plt.close()