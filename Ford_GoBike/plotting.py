import seaborn as sns
import matplotlib.pyplot as plt

def timeseries_plot(df, x, y, split, title, output_dir, save = True):
  '''Plots the time series plot with associated parameters
  Args:
  - df        (DataFrame): The dataframe containing all data
  - x         (Series)   : The series containing the time-related data
  - y         (Series)   : The series containing the data to plot against time
  - gender    (String)   : What gender data is contained within the dataframe
  - output_dir(String)   : The directory to save output to
  - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''

  fig = plt.figure(figsize = (15,10))
  if split:
    fig = sns.tsplot(df, condition = split, time = x, value = y, unit = split)
  else:
    fig = sns.tsplot(df, time = x, value = y, unit = y)
  plt.xlabel('Date')
  plt.ylabel('Rides')
  plt.tight_layout()
  fname = ' '.join([gender, 'average age'])
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()