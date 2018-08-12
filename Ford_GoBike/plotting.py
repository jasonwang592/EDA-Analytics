import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def bar_wrapper(df,x,y,title,xlab,ylab,output_dir,save,suffix,orientation='v'):
  '''Plots the barchart for corresponding to some value associated with a name
  Args:
    - df          (DataFrame): The dataframe containing all data
    - x           (String)   : Name of dataframe column to act as independent variable if vertical
    - y           (String)   : Name of dataframe column to act as dependent variable if vertical
    - title       (String)   : Title of the chart
    - xlab        (String)   : X-axis label
    - ylab        (String)   : Y-axis label
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
    - suffix      (String)   : Suffix to append to title, also the filename
    - orientation (String)   : 'v' for vertical bar chart, 'h' for horizontal bars
  '''

  fig = plt.figure(figsize = (15,9))
  if orientation == 'v':
    pal = sns.color_palette("RdYlGn_r", len(df[x].unique()))
    rank = df[y].argsort().argsort()
  else:
    pal = sns.color_palette("RdYlGn_r", len(df[y].unique()))
    rank = df[x].argsort().argsort()
  sns.barplot(data=df, x=x, y=y,palette=np.array(pal[::-1])[rank])
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(title)
  plt.yticks(fontsize=10)

  plt.tight_layout()
  fname = suffix

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()

def boxplot_wrapper(df,x,y,title,xlab,ylab,output_dir,save,suffix,hue=None,outliers=False):
  '''Plots the boxplot for the provided data.
  Args:
    - df          (DataFrame): The dataframe containing all data
    - x           (String)   : Name of dataframe column to act as independent variable if vertical
    - y           (String)   : Name of dataframe column to act as dependent variable if vertical
    - title       (String)   : Title of the chart
    - xlab        (String)   : X-axis label
    - ylab        (String)   : Y-axis label
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
    - suffix      (String)   : Suffix to append to title, also the filename
    - hue         (String)   : Variable by which to group data by
    - outliers    (Boolean)  : Show outliers if True
  '''
  fig = plt.figure(figsize = (15,9))
  g = sns.boxplot(x=x,y=y,data=df,showfliers=outliers,hue=hue)
  if hue:
    plt.legend(title=' '.join(hue.split('_')).capitalize())
  if outliers:
    title = '-'.join([title,'with outliers'])
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(title)
  plt.tight_layout()
  fname = suffix
  fname = '-'.join([fname,'boxplot'])
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()

def violinplot_wrapper(df,x,y,title,xlab,ylab,output_dir,save,suffix,hue=None,outliers=False):
  '''Plots the boxplot for the provided data.
  Args:
    - df          (DataFrame): The dataframe containing all data
    - x           (String)   : Name of dataframe column to act as independent variable if vertical
    - y           (String)   : Name of dataframe column to act as dependent variable if vertical
    - title       (String)   : Title of the chart
    - xlab        (String)   : X-axis label
    - ylab        (String)   : Y-axis label
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
    - suffix      (String)   : Suffix to append to title, also the filename
    - hue         (String)   : Variable by which to group data by
    - outliers    (Boolean)  : Show outliers if True
  '''
  fig = plt.figure(figsize = (15,9))
  g = sns.violinplot(x=x,y=y,data=df,showfliers=outliers,hue=hue,split=True)
  if outliers:
    title = '-'.join([title,'with outliers'])
  if hue:
    plt.legend(title=' '.join(hue.split('_')).capitalize())
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(title)
  plt.tight_layout()
  fname = suffix
  if hue:
    fname = '-'.join([fname,'violinplot'])
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()




