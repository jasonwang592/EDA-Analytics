import seaborn as sns
import matplotlib.pyplot as plt
import os

def bar_wrapper(df,x,y,title,xlab,ylab,output_dir,save,day=None):
  '''Plots the barchart for corresponding to some value associated with a name

  Args:
    - df        (DataFrame): The dataframe containing all data
    - gender    (String)   : What gender data is contained within the dataframe
    - output_dir(String)   : The directory to save output to
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''
  fig = plt.figure(figsize = (15,9))
  fname =' - '.join([title,day])
  sns.barplot(data=df, x=x, y=y,palette=sns.color_palette("RdYlGn_r", 10))
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(fname)
  plt.yticks(fontsize=10)

  plt.tight_layout()
  fname ='-'.join([title,day])

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + fname)
  else:
    plt.show()
  plt.close()
