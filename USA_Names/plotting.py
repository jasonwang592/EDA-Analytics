import seaborn as sns
import os
import matplotlib.pyplot as plt

def heatmapper(df, gender, output_dir, save = True):
  result = df.pivot(index = 'name', columns = 'decade', values = 'number')
  fig = plt.figure(figsize = (15, 9))
  fig = sns.heatmap(result, cmap = 'YlGnBu')
  plt.xticks(rotation = 90)
  plt.title(''.join['Top', len(df['name'].unique()), gender, 'Names'])

  if save:  
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + ','.join(['PCA Cumulative Variance', str(df.shape[1]) + ' components']))
  else:
    plt.show()
  plt.close()