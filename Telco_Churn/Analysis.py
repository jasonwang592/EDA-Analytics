import pandas as pd
import seaborn as sns
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

output_dir = 'output/'

def hist_wrapper(df,var,output_dir,ax=None,save=True):
  '''Simple helper to plot histograms for age or ride duration distribution
  Args:
    - df          (Series)   : The series for which to plot the histogram
    - var         (String)   : Variable for which to plot distribution
    - output_dir  (String)   : Base directory for this set of charts
    - save        (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''

  if ax is None:
    title = ' '.join(['Distribution of',var.capitalize()])
    fig = plt.figure(figsize = (15,9))
  else:
    title = var

  if df[var].dtype == 'object':
    sns.countplot(x=var,data=df,ax=ax)
  else:
    sns.distplot(df[var],kde=False,ax=ax)
  plt.xlabel(var.capitalize())
  plt.xticks(rotation=90)
  # plt.ylabel('Count')
  # plt.title(title)
  # plt.tight_layout()
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + title)
  # else:
    # plt.show()

df = pd.read_csv('Telco.csv')

print(df.head())
df['TotalCharges'] =  df['TotalCharges'].replace(" ","0")
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric)
y = df['Churn']

features = list(df.columns)
features.remove('customerID')
features.remove('Churn')
cont_features = ['TotalCharges','MonthlyCharges','tenure']
demo_features = ['gender','Partner','Dependents','SeniorCitizen']
cat_features = []
for f in features:
  if f not in (cont_features+demo_features):
    cat_features.append(f)
print(len(cat_features))
hist_dir = output_dir + 'histograms/'
if not os.path.exists(hist_dir):
  os.makedirs(hist_dir)

fig,axes = plt.subplots(4,3,figsize=(70,25))
for col,ax in zip(cat_features,axes.flatten()):
  hist_wrapper(df,col,hist_dir,ax=ax,save=False)
plt.tight_layout()
plt.savefig(hist_dir + 'Categorical')
plt.close()

fig,axes = plt.subplots(2,2,figsize=(30,30))
for col,ax in zip(demo_features,axes.flatten()):
  hist_wrapper(df,col,hist_dir,ax=ax,save=False)
plt.tight_layout()
plt.savefig(hist_dir + 'Demographic')
plt.close()

# hist_wrapper(df,'InternetService',hist_dir,save=False)