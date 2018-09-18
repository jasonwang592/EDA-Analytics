import numpy as np
import pickle
import pandas as pd
import analysis
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA


output_dir = 'output/Prediction/'

def downsample(df,features,target):
  '''Takes an imbalanced DataFrame in terms of target variable and downsamples
  the majority label
  Args:
    - df      (DataFrame): The DataFrame with unbalanced target variables
    - features(list)     : List of features we want to keep
    - target  (String)   : The column name of the unbalanced target variable
  '''
  over_label = df[target].value_counts().idxmax()
  under_label = df[target].value_counts().idxmin()
  over_count = df[target].value_counts()[over_label]
  under_count = df[target].value_counts()[under_label]

  df_over = df[df[target]==over_label]
  df_under = df[df[target]==under_label]
  df_over_undersampled = df_over.sample(under_count)
  balanced = pd.concat([df_over_undersampled,df_under],axis=0)
  y = balanced[target]
  X = balanced[features]
  X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=.3,
    random_state=21)
  return X_train,X_test,y_train,y_test

def upsample(df,features,target):
  '''Takes an imbalanced DataFrame in terms of target variable and upsamples
  the minority target using SMOTE
  Args:
    - df      (DataFrame): The DataFrame with unbalanced target variables
    - features(list)     : List of features we want to keep
    - target  (String)   : The column name of the unbalance target variable
  '''
  y = df[target]
  X = df[features]
  sm = SMOTE(random_state=12,ratio=1.0)
  X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=.3,
    random_state=21)
  X_train, y_train = sm.fit_sample(X_train,y_train)
  return X_train,X_test,y_train,y_test

def fit_tree(X_train,X_test,y_train,y_test,output_dir,filename=None,show_confusion=True):
  '''Splits DataFrame into train and test sets. Fits a decision tree model
  and uses the model to predict on test set
  Args:
    - X_train     (DataFrame): DataFrame with features for training
    - y_train     (Series)   : Series with target variable for training
    - X_test      (DataFrame): DataFrame with features for test
    - y_test      (Series)   : Series with target variable for test
    - output_dir  (String)   : Output directory to save confusion matrix plot
    - filename    (String)   : Name of file to write confusion matrix to
    - show_matrix (Boolean)  : Shows the confusion matrix if true
  '''
  clf = tree.DecisionTreeClassifier()
  clf.fit(X=X_train,y=y_train)
  print('Validation accuracy:{:.4f}'.format(clf.score(X=X_test,y=y_test)))

  y_pred = clf.predict(X_test)
  conf_matrix = confusion_matrix(y_true=y_test,y_pred=y_pred)
  if "Subscriber" in y_train:
    labels = ['Customer','Subscriber']
  else:
    labels = ['Male','Female']
  acc = "accuracy:{:.4f}".format(clf.score(X_test,y_test))
  prec = "precision:{:.4f}".format(precision_score(y_pred,y_test))
  rec = "recall:{:.4f}".format(recall_score(y_pred,y_test))
  text = '\n'.join([acc,prec,rec])

  fig = plt.figure(figsize = (15,9))
  ax = plt.subplot(111)
  cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
  fig.colorbar(cax)
  ax.text(-1,0,text)
  ax.set_xticklabels([''] + labels)
  ax.set_yticklabels([''] + labels)
  plt.xlabel('Predicted')
  plt.ylabel('Expected')
  plt.title(filename)
  plt.tight_layout()
  if show_confusion:
    plt.show()
  else:
    if not filename:
      raise ValueError("Provide filename argument when saving confusion matrix.")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + filename)
  plt.close()
  return clf

def tree_classifier(df,features,target,output_dir):
  '''Performs classification using a decision tree with upsampling/downsampling
  Args:
    - tree_df     (DataFrame): DataFrame containing data
    - features    (List)     : List of features to use for prediction
    - target      (String)   : Name of the target to classify
    - output_dir  (String)   : Output directory to write results to
  '''
  print('Encoding categorical variables...')
  tree_df = df.copy()
  if target == 'gender':
    tree_df = tree_df.loc[tree_df['gender']!='Other']
  le = preprocessing.LabelEncoder()
  for col in ['gender','day_of_week','user_type']:
    le.fit(df[col])
    tree_df[col] = le.transform(tree_df[col])
  print('Majority to minority ratio:{:.4f}'.format((tree_df[target].value_counts().tolist()[0])/(tree_df.shape[0])))
  #Use the opposite of the provided target as an additional feature
  if target == 'user_type':
    features.append('gender')
    output_dir = output_dir + 'UserTypeClassification/'
  else:
    features.append('user_type')
    output_dir = output_dir + 'GenderClassification/'

  X_train,X_test,y_train,y_test = train_test_split(
    tree_df[features],
    tree_df[target],
    test_size=.3,
    random_state=24)

  print('Fitting unbalanced tree...')
  clf = fit_tree(X_train,X_test,y_train,y_test,output_dir,filename='Unbalanced',show_confusion=False)
  preds = clf.predict(X_test)
  print("Unbalanced test accuracy:{:.4f}".format(clf.score(X_test,y_test)))
  print("Unbalanced precision:{:.4f}".format(precision_score(preds,y_test)))
  print("Unbalanced recall:{:.4f}".format(recall_score(preds,y_test)))

  df_train = pd.concat([X_train,y_train],axis=1)
  X_train,X_val,y_train,y_val = downsample(df_train,features,target)
  print('Fitting downsampled tree...')
  clf = fit_tree(X_train,X_val,y_train,y_val,output_dir,filename='Downsampled',show_confusion=False)
  preds = clf.predict(X_test)
  print("Downsampling test accuracy:{:.4f}".format(clf.score(X_test,y_test)))
  print("Downsampling precision:{:.4f}".format(precision_score(preds,y_test)))
  print("Downsampling recall:{:.4f}".format(recall_score(preds,y_test)))

  X_train,X_val,y_train,y_val = upsample(df_train,features,target)
  print('Fitting upsampled tree...')
  clf = fit_tree(X_train,X_test,y_train,y_test,output_dir,filename='Upsampled',show_confusion=False)
  print("Upsampling test accuracy:{:.4f}".format(clf.score(X_test,y_test)))
  print("Upsampling precision:{:.4f}".format(precision_score(preds,y_test)))
  print("Upsampling recall:{:.4f}".format(recall_score(preds,y_test)))

def difference(df):
  diff = list()
  for i in range(1, len(df)):
    value = df[i] - df[i - 1]
    diff.append(value)
  return pd.Series(diff)

def arima_model(df,p,d,q,output_dir=None,save=False,verbose=False,baseline=False):
  '''Creates ARIMA model for data and predicts for half of the data
  Args:
    - df          (DataFrame): DataFrame with features for training
    - p           (Range)    : ARIMA autoregressive parameter
    - d           (Range)    : ARIMA degree of differencing parameter
    - q           (Range)    : ARIMA order of MA parameter
    - output_dir  (String)   : Output directory to save graphs to
    - plot        (Boolean)  : Shows plots if true
    - verbose     (Boolean)  : Prints each line of prediction if true
    - baseline    (Boolean)  : Predicts baseline with lag 1
  '''
  station = str(df['start_station_name'].unique()[0])
  df = pd.Series(df['rides'])
  df = df.astype('float32')
  train_size = len(df)//2
  train, test = df[0:train_size], df[train_size:]
  plt.plot(train)
  plt.plot(test)
  plt.xlabel('Day')
  plt.ylabel('Number of Rides')
  plt.title(' '.join(['Ridership at',station]))
  plt.tight_layout()
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + 'Actual')
  plt.close()

  test.reset_index(drop=True)
  #walk-forward validation
  history = list(train)
  predictions = list()
  for i in range(len(test)):
    # predict
    if baseline:
      pred = history[-1]
    else:
      model = ARIMA(history, order=(p,d,q))
      model_fit = model.fit(disp=0)
      pred = model_fit.forecast()[0][0]
    predictions.append(pred)
    # observation
    act = test.iloc[i]
    history.append(act)
    if verbose:
      print('>Predicted={:.3f}, Expected={:.3f}'.format(pred,act))

  #print performance
  rmse = sqrt(mean_squared_error(test, predictions))
  if verbose:
    print('RMSE: {:.3f}'.format(rmse))

  #check if stationary
  if baseline:
    X = pd.Series(df)
    # difference data
    stationary = difference(X)
    stationary.index = X.index[1:]
    result = adfuller(stationary)
    print('ADF Statistic: {:f}'.format(result[0]))
    print('p-value: {:f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
      print('\t{:s}: {:.3f}'.format(key,value))

    #Plot stationarity
    plt.plot(stationary)
    if save:
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      plt.savefig(output_dir + 'Stationarity')
    plt.close()

    plt.figure(figsize = (15,9))
    plt.subplot(211)
    plot_acf(X, ax=plt.gca())
    plt.subplot(212)
    plot_pacf(X, ax=plt.gca())

    if save:
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      plt.savefig(output_dir + 'ACF-PACF')
    plt.close()

  #plot differenced data
  predictions = pd.Series(predictions)
  predictions.index = test.index
  title = '-'.join(['Forecasted vs. Actual Number of Rides',station])
  title = ','.join([title,''.join(['RMSE=',str(rmse)])])
  test.reset_index(drop=True)
  plt.figure(figsize = (15,9))
  plt.plot(train,color='blue')
  plt.plot(predictions,color='orange',label='predicted')
  plt.plot(test,color='green',label='observed')
  plt.xlabel('Day')
  plt.ylabel('Number of Rides')
  plt.title(title)
  plt.legend(loc=4)
  plt.tight_layout()

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + 'Forecasted')
  plt.close()

  return rmse

def grid_search_rmse(df,p_vals,d_vals,q_vals):
  '''Performs gridsearch within defined range to find best
  parameters for ARIMA on the data in df
  Args:
    - df          (DataFrame): DataFrame with features for training
    - p_vals      (Range)    : Range of values to test for ARIMA p parameter
    - d_vals      (Range)    : Range of values to test for ARIMA d parameter
    - q_test      (Range)    : Range of values to test for ARIMA q parameter
  '''
  rmse_list = []
  rmse_lowest = float('inf')
  params = ()
  for p in p_vals:
    for d in d_vals:
      for q in q_vals:
        print('Testing: p={:d}, d={:d}, q={:d}'.format(p,d,q))
        try:
          rmse = arima_model(df,p,d,q)
        except:
          print('Model does not converge.')
          continue
        if rmse < rmse_lowest:
          rmse_lowest = rmse
          print('RMSE={:.3f}, new low'.format(rmse))
          params = (p,d,q)
        else:
          print('RMSE={:.3f}'.format(rmse))
        rmse_list.append(rmse)
  print('Best ARIMA params: p={:d}, d={:d}, q={:d}'.format(params[0],params[1],params[2]))
  return params[0],params[1],params[2]

def eval_arima(df,station,output_dir):
  '''Performs gridsearch within defined range to find best
  parameters for ARIMA on the data in df
  Args:
    - df          (DataFrame): DataFrame all data
    - station     (String)   : Name of the station to perform time series analysis on
    - output_dir  (Strin)    : Output to write ARIMA results to
  '''

  arima_dir = output_dir + 'ARIMA/'
  df['date'] = df['start_time'].dt.day
  df.sort_values(['year','month_num','date'],inplace=True)
  station_df = df[df['start_station_name']==station]
  station_df = station_df.groupby(['start_station_name','year','month_num','date'])['bike_id'].count().reset_index(name='rides')
  station_df['date'] = station_df['year'].map(str)+'-'+station_df['month_num'].map(str)+'-'+station_df['date'].map(str)

  p_vals = range(0,8)
  d_vals = range(0,3)
  q_vals = range(0,2)
  arima_dir += station + '/'
  (p,d,q) = grid_search_rmse(station_df,p_vals,d_vals,q_vals)
  arima_model(station_df,p,d,q,arima_dir,save=True,verbose=True)

if __name__ == '__main__':
  df = analysis.pickler()

  print('Creating age column...')
  df = df.dropna(subset=['birth_year','gender','start_station_name','end_station_name'])
  df['age'] = df['birth_year'].apply(lambda x:pd.Timestamp.now().year-x)
  df['age'] = df['age'].map(int)
  df = df.loc[(df['region']=='San Francisco') &
    (df['age'] <= 70) &
    (df['duration'] <= 2000)]

  # Tree based classification of gender or user type
  tree_classifier(df,['duration','start_hour','day_of_week','age'],'gender',output_dir)

  STATION_NAME = 'San Francisco Ferry Building'
  #Perform time series analysis and ARIMA on given station
  eval_arima(df,STATION_NAME,output_dir)

