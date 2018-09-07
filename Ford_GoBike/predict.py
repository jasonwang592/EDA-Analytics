import numpy as np
import pickle
import pandas as pd
import analysis
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA


output_dir = 'output/prediction/'

def downsample(df,features,target):
  '''Takes an imbalanced DataFrame in terms of target variable and downsamples
  the majority label
  Args:
    - df      (DataFrame): The DataFrame with unbalanced target variables
    - features(list)     : List of features we want to keep
    - target  (String)   : The column name of the unbalance target variable
  '''
  over_count = df[target].value_counts()[0]
  under_count = df[target].value_counts()[1]
  over_label = df[target].value_counts().idxmax()
  under_label = df[target].value_counts().idxmin()

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

def fit_forest():

  clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)



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
  print('Fitting tree...')
  clf = tree.DecisionTreeClassifier()
  clf.fit(X=X_train,y=y_train)
  print('Validation accuracy: ' + str(clf.score(X=X_test,y=y_test)))

  y_pred = clf.predict(X_test)
  if show_confusion:
    conf_matrix = confusion_matrix(y_true=y_test,y_pred=y_pred)
    labels = ['Customer','Subscriber']
    fig = plt.figure()
    ax = plt.subplot(111)
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.tight_layout()
    plt.show()
  else:
    if not filename:
      raise ValueError("Provide filename argument when saving confusion matrix.")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + filename)
  plt.close()
  return clf

def tree_classifier(df):
  #Tree based models for classification of gender or user type
  #target variable should either be user_type or gender
  tree_df = df.copy()
  print('Encoding categorical variables...')
  le = preprocessing.LabelEncoder()
  for col in ['gender','day_of_week']:
    le.fit(df[col])
    tree_df[col] = le.transform(tree_df[col])

  features = ['duration','start_hour','day_of_week']
  target = 'user_type'

  if target == 'user_type':
    features.append('gender')
  else:
    features.append('user_type')

  X_train,X_test,y_train,y_test = train_test_split(
    tree_df[features],
    tree_df[target],
    test_size=.3,
    random_state=21)

  df_train = pd.concat([X_train,y_train],axis=1)
  X_train,X_val,y_train,y_val = downsample(df_train,features,target)
  clf = fit_tree(X_train,X_val,y_train,y_val,output_dir)
  print("Downsampling test accuracy: " + str(clf.score(X_test,y_test)))

  X_train,X_val,y_train,y_val = upsample(df_train,features,target)
  clf = fit_tree(X_train,X_test,y_train,y_test,output_dir)
  print("Upsampling test accuracy: " + str(clf.score(X_test,y_test)))

def difference(df):
  diff = list()
  for i in range(1, len(df)):
    value = df[i] - df[i - 1]
    diff.append(value)
  return pd.Series(diff)

def arima_model(df,p,d,q,plot=False,verbose=False):
  df = pd.Series(temp['rides'])
  df = df.astype('float32')
  train_size = len(df)//2
  train, test = df[0:train_size], df[train_size:]
  if plot:
    plt.plot(train)
    plt.plot(test)
    plt.xlabel('Day')
    plt.ylabel('Number of Rides')
    plt.title(' '.join(['Ridership at',station]))
    plt.tight_layout()
    plt.show()
    plt.close()
    test.reset_index(drop=True)

  #walk-forward validation
  history = list(train)
  predictions = list()
  baseline = False
  for i in range(len(test)):
    # predict
    if baseline:
      pred = history[-1]
    else:
      model = ARIMA(history, order=(p,d,q))
      model_fit = model.fit(disp=0)
      pred = model_fit.forecast()[0]
    predictions.append(pred)
    # observation
    act = test.iloc[i]
    history.append(act)
    # print('>Predicted={:.3f}, Expected={:3f}'.format(pred,act))

  #print performance
  rmse = sqrt(mean_squared_error(test, predictions))
  if verbose:
    print('RMSE: %.3f' % rmse)

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
    if plot:
      plt.plot(stationary)
      plt.figure(figsize = (15,9))
      plt.subplot(211)
      plot_acf(X, ax=plt.gca())
      plt.subplot(212)
      plot_pacf(X, ax=plt.gca())
      plt.show()
      plt.close()

  # plot differenced data
  if plot:
    predictions = pd.Series(predictions)
    predictions.index = test.index
    test.reset_index(drop=True)
    plt.figure(figsize = (15,9))
    plt.plot(train,color='blue')
    plt.plot(predictions,color='orange')
    plt.plot(test,color='green')
    plt.xlabel('Day')
    plt.ylabel('Number of Rides')
    plt.title('-'.join(['Forecasted vs. Actual Number of Rides',station]))
    plt.tight_layout()
    plt.show()
  return rmse

def grid_search_rmse(df,p_vals,d_vals,q_vals):
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

if __name__ == '__main__':
  df = analysis.pickler()

  print('Creating age column...')
  df = df.dropna(subset=['birth_year','gender','start_station_name','end_station_name'])
  df['age'] = df['birth_year'].apply(lambda x:pd.Timestamp.now().year-x)
  df['age'] = df['age'].map(int)
  df = df.loc[(df['region']=='San Francisco') &
    (df['age'] <= 70) &
    (df['duration'] <= 2000)]

  df['date'] = df['start_time'].dt.day
  df.sort_values(['year','month_num','date'],inplace=True)
  station = 'Powell St BART Station'
  temp =df[df['start_station_name']== station]
  temp = temp.groupby(['year','month_num','date'])['bike_id'].count().reset_index(name='rides')
  temp['date'] = temp['year'].map(str)+'-'+temp['month_num'].map(str)+'-'+temp['date'].map(str)

  p_vals = range(0,8)
  d_vals = range(0,3)
  q_vals = range(0,2)
  (p,d,q) = grid_search_rmse(df,p_vals,d_vals,q_vals)

  sys.exit()

  #Linear Model
  grouped_df = df.groupby(['start_hour','start_station_name','day_of_week'])['bike_id'].count().reset_index(name='rides')
  y = grouped_df['rides']
  temp = grouped_df[grouped_df['start_station_name']=='Powell St BART Station']
  sns.pairplot(temp)
  plt.show()
  X = grouped_df.drop('rides',axis=1)
  X = pd.get_dummies(X)

  X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=.3,
    random_state=21)

  lm = linear_model.LinearRegression()
  lm.fit(X,y)
  preds = lm.predict(X)
  print(preds[0:5],y[0:5])
  print(lm.score(X,y))
  sys.exit()

  # #GLM model for ride duration
  # features = ['start_hour','day_of_week','user_type','gender','age','start_station_name']
  # glm_df = df.copy()
  # print('Encoding categorical variables...')
  # le = preprocessing.LabelEncoder()
  # for col in ['day_of_week','user_type','gender','start_station_name']:
  #   le.fit(glm_df[col])
  #   glm_df[col] = le.transform(glm_df[col])
  # y = glm_df['duration']

  # print('One hot encoding variables...')
  # glm_df = glm_df[features]
  # one_hot_cols = ['start_station_name','gender','day_of_week','start_hour']
  # X = pd.get_dummies(glm_df,prefix=one_hot_cols,columns=one_hot_cols)

  # X_train,X_test,y_train,y_test = train_test_split(
  #   X,
  #   y,
  #   test_size=.3,
  #   random_state=21)
  # glm = linear_model.LinearRegression()
  # glm.fit(X,y)
  # preds = glm.predict(X)
  # print(preds[0:5],y[0:5])
  # print(glm.score(X,y))

