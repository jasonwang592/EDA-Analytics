import numpy as np
import pickle
import pandas as pd
import analysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import sys

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


if __name__ == '__main__':
  df = analysis.pickler()

  print('Creating age column...')
  df = df.dropna(subset=['birth_year','gender','start_station_name','end_station_name'])
  df['age'] = df['birth_year'].apply(lambda x:pd.Timestamp.now().year-x)
  df['age'] = df['age'].map(int)
  df = df.loc[(df['region']=='San Francisco') &
    (df['age'] <= 70) &
    (df['duration'] <= 2000)]

  print('Encoding categorical variables...')
  le = preprocessing.LabelEncoder()
  for col in ['gender','day_of_week']:
    le.fit(df[col])
    df[col] = le.transform(df[col])

  #target variable should either be user_type or gender
  features = ['duration','start_hour','day_of_week']
  target = 'user_type'

  if target == 'user_type':
    features.append('gender')
  else:
    features.append('user_type')

  X_train,X_test,y_train,y_test = train_test_split(
    df[features],
    df[target],
    test_size=.3,
    random_state=21)

  df_train = pd.concat([X_train,y_train],axis=1)
  X_train,X_val,y_train,y_val = downsample(df_train,features,target)
  clf = fit_tree(X_train,X_val,y_train,y_val,output_dir)
  print("Downsampling test accuracy: " + str(clf.score(X_test,y_test)))

  X_train,X_val,y_train,y_val = upsample(df_train,features,target)
  clf = fit_tree(X_train,X_test,y_train,y_test,output_dir)
  print("Upsampling test accuracy: " + str(clf.score(X_test,y_test)))