import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotting

def wavg(df, value, weights):
  v = df[value]
  w = df[weights]

  try:
    return (v*w).sum() / w.sum()
  except ZeroDivisionError:
    print('Weights are zero.')


year = datetime.datetime.now().year
df = pd.read_csv('names.csv')
df.drop('Unnamed: 0', inplace = True, axis = 1)
df['age'] = year - df['year']
df['decade'] = 1900 + (df['year'] - (df['year']//100 * 100))//10 * 10
gender_map = {'M': 'Male', 'F': 'Female'}
df['gender'] = df['gender'].map(gender_map)

set_size = 20
agg_gender = df.groupby('gender').count()
name_gender = df.groupby(['name', 'gender']).sum()
name_gender = name_gender.reset_index()
top_male_df = name_gender[name_gender['gender'] == 'Male'].sort_values('number', ascending = False).head(set_size)
top_female_df = name_gender[name_gender['gender'] == 'Female'].sort_values('number', ascending = False).head(set_size)

top_male_names = top_male_df['name']
top_female_names = top_female_df['name']

name_gender_age = df.groupby(['name', 'gender', 'year']).sum()
name_gender_age = name_gender_age.reset_index()
top_male_age = name_gender_age[(name_gender_age['name'].isin(top_male_names)) & (name_gender_age['gender'] == 'Male')]
top_female_age = name_gender_age[(name_gender_age['name'].isin(top_female_names)) & (name_gender_age['gender'] == 'Female')]

top_male_decade = top_male_age.groupby(['decade','name']).sum()
top_female_decade = top_female_age.groupby(['decade','name']).sum()
top_male_decade = top_male_decade.reset_index()
top_female_decade = top_female_decade.reset_index()
#plotting.heatmapper(top_male_decade, '/output/', save = False)
plotting.namePopularity(df, 'asda', 'output/', save = True)
