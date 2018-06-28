import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotting
import sys

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

set_size = 50
agg_gender = df.groupby('gender').count()
name_gender = df.groupby(['name', 'gender']).sum()
name_gender = name_gender.reset_index()
top_male_df = name_gender[name_gender['gender'] == 'Male'].sort_values('number', ascending = False).head(set_size)
top_female_df = name_gender[name_gender['gender'] == 'Female'].sort_values('number', ascending = False).head(set_size)

# Return the top set_size male and female names
top_male_names = top_male_df['name']
top_female_names = top_female_df['name']

#Find average age of the most common n names for each gender
common_male = df[(df['name'].isin(top_male_names)) & (df['gender'] == 'Male')]
common_female = df[(df['name'].isin(top_female_names)) & (df['gender'] == 'Male')]
wavg_male = pd.DataFrame(common_male.groupby('name').apply(wavg, 'age', 'number').reset_index())
wavg_female = pd.DataFrame(common_female.groupby('name').apply(wavg, 'age', 'number').reset_index())
wavg_male.columns = ['name', 'averageAge']
wavg_female.columns = ['name', 'averageAge']

wavg_male.sort_values('averageAge', inplace = True, ascending = False)
wavg_female.sort_values('averageAge', inplace = True, ascending = False)
plotting.bars(wavg_male, 'Male', 'output/')
plotting.bars(wavg_female,'Female', 'output/')

#Get the count of people for the top set_size names and gender within a year
name_gender_age = df.groupby(['name', 'gender', 'year']).sum()
name_gender_age = name_gender_age.reset_index()
top_male_age = name_gender_age[(name_gender_age['name'].isin(top_male_names)) & (name_gender_age['gender'] == 'Male')]
top_female_age = name_gender_age[(name_gender_age['name'].isin(top_female_names)) & (name_gender_age['gender'] == 'Female')]

#Group the top set_size names by decade
top_male_decade = top_male_age.groupby(['decade','name']).sum()
top_female_decade = top_female_age.groupby(['decade','name']).sum()
top_male_decade = top_male_decade.reset_index()
top_female_decade = top_female_decade.reset_index()
plotting.heatmapper(top_male_decade, 'Male', 'output/', save = True)
plotting.heatmapper(top_female_decade, 'Female', 'output/', save = True)

#Find the most gender neutral names based on a percentage split threshold and examine their trends
n = 20
l_threshold = 35
r_threshold = 65
male_names = set(df[df['gender'] == 'Male']['name'])
female_names = set(df[df['gender'] == 'Female']['name'])
common_names = list(male_names & female_names)
common_df = df[df['name'].isin(common_names)]
common_group = common_df.groupby(['name','gender']).agg({'number': 'sum'})
gender_group = common_group.groupby(level = 0).apply(lambda x: 100 * x/float(x.sum()))
top_n_common_names = common_df.groupby('name').sum().sort_values('number', ascending = False).reset_index()
gender_group = gender_group[(gender_group['number'] >= l_threshold) & (gender_group['number'] <= r_threshold)].reset_index()

top_n_shared_names = 10
rank = 1
top_shared_names = []
for name in top_n_common_names['name']:
  if name in gender_group['name'].unique():
    plotting.namePopularity(df, name, 'output/', rank)
    rank += 1
    top_n_shared_names -= 1
  if top_n_shared_names == 0:
    break

