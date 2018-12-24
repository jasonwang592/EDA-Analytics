import pandas as pd
import numpy as np

df = pd.read_csv('files/BlackFriday.csv')
print(df.head())

print(df.columns)
df.drop('User_ID',axis=1,inplace=True)
