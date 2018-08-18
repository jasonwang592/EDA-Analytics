import numpy as np
import pickle
import pandas as pd
import analysis


if __name__ == '__main__':
  df = analysis.pickler()
  numerical_df = df.copy()
  numerical_df = numerical_df.dropna(subset=['birth_year','gender'])
  numerical_df['age'] = numerical_df['birth_year'].apply(lambda x:pd.Timestamp.now().year-x)
  numerical_df['age'] = numerical_df['age'].map(int)
