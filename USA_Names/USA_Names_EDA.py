import pandas as pd
import numpy as np
from google.cloud import bigquery
import bq_helper
import matplotlib.pyplot as plt
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/jason.wang/Documents/Analytics Projects/EDA-Google-PK.json"

names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
print('pre query')
agg_names = names.query_to_pandas_safe(query)
print('Starting...')
# agg_names.to_csv("names.csv")
print('Done!')