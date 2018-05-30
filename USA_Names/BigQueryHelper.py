from google.cloud import bigquery
import bq_helper
import os

#replace with path to JSON file containing Google Service Account private key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/jason.wang/Documents/Analytics Projects/EDA-Google-PK.json"

names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = names.query_to_pandas_safe(query)
agg_names.to_csv("names.csv")