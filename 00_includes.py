# Databricks notebook source
# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Set config for database name, file paths, and table names
database_name = 'cchalc_e2eml'
# model_name = 'cchalc_e2eml_churn'

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/ibm-telco-churn/Telco-Customer-Churn.csv'.format(user)
# dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# Paths for various Delta tables
bronze_tbl_path = '/home/{}/ibm-telco-churn/bronze/'.format(user)
silver_tbl_path = '/home/{}/ibm-telco-churn/silver/'.format(user)
automl_tbl_path = '/home/{}/ibm-telco-churn/automl-silver/'.format(user)
telco_preds_path = '/home/{}/ibm-telco-churn/preds/'.format(user)

bronze_tbl_name = 'bronze_customers'
silver_tbl_name = 'silver_customers'
automl_tbl_name = 'gold_customers'
telco_preds_tbl_name = 'telco_preds'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS cchalc_e2eml
# MAGIC     COMMENT "CREATE A DATABASE WITH A LOCATION PATH"
# MAGIC     LOCATION "/Users/christopher.chalcraft@databricks.com/databases/cchalc_e2eml" --this must be a location on dbfs (i.e. not direct access)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE cchalc_e2eml

# COMMAND ----------

slack_webhook = "https://hooks.slack.com/services/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"
