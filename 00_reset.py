# Databricks notebook source
# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Set config for database name, file paths, and table names
database_name = 'cchalc_e2eml'

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

# # clean up feature store
# from databricks.feature_store import FeatureStoreClient

# fs = FeatureStoreClient()
# fs._catalog_client.delete_feature_table(f"{database_name}.churn_features")

# COMMAND ----------

# Delete the old database and tables if needed
# uncomment if you need a reset
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
# _ = spark.sql('CREATE DATABASE {}'.format(database_name))
# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+telco_preds_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS cchalc_e2eml
# MAGIC     COMMENT "CREATE A DATABASE WITH A LOCATION PATH"
# MAGIC     LOCATION "/Users/christopher.chalcraft@databricks.com/databases/cchalc_e2eml" --this must be a location on dbfs (i.e. not direct access)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE cchalc_e2eml
