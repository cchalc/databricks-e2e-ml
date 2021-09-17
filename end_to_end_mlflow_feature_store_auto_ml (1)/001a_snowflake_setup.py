# Databricks notebook source
# MAGIC %md
# MAGIC # Set up data and configure paths

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# Set config for database name, file paths, and table names
database_name = 'cchalc'
spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE;")
spark.sql(f"CREATE DATABASE {database_name} COMMENT 'This database is used to maintain Inventory';")
spark.sql(f"USE {database_name}")

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/ibm-telco-churn/Telco-Customer-Churn.csv'.format(user)
dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# Paths for various Delta tables
bronze_tbl_path = '/home/{}/ibm-telco-churn/bronze/'.format(user)
silver_tbl_path = '/home/{}/ibm-telco-churn/silver/'.format(user)
automl_tbl_path = '/home/{}/ibm-telco-churn/automl-silver/'.format(user)
telco_preds_path = '/home/{}/ibm-telco-churn/preds/'.format(user)

bronze_tbl_name = 'bronze_customers'
silver_tbl_name = 'silver_customers'
automl_tbl_name = 'gold_customers'
telco_preds_tbl_name = 'telco_preds'

# # Delete the old database and tables if needed
# _ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# # Create database to house tables
# _ = spark.sql('CREATE DATABASE {}'.format(database_name))
# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)

shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+telco_preds_path, ignore_errors=True)

# COMMAND ----------

dbutils.fs.ls(driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Data

# COMMAND ----------

df = (spark.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("/home/christopher.chalcraft@databricks.com/ibm-telco-churn/Telco-Customer-Churn.csv")
     )
      

# COMMAND ----------

display(df)

# COMMAND ----------

# skip snowflake and write directly to bronze
df.createOrReplaceTempView(bronze_tbl_name)
(spark.sql("select * from bronze_customers")
 .write
 .format("delta")
 .save(bronze_tbl_path)
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Snowflake configuration

# COMMAND ----------

# write to snowflake

