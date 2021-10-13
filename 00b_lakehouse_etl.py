# Databricks notebook source
# MAGIC %md
# MAGIC ### Setup
# MAGIC 
# MAGIC In this case we'll grab a CSV from the web, but we could also use Python or Spark to read data from databases or cloud storage.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load into Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC #### Path configs

# COMMAND ----------

# MAGIC %run ./00_reset

# COMMAND ----------

# MAGIC %fs ls /home/christopher.chalcraft@databricks.com/ibm-telco-churn/

# COMMAND ----------

# copy the data from driver to DBFS
# dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read and display

# COMMAND ----------

# Define schema
schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# Read CSV, write to Delta and take a look
bronze_df = spark.read.format('csv').schema(schema).option('header','true')\
               .load(driver_to_dbfs_path)

bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

display(bronze_df)

# COMMAND ----------

dbutils.fs.ls(bronze_tbl_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create bronze

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_name,bronze_tbl_path))

# COMMAND ----------


