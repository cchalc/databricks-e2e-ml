# Databricks notebook source
# MAGIC %md ---
# MAGIC title: End-to-End MLOps demo with MLFlow, Feature Store and Auto ML, part 1 - Feature Engineering
# MAGIC authors:
# MAGIC - Rafi Kurlansik
# MAGIC tags:
# MAGIC - python
# MAGIC - feature-engineering
# MAGIC - koalas
# MAGIC - feature-store
# MAGIC created_at: 2021-05-01
# MAGIC updated_at: 2021-05-01
# MAGIC tldr: End-to-end demo of Databricks for MLOps, including MLflow, the registry, webhooks, scoring, feature store and auto ML. Part 0 - feature engineering, with koalas, and creating feature store tables
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/10166861](https://demo.cloud.databricks.com/#notebook/10166861)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# Read into Spark
telcoDF = spark.table("ibm_telco_churn.bronze_customers")

display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Using `koalas` to scale my teammates' `pandas` code.

# COMMAND ----------

from databricks.feature_store import feature_table
import databricks.koalas as ks

def compute_churn_features(data):
  
  # Convert to koalas
  data = data.to_koalas()
  
  # OHE
  data = ks.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],dtype = 'int64')
  
  # Convert label to int and rename column
  data['churnString'] = data['churnString'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churnString': 'int32'})
  data = data.rename(columns = {'churnString': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compute and write features

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

churn_features_df = compute_churn_features(telcoDF)

churn_feature_table = fs.create_feature_table(
  name='ibm_telco_churn.churn_features',
  keys='customerID',
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the ibm_telco_churn.bronze_customers table in the lakehouse.  I created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=churn_features_df.to_spark(), name='ibm_telco_churn.churn_features', mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC As an alternative we could always write to Delta Lake:

# COMMAND ----------

# # Write out silver-level data to Delta lake
# trainingDF = spark.createDataFrame(training_df)

# trainingDF.write.format('delta').mode('overwrite').save(silver_tbl_path)

# # Create silver table
# spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,silver_tbl_name,silver_tbl_path))

# # Drop customer ID for AutoML
# automlDF = trainingDF.drop('customerID')

# # Write out silver-level data to Delta lake
# automlDF.write.format('delta').mode('overwrite').save(automl_tbl_path)

# # Create silver table
# _ = spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,automl_tbl_name,automl_tbl_path))

# COMMAND ----------


