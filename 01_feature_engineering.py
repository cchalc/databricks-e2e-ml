# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true">

# COMMAND ----------

# MAGIC %run ./00_includes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# Read into Spark
telcoDF = spark.table(f"{database_name}.bronze_customers")

display(telcoDF)

# COMMAND ----------

# import pyspark.pandas as ps
# tf = telcoDF.to_pandas_on_spark()
# dummy = ps.get_dummies(tf,
#                        columns=['gender', 'partner', 'dependents',
#                                 'phoneService', 'multipleLines', 'internetService',
#                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
#                                 'techSupport', 'streamingTV', 'streamingMovies',
#                                 'contract', 'paperlessBilling', 'paymentMethod'],
#                        dtype = 'int64'
#                       )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Pandas API on Spark

# COMMAND ----------

from databricks.feature_store import feature_table
import pyspark.pandas as ps

def compute_churn_features(data):
  
  # Convert to pandas
  data = data.to_pandas_on_spark()
  
  # OHE
  data = ps.get_dummies(data, 
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
  data.columns = data.columns.str.replace(' ', '', regex=True)
  data.columns = data.columns.str.replace('(', '-', regex=True)
  data.columns = data.columns.str.replace(')', '', regex=True)
  
  # Drop missing values
  data = data.dropna()
  data = data.to_spark()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compute and write features
# MAGIC - [Feature Store Python API Reference](https://docs.databricks.com/dev-tools/api/python/latest/index.html)
# MAGIC - [Work with Feature Store Tables](https://docs.databricks.com/applications/machine-learning/feature-store/feature-tables.html#register-an-existing-delta-table-as-a-feature-table)

# COMMAND ----------

# # clean up feature store
# from databricks.feature_store import FeatureStoreClient

# fs = FeatureStoreClient()
# fs._catalog_client.delete_feature_table(f"{database_name}.churn_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC --drop table churn_features

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
# fs._catalog_client.delete_feature_table(f"{database_name}.churn_features")

churn_features_df = compute_churn_features(telcoDF)

churn_feature_table = fs.create_table(
  name=f'{database_name}.churn_features',
  primary_keys=['customerID'],
  df=churn_features_df,
  description='These features are derived from the ibm_telco_churn.bronze_customers table in the lakehouse.  I created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(
  name=f'{database_name}.churn_features',
  df=churn_features_df,
  mode='overwrite'
)

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


