# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Batch Inference
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step6.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC 
# MAGIC Loading as a Spark UDF to set us up for future scale.

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/hhar_churn/staging") # may need to replace with your own model name

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Features

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
features = fs.read_table('ibm_telco_churn.churn_features')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference

# COMMAND ----------

predictions = features.withColumn('predictions', model(*features.columns))
display(predictions.select("customerId", "predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to Delta Lake

# COMMAND ----------

predictions.write.format("delta").mode("append").saveAsTable("ibm_telco_churn.churn_preds")
