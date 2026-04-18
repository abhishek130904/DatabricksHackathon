# Databricks notebook source
# MAGIC %pip install sentence-transformers faiss-cpu

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# COMMAND ----------

from pyspark.sql import SparkSession

# Recreate Spark session if lost
spark = SparkSession.builder.getOrCreate()

df = spark.table("pcm_dataset_2")
df.show(5)

# COMMAND ----------

dbutils.widgets.text("answer", "", "Enter your answer")

# COMMAND ----------

user_answer = dbutils.widgets.get("answer")