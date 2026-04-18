# Databricks notebook source
spark.sql("USE workspace.default")

# COMMAND ----------

df = spark.sql("SELECT * FROM upi_transactions_2024")

# COMMAND ----------

df = spark.table("upi_transactions_2024")
df.show(5)

# COMMAND ----------

df = df.select(
    "amount (INR)",
    "hour_of_day",
    "transaction type",
    "merchant_category",
    "device_type",
    "network_type",
    "fraud_flag"
)

# COMMAND ----------

from pyspark.sql.functions import col

df = df.withColumnRenamed("amount (INR)", "amount")
df = df.withColumnRenamed("transaction type", "transaction_type")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexers = [
    StringIndexer(inputCol="transaction_type", outputCol="transaction_type_index"),
    StringIndexer(inputCol="merchant_category", outputCol="merchant_category_index"),
    StringIndexer(inputCol="device_type", outputCol="device_type_index"),
    StringIndexer(inputCol="network_type", outputCol="network_type_index"),
]

# COMMAND ----------

for indexer in indexers:
    df = indexer.fit(df).transform(df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "amount",
        "hour_of_day",
        "transaction_type_index",
        "merchant_category_index",
        "device_type_index",
        "network_type_index"
    ],
    outputCol="features"
)

df = assembler.transform(df)

# COMMAND ----------

train, test = df.randomSplit([0.8, 0.2])

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="fraud_flag", featuresCol="features")
model = lr.fit(train)

# COMMAND ----------

predictions = model.transform(test)

predictions.select("amount", "prediction", "fraud_flag").show(50)

# COMMAND ----------

predictions = model.transform(test)

predictions.select("amount", "prediction", "fraud_flag").show(10)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="fraud_flag")
print("Accuracy:", evaluator.evaluate(predictions))

# COMMAND ----------

