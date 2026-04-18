# Databricks notebook source
# MAGIC   %pip install faiss-cpu sentence-transformers langchain pypdf

# COMMAND ----------

from pypdf import PdfReader

pdf_path = "/Volumes/workspace/default/pdf_files/ncert-class-12-chemistry-the-p-block-elements.pdf"
reader = PdfReader(pdf_path)

text_data = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages)]

df = spark.createDataFrame(text_data, ["page", "text"])

# ── Chunking ──────────────────────────────────────────────────────────────────
from pyspark.sql.functions import explode, col, expr
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf

# FIX 1: no default arg in UDF body — avoids serialization issues
def chunk_text(text):
    chunk_size = 500
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunk_udf = udf(chunk_text, ArrayType(StringType()))

chunked_df = df.withColumn("chunks", chunk_udf(df["text"])) \
               .select("page", explode("chunks").alias("chunk"))

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
pdf_df = chunked_df.toPandas()

embeddings = model.encode(
    pdf_df["chunk"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)
pdf_df["embedding"] = embeddings.tolist()


# COMMAND ----------

# DBTITLE 1,Save to Delta table
spark_df = spark.createDataFrame(pdf_df)

spark_df = spark_df \
    .withColumn("embedding", col("embedding").cast("array<float>")) \
    .withColumn("id", expr("uuid()"))

spark_df.write \
    .option("delta.enableChangeDataFeed", "true") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.pdf_embeddings2")

# COMMAND ----------

spark.table("workspace.default.pdf_embeddings2").show(5)

# COMMAND ----------

spark.table("workspace.default.pdf_embeddings2").printSchema()

# COMMAND ----------

spark.sql("DESCRIBE TABLE workspace.default.pdf_embeddings2").show(truncate=False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

endpoint_name = "vs_endpoint"
index_name    = "workspace.default.pdf_embeddings_index2"
source_table  = "workspace.default.pdf_embeddings2"

# FIX 2: create index if it doesn't exist yet
try:
    index = vsc.get_index(endpoint_name, index_name)
    print("Index already exists, reusing.")
except Exception:
    print("Index not found — creating...")
    index = vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",          # or "CONTINUOUS"
        primary_key="id",
        embedding_dimension=384,            # bge-small-en-v1.5 output dim
        embedding_vector_column="embedding"
    )

# FIX 3: correct polling — detailed_state is the string, not a bool "ready" key
READY_STATES = {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}

while True:
    desc          = index.describe()
    detailed_state = desc.get("status", {}).get("detailed_state", "UNKNOWN")
    print("Status:", detailed_state)

    if detailed_state in READY_STATES:
        break

    time.sleep(10)

print("✅ Index READY — you can query now")

# COMMAND ----------

print(index.describe())

# COMMAND ----------

index = vsc.get_index(endpoint_name, index_name)
index.describe()

# COMMAND ----------

spark.table("workspace.default.pdf_embeddings").count()

# COMMAND ----------

spark.table("workspace.default.pdf_embeddings").printSchema()

# COMMAND ----------

import time

while True:
    status = index.describe()
    
    print(status["status"]["detailed_state"], "| Ready:", status["status"]["ready"])
    
    if status["status"]["ready"]:
        print("✅ INDEX READY")
        break
    
    time.sleep(10)

# COMMAND ----------

# DBTITLE 1,Cell 18
# Check if index is ready before syncing
status = index.describe()

if status["status"]["ready"]:
    index.sync()
    print("✅ Index synced successfully")
else:
    print(f"⚠️ Index is not ready yet. Current state: {status['status']['detailed_state']}")
    print(f"Message: {status['status']['message']}")
    print("\nPlease wait for the index to be ready before syncing.")

# COMMAND ----------

# DBTITLE 1,RAG Query Function
def query_vector_index(query_text, num_results=3):
    """Query the vector index and return relevant chunks."""
    # Generate embedding using the SAME model as Cell 3
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    
    # Search the vector index
    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["page", "chunk"],
        num_results=num_results
    )
    
    return results

# COMMAND ----------

# DBTITLE 1,RAG with LLM Response
def rag_query(question, num_chunks=3):
    from openai import OpenAI

    # Use dbutils to get token (same as Cell 20)
    client = OpenAI(
        api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
        base_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
    )

    # Step 1: Retrieve
    search_results = query_vector_index(question, num_results=num_chunks)

    if 'result' in search_results and 'data_array' in search_results['result']:
        contexts = [row[1] for row in search_results['result']['data_array']]
        pages = [row[0] for row in search_results['result']['data_array']]
    else:
        return {"answer": "No context found", "sources": []}

    # Step 2: Build context
    context_str = "\n\n".join([f"[Page {p}] {c}" for p, c in zip(pages, contexts)])

    # Step 3: Prompt
    prompt = f"""
Answer ONLY using the context.

Context:
{context_str}

Question:
{question}

Answer:
"""

    try:
        # Step 4: LLM call
        response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"LLM Error: {str(e)}\n\nContext:\n{context_str}"

    return {
        "answer": answer,
        "sources": [{"page": p, "text": c[:200] + "..."} for p, c in zip(pages, contexts)]
    }

# COMMAND ----------

# DBTITLE 1,Test RAG System
# Test the RAG system
question = "Explain the anomalous behavior of nitrogen compared to other group 15 elements"

result = rag_query(question)

print("ANSWER:")
print(result["answer"])
print("\n" + "="*80)
print("\nSOURCES:")
for i, source in enumerate(result["sources"], 1):
    print(f"\n{i}. Page {source['page']}:")
    print(f"   {source['text']}")

# COMMAND ----------

user_profiles = [
    "visual_impairment",
    "hearing_impairment",
    "dyslexia",
    "adhd",
    "low_connectivity",
    "default"
]

# COMMAND ----------

from openai import OpenAI
import os

# Initialize OpenAI client for Databricks Foundation Models
client = OpenAI(
    api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
    base_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
)

# COMMAND ----------

def retrieve_context(query, num_results=3):
    """Retrieve relevant context from vector index."""
    # Use your existing query function
    search_results = query_vector_index(query, num_results=num_results)
    
    # Extract chunks and pages
    if 'result' in search_results and 'data_array' in search_results['result']:
        contexts = [row[1] for row in search_results['result']['data_array']]
        pages = [row[0] for row in search_results['result']['data_array']]
        
        # Format context
        context_str = "\n\n".join([f"[Page {p}] {c}" for p, c in zip(pages, contexts)])
        sources = [{"page": p, "text": c} for p, c in zip(pages, contexts)]
        
        return context_str, sources
    else:
        return "No context found.", []

# COMMAND ----------

def detect_user_profile(query):
    prompt = f"""
Classify the user need.

Categories:
- visual_impairment
- hearing_impairment
- dyslexia
- adhd
- low_connectivity
- default

Query: {query}

Return ONLY one category.
"""

    response = client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# COMMAND ----------

def get_style(profile):
    styles = {
        "visual_impairment": "Use highly descriptive explanations. Avoid references to visuals.",
        
        "hearing_impairment": "Use clear written explanations. No reliance on audio.",
        
        "dyslexia": "Use short sentences, simple words, and bullet points.",
        
        "adhd": "Break answer into small steps. Keep it engaging and concise.",
        
        "low_connectivity": "Give a short, essential answer only.",
        
        "default": "Answer normally."
    }
    
    return styles.get(profile, styles["default"])

# COMMAND ----------

def adaptive_rag(query):
    
    profile = detect_user_profile(query)
    
    context, sources = retrieve_context(query)
    
    style = get_style(profile)
    
    prompt = f"""
You are an AI tutor.

Instructions:
- Answer ONLY from context
- Adapt style: {style}

Context:
{context}

Question:
{query}

Answer:
"""
    
    response = client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "profile": profile,
        "answer": response.choices[0].message.content
    }

# COMMAND ----------

# Test case for adaptive_rag function
test_query = "Explain the anomalous behavior of nitrogen compared to other group 15 elements for someone with visually impaired."

result = adaptive_rag(test_query)

print("Detected Profile:", result["profile"])
print("Adapted Answer:", result["answer"])

# COMMAND ----------

# MAGIC %pip install requests

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Cell 27
import requests
import base64
import os

# Try to get API key from secrets, fall back to hardcoded if not configured
try:
    SARVAM_API_KEY = dbutils.secrets.get(scope="my-secrets", key="sarvam-api-key")
    print("✅ Using API key from Databricks Secrets")
except:
    SARVAM_API_KEY = "sk_73frjkbk_h9u67wusG8JiAh2z4VBZ5EuH"
    print("⚠️ Using hardcoded API key. For better security, store in Databricks Secrets.")

def text_to_speech(text, filename="output.wav"):
    
    # Use /tmp directory which is always writable
    filepath = f"/tmp/{filename}"
    
    url = "https://api.sarvam.ai/text-to-speech"
    
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text,
        "voice": "anushka",
        "language": "en-IN"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        print(f"Response keys: {list(response_json.keys())}")
        
        # Try different possible keys
        if "audio" in response_json:
            audio_base64 = response_json["audio"]
        elif "audios" in response_json:
            audio_base64 = response_json["audios"][0]
        elif "data" in response_json:
            audio_base64 = response_json["data"]
        else:
            print(f"❌ Unexpected response format: {response_json}")
            return None
        
        audio_bytes = base64.b64decode(audio_base64)
        
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio saved as {filepath}")
        return filepath
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None

# COMMAND ----------

def rag_with_audio(query):
    
    result = rag_query(query)
    answer = result["answer"]
    
    # Convert answer → speech
    audio_file = text_to_speech(answer)
    
    return {
        "answer": answer,
        "audio": audio_file,
        "sources": result["sources"]
    }

# COMMAND ----------

from IPython.display import Audio

result = rag_with_audio("Explain anomalous behavior of nitrogen")

display(Audio(result["audio"]))

# COMMAND ----------

