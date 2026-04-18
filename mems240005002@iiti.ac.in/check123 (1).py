# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install sentence-transformers databricks-vectorsearch -q

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install pycryptodome

# COMMAND ----------

# MAGIC %pip install PyPDF2
# MAGIC

# COMMAND ----------

# MAGIC %pip install pypdf

# COMMAND ----------

# =========================
# 0. INSTALL (RUN ONCE)
# =========================
# %pip install pypdf sentence-transformers
# dbutils.library.restartPython()

# =========================
# 1. INPUT MULTIPLE PDF FILES
# =========================
pdf_paths = [
    "/Volumes/workspace/default/pdf_files/ncert-class-12-chemistry-the-p-block-elements.pdf"
  
]

# =========================
# 2. READ PDF + EXTRACT TEXT
# =========================
from pypdf import PdfReader

all_text_data = []

for file_id, pdf_path in enumerate(pdf_paths):
    reader = PdfReader(pdf_path)
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        all_text_data.append((file_id, pdf_path, page_num, text))

# Create Spark DataFrame
df = spark.createDataFrame(
    all_text_data,
    ["file_id", "file_name", "page", "text"]
)

# =========================
# 3. CHUNK TEXT
# =========================
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import ArrayType, StringType

def chunk_text(text):
    chunk_size = 500
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunk_udf = udf(chunk_text, ArrayType(StringType()))

chunked_df = df.withColumn("chunks", chunk_udf(df["text"])) \
               .select("file_id", "file_name", "page", explode("chunks").alias("chunk"))

# =========================
# 4. GENERATE EMBEDDINGS
# =========================
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Convert to Pandas
pdf_df = chunked_df.toPandas()

# Generate embeddings
embeddings = model.encode(
    pdf_df["chunk"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

pdf_df["embedding"] = embeddings.tolist()

# Fix datatype
pdf_df["embedding"] = pdf_df["embedding"].apply(
    lambda x: [float(i) for i in x]
)

# Add unique ID
pdf_df["id"] = range(len(pdf_df))

# =========================
# 5. SAVE TO DELTA TABLE
# =========================
spark_df = spark.createDataFrame(pdf_df)

table_name = "workspace.default.multi_pdf_embeddings"

# Reset table
spark.sql(f"DROP TABLE IF EXISTS {table_name}")

spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(table_name)

print("✅ All PDF embeddings stored in Delta table")

# =========================
# 6. CREATE / USE VECTOR INDEX
# =========================
from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

endpoint_name = "vs_endpoint"
index_name    = "workspace.default.multi_pdf_index"
source_table  = table_name

try:
    index = vsc.get_index(endpoint_name, index_name)
    print("Index already exists, reusing.")
except Exception:
    print("Index not found — creating...")
    index = vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_dimension=384,
        embedding_vector_column="embedding"
    )

# Wait until index is ready
READY_STATES = {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}

while True:
    desc = index.describe()
    state = desc.get("status", {}).get("detailed_state", "UNKNOWN")
    print("Status:", state)

    if state in READY_STATES:
        break

    time.sleep(10)

print("✅ Vector Index READY — You can query now!")

# COMMAND ----------

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from databricks.vector_search.client import VectorSearchClient
import requests, base64

# ---- CONFIG ----
ENDPOINT_NAME = "vs_endpoint"
INDEX_NAME = "workspace.default.chemistry_pblock_index"

# ---- INIT ----
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = OpenAI(
    api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
    base_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
)

vsc = VectorSearchClient()
index = vsc.get_index(ENDPOINT_NAME, INDEX_NAME)

# ---- SARVAM (SECURE) ----
try:
    SARVAM_API_KEY = dbutils.secrets.get(
        scope="my-secrets",
        key="sarvam-api-key"
    )
    print("✅ Loaded Sarvam API key from Databricks Secrets")

except Exception as e:
    print("❌ Failed to load Sarvam API key:", str(e))
    SARVAM_API_KEY = None

# COMMAND ----------

def retrieve_context(query, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["page", "chunk"],
        num_results=k
    )

    if 'result' not in results:
        return "", []

    data = results['result']['data_array']

    contexts = [row[1] for row in data]
    pages = [row[0] for row in data]

    context_text = "\n\n".join([f"[Page {p}] {c}" for p, c in zip(pages, contexts)])

    sources = [{"page": p, "text": c[:200]} for p, c in zip(pages, contexts)]

    return context_text, sources

# COMMAND ----------

 def detect_profile(query):
    prompt = f"""
Classify the user need:

Options:
visual_impairment
hearing_impairment
dyslexia
adhd
low_connectivity
default

Query: {query}

Return ONLY one label.
"""

    res = client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()

# COMMAND ----------

def get_style(profile):
    styles = {
        "visual_impairment": "Explain clearly with rich descriptive narration.",
        "hearing_impairment": "Use clear structured text.",
        "dyslexia": "Use short sentences and bullet points.",
        "adhd": "Break into small engaging steps.",
        "low_connectivity": "Give very short answer.",
        "default": "Normal explanation."
    }
    return styles.get(profile, styles["default"])

# COMMAND ----------

def generate_answer(query, context, style):
    
    prompt = f"""
You are an AI tutor.

Rules:
- Answer ONLY from context
- Style: {style}

Context:
{context}

Question:
{query}

Answer:
"""

    res = client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content

# COMMAND ----------

def text_to_speech(text, filename_prefix="audio"):
    import requests, base64, time

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

    if response.status_code != 200:
        print("❌ TTS Error:", response.text)
        return None

    data = response.json()

    audio_base64 = (
        data.get("audio") or 
        (data.get("audios")[0] if data.get("audios") else None)
    )

    if not audio_base64:
        print("❌ No audio returned")
        return None

    audio_bytes = base64.b64decode(audio_base64)

    filepath = f"/tmp/{filename_prefix}_{int(time.time())}.wav"

    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filepath

# COMMAND ----------

def split_text(text, max_length=900):
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length

        # Avoid cutting in middle of word
        if end < len(text):
            while end > start and text[end] != " ":
                end -= 1

        chunks.append(text[start:end].strip())
        start = end

    return chunks

# COMMAND ----------

# MAGIC %pip install pydub

# COMMAND ----------

from pydub import AudioSegment
import time

def generate_full_audio(text):
    chunks = split_text(text)
    files = []

    for i, chunk in enumerate(chunks):
        f = text_to_speech(chunk, f"part_{i}")
        if f:
            files.append(f)

    if not files:
        return None

    combined = AudioSegment.empty()

    for f in files:
        combined += AudioSegment.from_wav(f)
        combined += AudioSegment.silent(duration=500)  # smooth pause

    final_path = f"/tmp/final_{int(time.time())}.wav"
    combined.export(final_path, format="wav")

    return final_path

# COMMAND ----------

# audio = text_to_speech("Hello Adarsh, this is a test audio")

# print("Audio path:", audio)

def text_to_speech(text, filename_prefix="audio"):
    import requests, base64, time

    global CURRENT_LANG  # 👈 use selected language

    url = "https://api.sarvam.ai/text-to-speech"

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice": "anushka",
        "language": CURRENT_LANG   # 👈 THIS is the only change
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("❌ TTS Error:", response.text)
        return None

    data = response.json()

    audio_base64 = (
        data.get("audio") or 
        (data.get("audios")[0] if data.get("audios") else None)
    )

    if not audio_base64:
        print("❌ No audio returned")
        return None

    audio_bytes = base64.b64decode(audio_base64)

    filepath = f"/tmp/{filename_prefix}_{int(time.time())}.wav"

    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filepath

# COMMAND ----------

def translate_text(text, target_lang):
    import requests

    chunks = split_text(text, max_length=800)  # safer than 900
    translated_chunks = []

    for chunk in chunks:

        # 🔥 HARD SAFETY (never exceed limit)
        if len(chunk) > 950:
            chunk = chunk[:950]

        url = "https://api.sarvam.ai/translate"

        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": chunk,
            "source_language_code": "en-IN",
            "target_language_code": target_lang
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print("❌ Translation Error:", response.text)

            # 🔥 CRITICAL FIX: DO NOT return English fallback
            # Instead try smaller chunks
            smaller_chunks = split_text(chunk, max_length=400)

            for small in smaller_chunks:
                payload["input"] = small
                res = requests.post(url, headers=headers, json=payload)

                if res.status_code == 200:
                    translated_chunks.append(
                        res.json().get("translated_text", small)
                    )
                else:
                    translated_chunks.append("")  # avoid English leak

            continue

        translated_chunks.append(
            response.json().get("translated_text", "")
        )

    return " ".join(translated_chunks)

# COMMAND ----------

def disability_aware_rag(query, user_lang="en-IN"):

    global CURRENT_LANG
    CURRENT_LANG = user_lang

    # 1. Detect profile
    profile = detect_profile(query)

    # 2. Retrieve context
    context, sources = retrieve_context(query)

    # 3. Get style
    style = get_style(profile)

    # 4. Generate answer (LLM)
    answer = generate_answer(query, context, style)

    # 5. Translate if needed
    translated_answer = answer
    if user_lang != "en-IN":
        translated_answer = translate_text(answer, user_lang)

    # 6. Generate audio
    audio_path = None
    if profile == "visual_impairment":
        short_answer = translated_answer[:1000]
        audio_path = generate_full_audio(short_answer)

    # ✅ FINAL RETURN (VERY IMPORTANT)
    return {
        "profile": profile,
        "answer": translated_answer,
        "audio": audio_path,
        "sources": sources
    }

# COMMAND ----------

# result = disability_aware_rag(
#     "Explain anomalous behavior of nitrogen for visually impaired student",
#     user_lang="hi-IN"
# )

# print(result["answer"])

# COMMAND ----------

from IPython.display import Audio, display

query = "Explain anomalous behavior of nitrogen for visually impaired student"

result = disability_aware_rag(
    query,
    user_lang="mr-IN"   # ✅ ADD THIS
)

print("Profile:", result["profile"])
print("\nAnswer:\n", result["answer"])

if result["audio"]:
    display(Audio(result["audio"]))

# COMMAND ----------

