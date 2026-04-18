# 🚀 Vidya Setu – AI-Powered Inclusive Learning Platform

**"Smart. Inclusive. Multilingual. Education Reimagined."**

Vidya Setu (विद्या सेतु – Bridge of Knowledge) is an AI-powered educational platform that delivers personalized, multilingual, and accessible learning for every Indian student using Databricks, RAG, and LLMs.

---

## 🧠 What It Does

Vidya Setu allows students to upload study material (PDFs) and interact with an AI tutor that provides context-aware answers, adaptive quizzes, and multilingual explanations.

---

## 🏗️ Architecture Diagram

![Architecture Diagram](./architecture.png)

### 🔗 Databricks Flow

1. PDF Upload → Stored in DBFS / Unity Catalog  
2. Chunking → Text split into smaller parts  
3. Embeddings → Generated using BGE model  
4. Vector Search → Stored in Databricks Vector Index  
5. Query → User question processed via RAG  
6. LLM → Generates final contextual response  

---

## ⚙️ Tech Stack

- Databricks (Vector Search, DBFS, Notebook)
- LangChain
- Groq API (LLaMA 3)
- BGE Embeddings
- Streamlit (Frontend)
- Python

---

## 📁 Project Structure
vidya-setu/
│── app.py
│── rag_pipeline.py
│── requirements.txt
│── architecture.png
│── README.md


---

## 🖥️ How to Run (Exact Commands)



```bash
### 1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/vidya-setu.git
cd vidya-setu

###2️⃣ Install Dependencies
pip install -r requirements.txt

###3️⃣ Add Environment Variables
export GROQ_API_KEY="your_key_here"
export DATABRICKS_HOST="your_host"
export DATABRICKS_TOKEN="your_token"

###4️⃣ Run Application
streamlit run app.py

🎯 Demo Steps
Open the Streamlit app in browser
Upload a PDF (NCERT / notes)
Wait for embedding + indexing
Ask a question like:  Explain nitrogen anomaly
View:
 AI-generated answer
 Context from PDF
 Multilingual explanation (optional)
