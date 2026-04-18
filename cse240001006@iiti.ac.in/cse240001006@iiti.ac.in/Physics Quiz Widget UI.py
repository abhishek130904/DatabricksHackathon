# Databricks notebook source
# DBTITLE 1,Gradio Quiz Instructions
# MAGIC %md
# MAGIC # 🧪 Physics Quiz - Gradio Web App
# MAGIC
# MAGIC ## 🚀 Super Simple Setup (3 Steps!)
# MAGIC
# MAGIC ### Step 1: Install Gradio
# MAGIC Run **Cell 1** below (skip if already done)
# MAGIC
# MAGIC ### Step 2: Start the App ▶️
# MAGIC Run **Cell 2** - this starts the Gradio server
# MAGIC - The cell will keep running (that's normal!)
# MAGIC - Wait for "Running on local URL..." message
# MAGIC
# MAGIC ### Step 3: Get Your Quiz URL 🔗
# MAGIC Run **Cell 3** - this shows your personalized quiz URL
# MAGIC - **Click the URL** (looks like `https://...databricks.com/...7860/`)
# MAGIC - Your quiz opens in a new browser tab!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✨ What Makes Gradio Great?
# MAGIC
# MAGIC * 💻 **Clean, simple interface** - easier than Streamlit!
# MAGIC * 🎯 **Adaptive difficulty** - Easy → Medium → Hard
# MAGIC * ✅ **Instant feedback** - see if you're right immediately
# MAGIC * 📊 **Live stats** - score and question count
# MAGIC * 🎨 **Professional look** - with emojis and colors
# MAGIC * ⏩ **Press Enter** - to quickly submit answers
# MAGIC * 🔄 **One-click restart** - start new quiz anytime
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎮 How to Play
# MAGIC
# MAGIC 1. Click **Start New Quiz** button
# MAGIC 2. Read the question
# MAGIC 3. Type your answer: **1**, **2**, **3**, or **4**
# MAGIC 4. Press **Enter** or click **Submit Answer**
# MAGIC 5. See feedback and automatically get next question!
# MAGIC 6. Watch your level change based on performance
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Troubleshooting
# MAGIC
# MAGIC * **App loading slow?** First load takes 30-60 seconds (loading AI model)
# MAGIC * **URL not working?** Make sure Cell 2 is still **running** (not finished)
# MAGIC * **Want to stop?** Click **Cancel/Interrupt** on Cell 2
# MAGIC * **Need to restart?** Cancel Cell 2, then run it again
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **👇 START NOW: Run Cell 1 → Cell 2 → Cell 3 → Click URL!**

# COMMAND ----------

# MAGIC %pip install gradio

# COMMAND ----------

import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws

# Load data and create index
print("Loading quiz data...")
spark = SparkSession.builder.getOrCreate()
df = spark.table("pcm_dataset_2").limit(200)

df = df.withColumn(
    "text",
    concat_ws(" ", col("question"), col("subject"), col("difficulty"))
)

pdf = df.select("subject", "question", "difficulty", "answer", "text").toPandas()

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
pdf["embedding"] = pdf["text"].apply(lambda x: model.encode(x))

print("Creating search index...")
embeddings = np.vstack(pdf["embedding"].values).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Quiz data loaded successfully!")

# Helper functions
def extract_option_number(ans):
    ans = str(ans).strip()
    if ans.startswith("("):
        return ans.split(")")[0].replace("(", "").strip()
    return ans

def get_next_level(current, correct):
    levels = ["easy", "medium", "hard"]
    idx = levels.index(current)
    if correct:
        return levels[min(idx + 1, len(levels) - 1)]
    else:
        return levels[max(idx - 1, 0)]

def get_new_question(query, exclude_indices):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=len(pdf))
    
    for i in I[0]:
        if i not in exclude_indices:
            return pdf.iloc[i], i
    return None, None

# Quiz state initialization
def init_quiz():
    return {
        "level": "easy",
        "asked_indices": set(),
        "score": 0,
        "total": 0,
        "current_question": None,
        "current_idx": None
    }

# Main quiz logic
def start_quiz():
    state = init_quiz()
    query = f"{state['level']} physics question"
    q, idx = get_new_question(query, state['asked_indices'])
    
    if q is None:
        return "No questions available!", "", "", "", state
    
    state['current_question'] = q
    state['current_idx'] = idx
    state['asked_indices'].add(idx)
    
    level_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    level_display = f"{level_emoji.get(state['level'], '')} Level: {state['level'].upper()}"
    stats = f"Score: {state['score']}/{state['total']} | Questions: {len(state['asked_indices'])}"
    
    return q['question'], level_display, stats, "", state

def submit_answer(answer, state):
    if not answer or not answer.strip():
        return "⚠️ Please enter an answer (1, 2, 3, or 4)", state['current_question']['question'], state
    
    answer = answer.strip()
    
    if not answer.isdigit() or int(answer) not in [1, 2, 3, 4]:
        return "⚠️ Please enter only 1, 2, 3, or 4", state['current_question']['question'], state
    
    correct_option = extract_option_number(state['current_question']['answer'])
    is_correct = answer == correct_option
    
    state['total'] += 1
    if is_correct:
        state['score'] += 1
        feedback = f"✅ Correct! The answer is option {correct_option}."
        new_level = get_next_level(state['level'], True)
    else:
        feedback = f"❌ Incorrect. The correct answer is option {correct_option}."
        new_level = get_next_level(state['level'], False) if state['level'] != "easy" else "easy"
    
    state['level'] = new_level
    
    query = f"{state['level']} physics question"
    q, idx = get_new_question(query, state['asked_indices'])
    
    if q is None:
        accuracy = (state['score'] / state['total'] * 100) if state['total'] > 0 else 0
        return (
            f"{feedback}\n\n🎉 Quiz Complete!\nFinal Score: {state['score']}/{state['total']} ({accuracy:.1f}%)\n\nClick 'Start New Quiz' to play again!",
            "No more questions available",
            state
        )
    
    state['current_question'] = q
    state['current_idx'] = idx
    state['asked_indices'].add(idx)
    
    return feedback, q['question'], state

def update_display(state):
    if state.get('current_question') is None:
        return "", "Click 'Start Quiz' to begin!", "Score: 0/0"
    
    level_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    level_display = f"{level_emoji.get(state['level'], '')} Level: {state['level'].upper()}"
    stats = f"Score: {state['score']}/{state['total']} | Questions: {len(state['asked_indices'])}"
    
    return level_display, state['current_question']['question'], stats

# Create Gradio interface
with gr.Blocks(title="Physics Quiz", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Adaptive Physics Quiz")
    gr.Markdown("Answer questions and watch your difficulty level adapt to your performance!")
    
    quiz_state = gr.State(init_quiz())
    
    with gr.Row():
        level_display = gr.Textbox(label="Current Level", value="", interactive=False)
        stats_display = gr.Textbox(label="Stats", value="Score: 0/0", interactive=False)
    
    question_display = gr.Textbox(
        label="Question",
        value="Click 'Start Quiz' to begin!",
        interactive=False,
        lines=5
    )
    
    with gr.Row():
        answer_input = gr.Textbox(
            label="Your Answer",
            placeholder="Enter 1, 2, 3, or 4",
            scale=3
        )
        submit_btn = gr.Button("Submit Answer", variant="primary", scale=1)
    
    feedback_display = gr.Textbox(label="Feedback", value="", interactive=False, lines=3)
    
    with gr.Row():
        start_btn = gr.Button("🚀 Start New Quiz", variant="secondary")
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tips:** Answer correctly to level up (Easy → Medium → Hard). Wrong answers level you down!")
    
    def start_handler():
        question, level, stats, feedback, new_state = start_quiz()
        return new_state, question, level, stats, "", ""
    
    def submit_handler(answer, state):
        feedback, next_question, new_state = submit_answer(answer, state)
        level, _, stats = update_display(new_state)
        return new_state, next_question, level, stats, feedback, ""
    
    start_btn.click(
        start_handler,
        inputs=[],
        outputs=[quiz_state, question_display, level_display, stats_display, feedback_display, answer_input]
    )
    
    submit_btn.click(
        submit_handler,
        inputs=[answer_input, quiz_state],
        outputs=[quiz_state, question_display, level_display, stats_display, feedback_display, answer_input]
    )
    
    answer_input.submit(
        submit_handler,
        inputs=[answer_input, quiz_state],
        outputs=[quiz_state, question_display, level_display, stats_display, feedback_display, answer_input]
    )

print("\n✅ Launching Gradio app...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# COMMAND ----------

# DBTITLE 1,Get Gradio URL
from IPython.display import HTML, display
import re

print("\n" + "="*80)
print("✅ GRADIO QUIZ APP IS RUNNING!")
print("="*80)
print("\n🌐 YOUR QUIZ URL:")
print("\nThe Gradio app is running on port 7860.")
print("To access it, you need your Databricks proxy URL.")
print("\n📋 COPY THIS URL PATTERN:")
print("\nhttps://<your-workspace>.cloud.databricks.com/driver-proxy/o/<org-id>/<cluster-id>/7860/")
print("\n" + "="*80)
print("\n🔍 HOW TO GET YOUR EXACT URL:")
print("\n1. Look at your browser's current URL")
print("2. It should look like:")
print("   https://adb-123456789.12.azuredatabricks.net/... (Azure)")
print("   OR")
print("   https://dbc-abc123-def.cloud.databricks.com/... (AWS/GCP)")
print("\n3. Find the part with '/driver-proxy/o/{numbers}/{numbers}/'")
print("4. Replace the last port number with 7860")
print("\nExample:")
print("If your notebook URL ends with: .../driver-proxy/o/1234567/8901234/8888/")
print("Your quiz URL should be:        .../driver-proxy/o/1234567/8901234/7860/")
print("\n" + "="*80)

# Try to get the URL from displayHTML
display(HTML('''
<div style="padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; font-family: Arial, sans-serif; margin: 20px 0;">
    <h2 style="margin-top: 0; font-size: 24px;">🎮 Gradio Quiz is Live!</h2>
    <p style="font-size: 16px; margin: 15px 0;">Your quiz is running on <strong>port 7860</strong></p>
    <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; margin: 15px 0;">
        <p style="margin: 5px 0; font-size: 14px;"><strong>📍 To access your quiz:</strong></p>
        <ol style="margin: 10px 0; padding-left: 20px; font-size: 14px;">
            <li>Copy your current notebook URL from the address bar</li>
            <li>Find the <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">/driver-proxy/o/{org}/{cluster}/</code> part</li>
            <li>Change the last number to <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">7860</code></li>
        </ol>
    </div>
    <p style="font-size: 13px; margin-top: 15px; opacity: 0.9;">💡 Keep Cell 3 running while using the quiz!</p>
</div>
<script>
    // Try to construct and display the URL
    const url = window.location.href;
    const match = url.match(/(https:\/\/[^\/]+\/.*\/driver-proxy\/o\/\d+\/\d+)\/);
    if (match) {
        const quizUrl = match[1] + '/7860/';
        document.write('<div style="background: #d4edda; padding: 20px; border-radius: 8px; border: 2px solid #28a745; margin: 20px 0;">');
        document.write('<h3 style="color: #155724; margin-top: 0;">✅ Auto-detected URL:</h3>');
        document.write('<p style="background: white; padding: 15px; border-radius: 6px; font-family: monospace; word-break: break-all; font-size: 14px; color: #333;">' + quizUrl + '</p>');
        document.write('<a href="' + quizUrl + '" target="_blank" style="display: inline-block; padding: 12px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; margin-top: 10px;">🚀 CLICK HERE TO OPEN QUIZ</a>');
        document.write('</div>');
    }
</script>
'''))

print("\n💡 TIPS:")
print("   • Cell 3 must stay running for the quiz to work")
print("   • Click 'Start New Quiz' button when the page loads")
print("   • Type 1, 2, 3, or 4 and press Enter")
print("\n" + "="*80)