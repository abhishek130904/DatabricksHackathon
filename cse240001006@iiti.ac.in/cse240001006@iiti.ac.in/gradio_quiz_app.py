import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws

# Load data and create index (cached)
print("Loading quiz data...")
spark = SparkSession.builder.getOrCreate()
df = spark.table("pcm_dataset_2").limit(200)  # Load 200 questions for faster startup

df = df.withColumn(
    "text",
    concat_ws(" ", col("question"), col("subject"), col("difficulty"))
)

pdf = df.select("subject", "question", "difficulty", "answer", "text").toPandas()

# Load model and create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
pdf["embedding"] = pdf["text"].apply(lambda x: model.encode(x))

# Create FAISS index
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
    
    # Validate input
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
    
    # Get next question
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
    
    # State
    quiz_state = gr.State(init_quiz())
    
    # Display
    with gr.Row():
        level_display = gr.Textbox(label="Current Level", value="", interactive=False)
        stats_display = gr.Textbox(label="Stats", value="Score: 0/0", interactive=False)
    
    question_display = gr.Textbox(
        label="Question",
        value="Click 'Start Quiz' to begin!",
        interactive=False,
        lines=5
    )
    
    # Input
    with gr.Row():
        answer_input = gr.Textbox(
            label="Your Answer",
            placeholder="Enter 1, 2, 3, or 4",
            scale=3
        )
        submit_btn = gr.Button("Submit Answer", variant="primary", scale=1)
    
    feedback_display = gr.Textbox(label="Feedback", value="", interactive=False, lines=3)
    
    # Control buttons
    with gr.Row():
        start_btn = gr.Button("🚀 Start New Quiz", variant="secondary")
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tips:** Answer correctly to level up (Easy → Medium → Hard). Wrong answers level you down!")
    
    # Event handlers
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
    
    # Allow Enter key to submit
    answer_input.submit(
        submit_handler,
        inputs=[answer_input, quiz_state],
        outputs=[quiz_state, question_display, level_display, stats_display, feedback_display, answer_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)