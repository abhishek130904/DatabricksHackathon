import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
import time
import io
import requests
import base64

# Try to import ML dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import translation dependencies
try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    import torch
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# Try to import PDF and RAG dependencies
try:
    from pypdf import PdfReader
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Try to import Databricks SDK for LLM
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ============================================================================
# SARVAM AI TEXT-TO-SPEECH CONFIGURATION
# ============================================================================

# Securely retrieve API key from Databricks Secrets
SARVAM_API_KEY = ""
TTS_AVAILABLE = False

# Method 1: Try dbutils (works in notebooks)
try:
    from databricks.sdk.runtime import dbutils
    SARVAM_API_KEY = dbutils.secrets.get(scope="vidya-setu", key="sarvam-api-key")
    TTS_AVAILABLE = True
except Exception:
    pass

# Method 2: Try environment variable (works in Databricks Apps via app.yaml)
if not TTS_AVAILABLE:
    SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
    if SARVAM_API_KEY:
        TTS_AVAILABLE = True

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

def text_to_speech(text, target_language="hi-IN"):
    """Convert text to speech using Sarvam AI"""
    # Check if API key is available
    if not SARVAM_API_KEY or not TTS_AVAILABLE:
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }
        
        payload = {
            "inputs": [text],
            "target_language_code": target_language,
            "speaker": "priya",  # v3-compatible female speaker
            "pace": 1.0,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v3"  # Updated to v3
        }
        
        response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if "audios" in result and len(result["audios"]) > 0:
                return result["audios"][0]
        return None
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

st.set_page_config(page_title="Vidya Setu", page_icon="🎯", layout="wide")

# ============================================================================
# TOP NAVIGATION BAR
# ============================================================================

st.markdown("""
<style>
.top-bar {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 1rem 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Quiz"

# Create navigation bar
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("# 🎯 Vidya Setu")

with col2:
    if st.button("📝 Quiz", key="nav_quiz", use_container_width=True):
        st.session_state.current_page = "Quiz"
        st.rerun()

with col3:
    if st.button("🎓 Personalized Learning", key="nav_personalized", use_container_width=True):
        st.session_state.current_page = "Personalized Learning"
        st.rerun()

st.markdown("---")

# Language configuration
SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "flag": "🇬🇧", "native": "English"},
    "Hindi": {"code": "hi", "flag": "🇮🇳", "native": "हिंदी"},
    "Tamil": {"code": "ta", "flag": "🇮🇳", "native": "தமிழ்"},
    "Telugu": {"code": "te", "flag": "🇮🇳", "native": "తెలుగు"},
    "Bengali": {"code": "bn", "flag": "🇮🇳", "native": "বাংলা"},
    "Marathi": {"code": "mr", "flag": "🇮🇳", "native": "मराठी"},
    "Gujarati": {"code": "gu", "flag": "🇮🇳", "native": "ગુજરાતી"},
    "Kannada": {"code": "kn", "flag": "🇮🇳", "native": "ಕನ್ನಡ"},
    "Malayalam": {"code": "ml", "flag": "🇮🇳", "native": "മലയാളം"},
    "Punjabi": {"code": "pa", "flag": "🇮🇳", "native": "ਪੰਜਾਬੀ"},
}

# ============================================================================
# QUIZ PAGE (Original Implementation - UNCHANGED)
# ============================================================================

if st.session_state.current_page == "Quiz":
    
    st.markdown("*ML-Powered Adaptive Learning System with Multilingual Support*")

    # Show TTS warning if not configured
    if not TTS_AVAILABLE:
        st.warning("⚠️ Text-to-Speech is disabled: Sarvam AI API key not configured. Set up Databricks Secrets to enable.")

    
    # Load quiz data
    @st.cache_data
    def load_data():
        csv_path = os.path.join(os.path.dirname(__file__), "pcmDataset.csv")
        df = pd.read_csv(csv_path)
        df["text"] = df["subject"] + " " + df["question"] + " " + df["difficulty"]
        return df

    df = load_data()

    # User Profile Class
    class UserProfile:
        def __init__(self):
            self.history = []
            self.topic_mastery = defaultdict(lambda: {"correct": 0, "total": 0, "mastery_score": 0.0})
            self.difficulty_performance = defaultdict(lambda: {"correct": 0, "total": 0})
            self.asked_indices = set()
            self.current_level = "easy"
            
        def add_response(self, question_idx, correct, subject, difficulty, response_time=0):
            self.history.append({
                "idx": question_idx,
                "correct": correct,
                "subject": subject,
                "difficulty": difficulty,
                "time": response_time,
                "timestamp": datetime.now()
            })
            
            self.asked_indices.add(question_idx)
            
            self.topic_mastery[subject]["total"] += 1
            if correct:
                self.topic_mastery[subject]["correct"] += 1
            
            topic = self.topic_mastery[subject]
            if topic["total"] > 0:
                recent = [h for h in self.history[-5:] if h["subject"] == subject]
                if recent:
                    recent_accuracy = sum(1 for r in recent if r["correct"]) / len(recent)
                    overall_accuracy = topic["correct"] / topic["total"]
                    topic["mastery_score"] = 0.7 * recent_accuracy + 0.3 * overall_accuracy
            
            self.difficulty_performance[difficulty]["total"] += 1
            if correct:
                self.difficulty_performance[difficulty]["correct"] += 1
        
        def get_weak_topics(self, top_n=2):
            if not self.topic_mastery:
                return ["Physics"]
            
            topics = [(topic, scores["mastery_score"]) 
                      for topic, scores in self.topic_mastery.items() 
                      if scores["total"] >= 2]
            
            if not topics:
                return [list(self.topic_mastery.keys())[0]]
            
            topics.sort(key=lambda x: x[1])
            return [t[0] for t in topics[:top_n]]
        
        def predict_optimal_difficulty(self):
            if len(self.history) < 3:
                return "easy"
            
            recent = self.history[-5:]
            recent_accuracy = sum(1 for r in recent if r["correct"]) / len(recent)
            
            levels = ["easy", "medium", "hard"]
            current_perf = self.difficulty_performance.get(self.current_level, {"correct": 0, "total": 1})
            current_accuracy = current_perf["correct"] / max(current_perf["total"], 1)
            
            if recent_accuracy >= 0.75 and current_accuracy >= 0.70:
                idx = levels.index(self.current_level)
                return levels[min(idx + 1, 2)]
            elif recent_accuracy <= 0.40:
                idx = levels.index(self.current_level)
                return levels[max(idx - 1, 0)]
            else:
                return self.current_level
        
        def get_stats(self):
            if not self.history:
                return {"total": 0, "correct": 0, "accuracy": 0}
            
            total = len(self.history)
            correct = sum(1 for h in self.history if h["correct"])
            accuracy = correct / total if total > 0 else 0
            
            return {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "topic_mastery": dict(self.topic_mastery),
                "difficulty_performance": dict(self.difficulty_performance)
            }

    # Initialize session state
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = UserProfile()

    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"

    if "translation_cache" not in st.session_state:
        st.session_state.translation_cache = {}

    if "generate_audio" not in st.session_state:
        st.session_state.generate_audio = False

    user_profile = st.session_state.user_profile

    # Load M2M100 Translation Model
    @st.cache_resource
    def load_translation_model():
        if not TRANSLATION_AVAILABLE:
            return None, None, False
        
        try:
            model_name = "facebook/m2m100_418M"
            
            with st.spinner("🌐 Loading multilingual model (M2M100)..."):
                tokenizer = M2M100Tokenizer.from_pretrained(model_name)
                model = M2M100ForConditionalGeneration.from_pretrained(model_name)
                
                if torch.cuda.is_available():
                    model = model.cuda()
            
            return model, tokenizer, True
        except Exception as e:
            st.sidebar.error(f"Translation error: {str(e)[:100]}")
            return None, None, False

    translation_model, translation_tokenizer, translation_available = load_translation_model()

    def translate_text(text, target_lang_code, use_cache=True):
        if not translation_available or target_lang_code == "en":
            return text
        
        cache_key = f"{text}_{target_lang_code}"
        if use_cache and cache_key in st.session_state.translation_cache:
            return st.session_state.translation_cache[cache_key]
        
        try:
            translation_tokenizer.src_lang = "en"
            
            encoded = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            generated_tokens = translation_model.generate(
                **encoded,
                forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code),
                max_length=512,
                num_beams=5
            )
            
            translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            if use_cache:
                st.session_state.translation_cache[cache_key] = translated_text
            
            return translated_text
        except Exception as e:
            st.warning(f"Translation failed: {str(e)[:60]}")
            return text

    # Load AI components
    @st.cache_resource
    def load_ai_components():
        if not ML_AVAILABLE:
            return None, None, None, False
        
        try:
            model = SentenceTransformer('all-mpnet-base-v2')
            
            embeddings = np.vstack(
                df["text"].apply(lambda x: model.encode(x, show_progress_bar=False)).values
            ).astype("float32")
            
            faiss.normalize_L2(embeddings)
            
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            return model, index, embeddings, True
        except Exception as e:
            st.sidebar.warning(f"⚠️ AI features unavailable: {str(e)[:80]}")
            return None, None, None, False

    with st.spinner("🔄 Loading AI models..."):
        model, index, embeddings, ai_available = load_ai_components()

    def retrieve_question_smart(query_text, k=10, diversity_factor=0.3):
        if not ai_available or model is None or index is None or not ML_AVAILABLE:
            level_df = df[df["difficulty"] == user_profile.current_level]
            if len(level_df) > 0:
                available = level_df[~level_df.index.isin(user_profile.asked_indices)]
                if len(available) > 0:
                    return available.sample(1).iloc[0], available.index[0]
            return df.sample(1).iloc[0], 0
        
        query_vec = model.encode([query_text], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(query_vec)
        
        D, I = index.search(query_vec, k=min(k * 3, len(df)))
        
        candidates = [(i, d) for i, d in zip(I[0], D[0]) if i not in user_profile.asked_indices]
        
        if not candidates:
            user_profile.asked_indices.clear()
            candidates = [(i, d) for i, d in zip(I[0], D[0])]
        
        scored_candidates = []
        weak_topics = user_profile.get_weak_topics()
        recent_embeddings = []
        
        if len(user_profile.history) > 0:
            recent_indices = [h["idx"] for h in user_profile.history[-3:]]
            recent_embeddings = [embeddings[idx] for idx in recent_indices if idx < len(embeddings)]
        
        for idx, similarity in candidates[:k]:
            question = df.iloc[idx]
            score = float(similarity)
            
            if question["subject"].lower() in [t.lower() for t in weak_topics]:
                score += 0.2
            
            if recent_embeddings:
                q_emb = embeddings[idx].reshape(1, -1)
                recent_emb = np.array(recent_embeddings)
                similarities = cosine_similarity(q_emb, recent_emb)[0]
                max_sim = np.max(similarities)
                diversity_penalty = max_sim * diversity_factor
                score -= diversity_penalty
            
            scored_candidates.append((idx, score, question))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if scored_candidates:
            best_idx, best_score, best_question = scored_candidates[0]
            return best_question, best_idx
        else:
            return df.iloc[0], 0

    # Sidebar
    with st.sidebar:
        st.header("📊 Performance Analytics")
        
        st.divider()
        st.subheader("🌐 Language / भाषा")
        
        lang_options = [f"{info['flag']} {lang} ({info['native']})" 
                        for lang, info in SUPPORTED_LANGUAGES.items()]
        lang_names = list(SUPPORTED_LANGUAGES.keys())
        
        current_idx = lang_names.index(st.session_state.selected_language)
        
        selected_display = st.selectbox(
            "Choose your language:",
            lang_options,
            index=current_idx,
            key="lang_selector"
        )
        
        new_lang = lang_names[lang_options.index(selected_display)]
        
        if new_lang != st.session_state.selected_language:
            st.session_state.selected_language = new_lang
            if "question" in st.session_state:
                del st.session_state.question
            if "question_idx" in st.session_state:
                del st.session_state.question_idx
            st.rerun()
        
        if not translation_available and st.session_state.selected_language != "English":
            st.warning("⚠️ Translation model loading...")
        
        st.divider()
        
        stats = user_profile.get_stats()
        
        if stats["total"] > 0:
            st.metric("Total Questions", stats["total"])
            st.metric("Score", f"{stats['correct']}/{stats['total']}")
            st.metric("Accuracy", f"{stats['accuracy']*100:.1f}%")
            
            st.divider()
            
            st.subheader("📚 Topic Mastery")
            for topic, scores in stats['topic_mastery'].items():
                mastery_pct = scores['mastery_score'] * 100
                st.progress(mastery_pct / 100, text=f"{topic}: {mastery_pct:.0f}%")
            
            st.divider()
            
            st.subheader("⭐ Difficulty Levels")
            for diff in ["easy", "medium", "hard"]:
                if diff in stats['difficulty_performance']:
                    perf = stats['difficulty_performance'][diff]
                    acc = (perf['correct'] / perf['total'] * 100) if perf['total'] > 0 else 0
                    st.text(f"{diff.capitalize()}: {acc:.0f}% ({perf['correct']}/{perf['total']})")
            
            st.divider()
            
            weak = user_profile.get_weak_topics()
            st.info(f"🎯 Focus: {', '.join(weak)}")
            
        else:
            st.info("Start answering questions to see your stats!")
        
        st.divider()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if ai_available:
                st.success("✅ AI")
            else:
                st.warning("⚠️ AI")
        
        with col_s2:
            if translation_available:
                st.success("✅ Trans")
            else:
                st.warning("⚠️ Trans")
        
        st.caption(f"Questions: {len(df)}")
        
        if st.button("🔄 Reset Quiz"):
            st.session_state.user_profile = UserProfile()
            st.session_state.translation_cache = {}
            if "question" in st.session_state:
                del st.session_state.question
            if "question_idx" in st.session_state:
                del st.session_state.question_idx
            st.rerun()

    optimal_difficulty = user_profile.predict_optimal_difficulty()
    user_profile.current_level = optimal_difficulty

    if "question" not in st.session_state:
        weak_topics = user_profile.get_weak_topics()
        query = f"{optimal_difficulty} {weak_topics[0]} physics question"
        q, q_idx = retrieve_question_smart(query)
        st.session_state.question = q
        st.session_state.question_idx = q_idx

    q = st.session_state.question
    q_idx = st.session_state.question_idx

    target_lang_code = SUPPORTED_LANGUAGES[st.session_state.selected_language]["code"]

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📘 Question")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.caption(f"**Subject:** {q['subject']}")
        with meta_col2:
            st.caption(f"**Difficulty:** {q['difficulty'].upper()}")
        with meta_col3:
            st.caption(f"**ML Level:** {optimal_difficulty.upper()}")
        
        st.markdown("---")
        
        question_text = q['question']
        if st.session_state.selected_language != "English":
            if translation_available:
                with st.spinner(f"Translating to {st.session_state.selected_language}..."):
                    question_text = translate_text(question_text, target_lang_code)
            else:
                st.info(f"💡 Translation to {st.session_state.selected_language} unavailable. Showing English.")
        
        st.markdown(f"### {question_text}")
        
        
        # Text-to-Speech Button - Improved Layout
        st.markdown("---")
        
        col_tts_btn, col_tts_space = st.columns([2, 3])
        with col_tts_btn:
            if st.button("🔊 Read Question Aloud", key="tts_btn", use_container_width=True, disabled=not TTS_AVAILABLE):
                st.session_state.generate_audio = True
        
        # Generate and display audio outside the button column (full width)
        if st.session_state.get("generate_audio", False):
            with st.spinner("🎵 Generating audio..."):
                try:
                    lang_code = SUPPORTED_LANGUAGES[st.session_state.selected_language]["code"]
                    # Map language codes for Sarvam AI
                    sarvam_lang_map = {
                        "en": "en-IN",
                        "hi": "hi-IN",
                        "ta": "ta-IN",
                        "te": "te-IN",
                        "bn": "bn-IN",
                        "mr": "mr-IN",
                        "gu": "gu-IN",
                        "kn": "kn-IN",
                        "ml": "ml-IN",
                        "pa": "pa-IN"
                    }
                    sarvam_lang = sarvam_lang_map.get(lang_code, "hi-IN")
                    audio_base64 = text_to_speech(question_text, sarvam_lang)
                    
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        st.audio(audio_bytes, format="audio/wav", start_time=0)
                        st.success("✅ Audio ready! Click play above to listen.")
                    else:
                        st.error("❌ Failed to generate audio. Please try again.")
                except Exception as e:
                    st.error(f"❌ TTS Error: {str(e)[:200]}")
                finally:
                    st.session_state.generate_audio = False
        
        st.markdown("---")
        
        if st.session_state.selected_language != "English" and translation_available:
            with st.expander("🔍 View Original (English)"):
                st.caption(q['question'])
        
        user_answer = st.text_input("Your answer:", key="answer_input")
        
        if st.button("Submit Answer", type="primary"):
            start_time = time.time()
            
            def extract_option(ans):
                ans = str(ans).strip()
                if ans.startswith("("):
                    return ans.split(")")[0].replace("(", "").strip()
                return ans.lower()
            
            correct_option = extract_option(q["answer"])
            user_option = extract_option(user_answer)
            correct = user_option == correct_option
            
            response_time = time.time() - start_time
            
            user_profile.add_response(
                question_idx=q_idx,
                correct=correct,
                subject=q["subject"],
                difficulty=optimal_difficulty,
                response_time=response_time
            )
            
            if correct:
                st.success("✅ **Correct!** Well done!")
                st.balloons()
            else:
                correct_answer_text = q["answer"]
                if st.session_state.selected_language != "English" and translation_available:
                    correct_answer_text = translate_text(correct_answer_text, target_lang_code)
                
                st.error(f"❌ **Incorrect!** The correct answer is: **{correct_answer_text}**")
            
            if "explanation" in df.columns and pd.notna(q.get("explanation")):
                with st.expander("📖 View Explanation"):
                    explanation_text = q["explanation"]
                    if st.session_state.selected_language != "English" and translation_available:
                        explanation_text = translate_text(explanation_text, target_lang_code)
                    st.write(explanation_text)
            
            optimal_difficulty = user_profile.predict_optimal_difficulty()
            user_profile.current_level = optimal_difficulty
            weak_topics = user_profile.get_weak_topics()
            query = f"{optimal_difficulty} {weak_topics[0]} physics question"
            
            next_q, next_idx = retrieve_question_smart(query)
            st.session_state.question = next_q
            st.session_state.question_idx = next_idx
            
            st.info("Loading next question...")
            time.sleep(0.5)
            st.rerun()

    with col2:
        st.subheader("💡 Context")
        st.info(f"**Level:** {q['difficulty']}\n\n**Subject:** {q['subject']}")
        
        if stats["total"] > 0:
            st.metric("Current Streak", 
                      sum(1 for h in user_profile.history[-5:] if h["correct"]))
        
        if st.button("⏭️ Skip Question"):
            weak_topics = user_profile.get_weak_topics()
            query = f"{optimal_difficulty} {weak_topics[0]} physics question"
            next_q, next_idx = retrieve_question_smart(query)
            st.session_state.question = next_q
            st.session_state.question_idx = next_idx
            st.rerun()

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        flag = SUPPORTED_LANGUAGES[st.session_state.selected_language]["flag"]
        st.caption(f"{flag} {st.session_state.selected_language}")
    with col_b:
        st.caption("🧠 ML-Powered + 🌐 Multilingual")
    with col_c:
        if translation_available:
            st.caption("✨ M2M100")
        else:
            st.caption("📝 English Only")

# ============================================================================
# PERSONALIZED LEARNING PAGE - WITH REAL RAG IMPLEMENTATION
# ============================================================================

elif st.session_state.current_page == "Personalized Learning":
    
    st.markdown("*AI-Powered Adaptive RAG System for Personalized Education*")
    
    # Initialize session state
    if "pers_step" not in st.session_state:
        st.session_state.pers_step = "upload"
    
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    if "selected_disability" not in st.session_state:
        st.session_state.selected_disability = "default"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "pdf_embeddings" not in st.session_state:
        st.session_state.pdf_embeddings = None
    
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []
    
    # Load RAG model
    @st.cache_resource
    def load_rag_model():
        if not ML_AVAILABLE:
            return None, False
        try:
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            return model, True
        except Exception as e:
            st.error(f"Failed to load RAG model: {str(e)}")
            return None, False
    
    rag_model, rag_model_available = load_rag_model()
    
    # Initialize Databricks Workspace Client - FIXED USING SDK API
    @st.cache_resource
    def get_workspace_client():
        if not LLM_AVAILABLE:
            return None, False
        
        try:
            # Create WorkspaceClient - authenticates automatically
            w = WorkspaceClient()
            
            # Verify authentication by getting current user
            current_user = w.current_user.me()
            
            return w, True
            
        except Exception as e:
            st.error(f"Failed to initialize Databricks client: {str(e)}")
            return None, False
    
    workspace_client, workspace_available = get_workspace_client()
    
    # PDF Processing Function
    def process_pdf(uploaded_file):
        """Process PDF and create embeddings"""
        try:
            # Read PDF
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            
            # Extract text from pages
            chunks = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                
                # Simple chunking (split by sentences/paragraphs)
                chunk_size = 500
                overlap = 100
                
                for i in range(0, len(text), chunk_size - overlap):
                    chunk_text = text[i:i + chunk_size].strip()
                    if len(chunk_text) > 50:  # Minimum chunk size
                        chunks.append({
                            "page": page_num + 1,
                            "text": chunk_text
                        })
            
            if not rag_model_available or rag_model is None:
                return chunks, None
            
            # Create embeddings
            texts = [c["text"] for c in chunks]
            embeddings = rag_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            return chunks, embeddings
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return [], None
    
    # Retrieve context function
    def retrieve_context(query, num_results=3):
        """Retrieve relevant chunks from PDF"""
        if st.session_state.pdf_embeddings is None or not rag_model_available:
            return "No context available.", []
        
        try:
            # Encode query
            query_embedding = rag_model.encode([query], normalize_embeddings=True)[0]
            
            # Compute similarities
            similarities = np.dot(st.session_state.pdf_embeddings, query_embedding)
            
            # Get top results
            top_indices = np.argsort(similarities)[-num_results:][::-1]
            
            # Format context
            contexts = []
            sources = []
            
            for idx in top_indices:
                chunk = st.session_state.pdf_chunks[idx]
                contexts.append(f"[Page {chunk['page']}] {chunk['text']}")
                sources.append({"page": chunk["page"], "text": chunk["text"][:200] + "..."})
            
            context_str = "\n\n".join(contexts)
            return context_str, sources
            
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving context.", []
    
    # Get style based on disability profile
    def get_style(profile):
        styles = {
            "visual_impairment": "Use highly descriptive explanations. Avoid references to visuals. Describe everything in detail.",
            
            "hearing_impairment": "Use clear written explanations. No reliance on audio. Provide complete written context.",
            
            "dyslexia": "Use short sentences and simple words. Break into bullet points. Avoid complex vocabulary.",
            
            "adhd": "Break answer into small numbered steps. Keep it engaging and concise. Use clear structure.",
            
            "default": "Answer normally with balanced explanation and examples."
        }
        
        return styles.get(profile, styles["default"])
    
    # Adaptive RAG function using Databricks SDK
    def adaptive_rag(query, profile):
        """Main RAG function with profile adaptation - Uses Databricks SDK"""
        
        if not workspace_available or workspace_client is None:
            return {
                "profile": profile,
                "answer": "⚠️ Databricks workspace client is not available. Please check your environment.",
                "sources": []
            }
        
        # Retrieve context
        context, sources = retrieve_context(query)
        
        if context == "No context available.":
            return {
                "profile": profile,
                "answer": "⚠️ No PDF content available. Please upload and process a PDF first.",
                "sources": []
            }
        
        # Get adaptation style
        style = get_style(profile)
        
        # Build prompt
        prompt = f"""You are an AI tutor helping a student understand educational material.

Instructions:
- Answer ONLY using the context provided below
- Adapt your response style: {style}
- Be accurate and cite page numbers when relevant

Context:
{context}

Question:
{query}

Answer:"""
        
        try:
            # Call LLM using Databricks SDK (native API)
            messages = [
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=prompt
                )
            ]
            
            response = workspace_client.serving_endpoints.query(
                name="databricks-meta-llama-3-3-70b-instruct",
                messages=messages,
                max_tokens=500,
                temperature=0.2
            )
            
            answer = response.choices[0].message.content
            
            return {
                "profile": profile,
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "profile": profile,
                "answer": f"⚠️ Error generating response: {str(e)[:200]}\n\nThe context was successfully retrieved from your PDF, but the LLM call failed. Please check if the serving endpoint 'databricks-meta-llama-3-3-70b-instruct' is available.",
                "sources": sources
            }
    
    # STEP 1: Upload and Configure
    if st.session_state.pers_step == "upload":
        
        st.subheader("📚 Upload Learning Material")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📄 Upload PDF Document")
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload educational content (textbooks, notes, etc.)"
            )
            
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.session_state.uploaded_pdf = uploaded_file
        
        with col2:
            st.markdown("### ♿ Accessibility Profile")
            
            disability_options = {
                "default": "🔵 Default",
                "visual_impairment": "👁️ Visual Impairment",
                "hearing_impairment": "👂 Hearing Impairment",
                "dyslexia": "📖 Dyslexia",
                "adhd": "🧠 ADHD"
            }
            
            selected = st.selectbox(
                "Select your learning profile:",
                options=list(disability_options.keys()),
                format_func=lambda x: disability_options[x],
                help="Choose a profile for personalized content adaptation"
            )
            
            st.session_state.selected_disability = selected
            
            # Show profile description
            profile_descriptions = {
                "default": "Standard learning format with balanced explanations.",
                "visual_impairment": "Highly descriptive explanations without visual references.",
                "hearing_impairment": "Clear written explanations with no audio dependency.",
                "dyslexia": "Short sentences, simple words, and bullet points.",
                "adhd": "Concise steps with engaging and focused content."
            }
            
            st.info(f"**Profile:** {profile_descriptions[selected]}")
        
        st.markdown("---")
        
        # Status indicators
        col_status1, col_status2, col_status3 = st.columns(3)
        with col_status1:
            if RAG_AVAILABLE:
                st.success("✅ PDF Processing")
            else:
                st.error("❌ PDF Processing")
        
        with col_status2:
            if rag_model_available:
                st.success("✅ Embeddings")
            else:
                st.error("❌ Embeddings")
        
        with col_status3:
            if workspace_available:
                st.success("✅ LLM Ready")
            else:
                st.warning("⚠️ LLM")
        
        # Next button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            if st.button("▶️ Next: Start Learning", type="primary", use_container_width=True):
                if "uploaded_pdf" not in st.session_state:
                    st.error("❌ Please upload a PDF file first!")
                elif not RAG_AVAILABLE:
                    st.error("❌ PDF processing library not available!")
                elif not rag_model_available:
                    st.error("❌ Embeddings model not available!")
                else:
                    # Process PDF
                    with st.spinner("🔄 Processing PDF and creating embeddings..."):
                        chunks, embeddings = process_pdf(st.session_state.uploaded_pdf)
                        
                        if chunks:
                            st.session_state.pdf_chunks = chunks
                            st.session_state.pdf_embeddings = embeddings
                            st.session_state.pdf_processed = True
                            st.session_state.pers_step = "chat"
                            st.success(f"✅ Processed {len(chunks)} chunks from PDF")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to process PDF. Please try another file.")
    
    # STEP 2: Chat Interface
    elif st.session_state.pers_step == "chat":
        
        # Sidebar with settings
        with st.sidebar:
            st.header("⚙️ Learning Settings")
            
            st.metric("Profile", st.session_state.selected_disability.replace("_", " ").title())
            
            if "uploaded_pdf" in st.session_state:
                st.metric("Document", st.session_state.uploaded_pdf.name[:20] + "...")
            
            st.metric("Chunks", len(st.session_state.pdf_chunks))
            
            st.divider()
            
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                if rag_model_available:
                    st.success("✅ RAG")
                else:
                    st.error("❌ RAG")
            
            with col_llm2:
                if workspace_available:
                    st.success("✅ LLM")
                else:
                    st.warning("⚠️ LLM")
            
            st.divider()
            
            if st.button("🔄 Upload New PDF"):
                st.session_state.pers_step = "upload"
                st.session_state.chat_history = []
                st.session_state.pdf_chunks = []
                st.session_state.pdf_embeddings = None
                st.rerun()
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        st.subheader("💬 Ask Questions About Your Material")
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["question"])
            
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                
                if "sources" in chat and chat["sources"]:
                    with st.expander("📚 Sources"):
                        for j, source in enumerate(chat["sources"], 1):
                            st.caption(f"**Page {source['page']}:** {source['text']}")
                
                if "profile" in chat:
                    st.caption(f"*Adapted for: {chat['profile'].replace('_', ' ').title()}*")
        
        # Input area
        st.markdown("---")
        
        user_question = st.text_input(
            "Ask a question about the uploaded material:",
            placeholder="E.g., What is the main concept explained in the document?",
            key="user_question_input"
        )
        
        col_ask1, col_ask2 = st.columns([4, 1])
        
        with col_ask2:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        if ask_button and user_question:
            
            with st.spinner("🤔 Retrieving context and generating answer..."):
                
                # Call actual adaptive RAG
                result = adaptive_rag(user_question, st.session_state.selected_disability)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": result["answer"],
                    "profile": result["profile"],
                    "sources": result.get("sources", [])
                })
                
                st.rerun()
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            * What is the main topic covered in this document?
            * Explain the concept from page 5
            * What are the key formulas or equations mentioned?
            * Summarize the important points
            * How does this concept work?
            """)
        
        # Instructions
        st.info("""
        **💡 How it works:**
        1. Your question is converted to an embedding (BGE-small)
        2. Relevant sections are retrieved using similarity search
        3. Context is sent to Llama 3.3 70B with your profile
        4. An adapted answer is generated with source citations
        """)

