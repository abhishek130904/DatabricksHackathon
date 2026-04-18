import streamlit as st

# ---------- STATE ----------
if "level" not in st.session_state:
    st.session_state.level = "easy"

if "asked" not in st.session_state:
    st.session_state.asked = set()

# ---------- FUNCTIONS ----------
levels = ["easy", "medium", "hard"]

def get_next_level(current, correct):
    idx = levels.index(current)
    return levels[min(idx + 1, len(levels) - 1)] if correct else levels[max(idx - 1, 0)]

def extract_option_number(ans):
    ans = str(ans).strip()
    if ans.startswith("("):
        return ans.split(")")[0].replace("(", "").strip()
    return ans

# 👉 USE YOUR EXISTING FUNCTIONS HERE
# retrieve_question, index, pdf, model should already exist

def get_new_question(query, exclude_indices):
    q = retrieve_question(query)
    idx = pdf.index.get_loc(q.name)

    if idx not in exclude_indices:
        return q, idx

    D, I = index.search(model.encode([query]).astype("float32"), k=len(pdf))
    for i in I[0]:
        if i not in exclude_indices:
            return pdf.iloc[i], i

    return None, None


# ---------- UI ----------
st.title("🧠 Physics Quiz App")

query = f"{st.session_state.level} physics question"
q, idx = get_new_question(query, st.session_state.asked)

if q:
    st.session_state.asked.add(idx)

    st.subheader(f"Level: {st.session_state.level.upper()}")
    st.write(q["question"])

    st.info("👉 Enter ONLY the option number (1, 2, 3...)")

    user_answer = st.text_input("Your answer:")

    if st.button("Submit"):
        correct_option = extract_option_number(q["answer"])
        correct = user_answer.strip() == correct_option

        if correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Wrong! Correct answer is {correct_option}")

        # level update
        if st.session_state.level == "easy" and not correct:
            pass
        else:
            st.session_state.level = get_next_level(st.session_state.level, correct)