# Databricks notebook source
# MAGIC %run "/Workspace/Users/me240003058@iiti.ac.in/quiz"

# COMMAND ----------

# ---------- STATE ----------
if "current_level" not in globals():
    current_level = "easy"

if "asked_indices" not in globals():
    asked_indices = set()

# ---------- INPUT BOX ----------
dbutils.widgets.removeAll()
dbutils.widgets.text("answer", "", "Enter option number")

# ---------- GET QUESTION ----------
query = f"{current_level} physics question"
q, idx = get_new_question(query, asked_indices)

if q is None:
    print("No more questions available.")
else:
    asked_indices.add(idx)

    print("\nLevel:", current_level.upper())
    print("Question:", q["question"])
    print("\n👉 Enter ONLY option number (1,2,3...)")

    # ---------- GET ANSWER ----------
    user_answer = dbutils.widgets.get("answer").strip()

    if user_answer != "":
        correct_option = extract_option_number(q["answer"])
        correct = user_answer == correct_option

        if correct:
            print("✅ Correct!")
        else:
            print(f"❌ Wrong! Correct answer is {correct_option}")

        # ---------- UPDATE LEVEL ----------
        if current_level == "easy" and not correct:
            current_level = "easy"
        else:
            current_level = get_next_level(current_level, correct)