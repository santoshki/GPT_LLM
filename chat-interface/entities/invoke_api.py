from groq import Groq
import os
from dotenv import load_dotenv
import json
import re
import glob
import difflib

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY is not set in environment variables")

# -------------------------
# Groq Client
# -------------------------
client = Groq(api_key=api_key)

chat_history = []

# -------------------------
# Load Knowledge Base
# -------------------------
def load_all_knowledge(folder_path):
    knowledge_entries = []
    files = glob.glob(os.path.join(folder_path, "*.json"))

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            category = os.path.basename(file).replace(".json", "")

            for entry in data:
                entry["category"] = category
                knowledge_entries.append(entry)

    return knowledge_entries


# 👉 CHANGE THIS PATH FOR RENDER (use relative path)
KB_PATH = os.getenv("KB_PATH", "./../knowledge-base")

knowledge_base = load_all_knowledge(KB_PATH)
print(f"✅ Loaded {len(knowledge_base)} KB entries.")

# -------------------------
# Helpers
# -------------------------
STOP_WORDS = {
    "what","where","when","how","is","are","do","does","did",
    "the","a","an","you","your","i","me","about"
}

ACK_WORDS = {"ok", "okay", "hmm", "fine", "alright", "will do", "sure", "thanks", "thank you"}

conversation_state = {"last_category": None}


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


# -------------------------
# KB Retrieval
# -------------------------
def retrieve(query, min_score_threshold=4):
    query_norm = normalize(query)
    query_words = {w for w in query_norm.split() if w not in STOP_WORDS}

    if query_norm in ACK_WORDS:
        return "Take care. Let me know if you need anything else."

    best_match = None
    best_score = 0

    for entry in knowledge_base:
        question_norm = normalize(entry["question"])
        question_words = {w for w in question_norm.split() if w not in STOP_WORDS}

        keyword_words = set(normalize(" ".join(entry.get("keywords", []))).split())

        overlap_score = len(query_words & question_words)
        keyword_score = len(query_words & keyword_words) * 3
        fuzzy_score = int(difflib.SequenceMatcher(None, query_norm, question_norm).ratio() * 4)

        total_score = overlap_score + keyword_score + fuzzy_score

        if total_score > best_score:
            best_score = total_score
            best_match = entry

    if best_match and best_score >= min_score_threshold:
        conversation_state["last_category"] = best_match["category"]
        return best_match["answer"]

    return None


# -------------------------
# Rephrase KB Answer using Groq
# -------------------------
def rephrase_with_groq(user_input, kb_answer):
    global chat_history

    try:
        messages = [
            {
                "role": "system",
                "content": """
You are a helpful assistant.

Rephrase the given answer in a natural, conversational way.

Rules:
- Do NOT change meaning
- Do NOT add new information
- Keep it clear and user-friendly
"""
            },
            {
                "role": "user",
                "content": f"""
User Question: {user_input}

Knowledge Base Answer: {kb_answer}

Rephrase the answer:
"""
            }
        ]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_completion_tokens=300
        )

        response = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Groq Rephrase Error: {e}")
        return kb_answer  # ✅ fallback

    # Save history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    chat_history = chat_history[-10:]

    return response


# -------------------------
# Groq Fallback (with memory)
# -------------------------
def groq_fallback(user_input):
    global chat_history

    try:
        messages = [
            {
                "role": "system",
                "content": """
You are a helpful assistant.

Rules:
- Use conversation history if helpful
- If unsure, say you don’t know
"""
            }
        ] + chat_history + [
            {"role": "user", "content": user_input}
        ]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024
        )

        response = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Groq Fallback Error: {e}")
        return "I'm having trouble connecting right now. Please try again."

    # Save history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    chat_history = chat_history[-10:]

    return response


# -------------------------
# Final Unified Function (RAG + Rephrase)
# -------------------------
def generate_response(user_input):
    try:
        kb_response = retrieve(user_input)

        if kb_response:
            print("✅ Answer found in Knowledge Base!")
            return rephrase_with_groq(user_input, kb_response)

        print("🤖 No KB match. Using LLM fallback...")
        return groq_fallback(user_input)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return "Something went wrong. Please try again."