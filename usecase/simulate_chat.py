import torch
import torch.nn as nn
import json
import re
import glob
import difflib

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

STOP_WORDS = {
    "what", "where", "when", "how", "is", "are",
    "do", "does", "did", "the", "a", "an",
    "you", "your", "i", "me", "about"
}

# -------------------------
# Load Vocabulary
# -------------------------
with open("../vocab2.json", "r") as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}
vocab_size = len(stoi)
print("Vocabulary loaded.")

# -------------------------
# Hyperparameters
# -------------------------
d_model = 128
n_heads = 4
n_layers = 2
dropout = 0.1
block_size = 128

# -------------------------
# Model Definition
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


# -------------------------
# Load model
# -------------------------
model = SimpleGPT().to(device)
model.load_state_dict(
    torch.load("../mini_llm_v2.pt", map_location=device, weights_only=True)
)
model.eval()
print("Model loaded successfully!")

# -------------------------
# Math computation
# -------------------------
def compute_math(expr):
    try:
        expr_clean = re.sub(r"[^0-9\+\-\*\/\.\s]", "", expr)
        return str(eval(expr_clean))
    except:
        return None


# -------------------------
# Text generation fallback
# -------------------------
def generate(prompt, max_new_tokens=100):
    context = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = prompt

    for _ in range(max_new_tokens):
        context_cond = context[:, -block_size:]
        with torch.no_grad():
            logits = model(context_cond)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        context = torch.cat((context, next_token), dim=1)
        next_char = itos[int(next_token)]
        generated_text += next_char

        if next_char == "\n" or next_char == ".":
            break

    # Extract everything after the '=' sign if present
    if '=' in generated_text:
        return generated_text.split('=')[-1].strip()
    return generated_text[len(prompt):]


# -------------------------
# Load Knowledge Base ONCE
# -------------------------
# def load_knowledge(path):
#     with open(path, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f.readlines() if line.strip()]
#
#     # Group into question-answer pairs
#     knowledge_pairs = []
#     for i in range(0, len(lines), 2):
#         if i + 1 < len(lines):
#             question = lines[i]
#             answer = lines[i + 1]
#             knowledge_pairs.append((question, answer))
#
#     return knowledge_pairs

def load_all_knowledge(folder_path):
    knowledge_entries = []
    files = glob.glob(folder_path + "/*.json")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Add category automatically from filename
            category = file.split("\\")[-1].replace(".json", "")

            for entry in data:
                entry["category"] = category
                knowledge_entries.append(entry)

    return knowledge_entries


knowledge_base = load_all_knowledge("C:\\Users\\santo\\PycharmProjects\\GPT\\knowledge-base")
print(f"Loaded {len(knowledge_base)} total knowledge entries.")


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def retrieve(query):
    query_norm = normalize(query)
    query_words = {
        w for w in query_norm.split()
        if w not in STOP_WORDS
    }

    best_match = None
    best_score = 0

    for entry in knowledge_base:
        question_norm = normalize(entry["question"])
        question_words = {
            w for w in question_norm.split()
            if w not in STOP_WORDS
        }

        keyword_words = set(entry.get("keywords", []))

        overlap_score = len(query_words & question_words)
        keyword_score = len(query_words & keyword_words) * 2

        total_score = overlap_score + keyword_score

        if total_score > best_score:
            best_score = total_score
            best_match = entry

    if best_score >= 1:
        return best_match["answer"]

    return None
# -------------------------
# Chat loop
# -------------------------
print("\nMini LLM Ready! Type 'exit' to quit.\n")

conversation_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break


    # Math detection (with optional spaces)
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", user_input):
        answer = compute_math(user_input)
        if answer is not None:
            print("Model:", answer)
            continue

    else:

        conversation_history.append(f"User: {user_input}")
        conversation_history.append("Bot:")

        # Keep only last N turns (to fit block_size)
        conversation_history = conversation_history[-8:]

        context = "\n".join(conversation_history)

        # Generate response using FULL context
        response = generate(context, max_new_tokens=150)

        print("Model:", response)

        # Append generated response properly
        conversation_history[-1] = f"Bot: {response}"