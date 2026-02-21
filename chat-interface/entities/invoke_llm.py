import torch
import torch.nn as nn
import json
import re
from huggingface_hub import hf_hub_download

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Download model and vocab from Hugging Face
# -------------------------
repo_id = "Santoshki/mini-llm-alphav1"
model_path = hf_hub_download(repo_id=repo_id, filename="mini_llm.pt")
vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json")

# -------------------------
# Load vocabulary
# -------------------------
with open(vocab_path, "r") as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}
vocab_size = len(stoi)

# -------------------------
# Hyperparameters (must match training)
# -------------------------
d_model = 128
n_heads = 4
n_layers = 2
dropout = 0.1
block_size = 32

# -------------------------
# Model definition
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
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# -------------------------
# Math computation fallback
# -------------------------
def compute_math(expr):
    try:
        expr_clean = re.sub(r"[^0-9\+\-\*\/\.\s]", "", expr)
        return str(eval(expr_clean))
    except:
        return None


# -------------------------
# Generate function (argmax, like working chat)
# -------------------------
def get_model_response(prompt, max_new_tokens=100):
    """
    Input: prompt (str)
    Output: model response (str)
    """
    # Math fallback
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", prompt):
        answer = compute_math(prompt)
        if answer is not None:
            return answer

    # Otherwise model generation
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
        if next_char == "\n":
            break

    return generated_text[len(prompt):]  # return only the generated part
