import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

REPO_ID = "Santoshki/mini-llm-alphav1"
MODEL_FILENAME = "mini_llm_v2.pt"
VOCAB_FILENAME = "vocab2.json"

d_model = 128
n_heads = 4
n_layers = 2
dropout = 0.1
block_size = 256

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
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
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
    def __init__(self, vocab_size, seq_len=None):
        super().__init__()
        self.seq_len = block_size if seq_len is None else seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

def load_model_from_hf():
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename=VOCAB_FILENAME, repo_type="model")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, repo_type="model")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    vocab_size = len(stoi)

    state = torch.load(model_path, map_location="cpu")
    actual_block_size = state["pos_emb.weight"].shape[0]
    global block_size
    block_size = actual_block_size

    model = SimpleGPT(vocab_size=vocab_size, seq_len=actual_block_size)
    model.load_state_dict(state)
    model.eval()
    return model, stoi, itos

def generate(model, prompt, stoi, itos, max_new_tokens=40, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_tokens = [stoi[ch] for ch in prompt]
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]

            logits = model(idx)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_char = itos[int(next_token.item())]
            generated += next_char
            idx = torch.cat([idx, next_token], dim=1)

    return generated

if __name__ == "__main__":
    model, stoi, itos = load_model_from_hf()
    print("Model loaded successfully. Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question or type 'quit' to exit.")
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        prompt = f"User: {question}\nAssistant: "

        result = generate(
            model,
            prompt,
            stoi,
            itos,
            max_new_tokens=60,
            temperature=0.5,
        )

        bot_reply = result[len(prompt):].strip()

        # Stop if the model begins another conversation turn.
        for marker in ("\nUser:", "\nAssistant:"):
            if marker in bot_reply:
                bot_reply = bot_reply.split(marker, 1)[0].strip()

        print(f"Bot: {bot_reply}\n")