import torch
import torch.nn as nn
import json
import os

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Hyperparameters (UPGRADED)
# -------------------------
d_model = 128          # Increased model size
n_heads = 4
n_layers = 2           # Deeper network
dropout = 0.1
block_size = 128       # Larger context window for conversation
batch_size = 32
learning_rate = 3e-4
max_iters = 5000      # Train longer
eval_interval = 500

# -------------------------
# Load Text Data (ADD conversation.txt)
# -------------------------
training_data_filepath = "C:\\Users\\santo\\PycharmProjects\\GPT\\training-data"

files = [
    training_data_filepath + "\\arithmetic.txt",
    training_data_filepath + "\\conversation_memory.txt"
]

text = ""
for file in files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            text += f.read() + "\n\n"
    else:
        print(f"Warning: {file} not found.")

print("Dataset length:", len(text))

# -------------------------
# Build Vocabulary (Character-level)
# -------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Save vocab
with open("../vocab.json", "w", encoding="utf-8") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# Train/Val split
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# -------------------------
# Batch Loader
# -------------------------
def get_batch(split_name):
    dataset = train_data if split_name == "train" else val_data

    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

# -------------------------
# Model Components
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
        scores = scores.masked_fill(mask == 0, float('-inf'))

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
            nn.Dropout(dropout)
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
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

# -------------------------
# Initialize Model
# -------------------------
model = SimpleGPT().to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Training Loop
# -------------------------
for step in range(max_iters):
    model.train()

    x, y = get_batch("train")
    logits = model(x)

    B, T, C = logits.shape
    loss = loss_fn(logits.view(B * T, C), y.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# -------------------------
# Save Model
# -------------------------
torch.save(model.state_dict(), "../mini_llm.pt")
print("\nTraining complete. Model saved as mini_llm.pt")