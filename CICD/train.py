
import json
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Paths and training configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "training-data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

block_size = 128
batch_size = 32
learning_rate = 3e-4
max_iters = 5000
eval_interval = 250
eval_batches = 20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Training on: {device}")

# --------------------------------------------------
# 2. Load the text corpus
# --------------------------------------------------

files = [
    DATA_DIR / "conversation_memory.txt",
    DATA_DIR / "arithmetic.txt",
]

text_parts = []

for path in files:
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            text_parts.append(content)

if not text_parts:
    raise ValueError(
        "No training text found. Check training-data files."
    )

text = "\n\n".join(text_parts)

# Build a character-level vocabulary.
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor(
    [stoi[ch] for ch in text],
    dtype=torch.long,
)

if len(data) <= block_size * 2:
    raise ValueError(
        "Dataset is too small. Add more training text."
    )

# --------------------------------------------------
# 3. Split into training and validation sequences
# --------------------------------------------------

split_idx = int(0.9 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError(
        "Training or validation data is too small "
        "for the configured block_size."
    )

# --------------------------------------------------
# 4. Generate batches
# --------------------------------------------------

def get_batch(dataset):
    max_start = len(dataset) - block_size - 1

    ix = torch.randint(
        low=0,
        high=max_start + 1,
        size=(batch_size,),
    )

    x = torch.stack([
        dataset[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        dataset[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# 5. Initialize model and optimizer
# --------------------------------------------------

model = SimpleGPT(vocab_size=vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

# --------------------------------------------------
# 6. Estimate training and validation loss
# --------------------------------------------------

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for name, dataset in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = torch.zeros(eval_batches)

        for k in range(eval_batches):
            x, y = get_batch(dataset)
            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            losses[k] = loss.item()

        results[name] = losses.mean().item()

    model.train()
    return results


# --------------------------------------------------
# 7. Train the model and save the best checkpoint
# --------------------------------------------------

best_val_loss = float("inf")
checkpoint_path = OUTPUT_DIR / "mini_llm_v2.pt"
vocab_path = OUTPUT_DIR / "vocab2.json"

model.train()

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()

        print(
            f"Step {step:5d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"stoi": stoi, "itos": {
                        str(i): ch for i, ch in itos.items()
                    }},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    x, y = get_batch(train_data)

    logits = model(x)

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Helps prevent unusually large gradients.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    optimizer.step()

print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {checkpoint_path}")
print(f"Vocabulary saved to: {vocab_path}")
