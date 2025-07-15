import sys
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B-base"  # Or your current model

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load dataset
print("Loading dataset...")
train_dataset = load_dataset("Dahoas/rm-static", split="train")

lengths = []
N = 100  # Number of samples to check
print(f"Analyzing the first {N} samples...")
for i in range(min(N, len(train_dataset))):
    prompt = train_dataset[i]["prompt"]
    chosen = prompt + train_dataset[i]["chosen"]
    tokens = tokenizer(chosen, truncation=False)["input_ids"]
    lengths.append(len(tokens))

print("Average length:", sum(lengths)/len(lengths))
print("Max length:", max(lengths))
print("Min length:", min(lengths))
print("Lengths (first 20):", lengths[:20]) 