import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import default_collate
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-0.6B-base"  # Can change to "Qwen3-0.6B-base" when available
BATCH_SIZE = 1
MAX_LENGTH = 256
LR = 5e-5  # Conservative LR for LoRA + reward modeling
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # important for padding

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# LoRA configuration and integration
lora_config = LoraConfig(
    r=8,                # Rank of LoRA matrices (common values: 4, 8, 16)
    lora_alpha=16,      # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Common for transformer models; adjust if needed
    lora_dropout=0.05,  # Dropout for LoRA layers
    bias="none",
    task_type="CAUSAL_LM"
)
base_model = get_peft_model(base_model, lora_config)

# Reward model = base + scalar reward head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.v_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
        # Ensure v_head is on the same device and dtype as base_model
        self.v_head = self.v_head.to(dtype=base_model.dtype, device=base_model.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        rewards = self.v_head(last_hidden).squeeze(-1)  # [batch, seq]
        # We take the reward at the final non-padding token
        mask_sum = attention_mask.sum(dim=1) - 1  # index of last non-pad token
        chosen_rewards = rewards[range(rewards.size(0)), mask_sum]
        return chosen_rewards

# Wrap the base model
model = RewardModel(base_model).to(device)

# Freeze original base model parameters but keep LoRA adapters trainable
for name, param in model.base_model.named_parameters():
    # Freeze original model parameters (not LoRA)
    if "lora" not in name.lower():
        param.requires_grad = False
    else:
        # Keep LoRA parameters trainable
        param.requires_grad = True

# After model creation, check trainable parameters and LoRA modules
print("\n--- Trainable Parameters ---")
trainable_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}, shape: {param.shape}")
        trainable_count += param.numel()
print(f"Total trainable parameters: {trainable_count:,}")
print("---------------------------\n")

# List LoRA modules in the model
print("--- LoRA Modules in Model ---")
lora_params_count = 0
for name, module in model.named_modules():
    if "lora" in name.lower() or "lora" in str(type(module)).lower():
        print(f"LoRA module: {name}, type: {type(module)}")
        # Check if this module has trainable parameters
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                print(f"  Trainable LoRA param: {param_name}, shape: {param.shape}")
                lora_params_count += param.numel()
print(f"Total trainable LoRA parameters: {lora_params_count:,}")
print("-----------------------------\n")

# Print all module names to help user check target_modules
print("--- All Module Names in Base Model ---")
for name, module in model.base_model.named_modules():
    print(name, type(module))
print("-----------------------------\n")

# Load and filter dataset
full_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train")

def keyword_filter(example):
    keywords = ["friendship", "relationship", "mental health"]
    combined_text = (example["prompt"] + example["chosen"] + example["rejected"]).lower()
    return any(kw in combined_text for kw in keywords)

filtered_dataset = full_dataset.filter(keyword_filter)

if len(filtered_dataset) == 0:
    print("No examples found with the given keywords!")
    exit()

split_ratio = 0.9
split = filtered_dataset.train_test_split(test_size=1 - split_ratio)
train_dataset = split["train"]
test_dataset = split["test"]

# Subset for faster experimentation
train_dataset = train_dataset.select(range(1000))
test_dataset = test_dataset.select(range(100))

# Preprocessing
def preprocess(example):
    prompt = example["prompt"]
    chosen = prompt + example["chosen"]
    rejected = prompt + example["rejected"]

    chosen_tokens = tokenizer(chosen, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    rejected_tokens = tokenizer(rejected, truncation=True, padding="max_length", max_length=MAX_LENGTH)

    return {
        "chosen_input_ids": list(chosen_tokens["input_ids"]),
        "chosen_attention_mask": list(chosen_tokens["attention_mask"]),
        "rejected_input_ids": list(rejected_tokens["input_ids"]),
        "rejected_attention_mask": list(rejected_tokens["attention_mask"]),
    }

processed_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names, num_proc=1)
processed_test = test_dataset.map(preprocess, remove_columns=test_dataset.column_names, num_proc=1)

# After preprocessing, print a few examples to check data correctness
for i in range(3):
    raw = train_dataset[i]
    print(f"Example {i}")
    print("PROMPT:", raw["prompt"])
    print("CHOSEN:", raw["chosen"])
    print("REJECTED:", raw["rejected"])
    print("---")

dataloader = DataLoader(processed_train, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer with different learning rates for different components
param_groups = []
# LoRA parameters - lower LR
lora_params = [p for name, p in model.named_parameters() if 'lora' in name.lower()]
if lora_params:
    param_groups.append({'params': lora_params, 'lr': LR * 0.5})  # 2.5e-5 for LoRA

# v_head parameters - higher LR
v_head_params = [p for name, p in model.named_parameters() if 'v_head' in name.lower()]
if v_head_params:
    param_groups.append({'params': v_head_params, 'lr': LR * 2})  # 1e-4 for v_head

optimizer = AdamW(param_groups)

# Training Loop
model.train()
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0
    batch_iter = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch in batch_iter:
        chosen_ids = torch.tensor(batch["chosen_input_ids"])
        if chosen_ids.dim() == 1:
            chosen_ids = chosen_ids.unsqueeze(0)
        chosen_ids = chosen_ids.to(device)
        chosen_mask = torch.tensor(batch["chosen_attention_mask"])
        if chosen_mask.dim() == 1:
            chosen_mask = chosen_mask.unsqueeze(0)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = torch.tensor(batch["rejected_input_ids"])
        if rejected_ids.dim() == 1:
            rejected_ids = rejected_ids.unsqueeze(0)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = torch.tensor(batch["rejected_attention_mask"])
        if rejected_mask.dim() == 1:
            rejected_mask = rejected_mask.unsqueeze(0)
        rejected_mask = rejected_mask.to(device)

        optimizer.zero_grad()
        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)

        # Print reward values for debugging (first few batches)
        if batch_iter.n < 5:  # Only print first 5 batches
            print(f"Batch {batch_iter.n}: chosen_reward={chosen_rewards.item():.4f}, rejected_reward={rejected_rewards.item():.4f}, diff={chosen_rewards.item() - rejected_rewards.item():.4f}")

        # Pairwise logistic loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        loss.backward()
        # Print gradients for v_head and LoRA
        print("--- Gradients after backward() ---")
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Grad {name}: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}")
        print("-----------------------------\n")
        optimizer.step()

        epoch_loss += loss.item()
        batch_iter.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")

    # --- Pairwise Accuracy Evaluation ---
    model.eval()
    correct = 0
    total = 0
    test_loader = DataLoader(processed_test, batch_size=1)
    with torch.no_grad():
        for test_batch in test_loader:
            chosen_ids = torch.tensor(test_batch["chosen_input_ids"])
            if chosen_ids.dim() == 1:
                chosen_ids = chosen_ids.unsqueeze(0)
            chosen_ids = chosen_ids.to(device)
            chosen_mask = torch.tensor(test_batch["chosen_attention_mask"])
            if chosen_mask.dim() == 1:
                chosen_mask = chosen_mask.unsqueeze(0)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = torch.tensor(test_batch["rejected_input_ids"])
            if rejected_ids.dim() == 1:
                rejected_ids = rejected_ids.unsqueeze(0)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = torch.tensor(test_batch["rejected_attention_mask"])
            if rejected_mask.dim() == 1:
                rejected_mask = rejected_mask.unsqueeze(0)
            rejected_mask = rejected_mask.to(device)
            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)
            correct += (chosen_reward > rejected_reward).sum().item()
            total += chosen_reward.size(0)
    pairwise_acc = correct / total if total > 0 else 0.0
    print(f"Pairwise Accuracy on Test Set: {pairwise_acc:.4f}")
    model.train()

# Save model
torch.save(model.state_dict(), "qwen_reward_model.pt")
print("✅ Reward model saved as qwen_reward_model.pt")

# After training, print a few examples of chosen/rejected pairs and their rewards
model.eval()
test_loader = DataLoader(processed_test, batch_size=1)
with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        if i >= 3:
            break
        chosen_ids = torch.tensor(test_batch["chosen_input_ids"])
        if chosen_ids.dim() == 1:
            chosen_ids = chosen_ids.unsqueeze(0)
        chosen_ids = chosen_ids.to(device)
        chosen_mask = torch.tensor(test_batch["chosen_attention_mask"])
        if chosen_mask.dim() == 1:
            chosen_mask = chosen_mask.unsqueeze(0)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = torch.tensor(test_batch["rejected_input_ids"])
        if rejected_ids.dim() == 1:
            rejected_ids = rejected_ids.unsqueeze(0)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = torch.tensor(test_batch["rejected_attention_mask"])
        if rejected_mask.dim() == 1:
            rejected_mask = rejected_mask.unsqueeze(0)
        rejected_mask = rejected_mask.to(device)
        chosen_reward = model(chosen_ids, chosen_mask)
        rejected_reward = model(rejected_ids, rejected_mask)
        # Robust scalar extraction
        if chosen_reward.numel() == 1:
            chosen_reward = chosen_reward.item()
        else:
            print("[WARN] chosen_reward has more than 1 element, taking the last value.")
            chosen_reward = chosen_reward[-1].item()
        if rejected_reward.numel() == 1:
            rejected_reward = rejected_reward.item()
        else:
            print("[WARN] rejected_reward has more than 1 element, taking the last value.")
            rejected_reward = rejected_reward[-1].item()
        raw = test_dataset[i]
        print(f"Test Example {i}")
        print("PROMPT:", raw["prompt"])
        print("CHOSEN:", raw["chosen"])
        print("REJECTED:", raw["rejected"])
        print(f"CHOSEN REWARD: {chosen_reward:.4f}")
        print(f"REJECTED REWARD: {rejected_reward:.4f}")
        print("---")

test_batch = next(iter(DataLoader(processed_test, batch_size=1)))
test_ids = torch.tensor(test_batch["chosen_input_ids"])
if test_ids.dim() == 1:
    test_ids = test_ids.unsqueeze(0)
test_ids = test_ids.to(device)
test_mask = torch.tensor(test_batch["chosen_attention_mask"])
if test_mask.dim() == 1:
    test_mask = test_mask.unsqueeze(0)
test_mask = test_mask.to(device)
with torch.no_grad():
    rewards = model(test_ids, test_mask)
print("Test batch rewards:", rewards)