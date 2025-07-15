import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import default_collate
from peft import LoraConfig, get_peft_model
import wandb
import json
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = "Qwen/Qwen1.5-0.5B"  # Can change to "Qwen3-0.6B-base" when available
BATCH_SIZE = 8
MAX_LENGTH = 512
LR = 1e-7  # Conservative LR for LoRA + reward modeling
EPOCHS = 50

wandb.init(
    entity="Week6",
    project="Summarise",  # You can change this project name
    config={
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LR,
        "epochs": EPOCHS,
    }
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # important for padding

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

def init_lora_weights(module):
    # Initialize LoRA adapter weights with small normal values
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        nn.init.normal_(module.lora_A.default.weight, mean=0.0, std=1e-3)
        nn.init.normal_(module.lora_B.default.weight, mean=0.0, std=1e-3)

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
# Apply custom LoRA initialization
base_model.apply(init_lora_weights)

# Reward model = base + scalar reward head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.v_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)  # Set bias=False
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

# Freeze original base model parameters but keep LoRA adapters and last 8 transformer blocks trainable
for name, param in model.base_model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    elif any(f"layers.{i}" in name for i in range(23, 15, -1)):
        param.requires_grad = True
    else:
        param.requires_grad = False

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

# Remove keyword filtering, use full dataset
full_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train")
filtered_dataset = full_dataset  # No filtering

split_ratio = 0.9
split = filtered_dataset.train_test_split(test_size=1 - split_ratio)
train_dataset = split["train"]
test_dataset = split["test"]

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Subset for faster experimentation
train_dataset = train_dataset.select(range(50000))
test_dataset = test_dataset.select(range(1000))

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
    param_groups.append({'params': lora_params, 'lr': LR})  # 1e-5 for LoRA

# v_head parameters - lower LR (no multiplier)
v_head_params = [p for name, p in model.named_parameters() if 'v_head' in name.lower()]
if v_head_params:
    param_groups.append({'params': v_head_params, 'lr': LR})  # 2e-5 for v_head

optimizer = AdamW(param_groups, weight_decay=0.05)

# --- Early stopping and best model saving setup ---
best_loss = float('inf')
epochs_no_improve = 0
patience = 2  # Stop if no improvement for 2 consecutive epochs

ACCUMULATION_STEPS = 4  # Simulate larger batch size

def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    # If x is a list of tensors, stack them
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)

# Training Loop
model.train()
margin = 0.05  # Smaller margin for weak separation and noisy data
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0
    batch_iter = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    optimizer.zero_grad()
    for step, batch in enumerate(batch_iter):
        chosen_ids = ensure_tensor(batch["chosen_input_ids"]).to(device)
        chosen_mask = ensure_tensor(batch["chosen_attention_mask"]).to(device)
        rejected_ids = ensure_tensor(batch["rejected_input_ids"]).to(device)
        rejected_mask = ensure_tensor(batch["rejected_attention_mask"]).to(device)

        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)

        # Print reward values for debugging (first few batches)
        if batch_iter.n < 5:  # Only print first 5 batches
            for i in range(chosen_rewards.shape[0]):
                print(f"Batch {batch_iter.n} Example {i}: chosen_reward={chosen_rewards[i].item():.4f}, rejected_reward={rejected_rewards[i].item():.4f}, diff={(chosen_rewards[i] - rejected_rewards[i]).item():.4f}")

        # Pairwise logistic loss with margin
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin)).mean()
        # L2 penalty on reward outputs
        l2_lambda = 0.001
        reward_outputs = torch.cat([chosen_rewards, rejected_rewards])
        l2_penalty = l2_lambda * (reward_outputs ** 2).mean()
        loss = loss + l2_penalty
        loss = loss / ACCUMULATION_STEPS  # Normalize loss for accumulation
        loss.backward()
        # Gradient clipping to prevent collapse (clip after accumulation)
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(batch_iter):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Print gradients for v_head and LoRA and unfrozen base model layers only
            print("--- Gradients after backward() ---")
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"Grad {name}: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}")
            print("-----------------------------\n")
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item() * ACCUMULATION_STEPS  # Undo normalization for reporting
        batch_iter.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")

    # --- Pairwise Accuracy Evaluation ---
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    test_loader = DataLoader(processed_test, batch_size=1)
    with torch.no_grad():
        for test_batch in test_loader:
            chosen_ids = ensure_tensor(test_batch["chosen_input_ids"]).to(device)
            chosen_mask = ensure_tensor(test_batch["chosen_attention_mask"]).to(device)
            rejected_ids = ensure_tensor(test_batch["rejected_input_ids"]).to(device)
            rejected_mask = ensure_tensor(test_batch["rejected_attention_mask"]).to(device)
            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)
            correct += (chosen_reward > rejected_reward).sum().item()
            total += chosen_reward.size(0)
            # Compute test loss for early stopping (with margin)
            test_loss += -torch.log(torch.sigmoid(chosen_reward - rejected_reward - margin)).mean().item()
    pairwise_acc = correct / total if total > 0 else 0.0
    avg_test_loss = test_loss / len(test_loader)
    print(f"Pairwise Accuracy on Test Set: {pairwise_acc:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    model.train()

    # Log v_head.weight gradient mean and std (from last batch of epoch)
    v_head_grad = None
    for name, param in model.named_parameters():
        if param.requires_grad and "v_head.weight" in name and param.grad is not None:
            v_head_grad = param.grad
            break
    if v_head_grad is not None:
        v_head_grad_mean = v_head_grad.mean().item()
        v_head_grad_std = v_head_grad.std().item()
    else:
        v_head_grad_mean = None
        v_head_grad_std = None

    wandb.log({
        "train_loss": avg_loss,
        "pairwise_accuracy": pairwise_acc,
        "test_loss": avg_test_loss,
        "v_head_grad_mean": v_head_grad_mean,
        "v_head_grad_std": v_head_grad_std,
        "epoch": epoch + 1,
    })

    # --- Loss-based Early Stopping ---
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "qwen_reward_model_best.pt")
        print("✅ Best model saved as qwen_reward_model_best.pt")
    else:
        epochs_no_improve += 1
        print(f"No improvement in test loss for {epochs_no_improve} epoch(s). Best loss: {best_loss:.4f}")
    # Early stopping is disabled: do not break the loop
    # if epochs_no_improve >= patience:
    #     print(f"⏹️ Early stopping: No improvement in test loss for {patience} consecutive epochs.")
    #     break

# Save model
torch.save(model.state_dict(), "qwen_reward_model.pt")
print("✅ Reward model saved as qwen_reward_model.pt")

# Save LoRA config if used
if lora_config is not None:
    # Only save serializable fields
    def make_json_serializable(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, set):
                out[k] = list(v)
            else:
                out[k] = v
        return out
    lora_config_dict = {k: v for k, v in lora_config.__dict__.items() if not k.startswith('_')}
    lora_config_dict = make_json_serializable(lora_config_dict)
    with open("lora_config.json", "w") as f:
        json.dump(lora_config_dict, f)
    print("✅ LoRA config saved as lora_config.json")

# After training, print a few examples of chosen/rejected pairs and their rewards
model.eval()
test_loader = DataLoader(processed_test, batch_size=1)
with torch.no_grad():
    chosen_rewards_list = []
    rejected_rewards_list = []
    for i, test_batch in enumerate(test_loader):
        if i >= 3:
            break
        chosen_ids = ensure_tensor(test_batch["chosen_input_ids"]).to(device)
        chosen_mask = ensure_tensor(test_batch["chosen_attention_mask"]).to(device)
        rejected_ids = ensure_tensor(test_batch["rejected_input_ids"]).to(device)
        rejected_mask = ensure_tensor(test_batch["rejected_attention_mask"]).to(device)
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
        chosen_rewards_list.append(chosen_reward.cpu().numpy())
        rejected_rewards_list.append(rejected_reward.cpu().numpy())

plt.hist(np.concatenate(chosen_rewards_list), bins=50, alpha=0.5, label='chosen')
plt.hist(np.concatenate(rejected_rewards_list), bins=50, alpha=0.5, label='rejected')
plt.legend()
plt.title('Reward Distribution')
plt.show()

test_batch = next(iter(DataLoader(processed_test, batch_size=1)))
test_ids = ensure_tensor(test_batch["chosen_input_ids"]).to(device)
test_mask = ensure_tensor(test_batch["chosen_attention_mask"]).to(device)
with torch.no_grad():
    rewards = model(test_ids, test_mask)
print("Test batch rewards:", rewards)


wandb.finish()

# --- Reward Model Loader Wrapper ---
def load_reward_model(model_path, base_model_name, lora_config_path=None, device="cpu"):
    """
    Loads the reward model with base, v_head, and LoRA config (if provided).
    """
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    import torch
    import json

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    if lora_config_path is not None:
        with open(lora_config_path, "r") as f:
            lora_cfg_dict = json.load(f)
        lora_cfg = LoraConfig(**lora_cfg_dict)
        base_model = get_peft_model(base_model, lora_cfg)
    model = RewardModel(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Example usage (uncomment to test):
# reward_model = load_reward_model("qwen_reward_model_best.pt", "Qwen/Qwen1.5-0.5B", "lora_config.json", device="cuda")