from unsloth import FastLanguageModel
import os
import json
import time
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from transformers import T5TokenizerFast, T5EncoderModel
from safetensors.torch import save_file, load_file
import numpy as np

# ========== Configuration ==========
TXT_FILE_PATH = "/mnt/f/q5_xxs_training_script/100k_distilled_prompts.txt"
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl/"
QWEN3_MODEL_NAME = "/mnt/f/q5_xxs_training_script/q5-xxs-v11"

OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/q5-xxs-v12"
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512
EPOCHS = 1
LEARNING_RATE = 2e-4
GRAD_CLIP = 0.5
CLAMP_ETA = 2e-6
SAVE_EVERY_X_STEPS = 0
EVAL_EVERY_EPOCHS = 1
SAVE_BEST_MODEL = True
PRINT_EVERY_X_BATCHES = 10

# ========== Pre-tokenized Dataset Class ==========
class PreTokenizedDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, max_length, is_eval=False, sample_rate=0.1):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if is_eval:
            # Sample 10% of the dataset for evaluation
            lines = random.sample(lines, min(int(len(lines) * sample_rate), len(lines)))

        # Pre-tokenize entire dataset upfront
        self.student_input_ids = []
        self.student_attention_mask = []
        self.teacher_input_ids = []
        self.teacher_attention_mask = []

        for line in lines:
            student_inputs = student_tokenizer(
                line,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            teacher_inputs = teacher_tokenizer(
                line,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            self.student_input_ids.append(student_inputs["input_ids"])
            self.student_attention_mask.append(student_inputs["attention_mask"])
            self.teacher_input_ids.append(teacher_inputs["input_ids"])
            self.teacher_attention_mask.append(teacher_inputs["attention_mask"])

        # Convert to tensors
        self.student_input_ids = torch.tensor(self.student_input_ids, dtype=torch.long)
        self.student_attention_mask = torch.tensor(self.student_attention_mask, dtype=torch.long)
        self.teacher_input_ids = torch.tensor(self.teacher_input_ids, dtype=torch.long)
        self.teacher_attention_mask = torch.tensor(self.teacher_attention_mask, dtype=torch.long)

    def __len__(self):
        return len(self.student_input_ids)

    def __getitem__(self, idx):
        return (
            self.student_input_ids[idx],
            self.student_attention_mask[idx],
            self.teacher_input_ids[idx],
            self.teacher_attention_mask[idx]
        )


# ========== Projection Layer ==========
class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=4096):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)

# ========== Load Qwen3 Model ==========
print("Loading Qwen3 model...")
student_model, student_tokenizer = FastLanguageModel.from_pretrained(
    QWEN3_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    dtype=None,
    local_files_only=True,
    revision="main",
    full_finetuning=True
)

student_tokenizer.padding_side = "right"
student_tokenizer.truncation_side = "right"

teacher_tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_NAME)
teacher_tokenizer.padding_side = "right"
teacher_tokenizer.truncation_side = "right"

# Move to correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
student_model.to(device)

# ========== Load T5 Teacher Model ==========
print("Loading T5-xxl model...")
teacher_model = T5EncoderModel.from_pretrained(
    T5_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
teacher_model.eval()

# ========== Initialize or Load Projection Layer ==========
projection_path = os.path.join(QWEN3_MODEL_NAME, "projection_layer.safetensors")
if os.path.exists(projection_path):
    print("Loading existing projection layer from", projection_path)
    projection = ProjectionLayer(input_dim=1024, output_dim=4096)
    state_dict = load_file(projection_path)
    projection.load_state_dict(state_dict)
else:
    print("Initializing new projection layer")
    projection = ProjectionLayer(input_dim=1024, output_dim=4096)

projection.to(device, dtype=torch.bfloat16)

# ========== Dataset and Dataloader ==========
print(f"Loading and pre-tokenizing dataset from {TXT_FILE_PATH}...")
train_dataset = PreTokenizedDataset(
    TXT_FILE_PATH,
    student_tokenizer,
    teacher_tokenizer,
    MAX_SEQ_LENGTH
)
eval_dataset = PreTokenizedDataset(
    TXT_FILE_PATH,
    student_tokenizer,
    teacher_tokenizer,
    MAX_SEQ_LENGTH,
    is_eval=True
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=0
)

total_steps = EPOCHS * len(train_dataloader)

# ========== Mixed Precision Training Setup ==========
autocast_dtype = torch.bfloat16
scaler = GradScaler(enabled=False)

# Optimizer includes student model and projection
optimizer = torch.optim.AdamW(
    [p for p in student_model.parameters() if p.requires_grad] + list(projection.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=CLAMP_ETA,
)

# ========== Training Loop ==========
print("Starting training with mixed precision...")
student_model.train()

mse_loss = torch.nn.MSELoss(reduction='none')

start_time = time.time()
global_step = 0
best_loss = float('inf')

for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Unpack pre-tokenized batch
        s_input_ids, s_att_mask, t_input_ids, t_att_mask = batch
        s_input_ids = s_input_ids.to(device)
        s_att_mask = s_att_mask.to(device)
        t_input_ids = t_input_ids.to(device)
        t_att_mask = t_att_mask.to(device)

        # Mixed precision context
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            student_outputs = student_model(
                input_ids=s_input_ids,
                attention_mask=s_att_mask,
                output_hidden_states=True
            )
            student_hidden = student_outputs.hidden_states[-1]
            projected_student = projection(student_hidden)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=t_input_ids,
                attention_mask=t_att_mask
            )
            teacher_hidden = teacher_outputs.last_hidden_state
            teacher_hidden = teacher_hidden.to(device)

        elementwise_loss = mse_loss(projected_student, teacher_hidden)
        # Average over hidden dimension to get per-token loss
        tokenwise_loss = elementwise_loss.mean(dim=-1)
        # Apply mask and average over tokens
        loss = (tokenwise_loss * s_att_mask).sum() / s_att_mask.sum()

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping and optimizer step
        clip_grad_norm_(
            [p for p in student_model.parameters() if p.requires_grad] + list(projection.parameters()),
            max_norm=GRAD_CLIP
        )

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        global_step += 1

        # Save model at every N steps
        if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
            print(f"\nSaving checkpoint at step {global_step}")
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
            os.makedirs(save_path, exist_ok=True)
            student_model.save_pretrained(save_path)
            student_tokenizer.save_pretrained(save_path)
            projection_state = projection.state_dict()
            projection_path = os.path.join(save_path, "projection_layer.safetensors")
            save_file(projection_state, projection_path)

        # Print progress and time metrics
        if batch_idx % PRINT_EVERY_X_BATCHES == 0:
            elapsed = time.time() - start_time
            remaining_batches = total_steps - global_step
            eta = (elapsed / global_step) * remaining_batches if global_step > 0 else 0

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Batch [{batch_idx}/{len(train_dataloader)}], "
                f"Step: {global_step}/{total_steps}, "
                f"Loss: {loss.item():.6f}, "
                f"Elapsed: {elapsed/60:.1f} min, "
                f"ETA: {eta/60:.1f} min"
            )

    # Run evaluation every N epochs
    if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
        student_model.eval()
        total_eval_loss = 0.0
        total_elements = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                s_input_ids, s_att_mask, t_input_ids, t_att_mask = batch
                s_input_ids = s_input_ids.to(device)
                s_att_mask = s_att_mask.to(device)
                t_input_ids = t_input_ids.to(device)
                t_att_mask = t_att_mask.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    student_outputs = student_model(
                        input_ids=s_input_ids,
                        attention_mask=s_att_mask,
                        output_hidden_states=True
                    )
                    student_hidden = student_outputs.hidden_states[-1]
                    projected_student = projection(student_hidden)

                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=t_input_ids,
                        attention_mask=t_att_mask
                    )
                    teacher_hidden = teacher_outputs.last_hidden_state
                    teacher_hidden = teacher_hidden.to(device)

                # Compute loss with mask (same as training)
                elementwise_loss = mse_loss(projected_student, teacher_hidden)
                tokenwise_loss = elementwise_loss.mean(dim=-1)
                loss_val = (tokenwise_loss * s_att_mask).sum() / s_att_mask.sum()

                total_eval_loss += loss_val.item()
                total_elements += s_att_mask.sum().item()

            avg_eval_loss = total_eval_loss / len(eval_dataloader)  # Average per batch
            print(f"\n[Validation] Epoch {epoch + 1} | Average MSE Loss: {avg_eval_loss:.6f}")

            # Save best model based on validation loss
            if SAVE_BEST_MODEL and avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                print(f"\n✅ New best model at loss {best_loss:.6f}, saving...")
                best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                student_model.save_pretrained(best_model_dir)
                student_tokenizer.save_pretrained(best_model_dir)
                projection_state = projection.state_dict()
                projection_path = os.path.join(best_model_dir, "projection_layer.safetensors")
                save_file(projection_state, projection_path)

        student_model.train()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

student_model.save_pretrained(OUTPUT_DIR)
student_tokenizer.save_pretrained(OUTPUT_DIR)

# Save projection layer
projection_state = projection.state_dict()
projection_path = os.path.join(OUTPUT_DIR, "projection_layer.safetensors")
save_file(projection_state, projection_path)

# Save projection config
projection_config = {
    "input_dim": 1024,
    "output_dim": 4096,
    "dtype": "bfloat16",
}
projection_config_path = os.path.join(OUTPUT_DIR, "projection_config.json")
with open(projection_config_path, "w") as f:
    json.dump(projection_config, f)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
