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
from tqdm import tqdm
import queue
import threading

# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt" # Format: one prompt per line of the txt file
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/q5_xxs_training_script/final-q5-xxs-v2/checkpoint_step_6000"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/final-q5-xxs-v3/"

USE_CACHED_EMBEDDINGS = True # If you cache the embeddings, T5-xxl won't be loaded when training and we'll pull from the cache instead. Embeddings are saved as bfloat16 and are around 4MB in size for each prompt in the dataset
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2" # The cache files will be kept and should be picked up on subsequent runs by referencing the dataset base name

USE_SEPARATE_EVALUATION_DATASET = True # If disabled, pulls 10% of the main dataset, but using unseen data is a better test of generalisation
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
MIN_LR = 5e-5
SAVE_EVERY_X_STEPS = 2000
EVAL_EVERY_EPOCHS = 1
SAVE_BEST_MODEL = True
PRINT_EVERY_X_BATCHES = 1
GRAD_ACCUM_STEPS = 1
HYBRID_LOSS = 0.50 # Increasing this will add cosine loss: 1.0 = full cosine; 0.0 = full Huber. Around 0.5 is good

ADAPTIVE_LOSS_RATIO = True # Will shift the hybrid loss ratio gradually to account for unbalanced loss
ADAPTIVE_LOSS_HUBER_THRESHOLD = 80.0 # If cosine loss if over n times the huber loss, then huber loss alignment is dominating
ADAPTIVE_LOSS_COS_THRESHOLD = 40.0 # If cosine loss is under n times the huber loss, then cosine loss alignment is dominating
ADAPTIVE_LOSS_RATIO_UPPER_BOUND = 0.8 # Highest bound of hybrid loss ratio scaling
ADAPTIVE_LOSS_RATIO_LOWER_BOUND = 0.2 # Lowest bound of hybrid loss ratio scaling
ADAPTIVE_LOSS_ADJUSTMENT_FACTOR = 0.005 # Adjustment addition/subtraction applied to hybrid loss when unbalanced loss

# ========== Dataset Class ==========
class PreTokenizedDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, max_length, teacher_model=None, is_eval=False, sample_rate=0.1, use_cached_embeddings=False, cache_path=None):
        if USE_SEPARATE_EVALUATION_DATASET and is_eval:
            file_path = EVALUATION_DATASET_PATH
            sample_rate = None

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if is_eval and sample_rate is not None:
            lines = random.sample(lines, min(int(len(lines) * sample_rate), len(lines)))

        self.student_input_ids = []
        self.student_attention_mask = []
        for line in lines:
            student_inputs = student_tokenizer(
                line,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            self.student_input_ids.append(student_inputs["input_ids"])
            self.student_attention_mask.append(student_inputs["attention_mask"])

        self.student_input_ids = torch.tensor(self.student_input_ids, dtype=torch.long)
        self.student_attention_mask = torch.tensor(self.student_attention_mask, dtype=torch.long)

        if use_cached_embeddings:
            base_name = os.path.basename(file_path)
            cache_folder = os.path.join(cache_path, base_name)
            validation_file = os.path.join(cache_folder, f"{base_name}.validation")

            if os.path.exists(validation_file):
                print(f"Loading cached embeddings from folder {cache_folder}")
                self.cache_folder = cache_folder
                self.num_samples = len(lines)
                self.hidden_dim = 4096
            else:
                print(f"Generating and caching embeddings for {file_path}")
                os.makedirs(cache_folder, exist_ok=True)
                self.num_samples = len(lines)
                self.hidden_dim = teacher_model.config.hidden_size

                for i, line in enumerate(tqdm(lines, desc="Generating embeddings")):
                    teacher_inputs = teacher_tokenizer(
                        line,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length
                    )
                    input_ids = torch.tensor(teacher_inputs["input_ids"], dtype=torch.long).unsqueeze(0)
                    att_mask = torch.tensor(teacher_inputs["attention_mask"], dtype=torch.long).unsqueeze(0)

                    with torch.no_grad():
                        outputs = teacher_model(
                            input_ids=input_ids.to(teacher_model.device),
                            attention_mask=att_mask.to(teacher_model.device)
                        )
                        embeddings = outputs.last_hidden_state.cpu()

                    # Save both embeddings and attention mask
                    embedding_file = os.path.join(cache_folder, f"{i}.pt")
                    mask_file = os.path.join(cache_folder, f"{i}_mask.pt")
                    torch.save(embeddings, embedding_file)
                    torch.save(att_mask.cpu(), mask_file)  # Cache the attention mask

                with open(validation_file, "w") as f:
                    pass
                print(f"Saved embeddings to folder {cache_folder}")
                self.cache_folder = cache_folder
        else:
            self.teacher_input_ids = []
            self.teacher_attention_mask = []
            for line in lines:
                teacher_inputs = teacher_tokenizer(
                    line,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                self.teacher_input_ids.append(teacher_inputs["input_ids"])
                self.teacher_attention_mask.append(teacher_inputs["attention_mask"])

            self.teacher_input_ids = torch.tensor(self.teacher_input_ids, dtype=torch.long)
            self.teacher_attention_mask = torch.tensor(self.teacher_attention_mask, dtype=torch.long)

        self.use_cached_embeddings = use_cached_embeddings
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.prefetch_thread = None
        self.stop_thread = False
        self.current_batch_idx = 0

    def __len__(self):
        return len(self.student_input_ids)

    def __getitem__(self, idx):
        if self.use_cached_embeddings:
            embedding_file = os.path.join(self.cache_folder, f"{idx}.pt")
            mask_file = os.path.join(self.cache_folder, f"{idx}_mask.pt")  # Load the mask file
            embeddings = torch.load(embedding_file)
            att_mask = torch.load(mask_file)  # Load the attention mask
            return (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                embeddings,
                att_mask  # Return the attention mask
            )
        else:
            return (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                self.teacher_input_ids[idx],
                self.teacher_attention_mask[idx]
            )

    def start_prefetching(self, batch_size):
        if self.use_cached_embeddings:
            self.prefetch_thread = threading.Thread(target=self._prefetch_batch, args=(batch_size,))
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()

    def stop_prefetching(self):
        self.stop_thread = True
        if self.prefetch_thread:
            self.prefetch_thread.join()

    def _prefetch_batch(self, batch_size):
        while not self.stop_thread:
            start_idx = self.current_batch_idx
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = list(range(start_idx, end_idx))

            embeddings = []
            masks = []
            for idx in batch_indices:
                embedding_file = os.path.join(self.cache_folder, f"{idx}.pt")
                mask_file = os.path.join(self.cache_folder, f"{idx}_mask.pt")
                embed = torch.load(embedding_file)
                mask = torch.load(mask_file)

                embeddings.append(embed.to(torch.bfloat16))
                masks.append(mask.to(torch.bfloat16))

            s_input_ids = self.student_input_ids[batch_indices]
            s_att_mask = self.student_attention_mask[batch_indices]
            t_embeddings = torch.stack(embeddings)
            t_att_masks = torch.stack(masks).squeeze(1)

            try:
                self.prefetch_queue.put((s_input_ids, s_att_mask, t_embeddings, t_att_masks), timeout=0.1)
            except queue.Full:
                continue

            self.current_batch_idx = end_idx
            if self.current_batch_idx >= len(self):
                self.current_batch_idx = 0

    def get_prefetched_batch(self):
        return self.prefetch_queue.get()

# ========== Projection Layer ==========
class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim=1024, intermediate_dim=4096, output_dim=4096):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(intermediate_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='linear')
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)

# ========== Loss Function ==========
class SequenceLevelHybridLoss(torch.nn.Module):
    def __init__(self, lambda_weight=0.0, student_hidden_size=None, teacher_hidden_size=None,
                 dynamic_ratio=False, lambda_decay=0.9999):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.dynamic_ratio = dynamic_ratio
        self.lambda_decay = lambda_decay
        self.current_lambda = lambda_weight

        # Learnable pooling layers with bfloat16 dtype
        if student_hidden_size is not None and teacher_hidden_size is not None:
            self.student_pooler = torch.nn.Linear(student_hidden_size, 1).to(torch.bfloat16)
            self.teacher_pooler = torch.nn.Linear(teacher_hidden_size, 1).to(torch.bfloat16)
        else:
            raise ValueError("Hidden sizes must be specified for learnable pooling")

        self.huber = torch.nn.HuberLoss(reduction='none')
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output, teacher_output, student_mask, teacher_mask):
        # Ensure all inputs are in bfloat16 and on the same device
        device = student_output.device
        dtype = torch.bfloat16

        student_output = student_output.to(dtype)
        teacher_output = teacher_output.to(dtype)
        student_mask = student_mask.to(dtype).to(device)
        teacher_mask = teacher_mask.to(dtype).to(device)

        # Learn attention weights for pooling
        student_logits = self.student_pooler(student_output).squeeze(-1)
        teacher_logits = self.teacher_pooler(teacher_output).squeeze(-1)

        # Apply masks (set masked positions to large negative value)
        student_logits = student_logits.masked_fill(~student_mask.bool(), -1e9)
        teacher_logits = teacher_logits.masked_fill(~teacher_mask.bool(), -1e9)

        student_weights = torch.softmax(student_logits, dim=1)
        teacher_weights = torch.softmax(teacher_logits, dim=1)

        # Weighted pooling
        student_pooled = (student_output * student_weights.unsqueeze(-1)).sum(dim=1)
        teacher_pooled = (teacher_output * teacher_weights.unsqueeze(-1)).sum(dim=1)

        # Compute losses on pooled representations
        huber_loss = self.huber(student_pooled, teacher_pooled).mean()
        cos_sim = self.cos(student_pooled, teacher_pooled)
        cos_loss = (1 - cos_sim).mean()

        # Dynamic loss ratio adjustment
        if self.dynamic_ratio:
            # Adjust lambda based on relative loss magnitudes
            loss_ratio = cos_loss / (huber_loss + 1e-8)
            adjustment_factor = ADAPTIVE_LOSS_ADJUSTMENT_FACTOR  # Small adjustment factor

            if loss_ratio > ADAPTIVE_LOSS_HUBER_THRESHOLD:  # Huber loss is dominating - increase cosine influence
                self.current_lambda = min(ADAPTIVE_LOSS_RATIO_UPPER_BOUND, self.current_lambda + adjustment_factor)
            elif loss_ratio < ADAPTIVE_LOSS_COS_THRESHOLD: # Cosine loss is dominating - decrease cosine influence
                self.current_lambda = max(ADAPTIVE_LOSS_RATIO_LOWER_BOUND, self.current_lambda - adjustment_factor)
            elif loss_ratio > original_loss_ratio:
                loss_ratio -= adjustment_factor
            elif loss_ratio < original_loss_ratio:
                loss_ratio += adjustment_factor

        # Combine losses
        hybrid_loss = (1 - self.current_lambda) * huber_loss + self.current_lambda * cos_loss

        return hybrid_loss, cos_loss, huber_loss, self.current_lambda

    def get_current_lambda(self):
        return self.current_lambda

# ========== Load Qwen3 Model ==========
print("Loading Qwen3 model...")
student_model, student_tokenizer = FastLanguageModel.from_pretrained(
    QWEN3_MODEL_NAME,
    max_seq_length=512,
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

device = "cuda" if torch.cuda.is_available() else "cpu"
student_model.to(device)

# ========== Load T5 Teacher Model ==========
teacher_model = None
if not USE_CACHED_EMBEDDINGS:
    print("Loading T5-xxl model...")
    teacher_model = T5EncoderModel.from_pretrained(
        T5_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    teacher_model.eval()
else:
    base_name = os.path.basename(DATASET_PATH)
    cache_folder = os.path.join(CACHE_PATH, base_name)
    validation_file = os.path.join(cache_folder, f"{base_name}.validation")
    if not os.path.exists(validation_file):
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
    try:
        state_dict = load_file(projection_path)
        projection = ProjectionLayer(input_dim=1024, intermediate_dim=4096, output_dim=4096)
        projection.load_state_dict(state_dict)
    except:
        print("Incompatible projection layer detected. Initializing new projection layer")
        projection = ProjectionLayer(input_dim=1024, intermediate_dim=4096, output_dim=4096)
else:
    print("Initializing projection layer")
    projection = ProjectionLayer(input_dim=1024, intermediate_dim=4096, output_dim=4096)

projection.to(device, dtype=torch.bfloat16)

hybrid_loss = SequenceLevelHybridLoss(
    lambda_weight=HYBRID_LOSS,
    student_hidden_size=4096,
    teacher_hidden_size=4096,
    dynamic_ratio=ADAPTIVE_LOSS_RATIO
).to(device, dtype=torch.bfloat16)

# ========== Dataset and Dataloader ==========
train_dataset = PreTokenizedDataset(
    DATASET_PATH,
    student_tokenizer,
    teacher_tokenizer,
    512,
    teacher_model=teacher_model,
    is_eval=False,
    sample_rate=None,
    use_cached_embeddings=USE_CACHED_EMBEDDINGS,
    cache_path=CACHE_PATH
)

eval_dataset = PreTokenizedDataset(
    EVALUATION_DATASET_PATH if USE_SEPARATE_EVALUATION_DATASET else DATASET_PATH,
    student_tokenizer,
    teacher_tokenizer,
    512,
    teacher_model=teacher_model,
    is_eval=True,
    sample_rate=None,
    use_cached_embeddings=USE_CACHED_EMBEDDINGS,
    cache_path=CACHE_PATH
)

if USE_CACHED_EMBEDDINGS and teacher_model is not None:
    print("Freeing T5-xxl memory since embeddings are cached")
    del teacher_model
    torch.cuda.empty_cache()

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

total_steps = EPOCHS * len(train_dataloader) // GRAD_ACCUM_STEPS

# ========== Training Setup ==========
autocast_dtype = torch.bfloat16
scaler = GradScaler(enabled=False)

optimizer = torch.optim.AdamW(
    [p for p in student_model.parameters() if p.requires_grad] + list(projection.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=MIN_LR,
)

# Start prefetching
if USE_CACHED_EMBEDDINGS:
    train_dataset.start_prefetching(BATCH_SIZE)
    eval_dataset.start_prefetching(BATCH_SIZE)

# ========== Training Loop ==========
print("Starting training with mixed precision and gradient accumulation...")
student_model.train()

start_time = time.time()
eval_delta_time = 0
global_step = 0
best_loss = float('inf')
accumulation_step = 0
grad_norm = 0
original_loss_ratio = HYBRID_LOSS

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        if USE_CACHED_EMBEDDINGS:
            # Use the prefetched batch - now includes attention mask
            s_input_ids, s_att_mask, t_embeddings, t_att_mask = train_dataset.get_prefetched_batch()
        else:
            s_input_ids, s_att_mask, t_input_ids, t_att_mask = batch

        s_input_ids = s_input_ids.to(device)
        s_att_mask = s_att_mask.to(device)
        t_att_mask = t_att_mask.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            student_outputs = student_model(
                input_ids=s_input_ids,
                attention_mask=s_att_mask,
                output_hidden_states=True
            )
            student_hidden = student_outputs.hidden_states[-1]
            projected_student = projection(student_hidden)

        if USE_CACHED_EMBEDDINGS:
            teacher_hidden = t_embeddings.to(device).squeeze(1)
        else:
            with torch.no_grad():
                t_input_ids = t_input_ids.to(device)
                t_att_mask = t_att_mask.to(device)
                teacher_outputs = teacher_model(
                    input_ids=t_input_ids,
                    attention_mask=t_att_mask
                )
                teacher_hidden = teacher_outputs.last_hidden_state
                teacher_hidden = teacher_hidden.to(device)

        loss, cos_loss, huber_loss, current_lambda = hybrid_loss(
            projected_student,
            teacher_hidden,
            s_att_mask,
            t_att_mask
        )
        scaler.scale(loss).backward()
        accumulation_step += 1

        if accumulation_step % GRAD_ACCUM_STEPS == 0:
            grad_norm = clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad] + list(projection.parameters()),
                max_norm=GRAD_CLIP
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                print(f"\nSaving checkpoint at step {global_step}")
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                os.makedirs(save_path, exist_ok=True)
                student_model.save_pretrained(save_path)
                student_tokenizer.save_pretrained(save_path)
                projection_state = projection.state_dict()
                projection_path = os.path.join(save_path, "projection_layer.safetensors")
                save_file(projection_state, projection_path)

        if batch_idx % PRINT_EVERY_X_BATCHES == 0:
            elapsed = time.time() - start_time - eval_delta_time
            remaining_batches = total_steps * GRAD_ACCUM_STEPS - (batch_idx + 1)
            eta = (elapsed / (batch_idx + 1)) * remaining_batches if batch_idx > 0 else 0

            print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], "
                    f"Batch [{batch_idx}/{len(train_dataloader)}], "
                    f"Step: {global_step}/{total_steps}, "
                    f"Hybrid Loss: {loss.item():.6f}, "
                    f"Huber Loss: {huber_loss.item():.6f}, "
                    f"Cos Loss: {cos_loss.item():.6f}, "
                    f"Loss Ratio: {current_lambda:.3f}, "
                    f"Grad Norm: {grad_norm:.6f}, "
                    f"Elapsed: {elapsed/60:.1f} min, "
                    f"ETA: {eta/60:.1f} min"
                )

    if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
        eval_start_time = time.time()
        student_model.eval()
        total_eval_loss = 0.0
        total_elements = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
                if USE_CACHED_EMBEDDINGS:
                    s_input_ids, s_att_mask, t_embeddings, t_att_mask = eval_dataset.get_prefetched_batch()
                else:
                    s_input_ids, s_att_mask, t_input_ids, t_att_mask = batch

                s_input_ids = s_input_ids.to(device)
                s_att_mask = s_att_mask.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    student_outputs = student_model(
                        input_ids=s_input_ids,
                        attention_mask=s_att_mask,
                        output_hidden_states=True
                    )
                    student_hidden = student_outputs.hidden_states[-1]
                    projected_student = projection(student_hidden)

                if USE_CACHED_EMBEDDINGS:
                    teacher_hidden = t_embeddings.to(device).squeeze(1)
                else:
                    with torch.no_grad():
                        t_input_ids = t_input_ids.to(device)
                        t_att_mask = t_att_mask.to(device)
                        teacher_outputs = teacher_model(
                            input_ids=t_input_ids,
                            attention_mask=t_att_mask
                        )
                        teacher_hidden = teacher_outputs.last_hidden_state
                        teacher_hidden = teacher_hidden.to(device)

                loss_val, cos_loss_val, huber_loss_val, current_lambda_val = hybrid_loss(
                    projected_student,
                    teacher_hidden,
                    s_att_mask,
                    t_att_mask
                )

                total_eval_loss += loss_val.item()
                total_elements += s_att_mask.sum().item()

            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            print(f"\n[Validation] Epoch {epoch + 1} | Average Hybrid Loss: {avg_eval_loss:.6f}")

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

        eval_end_time = time.time()
        eval_delta_time += (eval_end_time - eval_start_time)
        student_model.train()

# ========== Cleanup Prefetching Threads ==========
if USE_CACHED_EMBEDDINGS:
    train_dataset.stop_prefetching()
    eval_dataset.stop_prefetching()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

student_model.save_pretrained(OUTPUT_DIR)
student_tokenizer.save_pretrained(OUTPUT_DIR)

projection_state = projection.state_dict()
projection_path = os.path.join(OUTPUT_DIR, "projection_layer.safetensors")
save_file(projection_state, projection_path)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
