from unsloth import FastLanguageModel
import os
import json
import time
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler
from transformers import T5TokenizerFast, T5EncoderModel
from safetensors.torch import save_file, load_file
import numpy as np
from tqdm import tqdm
import queue
import threading
import gc
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt" # Each line of the dataset text file is taken as one prompt
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/q3-xxs-ALL/ultimate-q3-xxs-v1/"

USE_CACHED_EMBEDDINGS = True # Each T5-xxl embedding is cached; size per is 4MB so multiply by dataset size for capacity required
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2" # Cache is picked up on subsequent runs by reference to dataset file name
PREFETCH_FACTOR = 16

USE_SEPARATE_EVALUATION_DATASET = True # Otherwise we take some of the main dataset, but best to use unseen data
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EPOCHS = 1
MAX_LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
GRAD_CLIP = 1.0
SAVE_EVERY_X_STEPS = 0
SAVE_EVERY_X_RESTARTS = 1
PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_EPOCHS = 1
SAVE_BEST_MODEL = True

HUBER_LOSS = 0.70
COSINE_LOSS = 0.30

WARMUP_STEPS = 500
RESTART_PERIOD_STEPS = 1501

INTERMEDIATE_DIM = 1024 # We scale the output to this size before inputting it into the transformer encoder. Significantly affects the size of the projection layer and VRAM consumption during training
FEED_FORWARD_DIM = 1024

# ========== Dataset Class ==========
class PreTokenizedDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, max_length, teacher_model=None, is_eval=False, sample_rate=0.1, use_cached_embeddings=False, cache_path=None):
        self.max_length = max_length
        if USE_SEPARATE_EVALUATION_DATASET and is_eval:
            file_path = EVALUATION_DATASET_PATH
            sample_rate = None

        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

        if is_eval and sample_rate is not None:
            self.lines = random.sample(self.lines, min(int(len(self.lines) * sample_rate), len(self.lines)))

        student_inputs = student_tokenizer(
            self.lines,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        self.student_input_ids = torch.tensor(student_inputs["input_ids"], dtype=torch.long)
        self.student_attention_mask = torch.tensor(student_inputs["attention_mask"], dtype=torch.long)

        if use_cached_embeddings:
            base_name = os.path.basename(file_path)
            cache_folder = os.path.join(cache_path, base_name)
            validation_file = os.path.join(cache_folder, f"{base_name}.validation")

            if os.path.exists(validation_file):
                print(f"Loading cached embeddings from folder {cache_folder}")
                self.cache_folder = cache_folder
                self.num_samples = len(self.lines)
                self.hidden_dim = 4096

                self.embedding_files = [os.path.join(self.cache_folder, f"{i}.pt") for i in range(len(self.lines))]
                self.mask_files = [os.path.join(self.cache_folder, f"{i}_mask.pt") for i in range(len(self.lines))]
            else:
                print(f"Generating and caching embeddings for {file_path}")
                os.makedirs(cache_folder, exist_ok=True)
                self.num_samples = len(self.lines)
                self.hidden_dim = teacher_model.config.hidden_size

                for i, line in enumerate(tqdm(self.lines, desc="Generating embeddings")):
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

                    embedding_file = os.path.join(cache_folder, f"{i}.pt")
                    mask_file = os.path.join(cache_folder, f"{i}_mask.pt")
                    torch.save(embeddings, embedding_file)
                    torch.save(att_mask.cpu(), mask_file)

                with open(validation_file, "w") as f:
                    pass
                print(f"Saved embeddings to folder {cache_folder}")
                self.cache_folder = cache_folder

                self.embedding_files = [os.path.join(self.cache_folder, f"{i}.pt") for i in range(len(self.lines))]
                self.mask_files = [os.path.join(self.cache_folder, f"{i}_mask.pt") for i in range(len(self.lines))]
        else:
            teacher_inputs = teacher_tokenizer(
                self.lines,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            self.teacher_input_ids = torch.tensor(teacher_inputs["input_ids"], dtype=torch.long)
            self.teacher_attention_mask = torch.tensor(teacher_inputs["attention_mask"], dtype=torch.long)

        self.use_cached_embeddings = use_cached_embeddings
        self.line_index = list(range(len(self.lines)))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.use_cached_embeddings:
            embeddings = torch.load(self.embedding_files[idx], map_location='cpu')
            att_mask = torch.load(self.mask_files[idx], map_location='cpu')

            if embeddings.shape[1] < self.max_length:
                pad_length = self.max_length - embeddings.shape[1]
                embeddings = torch.cat([
                    embeddings,
                    torch.zeros(embeddings.shape[0], pad_length, embeddings.shape[2])
                ], dim=1)
                att_mask = torch.cat([
                    att_mask,
                    torch.zeros(att_mask.shape[0], pad_length)
                ], dim=1)
            elif embeddings.shape[1] > self.max_length:
                embeddings = embeddings[:, :self.max_length, :]
                att_mask = att_mask[:, :self.max_length]

            if embeddings.dim() == 3 and embeddings.shape[0] == 1:
                embeddings = embeddings.squeeze(0)
            if att_mask.dim() == 2 and att_mask.shape[0] == 1:
                att_mask = att_mask.squeeze(0)

            return (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                embeddings,
                att_mask
            )
        else:
            return (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                self.teacher_input_ids[idx],
                self.teacher_attention_mask[idx]
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'file_handles'):
            for file in self.file_handles:
                file.close()
            self.file_handles.clear()

# ========== Projection Layer ==========
class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_dim, output_dim=4096, dim_feedforward=None, num_layers=1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = transformer_dim
        self.linear_in = torch.nn.Linear(input_dim, transformer_dim)

        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )

        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.upscale = torch.nn.Linear(transformer_dim, output_dim)
        self.activation = torch.nn.GELU()
        self.linear_out = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.transformer(x)
        x = self.upscale(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x

# ========== Hybrid Loss ==========
class HybridLoss(torch.nn.Module):
    def __init__(self, huber_weight=0.70, cosine_weight=0.30):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber = torch.nn.HuberLoss(reduction='none')
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output, teacher_output, teacher_mask):
        device = student_output.device
        dtype = torch.bfloat16

        student_output = student_output.to(dtype)
        teacher_output = teacher_output.to(dtype)
        teacher_mask = teacher_mask.to(dtype).to(device)

        huber_loss = self.huber(student_output, teacher_output)
        huber_loss = huber_loss.mean(dim=-1)
        huber_loss = (huber_loss * teacher_mask).sum(dim=-1) / (teacher_mask.sum(dim=-1) + 1e-8)

        cos_sim = self.cos(student_output, teacher_output)
        cos_loss = 1 - cos_sim
        cos_loss = (cos_loss * teacher_mask).sum(dim=-1) / (teacher_mask.sum(dim=-1) + 1e-8)

        total_loss = (
            self.huber_weight * huber_loss.mean() +
            self.cosine_weight * cos_loss.mean()
        )

        return total_loss, huber_loss.mean(), cos_loss.mean()

# ========== Evaluation Function ==========
def evaluate_model(model, dataloader, projection, loss_fn, device, autocast_dtype):
    model.eval()
    projection.eval()
    loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'huber': 0.0,
        'cos': 0.0
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                s_input_ids, s_att_mask, t_embeddings, t_att_mask = batch
                s_input_ids = s_input_ids.to(device)
                s_att_mask = s_att_mask.to(device)
                t_att_mask = t_att_mask.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    student_outputs = model(
                        input_ids=s_input_ids,
                        attention_mask=s_att_mask,
                        output_hidden_states=True
                    )
                    student_hidden = student_outputs.hidden_states[-1]
                    projected_student = projection(student_hidden)

                if USE_CACHED_EMBEDDINGS:
                    teacher_hidden = t_embeddings.to(device).squeeze(1)
                else:
                    t_input_ids, t_att_mask = t_embeddings, t_att_mask
                    with torch.no_grad():
                        t_input_ids = t_input_ids.to(device)
                        t_att_mask = t_att_mask.to(device)
                        teacher_outputs = teacher_model(
                            input_ids=t_input_ids,
                            attention_mask=t_att_mask
                        )
                        teacher_hidden = teacher_outputs.last_hidden_state
                        teacher_hidden = teacher_hidden.to(device)

                loss, huber_loss, cos_loss = loss_fn(
                    projected_student,
                    teacher_hidden,
                    t_att_mask
                )

                total_losses['total'] += loss.item()
                total_losses['huber'] += huber_loss.item()
                total_losses['cos'] += cos_loss.item()

                del loss, huber_loss, cos_loss, projected_student, student_hidden, student_outputs
                if not USE_CACHED_EMBEDDINGS and 'teacher_hidden' in locals():
                    del teacher_hidden, teacher_outputs
                del s_input_ids, s_att_mask, t_att_mask, t_embeddings

            except Exception as e:
                if 'projected_student' in locals():
                    del projected_student
                if 'student_hidden' in locals():
                    del student_hidden
                if 'student_outputs' in locals():
                    del student_outputs
                if 'teacher_hidden' in locals():
                    del teacher_hidden
                if 'teacher_outputs' in locals():
                    del teacher_outputs
                del s_input_ids, s_att_mask, t_att_mask, t_embeddings
                raise e

    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches

    model.train()
    projection.train()
    loss_fn.train()

    return total_losses

# ========== Miscellaneous Functions ==========
def get_memory_usage():
    if torch.cuda.is_available():
        memory = torch.cuda.mem_get_info()
        memory_mib = []
        for item in memory:
            memory_mib.append(item/1048576)
        memory_used = memory_mib[1]-memory_mib[0]
        memory_mib.append(memory_used)
        return memory_mib

def save_trained_model(save_path, model, tokenizer, projection, qwen_embedding_dim):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    projection_state = projection.state_dict()
    projection_path = os.path.join(save_path, "projection_layer.safetensors")
    save_file(projection_state, projection_path)

    projection_config_path = os.path.join(save_path, "projection_config.json")
    save_projection_config(projection_config_path, qwen_embedding_dim)

def save_projection_config(projection_config_path, embedding_dim):
    projection_config = {
        "input_dim": embedding_dim,
        "transformer_dim": INTERMEDIATE_DIM,
        "output_dim": 4096,
        "dim_feedforward": FEED_FORWARD_DIM,
        "num_layers": 1
    }
    with open(projection_config_path, "w") as f:
        json.dump(projection_config, f)

def exit_dataloader():
    train_dataloader = None
    train_dataset = None
    eval_dataloader = None
    eval_dataset = None
    del train_dataloader
    del train_dataset
    del eval_dataloader
    del eval_dataset
    gc.collect()

# ========== Load Qwen3 Model ==========
gc.collect
print("Loading Qwen3 model...")
student_model, student_tokenizer = FastLanguageModel.from_pretrained(
    QWEN3_MODEL_NAME,
    max_seq_length=512,
    load_in_4bit=False,
    dtype=torch.bfloat16,
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

try:
    with open(os.path.join(QWEN3_MODEL_NAME, "config.json"), "r") as f:
        qwen_config = json.load(f)
    qwen_embedding_dim = qwen_config["hidden_size"]
except Exception as e:
    print(f"Error loading Qwen config: {e}")
    qwen_embedding_dim = 1024

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
        projection = ProjectionLayer(
            input_dim=qwen_embedding_dim,
            transformer_dim=INTERMEDIATE_DIM,
            output_dim=4096,
            dim_feedforward=FEED_FORWARD_DIM
        )
        projection.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading projection layer: {e}")
        print("Initializing new projection layer")
        projection = ProjectionLayer(
            input_dim=qwen_embedding_dim,
            transformer_dim=INTERMEDIATE_DIM,
            output_dim=4096,
            dim_feedforward=FEED_FORWARD_DIM
        )
else:
    print("Initializing projection layer")
    projection = ProjectionLayer(
        input_dim=qwen_embedding_dim,
        transformer_dim=INTERMEDIATE_DIM,
        output_dim=4096,
        dim_feedforward=FEED_FORWARD_DIM
    )

projection.to(device, dtype=torch.bfloat16)

losses = [HUBER_LOSS, COSINE_LOSS]
sum_loss = sum(losses)
normalized_losses = [loss / sum_loss for loss in losses]

hybrid_loss = HybridLoss(
    huber_weight=normalized_losses[0],
    cosine_weight=normalized_losses[1],
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

with train_dataset as train_ds, eval_dataset as eval_ds:
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=min(4, os.cpu_count()//2) if torch.cuda.is_available() else 0,
            persistent_workers=False,
            prefetch_factor=PREFETCH_FACTOR,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=min(4, os.cpu_count()//2) if torch.cuda.is_available() else 0,
            persistent_workers=False,
            prefetch_factor=PREFETCH_FACTOR,
        )
        total_steps = (len(train_dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
        # ========== Training Setup ==========
        autocast_dtype = torch.bfloat16
        scaler = GradScaler('cuda', enabled=False)

        optimizer = torch.optim.AdamW(
            [p for p in student_model.parameters() if p.requires_grad] +
            list(projection.parameters()),
            lr=MAX_LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=RESTART_PERIOD_STEPS,
            T_mult=1,
            eta_min=MIN_LEARNING_RATE,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((MIN_LEARNING_RATE / MAX_LEARNING_RATE) + (step / (WARMUP_STEPS+1)) * (1 - MIN_LEARNING_RATE / MAX_LEARNING_RATE), 1.0)
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[WARMUP_STEPS]
        )

        # ========== Training Loop ==========
        print("Starting training...")
        student_model.train()

        start_time = time.time()
        eval_delta_time = 0
        global_step = 0
        best_loss = float('inf')
        accumulation_step = 0
        grad_norm = 0
        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            steps_completed_this_epoch = 0
            accumulation_step = 0

            print(f"Starting epoch {epoch + 1}/{EPOCHS}")
            print(f"Total batches in epoch: {len(train_dataloader)}")

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    s_input_ids, s_att_mask, t_embeddings, t_att_mask = batch
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
                        t_input_ids, t_att_mask = t_embeddings, t_att_mask
                        with torch.no_grad():
                            t_input_ids = t_input_ids.to(device)
                            t_att_mask = t_att_mask.to(device)
                            teacher_outputs = teacher_model(
                                input_ids=t_input_ids,
                                attention_mask=t_att_mask
                            )
                            teacher_hidden = teacher_outputs.last_hidden_state
                            teacher_hidden = teacher_hidden.to(device)

                    loss, huber_loss, cos_loss = hybrid_loss(
                        projected_student,
                        teacher_hidden,
                        t_att_mask
                    )

                    scaled_loss = loss / GRAD_ACCUM_STEPS
                    scaler.scale(scaled_loss).backward()
                    accumulation_step += 1

                    if accumulation_step >= GRAD_ACCUM_STEPS or batch_idx == len(train_dataset) - 1:
                        grad_norm = clip_grad_norm_(
                            [p for p in student_model.parameters() if p.requires_grad] +
                            list(projection.parameters()),
                            max_norm=GRAD_CLIP
                        )

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                        global_step += 1
                        steps_completed_this_epoch += 1
                        accumulation_step = 0

                        current_loss = loss.item()
                        current_huber = huber_loss.item()
                        current_cos = cos_loss.item()

                        if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                            elapsed = time.time() - start_time - eval_delta_time
                            remaining_steps = total_steps - global_step
                            eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0
                            vram_free, vram_total, vram_used = get_memory_usage()
                            current_lr = scheduler.get_last_lr()[0]

                            print(
                                f"Epoch [{epoch + 1}/{EPOCHS}], "
                                f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                                f"Step: {global_step}/{total_steps}, "
                                f"Total Loss: {current_loss:.6f}, "
                                f"Huber Loss: {current_huber:.6f}, "
                                f"Cosine Loss: {current_cos:.6f}, "
                                f"Grad Norm: {grad_norm:.6f}, "
                                f"Learning Rate: {current_lr:.6f}, "
                                f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
                                f"Elapsed: {elapsed/60:.1f} min, "
                                f"ETA: {eta/60:.1f} min"
                            )

                        if SAVE_EVERY_X_RESTARTS > 0 and global_step > WARMUP_STEPS and (global_step - WARMUP_STEPS) % RESTART_PERIOD_STEPS == 0:
                            restart_count += 1
                            if restart_count % SAVE_EVERY_X_RESTARTS == 0:
                                print(f"\nSaving checkpoint at restart {restart_count}\n")
                                save_path = os.path.join(OUTPUT_DIR, f"restart_{restart_count}")
                                save_trained_model(save_path, student_model, student_tokenizer, projection, qwen_embedding_dim)

                        if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                            print(f"\nSaving checkpoint at step {global_step}")
                            save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                            save_trained_model(save_path, student_model, student_tokenizer, projection, qwen_embedding_dim)

                        del loss, scaled_loss, student_outputs, student_hidden, projected_student
                        del teacher_hidden, huber_loss, cos_loss
                        if 't_input_ids' in locals():
                            del t_input_ids, t_att_mask, teacher_outputs

                    del s_input_ids, s_att_mask, t_att_mask, t_embeddings

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    if 'loss' in locals():
                        del loss
                    if 'scaled_loss' in locals():
                        del scaled_loss
                    if 'student_outputs' in locals():
                        del student_outputs
                    if 'student_hidden' in locals():
                        del student_hidden
                    if 'projected_student' in locals():
                        del projected_student
                    if 'teacher_hidden' in locals():
                        del teacher_hidden
                    if 'huber_loss' in locals():
                        del huber_loss
                    if 'cos_loss' in locals():
                        del cos_loss
                    if 't_input_ids' in locals():
                        del t_input_ids, t_att_mask, teacher_outputs
                    del s_input_ids, s_att_mask, t_att_mask, t_embeddings

                    accumulation_step = 0
                    optimizer.zero_grad()
                    continue

            print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")

            if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
                eval_start_time = time.time()

                eval_metrics = evaluate_model(
                    student_model, eval_dataloader, projection, hybrid_loss, device, autocast_dtype
                )

                avg_eval_loss = eval_metrics['total']
                print(f"\n[Validation] Epoch {epoch + 1}")
                print(f"  Average Total Loss: {avg_eval_loss:.6f}")
                print(f"  Huber Loss: {eval_metrics['huber']:.6f}")
                print(f"  Cosine Loss: {eval_metrics['cos']:.6f}")

                if SAVE_BEST_MODEL and avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    print(f"\n✅ New best model at loss {best_loss:.6f}, saving...")
                    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
                    save_trained_model(best_model_dir, student_model, student_tokenizer, projection, qwen_embedding_dim)

                eval_end_time = time.time()
                eval_delta_time += (eval_end_time - eval_start_time)
                student_model.train()

    except Exception as e:
        print(f"Exception during training: {e}")
        exit_dataloader()
        sys.exit(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Closing dataloaders and exiting")
        exit_dataloader()
        sys.exit(1)
    finally:
        exit_dataloader()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
save_trained_model(OUTPUT_DIR, student_model, student_tokenizer, projection, qwen_embedding_dim)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
