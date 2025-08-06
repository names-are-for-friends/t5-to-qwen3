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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt"
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/q5-xxs-ALL/ultimate-q5-xxs-v1/"

USE_CACHED_EMBEDDINGS = True
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2"

USE_SEPARATE_EVALUATION_DATASET = True
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
MIN_LR = 2e-5
SAVE_EVERY_X_STEPS = 500
EVAL_EVERY_EPOCHS = 1
SAVE_BEST_MODEL = True
PRINT_EVERY_X_STEPS = 1
GRAD_ACCUM_STEPS = 2
COSINE_LOSS = 0.30 # 1 = Full cosine, 0 = full huber
T5_EMBEDDING_DIM = 1024 # Change for the hidden size of the model/features of the embedding. 0.6B is 1024, 1.7B is 2048. Check config.json if unsure
T5_ADDITIONAL_PADDING_ATTENTION = 1 # Train to match the expectations of the target image gen model. Chroma = 1, most Flux variants = 3

# ========== Custom Prefetch DataLoader ==========
class PrefetchDataLoader:
    def __init__(self, dataloader, prefetch_factor=3):
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.thread = None
        self.stop_event = threading.Event()

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.stop_event.clear()
        self._start_prefetch()
        return self

    def __next__(self):
        if self.thread is None:
            self._start_prefetch()

        try:
            batch = self.queue.get(timeout=30)
            if isinstance(batch, Exception):
                raise batch
            return batch
        except queue.Empty:
            raise StopIteration()

    def _prefetch_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    batch = next(self.iterator)
                    # Use block=False to avoid hanging when queue is full
                    self.queue.put(batch, block=False)
                except StopIteration:
                    break
                except queue.Full:
                    # Queue is full, continue to next iteration
                    continue
                except Exception as e:
                    # Put exception in queue to be raised in main thread
                    try:
                        self.queue.put(e, block=False)
                    except queue.Full:
                        pass
                    break
        except Exception as e:
            try:
                self.queue.put(e, block=False)
            except queue.Full:
                pass

    def _start_prefetch(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.thread.start()

    def __len__(self):
        return len(self.dataloader)

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

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

                # Preload embedding paths for faster access
                self.embedding_files = []
                self.mask_files = []
                for i in range(len(lines)):
                    self.embedding_files.append(os.path.join(self.cache_folder, f"{i}.pt"))
                    self.mask_files.append(os.path.join(self.cache_folder, f"{i}_mask.pt"))
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

                # Preload embedding paths for faster access
                self.embedding_files = []
                self.mask_files = []
                for i in range(len(lines)):
                    self.embedding_files.append(os.path.join(self.cache_folder, f"{i}.pt"))
                    self.mask_files.append(os.path.join(self.cache_folder, f"{i}_mask.pt"))
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

    def __len__(self):
        return len(self.student_input_ids)

    def __getitem__(self, idx):
        if self.use_cached_embeddings:
            embeddings = torch.load(self.embedding_files[idx], map_location='cpu')
            att_mask = torch.load(self.mask_files[idx], map_location='cpu')
            # Remove batch dimension if present
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

# ========== T5 Mask Modification ==========
def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=1):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask

# ========== Projection Layer ==========
class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096):
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
    def __init__(self, lambda_weight=0.0, student_hidden_size=None, teacher_hidden_size=None):
        super().__init__()
        self.lambda_weight = lambda_weight
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

        # Apply attention mask modification for T5
        teacher_mask = modify_mask_to_attend_padding(
            teacher_mask,
            teacher_mask.shape[-1],
            num_extra_padding=T5_ADDITIONAL_PADDING_ATTENTION
        )

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

# ========== Initialize or Load Projection Layer and Poolers ==========
projection_path = os.path.join(QWEN3_MODEL_NAME, "projection_layer.safetensors")
poolers_path = os.path.join(QWEN3_MODEL_NAME, "poolers.safetensors")

if os.path.exists(projection_path) and os.path.exists(poolers_path):
    print("Loading existing projection layer and poolers from", QWEN3_MODEL_NAME)
    try:
        projection_state = load_file(projection_path)
        projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)
        projection.load_state_dict(projection_state)

        poolers_state = load_file(poolers_path)

        hybrid_loss = SequenceLevelHybridLoss(
            lambda_weight=COSINE_LOSS,
            student_hidden_size=4096,
            teacher_hidden_size=4096
        ).to(device, dtype=torch.bfloat16)

        hybrid_loss.load_state_dict(poolers_state)
    except Exception as e:
        print(f"Incompatible projection layer or poolers detected: {e}. Initializing new projection layer and poolers")
        projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)
        hybrid_loss = SequenceLevelHybridLoss(
            lambda_weight=COSINE_LOSS,
            student_hidden_size=4096,
            teacher_hidden_size=4096
        ).to(device, dtype=torch.bfloat16)
else:
    print("Initializing projection layer and poolers")
    projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)
    hybrid_loss = SequenceLevelHybridLoss(
        lambda_weight=COSINE_LOSS,
        student_hidden_size=4096,
        teacher_hidden_size=4096
    ).to(device, dtype=torch.bfloat16)

projection.to(device, dtype=torch.bfloat16)

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

# Optimized DataLoader with custom prefetching
base_train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=min(4, os.cpu_count() - 1) if torch.cuda.is_available() else 0,
    persistent_workers=True if torch.cuda.is_available() and min(4, os.cpu_count() - 1) > 0 else False
)

base_eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=min(2, os.cpu_count() - 1) if torch.cuda.is_available() else 0,
    persistent_workers=True if torch.cuda.is_available() and min(2, os.cpu_count() - 1) > 0 else False
)

# Wrap with custom prefetch dataloaders
train_dataloader = PrefetchDataLoader(base_train_dataloader, prefetch_factor=3)
eval_dataloader = PrefetchDataLoader(base_eval_dataloader, prefetch_factor=2)

total_steps = EPOCHS * len(train_dataloader) // GRAD_ACCUM_STEPS

# ========== Training Setup ==========
autocast_dtype = torch.bfloat16
scaler = GradScaler(enabled=False)

optimizer = torch.optim.AdamW(
    [p for p in student_model.parameters() if p.requires_grad] +
    list(projection.parameters()) +
    list(hybrid_loss.student_pooler.parameters()) +
    list(hybrid_loss.teacher_pooler.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=MIN_LR,
)

# ========== Training Loop ==========
print("Starting training with mixed precision and gradient accumulation...")
student_model.train()

start_time = time.time()
eval_delta_time = 0
global_step = 0
best_loss = float('inf')
accumulation_step = 0
grad_norm = 0

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Track actual steps completed
    steps_completed_this_epoch = 0

    for batch_idx, batch in enumerate(train_dataloader):
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
            # Fix: properly get teacher inputs from batch
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

        loss, cos_loss, huber_loss, current_lambda = hybrid_loss(
            projected_student,
            teacher_hidden,
            s_att_mask,
            t_att_mask
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(scaled_loss).backward()
        accumulation_step += 1

        # Only step optimizer after accumulation steps
        if accumulation_step % GRAD_ACCUM_STEPS == 0:
            grad_norm = clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad] + list(projection.parameters()) +
                list(hybrid_loss.student_pooler.parameters()) + list(hybrid_loss.teacher_pooler.parameters()),
                max_norm=GRAD_CLIP
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            steps_completed_this_epoch += 1

            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                print(f"\nSaving checkpoint at step {global_step}")
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                os.makedirs(save_path, exist_ok=True)
                student_model.save_pretrained(save_path)
                student_tokenizer.save_pretrained(save_path)

                # Save projection layer
                projection_state = projection.state_dict()
                projection_path = os.path.join(save_path, "projection_layer.safetensors")
                save_file(projection_state, projection_path)

                # Save poolers
                poolers_state = hybrid_loss.state_dict()
                poolers_path = os.path.join(save_path, "poolers.safetensors")
                save_file(poolers_state, poolers_path)

            if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                elapsed = time.time() - start_time - eval_delta_time
                remaining_steps = total_steps - global_step
                eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], "
                    f"Batch [{batch_idx}/{len(train_dataloader)}], "
                    f"Step: {global_step}/{total_steps}, "
                    f"Total Loss: {loss.item():.6f}, "
                    f"Huber Loss: {huber_loss.item():.6f}, "
                    f"Cos Loss: {cos_loss.item():.6f}, "
                    f"Grad Norm: {grad_norm:.6f}, "
                    f"Elapsed: {elapsed/60:.1f} min, "
                    f"ETA: {eta/60:.1f} min"
                )

    # Evaluation at the end of each epoch
    if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
        eval_start_time = time.time()
        student_model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
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

                loss_val, cos_loss_val, huber_loss_val, current_lambda_val = hybrid_loss(
                    projected_student,
                    teacher_hidden,
                    s_att_mask,
                    t_att_mask
                )

                total_eval_loss += loss_val.item()

            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            print(f"\n[Validation] Epoch {epoch + 1} | Average Total Loss: {avg_eval_loss:.6f}")

            if SAVE_BEST_MODEL and avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                print(f"\n✅ New best model at loss {best_loss:.6f}, saving...")
                best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                student_model.save_pretrained(best_model_dir)
                student_tokenizer.save_pretrained(best_model_dir)

                # Save projection layer
                projection_state = projection.state_dict()
                projection_path = os.path.join(best_model_dir, "projection_layer.safetensors")
                save_file(projection_state, projection_path)

                # Save poolers
                poolers_state = hybrid_loss.state_dict()
                poolers_path = os.path.join(best_model_dir, "poolers.safetensors")
                save_file(poolers_state, poolers_path)

        eval_end_time = time.time()
        eval_delta_time += (eval_end_time - eval_start_time)
        student_model.train()

# Clean up prefetch loaders
train_dataloader.close()
eval_dataloader.close()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

student_model.save_pretrained(OUTPUT_DIR)
student_tokenizer.save_pretrained(OUTPUT_DIR)

# Save projection layer
projection_state = projection.state_dict()
projection_path = os.path.join(OUTPUT_DIR, "projection_layer.safetensors")
save_file(projection_state, projection_path)

# Save poolers
poolers_state = hybrid_loss.state_dict()
poolers_path = os.path.join(OUTPUT_DIR, "poolers.safetensors")
save_file(poolers_state, poolers_path)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
