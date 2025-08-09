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
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt" # The correct format is one prompt per line of the text file
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/q5-xxs-ALL/ultimate-q5-xxs-v1/"

USE_CACHED_EMBEDDINGS = True # Embeddings are 4MB in size so multiply this by the dataset size for the total capacity required
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2" # Cache is picked up on subsequent runs by reference to dataset file name
PREFETCH_FACTOR = 8

USE_SEPARATE_EVALUATION_DATASET = True # If disabled, we'll just use 10% of the main dataset, but using unseen data is a better test of generalisation
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"
'''
Recommendation w/ Qwen3 0.6B:
24GB VRAM: BATCH SIZE = 64, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
16GB VRAM: BATCH SIZE = 32, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
12GB VRAM: BATCH SIZE = 16, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
'''
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1 # Accumulates the gradient across x batches before averaging to simulate larger batch size; does nothing when set to 1
EPOCHS = 1
LEARNING_RATE = 2e-4
MIN_LR = 2e-5
GRAD_CLIP = 1.0
SAVE_EVERY_X_STEPS = 500
PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_EPOCHS = 1
SAVE_BEST_MODEL = True

PER_TOKEN_HUBER_LOSS = 0.35 # Per-token huber loss is particularly important to match our projected Qwen3 embedding with the T5-xxl embedding given differing architecture & tokenization, but since it's one-to-one we can only compare with a uniform mask (obviously, the mask of the two, usually the Qwen3 mask)
SEQUENCE_HUBER_LOSS = 0.35 # Sequence-level loss provides a teacher-mask-to-student-mask alignment factor to the overall loss, hopefully keeping the total Qwen3 embedding aligned to the total T5-xxl embedding
PER_TOKEN_COSINE_LOSS = 0.30 # Ditto for cosine loss, except this helps with directional alignment
SEQUENCE_COSINE_LOSS = 0.30 # Looks lonely without a comment here

T5_EMBEDDING_DIM = 1024 # Change for the hidden size of the model/features of the embedding, eg. 0.6B is 1024, 1.7B is 2048. Check model's config.json if unsure
T5_ADDITIONAL_PADDING_ATTENTION = 1 # Train to match the expectations of the target image gen model. Chroma = 1, most Flux variants = 3

# ========== Custom Prefetch DataLoader ==========
class PrefetchDataLoader:
    def __init__(self, dataloader, prefetch_factor=3):
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.thread = None
        self.stop_event = threading.Event()
        self._exception = None

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.stop_event.clear()
        self._exception = None
        self._start_prefetch()
        return self

    def __next__(self):
        if self.thread is None or not self.thread.is_alive():
            self._start_prefetch()

        try:
            if self._exception:
                raise self._exception

            batch = self.queue.get(timeout=30)
            if isinstance(batch, Exception):
                raise batch
            return batch
        except queue.Empty:
            raise StopIteration()
        except Exception as e:
            self._exception = e
            raise e

    def _prefetch_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    batch = next(self.iterator)
                    if isinstance(batch, tuple):
                        cleaned_batch = []
                        for item in batch:
                            if hasattr(item, 'detach'):
                                cleaned_item = item.detach()
                                cleaned_batch.append(cleaned_item)
                            else:
                                cleaned_batch.append(item)
                        batch = tuple(cleaned_batch)
                    try:
                        self.queue.put(batch, block=False)
                    except queue.Full:
                        del batch
                        continue

                except StopIteration:
                    break
                except Exception as e:
                    try:
                        self.queue.put(e, block=False)
                    except queue.Full:
                        pass
                    break
        except Exception as e:
            self._exception = e
            try:
                self.queue.put(e, block=False)
            except queue.Full:
                pass

    def _start_prefetch(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1)

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def __len__(self):
        return len(self.dataloader)

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if hasattr(item, 'delete') or hasattr(item, 'cpu'):
                    del item
            except:
                break

    def __del__(self):
        self.close()
        try:
            while True:
                self.queue.get_nowait()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# ========== Dataset Class ==========
class PreTokenizedDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, max_length, teacher_model=None, is_eval=False, sample_rate=0.1, use_cached_embeddings=False, cache_path=None):
        self.max_length = max_length
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

                    embedding_file = os.path.join(cache_folder, f"{i}.pt")
                    mask_file = os.path.join(cache_folder, f"{i}_mask.pt")
                    torch.save(embeddings, embedding_file)
                    torch.save(att_mask.cpu(), mask_file)

                with open(validation_file, "w") as f:
                    pass
                print(f"Saved embeddings to folder {cache_folder}")
                self.cache_folder = cache_folder

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

            result = (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                embeddings,
                att_mask
            )

            del embeddings, att_mask
            return result
        else:
            return (
                self.student_input_ids[idx],
                self.student_attention_mask[idx],
                self.teacher_input_ids[idx],
                self.teacher_attention_mask[idx]
            )

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
class ComprehensiveHybridLoss(torch.nn.Module):
    def __init__(self, student_hidden_size=None, teacher_hidden_size=None,
                 per_token_huber_weight=0.35, sequence_huber_weight=0.35,
                 per_token_cosine_weight=0.30, sequence_cosine_weight=0.30):
        super().__init__()
        self.per_token_huber_weight = per_token_huber_weight
        self.sequence_huber_weight = sequence_huber_weight
        self.per_token_cosine_weight = per_token_cosine_weight
        self.sequence_cosine_weight = sequence_cosine_weight

        if student_hidden_size is not None and teacher_hidden_size is not None:
            self.student_pooler = torch.nn.Linear(student_hidden_size, 1).to(torch.bfloat16)
            self.teacher_pooler = torch.nn.Linear(teacher_hidden_size, 1).to(torch.bfloat16)
        else:
            raise ValueError("Hidden sizes must be specified for learnable pooling")

        self.huber = torch.nn.HuberLoss(reduction='none')
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output, teacher_output, student_mask, teacher_mask):
        device = student_output.device
        dtype = torch.bfloat16

        student_output = student_output.to(dtype)
        teacher_output = teacher_output.to(dtype)
        student_mask = student_mask.to(dtype).to(device)
        teacher_mask = teacher_mask.to(dtype).to(device)

        # Per-token losses
        per_token_huber_loss = self.huber(student_output, teacher_output)
        per_token_huber_loss = per_token_huber_loss.mean(dim=-1)
        per_token_huber_loss = (per_token_huber_loss * student_mask).sum(dim=-1) / (student_mask.sum(dim=-1) + 1e-8)

        per_token_cos_sim = self.cos(student_output, teacher_output)
        per_token_cos_loss = (1 - per_token_cos_sim)
        per_token_cos_loss = (per_token_cos_loss * student_mask).sum(dim=-1) / (student_mask.sum(dim=-1) + 1e-8)

        # Learn attention weights for sequence-level pooling
        student_logits = self.student_pooler(student_output).squeeze(-1)
        teacher_logits = self.teacher_pooler(teacher_output).squeeze(-1)

        # Apply masks
        student_logits = student_logits.masked_fill(~student_mask.bool(), -1e9)
        teacher_logits = teacher_logits.masked_fill(~teacher_mask.bool(), -1e9)

        student_weights = torch.softmax(student_logits, dim=1)
        teacher_weights = torch.softmax(teacher_logits, dim=1)

        # Weighted pooling
        student_pooled = (student_output * student_weights.unsqueeze(-1)).sum(dim=1)
        teacher_pooled = (teacher_output * teacher_weights.unsqueeze(-1)).sum(dim=1)

        # Sequence-level losses
        sequence_huber_loss = self.huber(student_pooled, teacher_pooled).mean()
        sequence_cos_sim = self.cos(student_pooled, teacher_pooled)
        sequence_cos_loss = (1 - sequence_cos_sim).mean()

        # Combine all losses
        total_loss = (
            self.per_token_huber_weight * per_token_huber_loss.mean() +
            self.sequence_huber_weight * sequence_huber_loss +
            self.per_token_cosine_weight * per_token_cos_loss.mean() +
            self.sequence_cosine_weight * sequence_cos_loss
        )

        return total_loss, per_token_huber_loss.mean(), sequence_huber_loss, per_token_cos_loss.mean(), sequence_cos_loss

# ========== Miscellaneous Helpers ==========
def evaluate_model(model, dataloader, projection, loss_fn, device, autocast_dtype):
    model.eval()
    projection.eval()
    loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'per_token_huber': 0.0,
        'sequence_huber': 0.0,
        'per_token_cos': 0.0,
        'sequence_cos': 0.0
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

                loss, per_token_huber, sequence_huber, per_token_cos, sequence_cos = loss_fn(
                    projected_student,
                    teacher_hidden,
                    s_att_mask,
                    t_att_mask
                )

                total_losses['total'] += loss.item()
                total_losses['per_token_huber'] += per_token_huber.item()
                total_losses['sequence_huber'] += sequence_huber.item()
                total_losses['per_token_cos'] += per_token_cos.item()
                total_losses['sequence_cos'] += sequence_cos.item()

                del loss, per_token_huber, sequence_huber, per_token_cos, sequence_cos, projected_student, student_hidden, student_outputs
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

def clip_gradients_individually(parameters, max_norm):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

def get_memory_usage():
    if torch.cuda.is_available():
        memory = torch.cuda.mem_get_info()
        memory_mib = []
        for item in memory:
            memory_mib.append(item/1048576)
        memory_used = memory_mib[1]-memory_mib[0]
        memory_mib.append(memory_used)
        return memory_mib

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
        projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)
        projection.load_state_dict(state_dict)
    except:
        print("Incompatible projection layer detected. Initializing new projection layer")
        projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)
else:
    print("Initializing projection layer")
    projection = ProjectionLayer(input_dim=T5_EMBEDDING_DIM, intermediate_dim=4096, output_dim=4096)

projection.to(device, dtype=torch.bfloat16)

hybrid_loss = ComprehensiveHybridLoss(
    student_hidden_size=4096,
    teacher_hidden_size=4096,
    per_token_huber_weight=PER_TOKEN_HUBER_LOSS,
    sequence_huber_weight=SEQUENCE_HUBER_LOSS,
    per_token_cosine_weight=PER_TOKEN_COSINE_LOSS,
    sequence_cosine_weight=SEQUENCE_COSINE_LOSS
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
train_dataloader = PrefetchDataLoader(base_train_dataloader, prefetch_factor=PREFETCH_FACTOR)
eval_dataloader = PrefetchDataLoader(base_eval_dataloader, prefetch_factor=PREFETCH_FACTOR)

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
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=MIN_LR,
)

# ========== Training Loop ==========
print("Starting training with mixed precision and gradient accumulation...")
student_model.train()

gc.collect()
torch.cuda.empty_cache()

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

        loss, per_token_huber_loss, sequence_huber_loss, per_token_cos_loss, sequence_cos_loss = hybrid_loss(
            projected_student,
            teacher_hidden,
            s_att_mask,
            t_att_mask
        )

        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(scaled_loss).backward()
        accumulation_step += 1

        if accumulation_step % GRAD_ACCUM_STEPS == 0:
            grad_norm = clip_gradients_individually(
                [p for p in student_model.parameters() if p.requires_grad] +
                list(projection.parameters()) +
                list(hybrid_loss.student_pooler.parameters()) +
                list(hybrid_loss.teacher_pooler.parameters()),
                max_norm=GRAD_CLIP
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            steps_completed_this_epoch += 1

            current_loss = loss.item()
            current_per_token_huber = per_token_huber_loss.item()
            current_sequence_huber = sequence_huber_loss.item()
            current_per_token_cos = per_token_cos_loss.item()
            current_sequence_cos = sequence_cos_loss.item()

            del loss, scaled_loss, student_outputs, student_hidden, projected_student
            del teacher_hidden, per_token_huber_loss, sequence_huber_loss, per_token_cos_loss, sequence_cos_loss
            if 't_input_ids' in locals():
                del t_input_ids, t_att_mask, teacher_outputs

            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                print(f"\nSaving checkpoint at step {global_step}")
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                os.makedirs(save_path, exist_ok=True)
                student_model.save_pretrained(save_path)
                student_tokenizer.save_pretrained(save_path)
                projection_state = projection.state_dict()
                projection_path = os.path.join(save_path, "projection_layer.safetensors")
                save_file(projection_state, projection_path)

            if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                elapsed = time.time() - start_time - eval_delta_time
                remaining_steps = total_steps - global_step
                eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0
                vram_free, vram_total, vram_used = get_memory_usage()

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                    f"Step: {global_step}/{total_steps}, "
                    f"Total Loss: {current_loss:.6f}, "
                    f"Per-Token Huber: {current_per_token_huber:.6f}, "
                    f"Sequence Huber: {current_sequence_huber:.6f}, "
                    f"Per-Token Cosine: {current_per_token_cos:.6f}, "
                    f"Sequence Cosine: {current_sequence_cos:.6f}, "
                    f"Grad Norm: {grad_norm:.6f}, "
                    f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
                    f"Elapsed: {elapsed/60:.1f} min, "
                    f"ETA: {eta/60:.1f} min"
                )

            if global_step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if accumulation_step % GRAD_ACCUM_STEPS != 0:
        grad_norm = clip_gradients_individually(
            [p for p in student_model.parameters() if p.requires_grad] +
            list(projection.parameters()) +
            list(hybrid_loss.student_pooler.parameters()) +
            list(hybrid_loss.teacher_pooler.parameters()),
            max_norm=GRAD_CLIP
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        steps_completed_this_epoch += 1
        accumulation_step = 0

    print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")

    if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
        eval_start_time = time.time()

        eval_metrics = evaluate_model(
            student_model, eval_dataloader, projection, hybrid_loss, device, autocast_dtype
        )

        avg_eval_loss = eval_metrics['total']
        print(f"\n[Validation] Epoch {epoch + 1}")
        print(f"  Average Total Loss: {avg_eval_loss:.6f}")
        print(f"  Per-Token Huber Loss: {eval_metrics['per_token_huber']:.6f}")
        print(f"  Sequence Huber Loss: {eval_metrics['sequence_huber']:.6f}")
        print(f"  Per-Token Cosine Loss: {eval_metrics['per_token_cos']:.6f}")
        print(f"  Sequence Cosine Loss: {eval_metrics['sequence_cos']:.6f}")

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

# Clean up prefetch loaders
train_dataloader.close()
eval_dataloader.close()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

student_model.save_pretrained(OUTPUT_DIR)
student_tokenizer.save_pretrained(OUTPUT_DIR)

projection_state = projection.state_dict()
projection_path = os.path.join(OUTPUT_DIR, "projection_layer.safetensors")
save_file(projection_state, projection_path)

projection_config = {
    "input_dim": T5_EMBEDDING_DIM,
    "intermediate_dim": 4096,
    "output_dim": 4096,
    "dtype": "bfloat16",
}
projection_config_path = os.path.join(OUTPUT_DIR, "projection_config.json")
with open(projection_config_path, "w") as f:
    json.dump(projection_config, f)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
