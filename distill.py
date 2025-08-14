from unsloth import FastLanguageModel
import os
import json
import time
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
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
PREFETCH_FACTOR = 16

USE_SEPARATE_EVALUATION_DATASET = True # If disabled, we'll just use 10% of the main dataset, but using unseen data is a better test of generalisation
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"
'''
Recommendation w/ Qwen3 0.6B:
24GB VRAM: BATCH SIZE = 64, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
16GB VRAM: BATCH SIZE = 32, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
12GB VRAM: BATCH SIZE = 16, GRAD ACCUM = 1, USE CACHED EMBEDDINGS = True
'''
BATCH_SIZE = 64 # Increasing this stabilises training by averaging the gradient, reducing the influence of outliers, but increases VRAM use and could somewhat inhibit useful learning from outliers. Generally we use as high as possible for VRAM budget
GRAD_ACCUM_STEPS = 1 # Accumulates the gradient across x batches before averaging to simulate larger batch size; does nothing when set to 1
EPOCHS = 1 # Number of runs over the full dataset
LEARNING_RATE = 2e-4 # Rate of learning. Too low and you stunt the learning, too high and you make excessive gradients and blow up the weights
MIN_LR = 2e-5 # Cosine scheduler reduces the learning rate towards this value over the run. This should help generalisation when using a sufficiently large dataset
GRAD_CLIP = 1.0 # Gradients of excess size are clipped to avoid excessive gradients, which leads to mangled output and further gradient explosion. The displayed grad norm while training is the value before clipping
SAVE_EVERY_X_STEPS = 500 # Save the model every x steps. Note that 1 step = 1 batch * grad_accum_steps
PRINT_EVERY_X_STEPS = 1 # Print logging every x steps. Note that 1 step = 1 batch * grad_accum_steps
EVAL_EVERY_EPOCHS = 1 # Evaluate every x epochs. This uses the evaluation dataset to test how the model performs - preferably on unseen data, so use a separate evaluation dataset
SAVE_BEST_MODEL = True # Enabling this makes sure that you always get the model with the highest evaluation score saved in a subfolder, in addition to the other saved models
'''
Note that losses are calculated as percentages of the sum before being used
Feel free to use any values so long as they are appropriately proportionate to each other;
the learning rate won't get blown out by excessive values
'''
HUBER_LOSS = 0.70 # Helps with magnitude alignment. We use the T5 mask for all losses since we're also using a transformer layer to project the Qwen3 embedding to match the length of the T5 output
COSINE_LOSS = 0.30 # Helps with directional alignment

QWEN_EMBEDDING_DIM = 1024 # Change for the hidden size of the model/features of the embedding, eg. 0.6B is 1024, 1.7B is 2048. Check model's config.json if unsure
T5_ADDITIONAL_PADDING_ATTENTION = 0

# ========== Custom Prefetch DataLoader ==========
class PrefetchDataLoader:
    def __init__(self, dataloader, prefetch_factor=3):
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.thread = None
        self.stop_event = threading.Event()
        self._exception = None
        self.restart_count = 0
        self.max_restarts = 5
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.stop_event.clear()
        self._exception = None
        self.restart_count = 0
        self._start_prefetch()
        return self

    def __next__(self):
        if self.thread is None or not self.thread.is_alive():
            if self._exception:
                raise self._exception
            else:
                self._start_prefetch()

        try:
            if self._exception:
                raise self._exception

            batch = self.queue.get(timeout=120)
            if isinstance(batch, Exception):
                raise batch
            return batch
        except queue.Empty:
            if self.thread and self.thread.is_alive():
                return self.__next__()
            else:
                # Properly handle StopIteration
                if self.iterator is not None:
                    try:
                        next(self.iterator)
                    except StopIteration:
                        pass
                raise StopIteration
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
        self.restart_count += 1
        if self.restart_count > self.max_restarts:
            raise RuntimeError(f"Prefetch thread restarted too many times ({self.max_restarts})")

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

# ========== T5 Mask Modification ==========
def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=1):
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())
        if current_seq_len < max_seq_length:
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask

# ========== Projection Layers ==========
class ProjectionLayers(torch.nn.Module):
    def __init__(self, input_dim=1024, transformer_dim=1024, output_dim=4096, dim_feedforward=4096, num_layers=1):
        super().__init__()
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
    def __init__(self, student_hidden_size=None, teacher_hidden_size=None,
                 huber_weight=0.70, cosine_weight=0.30):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight

        self.huber = torch.nn.HuberLoss(reduction='none')
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output, teacher_output, student_mask, teacher_mask):
        device = student_output.device
        dtype = torch.bfloat16

        student_output = student_output.to(dtype)
        teacher_output = teacher_output.to(dtype)
        student_mask = student_mask.to(dtype).to(device)
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
        'cos': 0.0,
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

                loss, huber_loss, cos_loss= loss_fn(
                    projected_student,
                    teacher_hidden,
                    s_att_mask,
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

def save_projection_config(projection_config_path):
    projection_config = {
        "input_dim": QWEN_EMBEDDING_DIM,
        "transformer_dim": QWEN_EMBEDDING_DIM,
        "output_dim": 4096,
        "dim_feedforward": QWEN_EMBEDDING_DIM,
        "num_layers": 1
    }
    with open(projection_config_path, "w") as f:
        json.dump(projection_config, f)

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

# ========== Initialize or Load Projection Layers ==========
projection_path = os.path.join(QWEN3_MODEL_NAME, "projection_layers.safetensors")

if os.path.exists(projection_path):
    print("Loading existing projection layers from", projection_path)
    try:
        state_dict = load_file(projection_path)
        projection = ProjectionLayers(
            input_dim=QWEN_EMBEDDING_DIM,
            transformer_dim=QWEN_EMBEDDING_DIM,
            dim_feedforward=QWEN_EMBEDDING_DIM,
            output_dim=4096
        )
        projection.load_state_dict(state_dict)
    except:
        print("Incompatible projection layers detected. Initializing new projection layers")
        projection = ProjectionLayers(
            input_dim=QWEN_EMBEDDING_DIM,
            transformer_dim=QWEN_EMBEDDING_DIM,
            dim_feedforward=QWEN_EMBEDDING_DIM,
            output_dim=4096
        )
else:
    print("Initializing projection layers")
    projection = ProjectionLayers(
        input_dim=QWEN_EMBEDDING_DIM,
        transformer_dim=QWEN_EMBEDDING_DIM,
        dim_feedforward=QWEN_EMBEDDING_DIM,
        output_dim=4096
    )

projection.to(device, dtype=torch.bfloat16)

losses = [HUBER_LOSS, COSINE_LOSS]
sum_loss = sum(losses)
normalized_losses = [loss / sum_loss for loss in losses]

hybrid_loss = HybridLoss(
    student_hidden_size=4096,
    teacher_hidden_size=4096,
    huber_weight=normalized_losses[0],
    cosine_weight=normalized_losses[1]
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

base_train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=0,
    persistent_workers=False
)

base_eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    persistent_workers=False
)

train_dataloader = PrefetchDataLoader(base_train_dataloader, prefetch_factor=PREFETCH_FACTOR)
eval_dataloader = PrefetchDataLoader(base_eval_dataloader, prefetch_factor=PREFETCH_FACTOR)

# ========== Training Setup ==========
autocast_dtype = torch.bfloat16
scaler = GradScaler(enabled=False)

optimizer = torch.optim.AdamW(
    [p for p in student_model.parameters() if p.requires_grad] +
    list(projection.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8
)

total_steps = (len(train_dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
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

            t_att_mask = modify_mask_to_attend_padding(
                t_att_mask,
                t_att_mask.shape[-1],
                num_extra_padding=T5_ADDITIONAL_PADDING_ATTENTION
            )

            loss, huber_loss, cos_loss = hybrid_loss(
                projected_student,
                teacher_hidden,
                s_att_mask,
                t_att_mask
            )

            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(scaled_loss).backward()
            accumulation_step += 1

            if accumulation_step >= GRAD_ACCUM_STEPS or batch_idx == len(train_dataloader) - 1:
                grad_norm = clip_gradients_individually(
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

                del loss, scaled_loss, student_outputs, student_hidden, projected_student
                del teacher_hidden, huber_loss, cos_loss
                if 't_input_ids' in locals():
                    del t_input_ids, t_att_mask, teacher_outputs

                if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                    print(f"\nSaving checkpoint at step {global_step}")
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    student_model.save_pretrained(save_path)
                    student_tokenizer.save_pretrained(save_path)

                    projection_state = projection.state_dict()
                    projection_path = os.path.join(save_path, "projection_layers.safetensors")
                    projection_config_path = os.path.join(save_path, "projection_config.json")
                    save_projection_config(projection_config_path)

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
                        f"Huber Loss: {current_huber:.6f}, "
                        f"Cosine Loss: {current_cos:.6f}, "
                        f"Grad Norm: {grad_norm:.6f}, "
                        f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
                        f"Elapsed: {elapsed/60:.1f} min, "
                        f"ETA: {eta/60:.1f} min"
                    )

                if global_step % 100 == 0 or global_step == 1:
                    gc.collect()
                    torch.cuda.empty_cache()

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
            # Reset accumulation step on error
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
            os.makedirs(best_model_dir, exist_ok=True)

            student_model.save_pretrained(best_model_dir)
            student_tokenizer.save_pretrained(best_model_dir)

            projection_state = projection.state_dict()
            projection_path = os.path.join(best_model_dir, "projection_layers.safetensors")
            projection_config_path = os.path.join(best_model_dir, "projection_config.json")
            save_projection_config(projection_config_path)

            save_file(projection_state, projection_path)


        eval_end_time = time.time()
        eval_delta_time += (eval_end_time - eval_start_time)
        student_model.train()



train_dataloader.close()
eval_dataloader.close()

# ========== Save Final Model ==========
print(f"\nSaving final model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

student_model.save_pretrained(OUTPUT_DIR)
student_tokenizer.save_pretrained(OUTPUT_DIR)

projection_state = projection.state_dict()
projection_path = os.path.join(OUTPUT_DIR, "projection_layers.safetensors")
projection_config_path = os.path.join(OUTPUT_DIR, "projection_config.json")
save_projection_config(projection_config_path)

save_file(projection_state, projection_path)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
