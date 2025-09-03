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
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt" # Each line of the dataset text file is taken as one prompt
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/QT-embedder-v1/"

USE_CACHED_EMBEDDINGS = True # Each T5-xxl embedding is cached; size per is 4MB so multiply by dataset size for capacity required
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2" # Cache is picked up on subsequent runs by reference to dataset file name
PREFETCH_FACTOR = 32

USE_SEPARATE_EVALUATION_DATASET = True # Otherwise we take some of the main dataset, but best to use unseen data
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

ENABLE_LOGGING = True
WRITE_TO_LOG_EVERY_X_STEPS = 10

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
GRAD_CLIP = 1.0

EPOCHS = 1

MAX_LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5

SAVE_EVERY_X_STEPS = 500
SAVE_EVERY_X_RESTARTS = 1
SAVE_EVERY_X_EPOCHS = 1

PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_X_EPOCHS = 1
SAVE_BEST_MODEL = True

PER_TOKEN_HUBER_LOSS = 0.50
PER_TOKEN_COSINE_LOSS = 0.20
SEQUENCE_HUBER_LOSS = 0.20
SEQUENCE_COSINE_LOSS = 0.10

WARMUP_STEPS = 501 # Set to 0 to disable warmup
RESTART_PERIOD_STEPS = 1150 # Set to 0 to use linear scheduler instead

SHUFFLE_DATASET = False # Random order probably better, but might incur bottlenecking due to random read overhead

ENABLE_STUDENT_WORD_DROPOUT = False # Removes words from the student prompt according to the below ratio
STUDENT_WORD_DROPOUT_RATIO = 0.00 # This probability is applied on a per-word basis. Words are defined as any section delineated by spaces
SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True

TRAIN_PROJECTION = True
TRAIN_MODEL = True

USE_LINEAR_PROJECTION = True # Use a simple linear projection layer instead of the mlp -> transformer projection. Might be useful for initially training the model, then swapping in the full projection and training that with the model frozen
USE_STUDENT_MASK_FOR_SEQUENCE_LOSS = True # Use the student mask for sequence loss. This is probably better for training with the linear projection, then you can swap to the teacher model when training the projection

TOKEN_ALIGNMENT_WINDOW = 3 # We try to match tokens for per-token loss by matching identical text content, looking ahead by this many tokens

FEED_FORWARD_DIM = 4096
TRANSFORMER_LAYERS = 1

# ========== Experimental Configuration ==========
'''
Experimental or testing options that may or may not be beneficial
'''
ENHANCED_DATASET = True # Will enable a secondary dataset that is swapped in according to the below ratios
ENHANCED_DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"

UNTAMPERED_STUDENT_AND_TEACHER_RATIO = 0.50 # No swapping
ENHANCED_TEACHER_EMBEDDING_RATIO = 0.00 # Teacher prompt or embedding is swapped for enhanced but the student is the same. I'm mostly using this to encourage some over-projection for experimental purposes, and I don't know how well it functions in reality
ENHANCED_STUDENT_AND_TEACHER_RATIO = 0.50 # Teacher and student prompt or embedding is swapped for enhanced

DEBUG_PRINT = False # Mostly just prints out student/teacher prompt pairings atm

# ========== Debug Function ==========
def debug(description, variable=None):
    if DEBUG_PRINT:
        print(f'{description}')
        if variable is not None:
            print(f'{variable}')

# ========== Token Alignment ==========
def ids_to_tokens(token_ids, tokenizer):
    """Convert token IDs to their string representations"""
    return tokenizer.convert_ids_to_tokens(token_ids)

def normalize_token(token):
    """
    Normalize tokens from different tokenizers to a comparable form.
    """
    # Remove common prefix markers
    token = token.replace('Ġ', '')  # Qwen3 word prefix
    token = token.replace('▁', '')  # T5 word prefix

    # Remove other common prefix/suffix markers
    token = token.replace('▔', '')
    token = token.replace('▃', '')
    token = token.replace('�', '')

    # Handle punctuation and special characters
    token = token.replace(' .', '.')
    token = token.replace(' ,', ',')
    token = token.replace(' !', '!')
    token = token.replace(' ?', '?')
    token = token.replace(' :', ':')
    token = token.replace(' ;', ';')

    # Handle parentheses and brackets
    token = token.replace(' (', '(')
    token = token.replace(' )', ')')
    token = token.replace(' [', '[')
    token = token.replace(' ]', ']')
    token = token.replace(' {', '{')
    token = token.replace(' }', '}')

    token = token.lower()

    return token

def token_based_alignment(student_input_ids, teacher_input_ids,
                         student_tokenizer, teacher_tokenizer,
                         window_size=5):
    """
    Align tokens based on their textual content.

    Args:
        student_input_ids: Tensor of student token IDs
        teacher_input_ids: Tensor of teacher token IDs
        student_tokenizer: Student model tokenizer
        teacher_tokenizer: Teacher model tokenizer
        window_size: How many tokens to look ahead for matches
    """
    # Convert token IDs to tokens (strings)
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    aligned_pairs = []

    # Create normalized versions for comparison
    normalized_student = [normalize_token(token) for token in student_tokens]
    normalized_teacher = [normalize_token(token) for token in teacher_tokens]

    for i, norm_stu in enumerate(normalized_student):
        # Skip special tokens (like padding, start, end tokens)
        if student_tokens[i] in [student_tokenizer.pad_token,
                                student_tokenizer.bos_token,
                                student_tokenizer.eos_token]:
            continue

        # Search in a window around the current position
        start_j = max(0, i - window_size)
        end_j = min(len(teacher_tokens), i + window_size + 1)

        for j in range(start_j, end_j):
            # Skip special tokens on teacher side too
            if teacher_tokens[j] in [teacher_tokenizer.pad_token,
                                    teacher_tokenizer.bos_token,
                                    teacher_tokenizer.eos_token]:
                continue

            norm_tea = normalized_teacher[j]

            if norm_stu == norm_tea:
                aligned_pairs.append((i, j))
                break  # take the first match

    return aligned_pairs

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

        self.enhanced_lines = []
        if ENHANCED_DATASET:
            with open(ENHANCED_DATASET_PATH, "r", encoding="utf-8") as f:
                self.enhanced_lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(self.enhanced_lines) < len(self.lines):
                self.enhanced_lines += self.lines[len(self.enhanced_lines):]
            elif len(self.enhanced_lines) > len(self.lines):
                self.enhanced_lines = self.enhanced_lines[:len(self.lines)]

        ratios = [UNTAMPERED_STUDENT_AND_TEACHER_RATIO]
        if ENHANCED_DATASET:
            ratios.append(ENHANCED_TEACHER_EMBEDDING_RATIO)
            ratios.append(ENHANCED_STUDENT_AND_TEACHER_RATIO)
        else:
            ratios.append(0)
            ratios.append(0)

        self.enabled_ratios = ratios

        self.num_ratios = len(self.enabled_ratios)

        self.student_raw_lines = self.lines
        self.teacher_raw_lines = self.lines
        self.enhanced_teacher_raw_lines = self.enhanced_lines

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length

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

                if ENHANCED_DATASET:
                    enhanced_base_name = os.path.basename(ENHANCED_DATASET_PATH)
                    enhanced_cache_folder = os.path.join(cache_path, enhanced_base_name)
                    enhanced_validation_file = os.path.join(enhanced_cache_folder, f"{enhanced_base_name}.validation")

                    if os.path.exists(enhanced_validation_file):
                        self.enhanced_cache_folder = enhanced_cache_folder
                        self.enhanced_embedding_files = [os.path.join(enhanced_cache_folder, f"{i}.pt") for i in range(len(self.enhanced_lines))]
                        self.enhanced_mask_files = [os.path.join(enhanced_cache_folder, f"{i}_mask.pt") for i in range(len(self.enhanced_lines))]
                    else:
                        print(f"Generating and caching enhanced embeddings for {ENHANCED_DATASET_PATH}")
                        os.makedirs(enhanced_cache_folder, exist_ok=True)
                        for i, line in enumerate(tqdm(self.enhanced_lines, desc="Generating enhanced embeddings")):
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

                            embedding_file = os.path.join(enhanced_cache_folder, f"{i}.pt")
                            mask_file = os.path.join(enhanced_cache_folder, f"{i}_mask.pt")
                            torch.save(embeddings, embedding_file)
                            torch.save(att_mask.cpu(), mask_file)

                        with open(enhanced_validation_file, "w") as f:
                            pass
                        self.enhanced_cache_folder = enhanced_cache_folder
                        self.enhanced_embedding_files = [os.path.join(enhanced_cache_folder, f"{i}.pt") for i in range(len(self.enhanced_lines))]
                        self.enhanced_mask_files = [os.path.join(enhanced_cache_folder, f"{i}_mask.pt") for i in range(len(self.enhanced_lines))]
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
                            input_ids=input_ids.to(teacher_model.devive),
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
            pass

        self.use_cached_embeddings = use_cached_embeddings
        self.line_index = list(range(len(self.lines)))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        def apply_dropout(line, word_dropout_ratio):
            if word_dropout_ratio > 0:
                words = line.split()
                kept_words = []
                for word in words:
                    if random.random() > word_dropout_ratio:
                        kept_words.append(word)
                line = " ".join(kept_words)
            return line

        if self.enabled_ratios:
            choice = random.choices(range(self.num_ratios), weights=self.enabled_ratios)[0]

            if choice == 0:  # PRIMARY_DATASET_RATIO
                debug("\n-----\nStudent line from primary dataset:")
                student_line = self.student_raw_lines[idx]
                teacher_line = self.teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                teacher_type = "original"

            elif choice == 1:  # ENHANCED_EMBEDDING_RATIO
                debug("\n-----\nStudent line from primary dataset:")
                student_line = self.student_raw_lines[idx]
                teacher_line = self.enhanced_teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                teacher_type = "enhanced"

            elif choice == 2:  # ENHANCED_PROMPT_AND_EMBEDDING_RATIO
                debug("\n-----\nStudent line from enhanced dataset:")
                student_line = self.enhanced_teacher_raw_lines[idx]
                teacher_line = self.enhanced_teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                teacher_type = "enhanced"
            else:
                debug("\n-----\nERROR: Student line invalid choice selection!")

        if choice == 1 and SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER == True:
            student_dropout_word = 0

        student_line = apply_dropout(student_line, student_dropout_word)

        debug(student_line)
        debug("\nTeacher line:", teacher_line)
        debug("-----")

        student_inputs = self.student_tokenizer(
            text=student_line,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        student_input_ids = torch.tensor(student_inputs["input_ids"], dtype=torch.long)
        student_attention_mask = torch.tensor(student_inputs["attention_mask"], dtype=torch.long)

        teacher_inputs = self.teacher_tokenizer(
            text=teacher_line,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        teacher_mask = torch.tensor(teacher_inputs["attention_mask"], dtype=torch.long)
        teacher_input_ids = torch.tensor(teacher_inputs["input_ids"], dtype=torch.long)

        if self.use_cached_embeddings:
            if teacher_type == "enhanced" and hasattr(self, 'enhanced_embedding_files'):
                embeddings = torch.load(self.enhanced_embedding_files[idx], map_location='cpu')
                att_mask = torch.load(self.enhanced_mask_files[idx], map_location='cpu')
            else:
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
                student_input_ids,
                student_attention_mask,
                teacher_input_ids,
                embeddings,
                teacher_mask,
                student_line,
                teacher_line
            )
        else:
            return (
                student_input_ids,
                student_attention_mask,
                teacher_input_ids,
                teacher_mask,
                student_line,
                teacher_line
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'file_handles'):
            for file in self.file_handles:
                file.close()
            self.file_handles.clear()

# ========== Linear Projection Layer ==========
class LinearProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

# ========== Transformer Projection Layer ==========
class TransformerProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_dim, output_dim=4096, dim_feedforward=2048, num_layers=1):
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

# ========== Loss Function ==========
class HybridLoss(torch.nn.Module):
    def __init__(self, student_hidden_size, teacher_hidden_size,
                 huber_weight=0.4, cosine_weight=0.3, token_huber_weight=0.2, token_cosine_weight=0.1,
                 huber_delta=1.0, student_tokenizer=None, teacher_tokenizer=None):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.token_huber_weight = token_huber_weight
        self.token_cosine_weight = token_cosine_weight

        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='none')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

    def forward(self, student_output, teacher_output, student_mask, teacher_mask,
                student_input_ids=None, teacher_input_ids=None, student_text=None, teacher_text=None):
        device = student_output.device
        batch_size = student_output.size(0)

        if student_mask.dtype != torch.bool:
            student_mask = student_mask.bool()
        if teacher_mask.dtype != torch.bool:
            teacher_mask = teacher_mask.bool()
        if USE_STUDENT_MASK_FOR_SEQUENCE_LOSS:
            seq_student_mask = student_mask
        else:
            seq_student_mask = teacher_mask

        student_pooled = (student_output * seq_student_mask.unsqueeze(-1)).sum(dim=1) / seq_student_mask.sum(dim=1, keepdim=True).clamp(min=1)
        teacher_pooled = (teacher_output * teacher_mask.unsqueeze(-1)).sum(dim=1) / teacher_mask.sum(dim=1, keepdim=True).clamp(min=1)

        seq_huber = self.huber_loss(student_pooled, teacher_pooled).mean()
        seq_cos_sim = self.cos_loss(student_pooled, teacher_pooled)
        seq_cos_loss = (1 - seq_cos_sim).mean()

        token_huber = torch.tensor(0.0, device=device)
        token_cos_loss = torch.tensor(0.0, device=device)
        num_aligned_tokens = 0
        num_aligned_seqs = 0

        for i in range(batch_size):
            student_valid = student_mask[i]
            teacher_valid = teacher_mask[i]
            student_seq = student_output[i][student_valid]
            teacher_seq = teacher_output[i][teacher_valid]

            if student_seq.size(0) == 0 or teacher_seq.size(0) == 0:
                continue

            # Use token-based alignment instead of greedy alignment
            path = token_based_alignment(
                student_input_ids[i], teacher_input_ids[i],
                self.student_tokenizer, self.teacher_tokenizer,
                window_size=TOKEN_ALIGNMENT_WINDOW
            )

            if len(path) == 0:
                continue

            num_aligned_seqs += 1

            student_aligned = []
            teacher_aligned = []
            for pair in path:
                stu_idx, tea_idx = pair
                # We need to make sure we're not accessing out-of-bounds indices
                if stu_idx < student_seq.size(0) and tea_idx < teacher_seq.size(0):
                    student_aligned.append(student_seq[stu_idx])
                    teacher_aligned.append(teacher_seq[tea_idx])

            if len(student_aligned) == 0:
                continue

            student_aligned = torch.stack(student_aligned)
            teacher_aligned = torch.stack(teacher_aligned)

            token_huber += self.huber_loss(student_aligned, teacher_aligned).mean()
            token_cos_sim = self.cos_loss(student_aligned, teacher_aligned)
            token_cos_loss += (1 - token_cos_sim).mean()
            num_aligned_tokens += len(path)

        if num_aligned_seqs > 0:
            token_huber /= num_aligned_seqs
            token_cos_loss /= num_aligned_seqs
        else:
            token_huber = torch.tensor(0.0, device=device)
            token_cos_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.huber_weight * seq_huber +
            self.cosine_weight * seq_cos_loss +
            self.token_huber_weight * token_huber +
            self.token_cosine_weight * token_cos_loss
        )

        return total_loss, seq_huber, seq_cos_loss, token_huber, token_cos_loss, num_aligned_tokens

# ========== Evaluation Function ==========
def evaluate_model(model, dataloader, projection, loss_fn, device, autocast_dtype):
    model.eval()
    projection.eval()
    loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'huber': 0.0,
        'cos': 0.0,
        't_huber': 0.0,
        't_cosine': 0.0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                if USE_CACHED_EMBEDDINGS:
                    s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask, student_text, teacher_text = batch
                else:
                    s_input_ids, s_mask, t_input_ids, t_mask, student_text, teacher_text = batch

                s_input_ids = s_input_ids.to(device)
                t_mask = t_mask.to(device)
                s_mask = s_mask.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    student_outputs = model(
                        input_ids=s_input_ids,
                        attention_mask=s_mask,
                        output_hidden_states=True
                    )
                    student_hidden = student_outputs.hidden_states[-1]
                    projected_student = projection(student_hidden)

                if USE_CACHED_EMBEDDINGS:
                    teacher_hidden = t_embeddings.to(device).squeeze(1)
                else:
                    t_input_ids = t_input_ids.to(device)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=t_input_ids,
                            attention_mask=t_mask
                        )
                        teacher_hidden = teacher_outputs.last_hidden_state
                        teacher_hidden = teacher_hidden.to(device)

                loss, huber_loss, cos_loss, t_huber_loss, t_cos_loss, _ = loss_fn(
                    projected_student,
                    teacher_hidden,
                    s_mask,
                    t_mask,
                    student_input_ids=s_input_ids,
                    teacher_input_ids=t_input_ids,
                    student_text=student_text,
                    teacher_text=teacher_text
                )

                total_losses['total'] += loss.item()
                total_losses['huber'] += huber_loss.item()
                total_losses['cos'] += cos_loss.item()
                total_losses['t_huber'] += t_huber_loss.item()
                total_losses['t_cosine'] += t_cos_loss.item()

                del loss, huber_loss, cos_loss, t_huber_loss, t_cos_loss, projected_student, student_hidden, student_outputs
                if not USE_CACHED_EMBEDDINGS and 'teacher_hidden' in locals():
                    del teacher_hidden, teacher_outputs
                del s_input_ids, s_mask, t_mask
                if USE_CACHED_EMBEDDINGS:
                    del t_embeddings
                else:
                    del t_input_ids

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
                del s_input_ids, s_mask, t_mask
                if USE_CACHED_EMBEDDINGS:
                    del t_embeddings
                else:
                    del t_input_ids
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
    if USE_LINEAR_PROJECTION:
        projection_config = {
            "input_dim": embedding_dim,
            "output_dim": 4096,
        }
    else:
        projection_config = {
            "input_dim": embedding_dim,
            "transformer_dim": embedding_dim,
            "output_dim": 4096,
            "dim_feedforward": FEED_FORWARD_DIM,
            "num_layers": TRANSFORMER_LAYERS,
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

def get_logging():
    log_line = (
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
        f"Step: {global_step}/{total_steps}, "
        f"Total Loss: {current_loss:.6f}, "
        f"PT Huber Loss: {current_t_huber:.6f}, "
        f"PT Cosine Loss: {current_t_cos:.6f}, "
        f"SQ Huber Loss: {current_huber:.6f}, "
        f"SQ Cosine Loss: {current_cos:.6f}, "
        f"Aligned Tokens: {num_aligned_tokens}, "
        f"Grad Norm: {grad_norm:.6f}, "
        f"Learning Rate: {current_lr:.6f}, "
        f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
        f"Elapsed: {elapsed/60:.1f} min, "
        f"ETA: {eta/60:.1f} min"
    )
    return log_line

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
    enhanced_base_name = os.path.basename(ENHANCED_DATASET_PATH)
    enhanced_cache_folder = os.path.join(CACHE_PATH, enhanced_base_name)
    enhanced_validation_file = os.path.join(enhanced_cache_folder, f"{enhanced_base_name}.validation")
    if not os.path.exists(validation_file) or (ENHANCED_DATASET and not os.path.exists(enhanced_validation_file)):
        teacher_model = T5EncoderModel.from_pretrained(
        T5_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
        teacher_model.eval()

# ========== Initialize or Load Projection Layer ==========
projection_path = os.path.join(QWEN3_MODEL_NAME, "projection_layer.safetensors")

if USE_LINEAR_PROJECTION:
    if os.path.exists(projection_path):
        print("Loading existing projection layer from", projection_path)
        try:
            state_dict = load_file(projection_path)
            projection = LinearProjectionLayer(
                input_dim=qwen_embedding_dim,
                output_dim=4096,
            )
            projection.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading projection layer: {e}")
            print("Initializing projection layer")
            projection = LinearProjectionLayer(
                input_dim=qwen_embedding_dim,
                output_dim=4096,
            )
    else:
        print("Initializing projection layer")
        projection = LinearProjectionLayer(
            input_dim=qwen_embedding_dim,
            output_dim=4096,
        )
else:
    if os.path.exists(projection_path):
        print("Loading existing projection layer from", projection_path)
        try:
            state_dict = load_file(projection_path)
            projection = TransformerProjectionLayer(
                input_dim=qwen_embedding_dim,
                transformer_dim=qwen_embedding_dim,
                output_dim=4096,
                dim_feedforward=FEED_FORWARD_DIM,
                num_layers=TRANSFORMER_LAYERS,
            )
            projection.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading projection layer: {e}")
            print("Initializing projection layer")
            projection = TransformerProjectionLayer(
                input_dim=qwen_embedding_dim,
                transformer_dim=qwen_embedding_dim,
                output_dim=4096,
                dim_feedforward=FEED_FORWARD_DIM,
                num_layers=TRANSFORMER_LAYERS,
            )
    else:
        print("Initializing projection layer")
        projection = TransformerProjectionLayer(
            input_dim=qwen_embedding_dim,
            transformer_dim=qwen_embedding_dim,
            output_dim=4096,
            dim_feedforward=FEED_FORWARD_DIM,
            num_layers=TRANSFORMER_LAYERS,
        )

losses = [SEQUENCE_HUBER_LOSS, SEQUENCE_COSINE_LOSS, PER_TOKEN_HUBER_LOSS, PER_TOKEN_COSINE_LOSS]
sum_loss = sum(losses)
normalised_losses = [loss / sum_loss for loss in losses]

hybrid_loss = HybridLoss(
    huber_weight=normalised_losses[0],
    cosine_weight=normalised_losses[1],
    token_huber_weight=normalised_losses[2],
    token_cosine_weight=normalised_losses[3],
    student_hidden_size=4096,
    teacher_hidden_size=4096,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer
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

with train_dataset as train_ds, eval_dataset as eval_ds:
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE_DATASET,
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

        global_step = 0
        accumulation_step = 0
        grad_norm = 0
        restart_count = 0

        if ENABLE_LOGGING:
            log_dir = os.path.join(OUTPUT_DIR, "logging")
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            log_filename = f"training_log_{current_time}.txt"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, log_filename)
            log_lines = []

        model_parameters = []
        projection_parameters = []
        if TRAIN_MODEL == True:
            model_parameters = [p for p in student_model.parameters() if p.requires_grad]
        if TRAIN_PROJECTION == True:
            projection_parameters = list(projection.parameters())
        if TRAIN_MODEL == False and TRAIN_PROJECTION == False:
            print("You can't disable both model and projection training! Enabling model training...")
            model_parameters = [p for p in student_model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            model_parameters +
            projection_parameters,
            lr=MAX_LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        if RESTART_PERIOD_STEPS == 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps - WARMUP_STEPS
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=RESTART_PERIOD_STEPS,
                T_mult=1,
                eta_min=MIN_LEARNING_RATE,
            )

        if WARMUP_STEPS > 0:
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
        student_model.train()

        start_time = time.time()
        eval_delta_time = 0
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            steps_completed_this_epoch = 0
            accumulation_step = 0

            print(f"Starting epoch {epoch + 1}/{EPOCHS}")
            print(f"Total batches in epoch: {len(train_dataloader)}")

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    if USE_CACHED_EMBEDDINGS:
                        s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask, student_text, teacher_text = batch
                    else:
                        s_input_ids, s_mask, t_input_ids, t_mask, student_text, teacher_text = batch

                    s_input_ids = s_input_ids.to(device)
                    s_mask = s_mask.to(device)
                    t_mask = t_mask.to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        student_outputs = student_model(
                            input_ids=s_input_ids,
                            attention_mask=s_mask,
                            output_hidden_states=True
                        )
                        student_hidden = student_outputs.hidden_states[-1]
                        projected_student = projection(student_hidden)

                    if USE_CACHED_EMBEDDINGS:
                        teacher_hidden = t_embeddings.to(device).squeeze(1)
                    else:
                        t_input_ids = t_input_ids.to(device)
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                input_ids=t_input_ids,
                                attention_mask=t_mask
                            )
                            teacher_hidden = teacher_outputs.last_hidden_state
                            teacher_hidden = teacher_hidden.to(device)
                            del teacher_outputs

                    loss, huber_loss, cos_loss, t_huber_loss, t_cos_loss, num_aligned_tokens = hybrid_loss(
                        projected_student,
                        teacher_hidden,
                        s_mask,
                        t_mask,
                        student_input_ids=s_input_ids,
                        teacher_input_ids=t_input_ids,
                        student_text=student_text,
                        teacher_text=teacher_text
                    )

                    scaled_loss = loss / GRAD_ACCUM_STEPS
                    scaler.scale(scaled_loss).backward()
                    accumulation_step += 1

                    if accumulation_step >= GRAD_ACCUM_STEPS or batch_idx == len(train_dataset) - 1:
                        grad_norm = clip_grad_norm_(
                            model_parameters +
                            projection_parameters,
                            max_norm=GRAD_CLIP
                        )

                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1
                        steps_completed_this_epoch += 1
                        accumulation_step = 0

                        current_loss = loss.item()
                        current_huber = huber_loss.item()
                        current_cos = cos_loss.item()
                        current_t_huber = t_huber_loss.item()
                        current_t_cos = t_cos_loss.item()

                        elapsed = time.time() - start_time - eval_delta_time
                        remaining_steps = total_steps - global_step
                        eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0
                        vram_free, vram_total, vram_used = get_memory_usage()
                        current_lr = scheduler.get_last_lr()[0]

                        if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                            print(get_logging())

                        if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                            print(f"\nSaving checkpoint at step {global_step}\n")
                            save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                            save_trained_model(save_path, student_model, student_tokenizer, projection, qwen_embedding_dim)

                        if RESTART_PERIOD_STEPS > 0:
                            if global_step > WARMUP_STEPS and (global_step - WARMUP_STEPS) % RESTART_PERIOD_STEPS == 0:
                                next_restart = restart_count + 1
                                if SAVE_EVERY_X_RESTARTS > 0 and next_restart % SAVE_EVERY_X_RESTARTS == 0:
                                    print(f"\nSaving checkpoint at restart {next_restart}\n")
                                    save_path = os.path.join(OUTPUT_DIR, f"restart_{next_restart}")
                                    save_trained_model(save_path, student_model, student_tokenizer, projection, qwen_embedding_dim)
                                restart_count += 1

                        if ENABLE_LOGGING:
                            log_lines.append(get_logging())
                            if global_step % WRITE_TO_LOG_EVERY_X_STEPS == 0:
                                with open(log_file, "a") as f:
                                    for line in log_lines:
                                        f.write(line + "\n")
                                log_lines.clear()

                        scheduler.step()
                        optimizer.zero_grad()

                        del loss, scaled_loss, student_outputs, student_hidden, projected_student, teacher_hidden, huber_loss, cos_loss, t_huber_loss, t_cos_loss
                        if 't_input_ids' in locals():
                            del t_input_ids

                    del s_input_ids, s_mask, t_mask
                    if USE_CACHED_EMBEDDINGS:
                        del t_embeddings
                    else:
                        del t_input_ids

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
                    if 't_huber_loss' in locals():
                        del t_huber_loss
                    if 't_cos_loss' in locals():
                        del t_cos_loss
                    if 't_input_ids' in locals():
                        del t_input_ids
                    del s_input_ids, s_mask, t_mask
                    if USE_CACHED_EMBEDDINGS:
                        del t_embeddings
                    else:
                        del t_input_ids

                    accumulation_step = 0
                    optimizer.zero_grad()
                    continue

            print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")

            next_epoch = epoch + 1
            if next_epoch % SAVE_EVERY_X_EPOCHS == 0:
                print(f"\nSaving checkpoint at epoch {next_epoch}\n")
                save_path = os.path.join(OUTPUT_DIR, f"epoch_{next_epoch}")
                save_trained_model(save_path, student_model, student_tokenizer, projection, qwen_embedding_dim)

            if next_epoch % EVAL_EVERY_X_EPOCHS == 0:
                eval_start_time = time.time()

                eval_metrics = evaluate_model(
                    student_model, eval_dataloader, projection, hybrid_loss, device, autocast_dtype)

                avg_eval_loss = eval_metrics['total']
                print(f"\n[Validation] Epoch {epoch + 1}")
                print(f"  Average Total Loss: {avg_eval_loss:.6f}")
                print(f"  Huber Loss: {eval_metrics['huber']:.6f}")
                print(f"  Cosine Loss: {eval_metrics['cos']:.6f}")
                print(f"  Token Huber Loss: {eval_metrics['t_huber']:.6f}")
                print(f"  Token Cosine Loss: {eval_metrics['t_cosine']:.6f}")

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
