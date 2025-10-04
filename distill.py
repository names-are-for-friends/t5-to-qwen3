from unsloth import FastLanguageModel
import os
import json
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
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
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.DEBUG)

# ========== Configuration ==========
# Paths
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt"
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saikou/QT-embedder-v1/"

# Caching
USE_CACHED_EMBEDDINGS = True
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2"
PREFETCH_FACTOR = 16

# Evaluation
USE_SEPARATE_EVALUATION_DATASET = True
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

# Logging
ENABLE_LOGGING = True
WRITE_TO_LOG_EVERY_X_STEPS = 10

# Training parameters
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
GRAD_CLIP = 1.0
EPOCHS = 2

# Learning rates
MAX_LEARNING_RATE_MODEL = 5e-5
MIN_LEARNING_RATE_MODEL = 5e-6
MAX_LEARNING_RATE_TRANSFORMER = 8e-5
MIN_LEARNING_RATE_TRANSFORMER = 8e-6
MAX_LEARNING_RATE_MLP = 1e-4
MIN_LEARNING_RATE_MLP = 1e-5
MAX_LEARNING_RATE_LINEAR = 1.2e-4
MIN_LEARNING_RATE_LINEAR = 1.2e-5
MAX_LEARNING_RATE_EXPANSION = 9e-5
MIN_LEARNING_RATE_EXPANSION = 9e-6

# Saving
SAVE_EVERY_X_STEPS = 0
SAVE_EVERY_X_RESTARTS = 1
SAVE_EVERY_X_EPOCHS = 1

# Printing
PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_X_EPOCHS = 1
SAVE_BEST_MODEL = True

# Loss weights
ALIGN_HUBER_WEIGHT = 0.7
ALIGN_COSINE_WEIGHT = 0.3
TOKEN_HUBER_WEIGHT = 0.0
TOKEN_COSINE_WEIGHT = 0.0
SEQUENCE_HUBER_WEIGHT = 0.0
SEQUENCE_COSINE_WEIGHT = 0.0

# Scheduler
WARMUP_STEPS = 500
RESTART_CYCLE_STEPS = 1000
REPEAT_WARMUP_AFTER_RESTART = True

# Dataset
SHUFFLE_DATASET = True

# Training flags
TRAIN_PROJECTION = True
TRAIN_MODEL = True

# Debugging
LOG_VRAM_USAGE = False
EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS = []

# Enhanced dataset
ENHANCED_DATASET = True
ENHANCED_DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"
UNTAMPERED_STUDENT_AND_TEACHER_RATIO = 0.50
ENHANCED_TEACHER_EMBEDDING_RATIO = 0.00
ENHANCED_STUDENT_AND_TEACHER_RATIO = 0.50

# Dropout
ENABLE_STUDENT_WORD_DROPOUT = False
STUDENT_WORD_DROPOUT_RATIO = 0.10
ENABLE_STUDENT_TOKEN_DROPOUT = False
STUDENT_TOKEN_DROPOUT_RATIO = 0.10
SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True

# Alignment
TOKEN_ALIGNMENT_WINDOW = 10

# Projection layers configuration
PROJECTION_LAYERS_CONFIG = [
    {
        "type": "transformer",
        "input_dim": 1024,
        "hidden_dim": 1024,
        "dim_feedforward": 4096,
        "file_num": 1,
    },
    {
        "type": "mlp",
        "input_dim": 1024,
        "hidden_dim": 1024,
        "file_num": 2,
    },
    {
        "type": "linear",
        "input_dim": 1024,
        "output_dim": 4096,
        "file_num": 3,
    }
]

# ========== Projection Layers ==========
class LinearProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        self.layer_type = "linear"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear(x)

class MLPProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.GELU()
        self.output_dim = hidden_dim
        self.layer_type = "mlp"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x

class TransformerProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dim_feedforward: int, num_layers: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_dim = hidden_dim
        self.layer_type = "transformer"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.linear(x)
        x = self.transformer(x)
        return x

class ContinuousExpansionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, teacher_dim: int):
        super().__init__()
        # Learn position embeddings for any position
        self.position_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 64),  # Input is normalized position [0,1]
            torch.nn.GELU(),
            torch.nn.Linear(64, input_dim)
        )

        # Learn how to transform at each position
        self.position_transform = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, teacher_dim),  # [token + position]
            torch.nn.GELU(),
            torch.nn.LayerNorm(teacher_dim)
        )

        # Learn to blend neighboring tokens
        self.neighbor_weights = torch.nn.Parameter(torch.ones(3))  # [prev, current, next]
        self.output_dim = teacher_dim
        self.layer_type = "expansion"

    def forward(self, student_emb, s_mask, t_mask, target_length=512):
        batch_size, s_len, _ = student_emb.shape
        # The actual length of the teacher sequence for each item in the batch
        t_lens = t_mask.sum(dim=1)
        t_len = t_lens.max()

        # Get the actual length of the student sequence for each item
        s_lens = s_mask.sum(dim=1)

        # Create a grid of target positions for all samples in the batch
        # Shape: (batch_size, t_len)
        target_positions = torch.linspace(0, 1, t_len, device=student_emb.device)
        pos_grid = target_positions.unsqueeze(0).expand(batch_size, -1)

        # Map target positions to corresponding student positions
        # Shape: (batch_size, t_len)
        s_pos_grid = pos_grid * (s_lens.unsqueeze(1).float() - 1)

        # Get integer indices for current, previous, and next tokens
        # Shape: (batch_size, t_len)
        curr_idx = s_pos_grid.long()
        prev_idx = torch.clamp(curr_idx - 1, 0, s_len - 1)
        next_idx = torch.clamp(curr_idx + 1, 0, s_len - 1)

        # Create batch indices for advanced indexing
        # Shape: (batch_size, t_len)
        batch_indices = torch.arange(batch_size, device=student_emb.device).unsqueeze(1).expand(-1, t_len)

        # Gather embeddings for all positions at once
        # Shape: (batch_size, t_len, input_dim)
        curr_emb = student_emb.gather(1, curr_idx.unsqueeze(-1).expand(-1, -1, student_emb.shape[-1]))
        prev_emb = student_emb.gather(1, prev_idx.unsqueeze(-1).expand(-1, -1, student_emb.shape[-1]))
        next_emb = student_emb.gather(1, next_idx.unsqueeze(-1).expand(-1, -1, student_emb.shape[-1]))

        # Zero out embeddings that came from original padding tokens
        # Shape: (batch_size, t_len, 1)
        curr_mask = s_mask.gather(1, curr_idx).unsqueeze(-1)
        prev_mask = s_mask.gather(1, prev_idx).unsqueeze(-1)
        next_mask = s_mask.gather(1, next_idx).unsqueeze(-1)

        curr_emb = curr_emb * curr_mask
        prev_emb = prev_emb * prev_mask
        next_emb = next_emb * next_mask

        # Blend neighbors
        weights = F.softmax(self.neighbor_weights, dim=0)
        blended = (weights[0] * prev_emb + weights[1] * curr_emb + weights[2] * next_emb)

        # Add positional information
        # pos_grid is (batch_size, t_len), reshape for the linear layer
        pos_emb = self.position_encoder(pos_grid.reshape(-1, 1)).view(batch_size, t_len, -1)

        # Combine and transform
        combined = torch.cat([blended, pos_emb], dim=-1)
        transformed = self.position_transform(combined) # Shape: (batch_size, t_len, teacher_dim)

        # Mask the output based on the original teacher sequence length
        output_mask = t_mask[:, :t_len].unsqueeze(-1) # Shape: (batch_size, t_len, 1)
        transformed = transformed * output_mask

        # Pad the sequence dimension to the target_length (512)
        if t_len < target_length:
            padding = (0, 0, 0, target_length - t_len)
            transformed = F.pad(transformed, padding, 'constant', 0)

        return transformed

# ========== Token Alignment ==========
def normalize_token(token: str, tokenizer_type: str) -> str:
    """Tokenizer-aware token normalization"""
    # Common space markers
    token = token.replace('Ġ', '').replace('▁', '')

    # Tokenizer-specific cleaning
    if tokenizer_type.lower() == "t5":
        token = token.replace('▔', '').replace('▃', '')
    elif tokenizer_type.lower() == "qwen":
        # Add Qwen-specific cleaning if needed
        pass

    # Common punctuation normalization
    token = token.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
    token = token.replace(' :', ':').replace(' ;', ';').replace(' (', '(').replace(' )', ')')
    token = token.replace(' [', '[').replace(' ]', ']').replace(' {', '{').replace(' }', '}')
    token = token.replace('�', '')

    return token.lower().strip()

def ids_to_tokens(token_ids: torch.Tensor, tokenizer) -> List[str]:
    """Convert token IDs to tokens"""
    return tokenizer.convert_ids_to_tokens(token_ids.cpu().numpy())

def token_based_alignment(student_input_ids: torch.Tensor, teacher_input_ids: torch.Tensor,
                         student_tokenizer, teacher_tokenizer,
                         window_size: int = 5) -> List[Tuple[int, int]]:
    """Improved token alignment with tokenizer awareness"""
    student_tokens = ids_to_tokens(student_input_ids, student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids, teacher_tokenizer)

    aligned_pairs = []

    normalized_student = [normalize_token(token, "qwen") for token in student_tokens]
    normalized_teacher = [normalize_token(token, "t5") for token in teacher_tokens]

    for i, norm_stu in enumerate(normalized_student):
        if student_tokens[i] in [student_tokenizer.pad_token,
                                student_tokenizer.bos_token,
                                student_tokenizer.eos_token]:
            continue

        start_j = max(0, i - window_size)
        end_j = min(len(teacher_tokens), i + window_size + 1)

        for j in range(start_j, end_j):
            if teacher_tokens[j] in [teacher_tokenizer.pad_token,
                                    teacher_tokenizer.bos_token,
                                    teacher_tokenizer.eos_token]:
                continue

            norm_tea = normalized_teacher[j]

            if norm_stu == norm_tea:
                aligned_pairs.append((i, j))
                break

    return aligned_pairs

# ========== Loss Function ==========
class HybridLoss(torch.nn.Module):
    """Hybrid loss with simplified single-phase weighting"""
    def __init__(self, huber_weight: float = 0.7, cosine_weight: float = 0.3,
                 token_huber_weight: float = 0.0, token_cosine_weight: float = 0.0,
                 sequence_huber_weight: float = 0.0, sequence_cosine_weight: float = 0.0,
                 huber_delta: float = 1.0,
                 student_tokenizer=None, teacher_tokenizer=None):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.token_huber_weight = token_huber_weight
        self.token_cosine_weight = token_cosine_weight
        self.sequence_huber_weight = sequence_huber_weight
        self.sequence_cosine_weight = sequence_cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='none')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                student_mask: torch.Tensor, teacher_mask: torch.Tensor,
                student_input_ids: Optional[torch.Tensor] = None,
                teacher_input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        device = student_output.device
        batch_size = student_output.size(0)

        # Initialize loss accumulators
        token_alignment_huber_sum = torch.tensor(0.0, device=device)
        token_alignment_cos_loss_sum = torch.tensor(0.0, device=device)
        token_huber_sum = torch.tensor(0.0, device=device)
        token_cos_loss_sum = torch.tensor(0.0, device=device)
        sequence_huber_sum = torch.tensor(0.0, device=device)
        sequence_cos_loss_sum = torch.tensor(0.0, device=device)

        # Counters for valid sequences
        num_token_alignment_seqs = 0
        num_token_seqs = 0
        num_sequence_seqs = 0
        num_aligned_tokens = 0

        for i in range(batch_size):
            # Token alignment
            if student_input_ids is not None and teacher_input_ids is not None:
                aligned_pairs_tok = token_based_alignment(
                    student_input_ids[i], teacher_input_ids[i],
                    self.student_tokenizer, self.teacher_tokenizer,
                    window_size=TOKEN_ALIGNMENT_WINDOW
                )

                if len(aligned_pairs_tok) > 0:
                    student_aligned_tok = []
                    teacher_aligned_tok = []
                    for (stu_idx, tea_idx) in aligned_pairs_tok:
                        student_aligned_tok.append(student_output[i, stu_idx])
                        teacher_aligned_tok.append(teacher_output[i, tea_idx])

                    if student_aligned_tok and teacher_aligned_tok:
                        student_aligned_tok = torch.stack(student_aligned_tok)
                        teacher_aligned_tok = torch.stack(teacher_aligned_tok)

                        token_alignment_huber_sum += self.huber_loss(student_aligned_tok, teacher_aligned_tok).mean()
                        cos_sim_align = self.cos_loss(student_aligned_tok, teacher_aligned_tok)
                        token_alignment_cos_loss_sum += (1 - cos_sim_align).mean()
                        num_token_alignment_seqs += 1
                        num_aligned_tokens += len(aligned_pairs_tok)

            # Token loss and sequence loss
            teacher_non_pad_mask = teacher_mask[i].bool()
            seq_len = teacher_output.shape[1]

            if teacher_non_pad_mask.any():
                student_output_non_pad = student_output[i, teacher_non_pad_mask]
                teacher_output_non_pad = teacher_output[i, teacher_non_pad_mask]

                token_huber_sum += self.huber_loss(student_output_non_pad, teacher_output_non_pad).mean()
                cos_sim_token = self.cos_loss(student_output_non_pad, teacher_output_non_pad)
                token_cos_loss_sum += (1 - cos_sim_token).mean()
                num_token_seqs += 1

                # Sequence loss
                stu_mean = student_output_non_pad.mean(dim=0)
                tea_mean = teacher_output_non_pad.mean(dim=0)

                sequence_huber_sum += self.huber_loss(stu_mean, tea_mean).mean()
                cos_sim_seq = self.cos_loss(stu_mean, tea_mean)
                sequence_cos_loss_sum += (1 - cos_sim_seq).mean()
                num_sequence_seqs += 1

        # Compute averaged losses
        token_alignment_huber = token_alignment_huber_sum / num_token_alignment_seqs if num_token_alignment_seqs > 0 else torch.tensor(0.0, device=device)
        token_alignment_cos_loss = token_alignment_cos_loss_sum / num_token_alignment_seqs if num_token_alignment_seqs > 0 else torch.tensor(0.0, device=device)
        token_huber = token_huber_sum / num_token_seqs if num_token_seqs > 0 else torch.tensor(0.0, device=device)
        token_cos_loss = token_cos_loss_sum / num_token_seqs if num_token_seqs > 0 else torch.tensor(0.0, device=device)
        sequence_huber = sequence_huber_sum / num_sequence_seqs if num_sequence_seqs > 0 else torch.tensor(0.0, device=device)
        sequence_cos_loss = sequence_cos_loss_sum / num_sequence_seqs if num_sequence_seqs > 0 else torch.tensor(0.0, device=device)

        total_loss = (
            self.huber_weight * token_alignment_huber +
            self.cosine_weight * token_alignment_cos_loss +
            self.token_huber_weight * token_huber +
            self.token_cosine_weight * token_cos_loss +
            self.sequence_huber_weight * sequence_huber +
            self.sequence_cosine_weight * sequence_cos_loss
        )

        printed_loss = (
            self.huber_weight * token_huber +
            self.cosine_weight * token_cos_loss +
            self.token_huber_weight * token_huber +
            self.token_cosine_weight * token_cos_loss +
            self.sequence_huber_weight * sequence_huber +
            self.sequence_cosine_weight * sequence_cos_loss
        )

        return total_loss, printed_loss, token_alignment_huber, token_alignment_cos_loss, token_huber, token_cos_loss, sequence_huber, sequence_cos_loss, num_aligned_tokens

# ========== Metrics ==========
def compute_cross_architecture_metrics(student_emb: torch.Tensor, teacher_emb: torch.Tensor,
                                     student_input_ids: torch.Tensor, teacher_input_ids: torch.Tensor,
                                     student_tokenizer, teacher_tokenizer) -> Dict[str, float]:
    """Compute metrics specifically for cross-architecture alignment"""
    metrics = {}

    # 1. Overall embedding similarity
    overall_cosine = F.cosine_similarity(student_emb.flatten(), teacher_emb.flatten(), dim=0).mean()
    metrics['overall_cosine_similarity'] = overall_cosine.item()

    # 2. Token-level alignment quality
    aligned_pairs = token_based_alignment(
        student_input_ids[0], teacher_input_ids[0],
        student_tokenizer, teacher_tokenizer
    )

    if aligned_pairs:
        student_aligned = student_emb[0, [p[0] for p in aligned_pairs]]
        teacher_aligned = teacher_emb[0, [p[1] for p in aligned_pairs]]
        alignment_quality = F.cosine_similarity(student_aligned, teacher_aligned, dim=-1).mean()
        metrics['token_alignment_quality'] = alignment_quality.item()
        metrics['num_aligned_tokens'] = len(aligned_pairs)
    else:
        metrics['token_alignment_quality'] = 0.0
        metrics['num_aligned_tokens'] = 0

    # 3. Vocabulary coverage
    student_tokens = set(student_tokenizer.convert_ids_to_tokens(student_input_ids[0].cpu().numpy()))
    teacher_tokens = set(teacher_tokenizer.convert_ids_to_tokens(teacher_input_ids[0].cpu().numpy()))

    # Normalize tokens for comparison
    norm_student = {normalize_token(t, "qwen") for t in student_tokens}
    norm_teacher = {normalize_token(t, "t5") for t in teacher_tokens}

    if norm_student:
        vocab_overlap = len(norm_student & norm_teacher) / len(norm_student)
        metrics['vocabulary_overlap'] = vocab_overlap
    else:
        metrics['vocabulary_overlap'] = 0.0

    # 4. Embedding statistics
    student_std = torch.std(student_emb)
    teacher_std = torch.std(teacher_emb)
    metrics['embedding_std_diff'] = torch.abs(student_std - teacher_std).item()

    return metrics

# ========== Dataset Class ==========
class PreTokenizedDataset(Dataset):
    """Dataset with improved memory management"""
    def __init__(self, file_path: str, student_tokenizer, teacher_tokenizer,
                 max_length: int, teacher_model=None, is_eval: bool = False,
                 sample_rate: float = 0.1, use_cached_embeddings: bool = False,
                 cache_path: Optional[str] = None):
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

                            # Explicit cleanup
                            del embeddings, input_ids, att_mask, outputs
                            torch.cuda.empty_cache()

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
                            input_ids=input_ids.to(teacher_model.device),
                            attention_mask=att_mask.to(teacher_model.device)
                        )
                        embeddings = outputs.last_hidden_state.cpu()

                    embedding_file = os.path.join(cache_folder, f"{i}.pt")
                    mask_file = os.path.join(cache_folder, f"{i}_mask.pt")
                    torch.save(embeddings, embedding_file)
                    torch.save(att_mask.cpu(), mask_file)

                    # Explicit cleanup
                    del embeddings, input_ids, att_mask, outputs
                    torch.cuda.empty_cache()

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

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple:
        def apply_dropout(line: str, word_dropout_ratio: float, token_dropout_ratio: float,
                         tokenizer=None) -> str:
            if word_dropout_ratio > 0:
                words = line.split()
                kept_words = []
                for word in words:
                    if random.random() > word_dropout_ratio:
                        kept_words.append(word)
                line = " ".join(kept_words)
            elif token_dropout_ratio > 0:
                tokens = tokenizer.tokenize(line)
                kept_tokens = []
                special_tokens = [tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token]
                if hasattr(tokenizer, 'cls_token'):
                    special_tokens.append(tokenizer.cls_token)
                if hasattr(tokenizer, 'sep_token'):
                    special_tokens.append(tokenizer.sep_token)

                for token in tokens:
                    if token in special_tokens:
                        kept_tokens.append(token)
                    else:
                        if random.random() > token_dropout_ratio:
                            kept_tokens.append(token)
                line = tokenizer.convert_tokens_to_string(kept_tokens)
            return line

        if self.enabled_ratios:
            choice = random.choices(range(self.num_ratios), weights=self.enabled_ratios)[0]

            if choice == 0:
                student_line = self.student_raw_lines[idx]
                teacher_line = self.teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                student_dropout_token = STUDENT_TOKEN_DROPOUT_RATIO if ENABLE_STUDENT_TOKEN_DROPOUT else 0
                teacher_type = "original"

            elif choice == 1:
                student_line = self.student_raw_lines[idx]
                teacher_line = self.enhanced_teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                student_dropout_token = STUDENT_TOKEN_DROPOUT_RATIO if ENABLE_STUDENT_TOKEN_DROPOUT else 0
                teacher_type = "enhanced"

            elif choice == 2:
                student_line = self.enhanced_teacher_raw_lines[idx]
                teacher_line = self.enhanced_teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                student_dropout_token = STUDENT_TOKEN_DROPOUT_RATIO if ENABLE_STUDENT_TOKEN_DROPOUT else 0
                teacher_type = "enhanced"

        if choice == 1 and SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER == True:
            student_dropout_word = 0
            student_dropout_token = 0

        student_line = apply_dropout(student_line, student_dropout_word, student_dropout_token, self.student_tokenizer)

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
            )
        else:
            return (
                student_input_ids,
                student_attention_mask,
                teacher_input_ids,
                teacher_mask,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'file_handles'):
            for file in self.file_handles:
                file.close()
            self.file_handles.clear()

# ========== Projection Function ==========
def get_projection_layers(restart_cycle: int, layers_to_load: int, qwen_embedding_dim: int) -> Tuple[List[torch.nn.Module], int]:
    """Get projection layers with inference-ready implementations"""
    projection_layers = []

    # Update input_dim for first layer
    if PROJECTION_LAYERS_CONFIG:
        PROJECTION_LAYERS_CONFIG[0]["input_dim"] = qwen_embedding_dim

    layers_to_load = len(PROJECTION_LAYERS_CONFIG)

    output_dim_prev = None
    for i in range(1, layers_to_load + 1):
        layer_config = PROJECTION_LAYERS_CONFIG[i-1]
        layer_num = layer_config["file_num"]
        layer_path = os.path.join(QWEN3_MODEL_NAME, f"projection_layer_{layer_num}.safetensors")

        # Determine input dim
        if i == 1:
            input_dim = qwen_embedding_dim
        else:
            input_dim = projection_layers[-1].output_dim if hasattr(projection_layers[-1], 'output_dim') else 4096

        # Override if specified in config
        if layer_config.get("input_dim", "auto") != "auto":
            input_dim = layer_config["input_dim"]

        file_num = layer_config["file_num"]

        if layer_config["type"] == "linear":
            output_dim = layer_config["output_dim"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = LinearProjectionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing linear layer {file_num}")
            else:
                projection_layer = LinearProjectionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                print(f"Initialising new linear layer {file_num}")
            projection_layer.output_dim = output_dim

        elif layer_config["type"] == "mlp":
            output_dim = layer_config["hidden_dim"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = MLPProjectionLayer(
                    input_dim=input_dim,
                    hidden_dim=output_dim,
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing MLP layer {file_num}")
            else:
                projection_layer = MLPProjectionLayer(
                    input_dim=input_dim,
                    hidden_dim=output_dim,
                )
                print(f"Initialising new MLP layer {file_num}")
            projection_layer.output_dim = output_dim

        elif layer_config["type"] == "transformer":
            output_dim = layer_config["hidden_dim"]
            dim_feedforward = layer_config["dim_feedforward"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = TransformerProjectionLayer(
                    input_dim=input_dim,
                    hidden_dim=output_dim,
                    dim_feedforward=dim_feedforward,
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing transformer layer {file_num}")
            else:
                projection_layer = TransformerProjectionLayer(
                    input_dim=input_dim,
                    hidden_dim=output_dim,
                    dim_feedforward=dim_feedforward,
                )
                print(f"Initialising new transformer layer {file_num}")
            projection_layer.output_dim = output_dim

        elif layer_config["type"] == "expansion":
            output_dim = layer_config["teacher_dim"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = ContinuousExpansionLayer(
                    input_dim=input_dim,
                    teacher_dim=output_dim
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing expansion layer {file_num}")
            else:
                projection_layer = ContinuousExpansionLayer(
                    input_dim=input_dim,
                    teacher_dim=output_dim
                )
                print(f"Initialising new expansion layer {file_num}")
            projection_layer.output_dim = output_dim

        projection_layer.file_num = layer_config["file_num"]
        projection_layers.append(projection_layer)

    return projection_layers, layers_to_load

def get_projection_parameters_by_type(projection_layers: List[torch.nn.Module]) -> Dict[str, List[torch.nn.Parameter]]:
    """Get projection parameters grouped by layer type"""
    params_by_type = {
        'transformer': [],
        'mlp': [],
        'linear': [],
        'expansion': []
    }

    for layer in projection_layers:
        if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
            layer_type = getattr(layer, 'layer_type', 'unknown')
            if layer_type in params_by_type:
                params_by_type[layer_type].extend([p for p in layer.parameters() if p.requires_grad])

    return params_by_type

# ========== Optimiser Initialisation ==========
def initialize_optimizer(parameters: List[torch.nn.Parameter], max_lr: float, min_lr: float) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        parameters,
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    if RESTART_CYCLE_STEPS > 0:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=RESTART_CYCLE_STEPS,
            T_mult=1,
            eta_min=min_lr,
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=1000 - WARMUP_STEPS
        )

    warmup_scheduler = None
    if WARMUP_STEPS > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((min_lr / max_lr) + (step / (WARMUP_STEPS)) * (1 - min_lr / max_lr), 1.0)
        )

    if warmup_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[WARMUP_STEPS]
        )
    else:
        scheduler = main_scheduler

    return optimizer, scheduler

def initialize_projection_optimizers(projection_layers: List[torch.nn.Module]) -> Dict[str, Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
    """Initialize separate optimizers for each projection layer type"""
    params_by_type = get_projection_parameters_by_type(projection_layers)

    optimizers = {}

    # Transformer layers
    if params_by_type['transformer']:
        optimizers['transformer'] = initialize_optimizer(
            params_by_type['transformer'],
            MAX_LEARNING_RATE_TRANSFORMER,
            MIN_LEARNING_RATE_TRANSFORMER
        )

    # MLP layers
    if params_by_type['mlp']:
        optimizers['mlp'] = initialize_optimizer(
            params_by_type['mlp'],
            MAX_LEARNING_RATE_MLP,
            MIN_LEARNING_RATE_MLP
        )

    # Linear layers
    if params_by_type['linear']:
        optimizers['linear'] = initialize_optimizer(
            params_by_type['linear'],
            MAX_LEARNING_RATE_LINEAR,
            MIN_LEARNING_RATE_LINEAR
        )

    # Expansion layers
    if params_by_type['expansion']:
        optimizers['expansion'] = initialize_optimizer(
            params_by_type['expansion'],
            MAX_LEARNING_RATE_EXPANSION,
            MIN_LEARNING_RATE_EXPANSION
        )

    return optimizers

# ========== Evaluation Function ==========
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, projection_layers: List[torch.nn.Module],
                  loss_fn: HybridLoss, device: str, autocast_dtype: torch.dtype,
                  student_tokenizer, teacher_tokenizer, teacher_model=None) -> Dict[str, float]:
    """Evaluate model with improved metrics and memory handling"""
    model.eval()
    for layer in projection_layers:
        layer.eval()
    loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'token_huber': 0.0,
        'token_cos': 0.0,
        'seq_huber': 0.0,
        'seq_cos': 0.0,
        'num_aligned_tokens': 0,
        'vocab_overlap': 0.0,
        'alignment_quality': 0.0
    }

    metric_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                if USE_CACHED_EMBEDDINGS:
                    s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask = batch
                    # Directly use cached embeddings without reloading
                    teacher_hidden = t_embeddings.to(device)
                else:
                    s_input_ids, s_mask, t_input_ids, t_mask = batch
                    # Generate embeddings on-the-fly
                    t_input_ids = t_input_ids.to(device)
                    teacher_outputs = teacher_model(
                        input_ids=t_input_ids,
                        attention_mask=t_mask
                    )
                    teacher_hidden = teacher_outputs.last_hidden_state.to(device)

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
                    projected_student = student_hidden

                    # Pass teacher embeddings and masks to projection layers
                    for layer in projection_layers:
                        if isinstance(layer, ContinuousExpansionLayer):
                            projected_student = layer(
                                projected_student,
                                s_mask=s_mask,
                                t_mask=t_mask,
                                target_length=512
                            )
                        else:
                            projected_student = layer(projected_student)

                loss, _, _, _, token_huber_loss, token_cos_loss, seq_huber_loss, seq_cos_loss, num_aligned = loss_fn(
                    projected_student,
                    teacher_hidden,
                    s_mask,
                    t_mask,
                    student_input_ids=s_input_ids,
                    teacher_input_ids=t_input_ids
                )

                # Collect metrics for first few samples
                if batch_idx < 5:  # Sample first 5 batches for metrics
                    metrics = compute_cross_architecture_metrics(
                        projected_student, teacher_hidden,
                        s_input_ids, t_input_ids,
                        student_tokenizer, teacher_tokenizer
                    )
                    metric_samples.append(metrics)

                total_losses['total'] += loss.item()
                total_losses['token_huber'] += token_huber_loss.item()
                total_losses['token_cos'] += token_cos_loss.item()
                total_losses['seq_huber'] += seq_huber_loss.item()
                total_losses['seq_cos'] += seq_cos_loss.item()
                total_losses['num_aligned_tokens'] += num_aligned

                # Cleanup
                del projected_student, student_hidden, student_outputs
                if not USE_CACHED_EMBEDDINGS:
                    del teacher_outputs
                del s_input_ids, s_mask, t_mask, teacher_hidden
                if not USE_CACHED_EMBEDDINGS:
                    del t_input_ids

            except Exception as e:
                logging.exception(f"Error in evaluation batch {batch_idx}: {e}")
                # Cleanup variables
                for var in ['projected_student', 'student_hidden', 'student_outputs',
                           'teacher_hidden', 'teacher_outputs', 's_input_ids', 's_mask',
                           't_mask', 't_input_ids']:
                    if var in locals():
                        del locals()[var]
                raise e

    # Compute averages
    num_batches = len(dataloader)
    for key in ['total', 'token_huber', 'token_cos', 'seq_huber', 'seq_cos']:
        total_losses[key] /= num_batches

    # Average metrics from samples
    if metric_samples:
        for key in ['vocab_overlap', 'alignment_quality']:
            total_losses[key] = sum(m[key] for m in metric_samples) / len(metric_samples)
        total_losses['num_aligned_tokens'] = total_losses['num_aligned_tokens'] / num_batches

    model.train()
    for layer in projection_layers:
        layer.train()
    loss_fn.train()

    return total_losses

# ========== Miscellaneous Functions ==========
def get_memory_usage() -> List[float]:
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        memory = torch.cuda.mem_get_info()
        memory_mib = []
        for item in memory:
            memory_mib.append(item/1048576)
        memory_used = memory_mib[1]-memory_mib[0]
        memory_mib.append(memory_used)
        return memory_mib
    return [0.0, 0.0, 0.0]

def save_trained_model(save_path: str, model: torch.nn.Module, tokenizer,
                      projection_layers: List[torch.nn.Module], qwen_embedding_dim: int) -> None:
    """Save trained model with all components"""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    for layer in projection_layers:
        layer_state = layer.state_dict()
        layer_path = os.path.join(save_path, f"projection_layer_{layer.file_num}.safetensors")
        save_file(layer_state, layer_path)

    projection_config_path = os.path.join(save_path, "projection_config.json")
    save_projection_config(projection_config_path, qwen_embedding_dim)

def save_projection_config(projection_config_path: str, embedding_dim: int) -> None:
    """Save projection configuration"""
    projection_config = {
        "layers": PROJECTION_LAYERS_CONFIG,
    }
    with open(projection_config_path, "w") as f:
        json.dump(projection_config, f)

def exit_dataloader() -> None:
    """Clean up dataloaders and datasets"""
    global train_dataloader, train_dataset, eval_dataloader, eval_dataset
    train_dataloader = None
    train_dataset = None
    eval_dataloader = None
    eval_dataset = None
    del train_dataloader, train_dataset, eval_dataloader, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()

def get_logging(epoch: int, batch_idx: int, global_step: int, total_steps: int,
               current_loss: float, current_huber: float, current_cos: float,
               current_seq_huber: float, current_seq_cos: float,
               grad_norm_model: float, grad_norm_proj: float,
               current_lr_model: float, current_lr_proj: float,
               elapsed: float, eta: float) -> str:
    """Generate logging string"""
    model_gn_line = ""
    model_lr_line = ""
    proj_gn_line = ""
    proj_lr_line = ""
    vram_line = ""
    batch_line = ""
    step_line = ""
    epoch_line = ""

    if TRAIN_MODEL:
        model_gn_line = f"Grad Norm Model: {grad_norm_model:.6f}, "
        model_lr_line = f"Model LR: {current_lr_model:.6f}, "
    if TRAIN_PROJECTION:
        proj_gn_line = f"Grad Norm Projection: {grad_norm_proj:.6f}, "
        proj_lr_line = f"Projection LR: {current_lr_proj:.6f}, "
    if LOG_VRAM_USAGE:
        vram_free, vram_total, vram_used = get_memory_usage()
        vram_line = f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
    epoch_line = f"Epoch [{epoch + 1}/{EPOCHS}], "
    step_line = f"Step: {global_step}/{total_steps}, "
    batch_line = f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "

    log_line = (
        f"{epoch_line}"
        f"{batch_line}"
        f"{step_line}"
        f"Total Loss: {current_loss:.6f}, "
        f"PT Huber: {current_huber:.6f}, "
        f"PT Cosine: {current_cos:.6f}, "
        f"SQ Huber: {current_seq_huber:.6f}, "
        f"SQ Cosine: {current_seq_cos:.6f}, "
        f"{model_gn_line}"
        f"{proj_gn_line}"
        f"{model_lr_line}"
        f"{proj_lr_line}"
        f"{vram_line}"
        f"Elapsed: {elapsed/60:.1f} min, "
        f"ETA: {eta/60:.1f} min"
    )
    return log_line

# ========== Main Training Script ==========
def main():
    """Main training function with improved memory management and inference-ready layers"""
    global total_steps, train_dataloader, train_dataset, eval_dataloader, eval_dataset

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

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

    # Initialize loss function
    loss_fn = HybridLoss(
        huber_weight=ALIGN_HUBER_WEIGHT,
        cosine_weight=ALIGN_COSINE_WEIGHT,
        token_huber_weight=TOKEN_HUBER_WEIGHT,
        token_cosine_weight=TOKEN_COSINE_WEIGHT,
        sequence_huber_weight=SEQUENCE_HUBER_WEIGHT,
        sequence_cosine_weight=SEQUENCE_COSINE_WEIGHT,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer
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
        teacher_model = None
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

            # Calculate total steps
            total_steps = (len(train_dataloader) // GRAD_ACCUM_STEPS) * EPOCHS

            # ========== Training Setup ==========
            autocast_dtype = torch.bfloat16
            scaler = GradScaler('cuda', enabled=False)

            global_step = 0
            accumulation_step = 0
            grad_norm = 0

            try:
                with open(os.path.join(QWEN3_MODEL_NAME, "config.json"), "r") as f:
                    qwen_config = json.load(f)
                qwen_embedding_dim = qwen_config["hidden_size"]
            except Exception as e:
                print(f"Error loading Qwen config: {e}")
                qwen_embedding_dim = 1024

            restart_cycle = 1
            layers_to_load = 1
            projection_layers, layers_to_load = get_projection_layers(restart_cycle, layers_to_load, qwen_embedding_dim)
            for layer in projection_layers:
                layer.to(device, dtype=torch.bfloat16)

            # Set excluded layers to not require grad
            for layer_idx in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
                if layer_idx <= len(projection_layers):
                    for param in projection_layers[layer_idx-1].parameters():
                        param.requires_grad = False

            # Initialize optimizers
            projection_optimizers = {}
            if TRAIN_PROJECTION:
                projection_optimizers = initialize_projection_optimizers(projection_layers)

            model_optimizer = None
            scheduler_model = None
            if TRAIN_MODEL:
                model_parameters = [p for p in student_model.parameters() if p.requires_grad]
                model_optimizer, scheduler_model = initialize_optimizer(model_parameters, MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)

            if ENABLE_LOGGING:
                log_dir = os.path.join(OUTPUT_DIR, "logging")
                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                log_filename = f"training_log_{current_time}.txt"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, log_filename)
                log_lines = []

            # ========== Training Loop ==========
            student_model.train()
            for layer in projection_layers:
                layer.train()

            start_time = time.time()
            eval_delta_time = 0
            best_loss = float('inf')

            for epoch in range(EPOCHS):
                if TRAIN_MODEL and model_optimizer: model_optimizer.zero_grad()
                for opt in projection_optimizers.values():
                    opt[0].zero_grad()

                steps_completed_this_epoch = 0
                accumulation_step = 0
                print(f"Starting epoch {epoch + 1}/{EPOCHS}")
                print(f"Total batches in epoch: {len(train_dataloader)}")

                for batch_idx, batch in enumerate(train_dataloader):
                    try:
                        if USE_CACHED_EMBEDDINGS:
                            s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask = batch
                        else:
                            s_input_ids, s_mask, t_input_ids, t_mask = batch

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
                            projected_student = student_hidden

                            for layer in projection_layers:
                                if isinstance(layer, ContinuousExpansionLayer):
                                        projected_student = layer(
                                            projected_student,
                                            s_mask=s_mask,
                                            t_mask=t_mask,
                                            target_length=512
                                        )
                                else:
                                    projected_student = layer(projected_student)

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

                        loss, printed_loss, align_huber_loss, align_cos_loss, token_huber_loss, token_cos_loss, seq_huber_loss, seq_cos_loss, num_aligned_tokens = loss_fn(
                            projected_student,
                            teacher_hidden,
                            s_mask,
                            t_mask,
                            student_input_ids=s_input_ids,
                            teacher_input_ids=t_input_ids
                        )

                        scaled_loss = loss / GRAD_ACCUM_STEPS
                        scaler.scale(scaled_loss).backward()
                        accumulation_step += 1

                        if accumulation_step >= GRAD_ACCUM_STEPS or batch_idx == len(train_dataloader) - 1:
                            # Clip gradients
                            if TRAIN_MODEL and model_optimizer:
                                grad_norm_model = clip_grad_norm_(
                                    [p for p in student_model.parameters() if p.requires_grad],
                                    max_norm=GRAD_CLIP
                                )
                            else:
                                grad_norm_model = 0.0

                            # Get all projection parameters for clipping
                            all_proj_params = []
                            for layer in projection_layers:
                                if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
                                    all_proj_params.extend([p for p in layer.parameters() if p.requires_grad])

                            grad_norm_proj = clip_grad_norm_(
                                all_proj_params if all_proj_params else [],
                                max_norm=GRAD_CLIP
                            )

                            # Step optimizers
                            if TRAIN_MODEL and model_optimizer:
                                scaler.step(model_optimizer)
                            for opt in projection_optimizers.values():
                                scaler.step(opt[0])
                            scaler.update()

                            global_step += 1
                            steps_completed_this_epoch += 1

                            accumulation_step = 0
                            if TRAIN_MODEL and model_optimizer:
                                model_optimizer.zero_grad()
                            for opt in projection_optimizers.values():
                                opt[0].zero_grad()

                            # Reset gradients for projection layers
                            for layer in projection_layers:
                                if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
                                    for param in layer.parameters():
                                        if param.grad is not None:
                                            param.grad = None

                            current_loss = loss.item()
                            current_printed_loss = printed_loss.item()
                            current_huber = token_huber_loss.item()
                            current_cos = token_cos_loss.item()
                            current_seq_huber = seq_huber_loss.item()
                            current_seq_cos = seq_cos_loss.item()

                            elapsed = time.time() - start_time - eval_delta_time
                            remaining_steps = total_steps - global_step
                            eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0

                            if TRAIN_MODEL and model_optimizer:
                                current_lr_model = model_optimizer.param_groups[0]['lr']
                            else:
                                current_lr_model = 0.0

                            if TRAIN_PROJECTION and projection_optimizers:
                                # Get average LR across projection optimizers
                                current_lr_proj = sum(opt[0].param_groups[0]['lr'] for opt in projection_optimizers.values()) / len(projection_optimizers)
                            else:
                                current_lr_proj = 0.0

                            if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                                print(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss, current_huber, current_cos,
                                    current_seq_huber, current_seq_cos,
                                    grad_norm_model, grad_norm_proj,
                                    current_lr_model, current_lr_proj,
                                    elapsed, eta
                                ))

                            if ENABLE_LOGGING:
                                log_lines.append(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss, current_huber, current_cos,
                                    current_seq_huber, current_seq_cos,
                                    grad_norm_model, grad_norm_proj,
                                    current_lr_model, current_lr_proj,
                                    elapsed, eta
                                ))
                                if global_step % WRITE_TO_LOG_EVERY_X_STEPS == 0:
                                    with open(log_file, "a") as f:
                                        for line in log_lines:
                                            f.write(line + "\n")
                                    log_lines.clear()

                            # Update schedulers
                            if TRAIN_MODEL and scheduler_model:
                                scheduler_model.step()
                            for opt in projection_optimizers.values():
                                opt[1].step()

                            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                                print(f"\nSaving checkpoint at step {global_step}\n")
                                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

                            # Define cycle length for restart-based training
                            if REPEAT_WARMUP_AFTER_RESTART or (restart_cycle == 1 and WARMUP_STEPS > 0):
                                cycle_length = WARMUP_STEPS + RESTART_CYCLE_STEPS
                            else:
                                cycle_length = RESTART_CYCLE_STEPS

                            if global_step % cycle_length == 0 and global_step > 0:
                                if SAVE_EVERY_X_RESTARTS > 0 and restart_cycle % SAVE_EVERY_X_RESTARTS == 0:
                                    print(f"\nSaving checkpoint at restart {restart_cycle}\n")
                                    save_path = os.path.join(OUTPUT_DIR, f"restart_{restart_cycle}")
                                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

                                restart_cycle += 1
                                if REPEAT_WARMUP_AFTER_RESTART:
                                    # Reinitialize optimizers
                                    if TRAIN_MODEL:
                                        model_optimizer, scheduler_model = initialize_optimizer(model_parameters, MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)
                                    if TRAIN_PROJECTION:
                                        projection_optimizers = initialize_projection_optimizers(projection_layers)

                            # Explicit cleanup after optimizer step
                            del student_outputs, student_hidden, projected_student, teacher_hidden
                            if 't_input_ids' in locals():
                                del t_input_ids
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logging.exception(f"Error in batch {batch_idx}: {e}")
                        # Clean up variables
                        for var in ['student_outputs', 'student_hidden', 'projected_student',
                                   'teacher_hidden', 't_input_ids', 's_input_ids', 's_mask',
                                   't_mask', 't_embeddings']:
                            if var in locals():
                                del locals()[var]
                        accumulation_step = 0
                        if TRAIN_MODEL and model_optimizer:
                            model_optimizer.zero_grad()
                        for opt in projection_optimizers.values():
                            opt[0].zero_grad()
                        continue

                    finally:
                        # Cleanup batch variables
                        del s_input_ids, s_mask, t_mask
                        if USE_CACHED_EMBEDDINGS:
                            del t_embeddings
                        else:
                            if 't_input_ids' in locals():
                                del t_input_ids

                print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")
                next_epoch = epoch + 1
                if next_epoch % SAVE_EVERY_X_EPOCHS == 0:
                    print(f"\nSaving checkpoint at epoch {next_epoch}\n")
                    save_path = os.path.join(OUTPUT_DIR, f"epoch_{next_epoch}")
                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

                if next_epoch % EVAL_EVERY_X_EPOCHS == 0:
                    eval_start_time = time.time()

                    # Load teacher model if needed for evaluation
                    if not USE_CACHED_EMBEDDINGS and teacher_model is None:
                        print("Loading T5-xxl model for evaluation...")
                        teacher_model = T5EncoderModel.from_pretrained(
                            T5_MODEL_NAME,
                            torch_dtype=torch.bfloat16,
                            device_map="auto"
                        )
                        teacher_model.eval()

                    eval_metrics = evaluate_model(
                        student_model, eval_dataloader, projection_layers, loss_fn, device, autocast_dtype,
                        student_tokenizer, teacher_tokenizer, teacher_model
                    )

                    # Unload teacher model if it was loaded just for evaluation
                    if not USE_CACHED_EMBEDDINGS:
                        del teacher_model
                        teacher_model = None
                        torch.cuda.empty_cache()

                    avg_eval_loss = eval_metrics['total']
                    print(f"\n[Validation] Epoch {epoch + 1}")
                    print(f"  Average Total Loss: {avg_eval_loss:.6f}")
                    print(f"  Token Huber Loss: {eval_metrics['token_huber']:.6f}")
                    print(f"  Token Cosine Loss: {eval_metrics['token_cos']:.6f}")
                    print(f"  Sequence Huber Loss: {eval_metrics['seq_huber']:.6f}")
                    print(f"  Sequence Cosine Loss: {eval_metrics['seq_cos']:.6f}")
                    print(f"  Avg Aligned Tokens: {eval_metrics['num_aligned_tokens']:.1f}")
                    print(f"  Vocabulary Overlap: {eval_metrics['vocab_overlap']:.3f}")
                    print(f"  Alignment Quality: {eval_metrics['alignment_quality']:.3f}")

                    if SAVE_BEST_MODEL and avg_eval_loss < best_loss:
                        best_loss = avg_eval_loss
                        print(f"\n✅ New best model at loss {best_loss:.6f}, saving...")
                        best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
                        save_trained_model(best_model_dir, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

                    eval_end_time = time.time()
                    eval_delta_time += (eval_end_time - eval_start_time)
                    student_model.train()

        except Exception as e:
            logging.exception("Exception during training.")
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
    save_trained_model(OUTPUT_DIR, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("✅ Training and saving completed successfully!")

if __name__ == "__main__":
    main()
