from unsloth import FastLanguageModel
import os
import json
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
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
QWEN3_MODEL_NAME = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saikou/QT-embedder-v61/restart_1/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saikou/QT-embedder-v62/"

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
MAX_LEARNING_RATE_TRANSFORMER = 1e-4
MIN_LEARNING_RATE_TRANSFORMER = 1e-5
MAX_LEARNING_RATE_MLP = 1.2e-4
MIN_LEARNING_RATE_MLP = 1.2e-5
MAX_LEARNING_RATE_LINEAR = 1.5e-4
MIN_LEARNING_RATE_LINEAR = 1.5e-5
MAX_LEARNING_RATE_INTERPOLATION = 1e-4
MIN_LEARNING_RATE_INTERPOLATION = 1e-5
MAX_LEARNING_RATE_POSITION = 2e-4
MIN_LEARNING_RATE_POSITION = 2e-5

# Saving
SAVE_EVERY_X_STEPS = 0
SAVE_EVERY_X_RESTARTS = 1
SAVE_EVERY_X_EPOCHS = 1

# Printing
PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_X_EPOCHS = 1
SAVE_BEST_MODEL = True

# Alignment weights & settings
TEXT_MATCH_HUBER_WEIGHT = 0.70
TEXT_MATCH_COSINE_WEIGHT = 0.30
POSITION_MAPPING_HUBER_WEIGHT = 0.70
POSITION_MAPPING_COSINE_WEIGHT = 0.30
POSITION_MAPPING_COVERAGE = 1.00 # Proportion coverage of non-text-matched area for position prediction matching
ALIGN_WINDOW = 2 # We look this many tokens around the approximate matching position when text matching

# Basic weights
TOKEN_HUBER_WEIGHT = 0.00
TOKEN_COSINE_WEIGHT = 0.00
SEQUENCE_HUBER_WEIGHT = 0.00
SEQUENCE_COSINE_WEIGHT = 0.00

# Scheduler
WARMUP_STEPS = 150
RESTART_CYCLE_STEPS = 350
REPEAT_WARMUP_AFTER_RESTART = False

# Dataset
SHUFFLE_DATASET = True

# Optimizer state preservation
REUSE_OPTIMIZER_STATE = True
SAVE_OPTIMIZER_STATES = True

# Debugging
LOG_VRAM_USAGE = False

# Dropout - this could potentially be used to train the sequential interpolation layer to over-project and infer embedding space from context, but is untested
ENABLE_STUDENT_WORD_DROPOUT = False
STUDENT_WORD_DROPOUT_RATIO = 0.10
ENABLE_STUDENT_TOKEN_DROPOUT = False
STUDENT_TOKEN_DROPOUT_RATIO = 0.10
SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True

# Enhanced dataset - experimental option that is likely not useful, earlier on in training I wanted to see what mimicking a longer sequence might do
ENHANCED_DATASET = True
ENHANCED_DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"
UNTAMPERED_STUDENT_AND_TEACHER_RATIO = 0.50
ENHANCED_TEACHER_EMBEDDING_RATIO = 0.00
ENHANCED_STUDENT_AND_TEACHER_RATIO = 0.50

# Training flags
TRAIN_PROJECTION = True
TRAIN_MODEL = True

# Layer arrangement
EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS = []
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
    },
    {
        "type": "transformer",
        "input_dim": 4096,
        "hidden_dim": 1024,
        "dim_feedforward": 4096,
        "file_num": 4,
    },
    {
        "type": "mlp",
        "input_dim": 1024,
        "hidden_dim": 1024,
        "file_num": 5,
    },
    {
        "type": "linear",
        "input_dim": 1024,
        "output_dim": 4096,
        "file_num": 6,
    },
    {
        "type": "transformer",
        "input_dim": 4096,
        "hidden_dim": 1024,
        "dim_feedforward": 4096,
        "file_num": 7,
    },
    {
        "type": "mlp",
        "input_dim": 1024,
        "hidden_dim": 1024,
        "file_num": 8,
    },
    {
        "type": "linear",
        "input_dim": 1024,
        "output_dim": 4096,
        "file_num": 9,
    },
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

class LearnedInterpolationLayer(torch.nn.Module):
    """Learned interpolation layer for stretching sequences using position-aware upsampling"""
    def __init__(self, input_dim: int, teacher_dim: int, max_length: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.teacher_dim = teacher_dim
        self.max_length = max_length

        # Position encodings for both sequences
        self.student_pos_encoder = torch.nn.Embedding(max_length, input_dim)
        self.teacher_pos_encoder = torch.nn.Embedding(max_length, teacher_dim)

        # Learnable interpolation network using 1D convolution for upsampling
        self.upsample_conv = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=teacher_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Learnable interpolation weights
        self.weight_predictor = torch.nn.Sequential(
            torch.nn.Linear(teacher_dim * 3, 512),  # student emb + 2 position embeddings
            torch.nn.GELU(),
            torch.nn.Linear(512, 3),  # weights for 3 points
            torch.nn.Softmax(dim=-1)
        )

        # Refinement layers
        self.refine = torch.nn.Sequential(
            torch.nn.Linear(teacher_dim, teacher_dim),
            torch.nn.GELU(),
            torch.nn.Linear(teacher_dim, teacher_dim)
        )

        # Output projection
        self.output_proj = torch.nn.Linear(teacher_dim, teacher_dim)
        self.layer_norm = torch.nn.LayerNorm(teacher_dim)
        self.output_dim = teacher_dim
        self.layer_type = "interpolation"

        # Add projection layer for dimension mismatch
        if self.input_dim != self.teacher_dim:
            self.input_to_teacher_proj = torch.nn.Linear(self.input_dim, self.teacher_dim)
        else:
            self.input_to_teacher_proj = None

        # Add layer normalization at the end
        self.final_norm = torch.nn.LayerNorm(teacher_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent collapse"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, student_emb, s_mask, t_mask, target_length=512):
        batch_size, s_len, _ = student_emb.shape
        t_len = t_mask.sum(dim=1).max()

        # Get actual sequence lengths
        s_lens = s_mask.sum(dim=1)  # [B]
        t_lens = t_mask.sum(dim=1)  # [B]

        # Avoid division by zero and ensure at least one token
        s_lens = s_lens.clamp(min=1)
        t_lens = t_lens.clamp(min=1)

        # Create position indices for the entire batch (using max lengths)
        s_pos = torch.arange(s_len, device=student_emb.device).unsqueeze(0).expand(batch_size, -1)  # [B, s_len]
        t_pos = torch.arange(t_len, device=student_emb.device).unsqueeze(0).expand(batch_size, -1)   # [B, t_len]

        # Normalize positions to [0, 1] per batch
        s_pos_norm = s_pos.float() / (s_lens.unsqueeze(1) - 1)  # [B, s_len]
        t_pos_norm = t_pos.float() / (t_lens.unsqueeze(1) - 1)  # [B, t_len]

        # Get position encodings
        s_pos_emb = self.student_pos_encoder(s_pos)  # [B, s_len, input_dim]
        t_pos_emb = self.teacher_pos_encoder(t_pos)  # [B, t_len, teacher_dim]

        # Add position info to student embeddings
        student_with_pos = student_emb + s_pos_emb

        # Apply 1D convolution for initial upsampling (helps with local patterns)
        student_conv = student_with_pos.transpose(1, 2)  # [B, input_dim, s_len]
        upsampled = self.upsample_conv(student_conv)  # [B, teacher_dim, s_len]
        upsampled = upsampled.transpose(1, 2)  # [B, s_len, teacher_dim]

        # Project to teacher dimension if needed (using pre-initialized layer)
        if self.input_to_teacher_proj is not None:
            student_with_pos = self.input_to_teacher_proj(student_with_pos)

        # For each teacher position, find 3 neighboring student positions for smooth interpolation
        t_pos_scaled = t_pos_norm * (s_lens.unsqueeze(1) - 1)  # [B, t_len] - positions in student scale
        t_pos_scaled = t_pos_scaled.long()  # [B, t_len]

        # Calculate max valid index per batch (ensure non-negative)
        max_index = (s_lens - 1).clamp(min=0)  # [B]

        # Use tensors for min and max in clamp
        min_val = torch.tensor(0, device=student_emb.device, dtype=torch.long)

        # Get 3 neighboring indices (prev, current, next) with proper per-batch clamping
        idx_prev = torch.clamp(t_pos_scaled - 1, min=min_val, max=max_index.unsqueeze(1))
        idx_curr = torch.clamp(t_pos_scaled, min=min_val, max=max_index.unsqueeze(1))
        idx_next = torch.clamp(t_pos_scaled + 1, min=min_val, max=max_index.unsqueeze(1))

        # Gather embeddings at these indices
        student_emb_prev = student_with_pos.gather(1, idx_prev.unsqueeze(-1).expand(-1, -1, self.teacher_dim))
        student_emb_curr = student_with_pos.gather(1, idx_curr.unsqueeze(-1).expand(-1, -1, self.teacher_dim))
        student_emb_next = student_with_pos.gather(1, idx_next.unsqueeze(-1).expand(-1, -1, self.teacher_dim))

        # Get corresponding upsampled embeddings
        upsampled_curr = upsampled.gather(1, idx_curr.unsqueeze(-1).expand(-1, -1, self.teacher_dim))

        # Use learned weights to combine the 3 points
        weight_input = torch.cat([student_emb_prev, student_emb_curr, student_emb_next], dim=-1)
        weights = self.weight_predictor(weight_input)  # [B, t_len, 3]

        # Interpolate
        interpolated = (
            weights[:, :, 0:1] * student_emb_prev +
            weights[:, :, 1:2] * student_emb_curr +
            weights[:, :, 2:3] * student_emb_next
        )

        # Blend with convolutional upsampling for better pattern preservation
        blend_factor = 0.7  # Can be made learnable
        interpolated = blend_factor * interpolated + (1 - blend_factor) * upsampled_curr

        # Refine the interpolated sequence
        interpolated = self.refine(interpolated)
        interpolated = self.layer_norm(interpolated)

        # Add teacher position encoding
        interpolated = interpolated + t_pos_emb

        # Final projection
        output = self.output_proj(interpolated)

        # Apply mask
        output_mask = t_mask[:, :t_len].unsqueeze(-1)
        output = output * output_mask

        # Pad to target length if needed
        if output.shape[1] < target_length:
            padding = (0, 0, 0, target_length - output.shape[1])
            output = F.pad(output, padding, 'constant', 0)

        output = self.final_norm(output)

        return output

# ========== Token Alignment ==========
def normalize_token(token):
    """Optimized token normalization with reduced string operations"""
    replacements = {
        'Ġ': '', '▁': '', '▔': '', '▃': '', '�': '',
        ' .': '.', ' ,': ',', ' !': '!', ' ?': '?',
        ' :': ':', ' ;': ';', ' (': '(', ' )': ')',
        ' [': '[', ' ]': ']', ' {': '{', ' }': '}'
    }
    for old, new in replacements.items():
        token = token.replace(old, new)
    return token.lower()

def ids_to_tokens(token_ids, tokenizer):
    """Optimized token conversion"""
    return tokenizer.convert_ids_to_tokens(token_ids)

def token_based_alignment(student_input_ids, teacher_input_ids, student_tokenizer, teacher_tokenizer, window_size=5, use_approx=False):
    """Token-based alignment"""
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    aligned_pairs = []
    normalized_student = [normalize_token(token) for token in student_tokens]
    normalized_teacher = [normalize_token(token) for token in teacher_tokens]

    # Precompute special token sets for faster lookup
    student_special = {student_tokenizer.pad_token, student_tokenizer.bos_token, student_tokenizer.eos_token}
    teacher_special = {teacher_tokenizer.pad_token, teacher_tokenizer.bos_token, teacher_tokenizer.eos_token}

    # Create mapping of normalized teacher tokens to positions
    token_to_teacher_positions = {}
    for j, norm_t in enumerate(normalized_teacher):
        if teacher_tokens[j] not in teacher_special:
            token_to_teacher_positions.setdefault(norm_t, []).append(j)

    # Iterate through student tokens
    for i, norm_s in enumerate(normalized_student):
        if student_tokens[i] in student_special:
            continue

        # Get approximate position and window
        t_pos_approx = int(i * len(teacher_tokens) / len(student_tokens)) if use_approx else i
        start_j = max(0, t_pos_approx - window_size) if use_approx else t_pos_approx
        end_j = min(len(teacher_tokens), t_pos_approx + window_size + 1)

        # Get candidate positions from mapping
        candidate_positions = token_to_teacher_positions.get(norm_s, [])
        matches = [j for j in candidate_positions if start_j <= j < end_j]

        if matches:
            closest_match = min(matches, key=lambda x: abs(x - t_pos_approx))
            aligned_pairs.append((i, closest_match))

    return aligned_pairs

# ========== Position Learner Network ==========
class PositionLearnerNetwork(torch.nn.Module):
    """
    Improved position learner that better handles sequences of differing lengths
    """
    def __init__(self, student_dim, teacher_dim, hidden_dim=256, max_distance_ratio=3.0):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.hidden_dim = hidden_dim
        self.max_distance_ratio = max_distance_ratio

        # Position encodings
        self.student_pos_encoder = torch.nn.Linear(1, hidden_dim)
        self.teacher_pos_encoder = torch.nn.Linear(1, hidden_dim)

        # Length ratio processing (more robust)
        self.length_ratio_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),  # [log_ratio, inverse_ratio]
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Cumulative distance modeling with proper scaling
        self.cumulative_distance_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),  # [position, length_ratio]
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Position predictor with bias for different sequence lengths
        position_input_dim = hidden_dim * 4  # s_pos, cum_dist, len_ratio, edge_pos
        self.position_predictor = torch.nn.Sequential(
            torch.nn.Linear(position_input_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # Predict position offset ratio
        )

        # Learnable parameters for length scaling
        self.length_scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.position_bias = torch.nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with better defaults"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        # Initialize length scaling to account for typical tokenization differences
        self.length_scale_factor.data.fill_(1.2)  # Teacher sequences often ~20% longer

    def compute_position_mapping(self, s_positions_norm, s_lengths, t_lengths):
        """
        Compute expected position mapping with proper length ratio handling
        """
        # Ensure lengths are at least 1 to avoid division by zero
        s_lengths_safe = s_lengths.clamp(min=1)
        t_lengths_safe = t_lengths.clamp(min=1)

        # Calculate length ratio (teacher_length / student_length)
        length_ratio = t_lengths_safe.float() / s_lengths_safe.float()
        log_ratio = torch.log(length_ratio.clamp(min=0.1, max=10.0))

        # Non-linear mapping that accounts for cumulative tokenization differences
        # Early tokens: closer alignment, Later tokens: more divergence
        base_mapping = s_positions_norm * length_ratio.unsqueeze(1)

        # Add quadratic term for cumulative differences
        quadratic_term = (s_positions_norm ** 2) * (length_ratio.unsqueeze(1) - 1.0)

        # Combine with learnable scaling
        expected_positions = base_mapping + quadratic_term * self.length_scale_factor

        # Normalize to [0, 1] range
        expected_positions = torch.clamp(expected_positions, 0.0, 1.0)

        return expected_positions, log_ratio

    def predict_positions_batch(self, s_positions, s_mask, t_mask):
        """
        Predict teacher positions with robust handling of differing sequence lengths
        """
        device = s_mask.device
        batch_size = s_mask.size(0)

        # Get sequence lengths from masks
        s_lengths = s_mask.sum(dim=1).long()  # [B]
        t_lengths = t_mask.sum(dim=1).long()  # [B]

        # Ensure lengths are at least 1 to avoid division by zero
        s_lengths = s_lengths.clamp(min=1)
        t_lengths = t_lengths.clamp(min=1)

        # Create position indices based on the provided positions
        # s_positions should be [B, N] where N is number of positions to predict
        max_pos = s_positions.max().item()

        # Create a mask for valid positions
        valid_pos_mask = s_positions < s_lengths.unsqueeze(1)  # [B, N]

        # Normalize positions for valid entries only
        s_positions_norm = s_positions.float()
        s_lengths_float = s_lengths.unsqueeze(1).float()  # [B, 1]

        # Avoid division by zero
        s_lengths_float = s_lengths_float.clamp(min=1)
        s_positions_norm = s_positions_norm / (s_lengths_float - 1)

        # Apply mask to ensure invalid positions remain 0
        s_positions_norm = s_positions_norm * valid_pos_mask.float()

        # Compute position mapping and length ratio features
        expected_positions_norm, log_ratio = self.compute_position_mapping(
            s_positions_norm, s_lengths, t_lengths
        )

        # Encode position features
        s_pos_emb = self.student_pos_encoder(s_positions_norm.unsqueeze(-1))
        cum_dist_emb = self.cumulative_distance_encoder(
            torch.cat([s_positions_norm.unsqueeze(-1), expected_positions_norm.unsqueeze(-1)], dim=-1)
        )

        # Encode length ratio features
        inverse_ratio = s_lengths.float() / t_lengths.float()
        ratio_features = torch.cat([log_ratio.unsqueeze(-1), inverse_ratio.unsqueeze(-1)], dim=-1)
        length_ratio_emb = self.length_ratio_encoder(ratio_features).unsqueeze(1)

        # Expand length ratio embedding to match number of positions
        num_positions = s_positions.size(1)
        length_ratio_emb = length_ratio_emb.expand(-1, num_positions, -1)

        # Combine features
        x_input = torch.cat([
            s_pos_emb,
            cum_dist_emb,
            length_ratio_emb,
            s_pos_emb  # Edge position encoding (same as position for simplicity)
        ], dim=-1)

        # Predict position offset ratio
        offset_ratio = self.position_predictor(x_input).squeeze(-1)

        # Apply position bias and scale
        predicted_positions_norm = expected_positions_norm + offset_ratio + self.position_bias

        # Apply mask to ensure invalid positions remain 0
        predicted_positions_norm = predicted_positions_norm * valid_pos_mask.float()

        # Ensure within valid range
        predicted_positions_norm = torch.clamp(predicted_positions_norm, 0.0, 1.0)

        # Convert to absolute positions
        predicted_positions = (predicted_positions_norm * (t_lengths - 1).unsqueeze(1)).long()

        # Clamp each sequence to its valid range and mask invalid positions
        max_positions = (t_lengths - 1).unsqueeze(1)
        min_positions = torch.zeros_like(max_positions)
        predicted_positions = torch.clamp(predicted_positions, min=min_positions, max=max_positions)

        # Apply mask to ensure invalid positions are set to 0
        predicted_positions = predicted_positions * valid_pos_mask.long()

        return predicted_positions

    def predict_positions_batch_continuous(self, s_positions, s_mask, t_mask):
        """
        Predict teacher positions (continuous normalized) with robust handling of differing sequence lengths.
        Returns normalized positions [0,1] for loss computation.
        """
        device = s_mask.device
        batch_size = s_mask.size(0)

        # Get sequence lengths from masks
        s_lengths = s_mask.sum(dim=1).long()
        t_lengths = t_mask.sum(dim=1).long()
        s_lengths = s_lengths.clamp(min=1)
        t_lengths = t_lengths.clamp(min=1)

        # Create position indices
        s_positions = s_positions.float()
        s_lengths_float = s_lengths.unsqueeze(1).float()
        s_positions_norm = s_positions / (s_lengths_float - 1)

        # Compute position mapping
        expected_positions_norm, _ = self.compute_position_mapping(
            s_positions_norm, s_lengths, t_lengths
        )

        # Encode features
        s_pos_emb = self.student_pos_encoder(s_positions_norm.unsqueeze(-1))
        cum_dist_emb = self.cumulative_distance_encoder(
            torch.cat([s_positions_norm.unsqueeze(-1), expected_positions_norm.unsqueeze(-1)], dim=-1)
        )

        # Encode length ratio
        inverse_ratio = s_lengths.float() / t_lengths.float()
        log_ratio = torch.log(t_lengths.float() / s_lengths.float())
        ratio_features = torch.cat([log_ratio.unsqueeze(-1), inverse_ratio.unsqueeze(-1)], dim=-1)
        length_ratio_emb = self.length_ratio_encoder(ratio_features).unsqueeze(1)
        num_positions = s_positions.size(1)
        length_ratio_emb = length_ratio_emb.expand(-1, num_positions, -1)

        # Combine features
        x_input = torch.cat([
            s_pos_emb,
            cum_dist_emb,
            length_ratio_emb,
            s_pos_emb
        ], dim=-1)

        # Predict position offset ratio
        offset_ratio = self.position_predictor(x_input).squeeze(-1)
        predicted_positions_norm = expected_positions_norm + offset_ratio + self.position_bias
        predicted_positions_norm = torch.clamp(predicted_positions_norm, 0.0, 1.0)

        # Create valid position mask
        valid_pos_mask = s_positions < s_lengths.unsqueeze(1)
        predicted_positions_norm = predicted_positions_norm * valid_pos_mask.float()

        return predicted_positions_norm

# ========== Loss Functions ==========
class AlignmentLoss(torch.nn.Module):
    def __init__(self,
                 student_tokenizer=None,
                 teacher_tokenizer=None,
                 window_size: int = 3,
                 use_approximate_position: bool = True,
                 # Text alignment loss weights
                 text_huber_weight: float = 0.7,
                 text_cosine_weight: float = 0.3,
                 # Position mapping loss weights
                 position_huber_weight: float = 0.7,
                 position_cosine_weight: float = 0.3,
                 # Coverage parameters
                 additional_coverage: float = 0.25,
                 # Distance modeling parameters
                 max_distance_ratio: float = 3.0,
                 # Position prediction loss weight
                 position_prediction_weight: float = 0.05):
        super().__init__()

        # Text matching parameters
        self.window_size = window_size
        self.use_approximate_position = use_approximate_position
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Loss weights for text alignment
        self.text_huber_weight = text_huber_weight
        self.text_cosine_weight = text_cosine_weight

        # Loss weights for position mapping
        self.position_huber_weight = position_huber_weight
        self.position_cosine_weight = position_cosine_weight

        # Coverage parameters
        self.additional_coverage = additional_coverage

        # Distance modeling
        self.max_distance_ratio = max_distance_ratio

        # Position prediction loss weight
        self.position_prediction_weight = position_prediction_weight

        # Position learner
        self.position_learner = PositionLearnerNetwork(
            student_dim=512,
            teacher_dim=512,
            max_distance_ratio=max_distance_ratio
        )

        # Loss functions
        self.huber_loss = torch.nn.HuberLoss(delta=1.0, reduction='mean')
        self.cosine_loss = torch.nn.CosineSimilarity(dim=-1)

    def compute_position_prediction_loss(self, text_aligned_pairs, s_mask, t_mask):
        """Compute position prediction loss using text-aligned pairs"""
        if len(text_aligned_pairs) < 2:
            return torch.tensor(0.0, device=s_mask.device, requires_grad=True), torch.tensor(0.0, device=s_mask.device, requires_grad=True)

        device = s_mask.device

        # Extract positions from text-aligned pairs
        s_positions = torch.tensor([pair[0] for pair in text_aligned_pairs], device=device)
        target_t_positions = torch.tensor([pair[1] for pair in text_aligned_pairs], device=device)

        # Create batch dimension
        s_positions = s_positions.unsqueeze(0)  # [1, num_pairs]
        target_t_positions = target_t_positions.unsqueeze(0)  # [1, num_pairs]

        # Add batch dimension to masks
        s_mask_batch = s_mask.unsqueeze(0)  # [1, seq_len]
        t_mask_batch = t_mask.unsqueeze(0)  # [1, seq_len]

        # Predict continuous normalized positions
        predicted_t_positions_norm = self.position_learner.predict_positions_batch_continuous(
            s_positions,
            s_mask_batch,
            t_mask_batch
        )  # [1, num_pairs]

        # Normalize target positions to [0,1]
        t_lengths = t_mask_batch.sum(dim=1).max().item()
        t_lengths = max(t_lengths, 1)
        target_t_positions_norm = target_t_positions.float() / (t_lengths - 1)  # [1, num_pairs]

        # Create valid mask - ensure it's 2D for proper broadcasting
        valid_mask = (target_t_positions >= 0) & (target_t_positions < t_lengths)  # [1, num_pairs]
        valid_mask &= (predicted_t_positions_norm >= 0) & (predicted_t_positions_norm <= 1)

        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True)

        # Squeeze only after creating the mask
        predicted_valid_norm = predicted_t_positions_norm.squeeze(0)[valid_mask.squeeze(0)]
        target_valid_norm = target_t_positions_norm.squeeze(0)[valid_mask.squeeze(0)]

        # Compute losses on continuous positions
        huber_loss = torch.nn.HuberLoss(delta=1.0, reduction='mean')
        position_loss = huber_loss(predicted_valid_norm, target_valid_norm)

        # Add L1 term (optional, but keep for smoothness)
        l1_loss = F.l1_loss(predicted_valid_norm, target_valid_norm)
        total_position_loss = position_loss + 0.5 * l1_loss

        return total_position_loss, position_loss

    def compute_huber_cosine_loss(self, student_embs, teacher_embs, huber_weight, cosine_weight):
        """Compute combined Huber and cosine loss"""
        if len(student_embs) == 0:
            return torch.tensor(0.0, device=student_embs.device, requires_grad=True), \
                   torch.tensor(0.0, device=student_embs.device, requires_grad=True), \
                   torch.tensor(0.0, device=student_embs.device, requires_grad=True)

        # Huber loss
        huber_loss = self.huber_loss(student_embs, teacher_embs)

        # Cosine loss
        cos_sim = self.cosine_loss(student_embs, teacher_embs)
        cosine_loss = (1 - cos_sim).mean()

        return huber_loss, cosine_loss

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                student_mask: torch.Tensor, teacher_mask: torch.Tensor,
                student_input_ids: Optional[torch.Tensor] = None,
                teacher_input_ids: Optional[torch.Tensor] = None):
        device = student_output.device
        batch_size = student_output.size(0)

        # Initialize losses
        total_text_huber_loss = torch.tensor(0.0, device=device)
        total_text_cosine_loss = torch.tensor(0.0, device=device)
        total_position_huber_loss = torch.tensor(0.0, device=device)
        total_position_cosine_loss = torch.tensor(0.0, device=device)
        total_position_prediction_loss = torch.tensor(0.0, device=device)

        # Initialize counters for logging
        total_text_aligned_tokens = 0
        total_student_tokens = 0
        total_text_aligned_pairs = 0
        total_position_mapped_tokens = 0

        for i in range(batch_size):
            # Get actual tokens (not padded)
            t_indices = (teacher_mask[i] == 1).nonzero(as_tuple=True)[0]
            s_indices = (student_mask[i] == 1).nonzero(as_tuple=True)[0]

            if len(t_indices) == 0 or len(s_indices) == 0:
                continue

            # Count total student tokens (excluding special tokens)
            valid_student_tokens = 0
            for idx in s_indices:
                if student_input_ids[i][idx].item() not in [
                    self.student_tokenizer.pad_token_id,
                    self.student_tokenizer.bos_token_id,
                    self.student_tokenizer.eos_token_id
                ]:
                    valid_student_tokens += 1
            total_student_tokens += valid_student_tokens

            # Step 1: Text-based alignment
            text_aligned_pairs = token_based_alignment(
                student_input_ids[i], teacher_input_ids[i],
                self.student_tokenizer, self.teacher_tokenizer,
                window_size=self.window_size,
                use_approx=self.use_approximate_position
            )

            # Count text-aligned tokens
            text_aligned_student_positions = {pair[0] for pair in text_aligned_pairs}
            text_aligned_tokens_count = sum(
                1 for idx in text_aligned_student_positions
                if student_input_ids[i][idx].item() not in [
                    self.student_tokenizer.pad_token_id,
                    self.student_tokenizer.bos_token_id,
                    self.student_tokenizer.eos_token_id
                ]
            )
            total_text_aligned_tokens += text_aligned_tokens_count
            total_text_aligned_pairs += len(text_aligned_pairs)

            # Step 2: Position prediction loss (NEW)
            if text_aligned_pairs:
                pos_pred_loss, pos_pred_mse = self.compute_position_prediction_loss(
                    text_aligned_pairs, student_mask[i], teacher_mask[i]
                )
                total_position_prediction_loss += pos_pred_loss
            else:
                pos_pred_mse = torch.tensor(0.0, device=device, requires_grad=True)

            # Step 3: Position mapping using UNALIGNED tokens
            position_aligned_pairs = []
            if self.additional_coverage > 0:
                # Get all valid student positions (excluding special tokens)
                all_valid_positions = []
                for idx in s_indices:
                    if student_input_ids[i][idx].item() not in [
                        self.student_tokenizer.pad_token_id,
                        self.student_tokenizer.bos_token_id,
                        self.student_tokenizer.eos_token_id
                    ]:
                        all_valid_positions.append(idx.item())

                # Get unaligned positions (those not in text_aligned_pairs)
                unaligned_positions = [pos for pos in all_valid_positions
                                    if pos not in text_aligned_student_positions]

                if unaligned_positions and text_aligned_pairs:  # Need both unaligned and some aligned for reference
                    # Calculate how many unaligned positions to sample
                    num_to_sample = max(1, int(len(unaligned_positions) * self.additional_coverage))
                    num_to_sample = min(num_to_sample, len(unaligned_positions))

                    # Randomly sample from unaligned positions
                    sampled_positions = random.sample(unaligned_positions, num_to_sample)

                    # Predict teacher positions using position learner
                    s_positions_tensor = torch.tensor(sampled_positions, device=device).unsqueeze(0)
                    predicted_t_positions = self.position_learner.predict_positions_batch(
                        s_positions_tensor,
                        student_mask[i].unsqueeze(0),
                        teacher_mask[i].unsqueeze(0)
                    )
                    predicted_t_positions = predicted_t_positions.squeeze(0).cpu().numpy()

                    position_aligned_pairs = list(zip(sampled_positions, predicted_t_positions))

            # Step 4: Compute losses for text-aligned tokens
            if text_aligned_pairs:
                student_embs = student_output[i, [pair[0] for pair in text_aligned_pairs]]
                teacher_embs = teacher_output[i, [pair[1] for pair in text_aligned_pairs]]

                text_huber_loss, text_cosine_loss = self.compute_huber_cosine_loss(
                    student_embs, teacher_embs,
                    self.text_huber_weight, self.text_cosine_weight
                )

                total_text_huber_loss += text_huber_loss
                total_text_cosine_loss += text_cosine_loss
            else:
                text_huber_loss = torch.tensor(0.0, device=device, requires_grad=True)
                text_cosine_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Step 5: Compute losses for position-mapped tokens (from unaligned positions)
            if position_aligned_pairs:
                student_embs = student_output[i, [pair[0] for pair in position_aligned_pairs]]
                teacher_embs = teacher_output[i, [pair[1] for pair in position_aligned_pairs]]

                position_huber_loss, position_cosine_loss = self.compute_huber_cosine_loss(
                    student_embs, teacher_embs,
                    self.position_huber_weight,
                    self.position_cosine_weight
                )

                total_position_huber_loss += position_huber_loss
                total_position_cosine_loss += position_cosine_loss
                total_position_mapped_tokens += len(position_aligned_pairs)
            else:
                position_huber_loss = torch.tensor(0.0, device=device, requires_grad=True)
                position_cosine_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Average across batch
        if batch_size > 0:
            total_text_huber_loss = total_text_huber_loss / batch_size
            total_text_cosine_loss = total_text_cosine_loss / batch_size
            total_position_huber_loss = total_position_huber_loss / batch_size
            total_position_cosine_loss = total_position_cosine_loss / batch_size
            total_position_prediction_loss = total_position_prediction_loss / batch_size

        # Combine all losses
        total_loss = (
            self.text_huber_weight * total_text_huber_loss +
            self.text_cosine_weight * total_text_cosine_loss +
            self.position_huber_weight * total_position_huber_loss +
            self.position_cosine_weight * total_position_cosine_loss +
            self.position_prediction_weight * total_position_prediction_loss
        )

        # Return alignment ratio for logging
        text_aligned_ratio = total_text_aligned_tokens / max(total_student_tokens, 1)

        if total_text_aligned_pairs > 0:
            position_mapping_ratio = total_position_mapped_tokens / max(total_student_tokens, 1)
        else:
            position_mapping_ratio = 0.0

        return (
            total_loss,
            total_text_huber_loss, total_text_cosine_loss,
            total_position_huber_loss, total_position_cosine_loss,
            text_aligned_ratio,
            position_mapping_ratio,
            total_position_prediction_loss
        )

    def get_position_learner_parameters(self):
        """Return all position-learner-related parameters for optimizer"""
        return list(self.parameters())

    def get_position_learner_parameters_by_type(self):
        """Return position learner parameters grouped by type for different learning rates"""
        params_by_type = {
            'position_learner': []     # For position learner network
        }

        params_by_type['position_learner'].extend(list(self.position_learner.parameters()))

        # Add clipping for alignment params
        for param in params_by_type['position_learner']:
            if param.requires_grad:
                param.register_hook(lambda grad: grad.clamp_(-1, 1))

        return params_by_type

class TokenLoss(torch.nn.Module):
    """Loss for token-level position matching with optional position targeting"""
    def __init__(self, huber_weight: float = 0.7, cosine_weight: float = 0.3,
                 huber_delta: float = 1.0):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                teacher_mask: torch.Tensor, student_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Compute token-level loss on specified positions
        """
        device = student_output.device
        position_mask = teacher_mask.bool()

        if not position_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True), 0

        # Flatten for easier computation
        student_flat = student_output.view(-1, student_output.size(-1))
        teacher_flat = teacher_output.view(-1, teacher_output.size(-1))
        mask_flat = position_mask.view(-1)

        # Apply mask
        student_masked = student_flat[mask_flat]
        teacher_masked = teacher_flat[mask_flat]

        if len(student_masked) == 0:
            return torch.tensor(0.0, device=student_output.device, requires_grad=True), torch.tensor(0.0, device=student_output.device, requires_grad=True), torch.tensor(0.0, device=student_output.device, requires_grad=True), 0

        # Compute losses
        huber_loss = self.huber_loss(student_masked, teacher_masked)
        cos_sim = self.cos_loss(student_masked, teacher_masked)
        cos_loss = (1 - cos_sim).mean()

        total_loss = (
            self.huber_weight * huber_loss +
            self.cosine_weight * cos_loss
        )

        num_tokens = len(student_masked)

        return total_loss, huber_loss, cos_loss, num_tokens

class SequenceLoss(torch.nn.Module):
    """Loss for extended positions using mean pooling across sequence dimension"""
    def __init__(self, huber_weight: float = 0.7, cosine_weight: float = 0.3,
                 huber_delta: float = 1.0):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta)
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                student_mask: torch.Tensor, teacher_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Compute mean pooling loss on specified positions
        """
        device = student_output.device
        position_mask = teacher_mask.bool()

        if not position_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True), 0

        # Apply position mask to embeddings
        masked_student = student_output * position_mask.unsqueeze(-1)
        masked_teacher = teacher_output * position_mask.unsqueeze(-1)

        # Mean pooling across sequence dimension
        # Sum across sequence, divide by number of actual tokens in each sequence
        pool_denominator = position_mask.sum(dim=1, keepdim=True).clamp(min=1)
        student_pooled = masked_student.sum(dim=1) / pool_denominator
        teacher_pooled = masked_teacher.sum(dim=1) / pool_denominator

        # Compute losses
        huber_loss = self.huber_loss(student_pooled, teacher_pooled)
        cos_sim = self.cos_loss(student_pooled, teacher_pooled)
        cos_loss = (1 - cos_sim).mean()

        total_loss = (
            self.huber_weight * huber_loss +
            self.cosine_weight * cos_loss
        )

        num_positions = position_mask.sum().item()

        return total_loss, huber_loss, cos_loss, num_positions

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

        elif layer_config["type"] == "interpolation":
            output_dim = layer_config["output_dim"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = LearnedInterpolationLayer(
                    input_dim=input_dim,
                    teacher_dim=output_dim,
                    max_length=512
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing interpolation layer {file_num}")
            else:
                projection_layer = LearnedInterpolationLayer(
                    input_dim=input_dim,
                    teacher_dim=output_dim,
                    max_length=512
                )
                print(f"Initialising new interpolation layer {file_num}")
            projection_layer.output_dim = output_dim

        projection_layer.file_num = layer_config["file_num"]
        projection_layers.append(projection_layer)

    return projection_layers, layers_to_load

def get_projection_parameters_by_type(projection_layers: List[torch.nn.Module]) -> Dict[str, List[torch.nn.Parameter]]:
    """Get projection parameters grouped by individual torch layer type"""
    params_by_type = {
        'transformer': [],
        'mlp': [],
        'linear': [],
        'interpolation': []
    }

    for layer_idx, layer in enumerate(projection_layers):
        if layer_idx + 1 in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
            continue

        if isinstance(layer, LearnedInterpolationLayer):
            group = 'interpolation'
            for param in layer.parameters():
                if param.requires_grad:
                    params_by_type[group].append(param)
            continue

        # Use named_children to avoid recursion into nested modules
        for name, submodule in layer.named_children():
            # Determine group based on actual PyTorch layer type
            if isinstance(submodule, torch.nn.Linear):
                group = 'linear'
            elif isinstance(submodule, torch.nn.TransformerEncoder):
                group = 'transformer'
            elif isinstance(submodule, torch.nn.Sequential):
                group = 'mlp'
            elif isinstance(submodule, torch.nn.Conv1d):
                group = 'mlp'
            elif isinstance(submodule, torch.nn.Embedding):
                group = 'linear'
            else:
                continue

            for param in submodule.parameters():
                if param.requires_grad:
                    params_by_type[group].append(param)

    return params_by_type

# ========== Optimiser Initialisation ==========
def initialize_optimizer(parameters: List[torch.nn.Parameter], max_lr: float, min_lr: float) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    # Apply multiplier for first restart cycle at initialization
    actual_max_lr = max_lr
    actual_min_lr = min_lr

    optimizer = torch.optim.AdamW(
        parameters,
        lr=actual_max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    if RESTART_CYCLE_STEPS > 0:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=RESTART_CYCLE_STEPS,
            T_mult=1,
            eta_min=actual_min_lr,
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
            lr_lambda=lambda step: min((actual_min_lr / actual_max_lr) + (step / (WARMUP_STEPS)) * (1 - actual_min_lr / actual_max_lr), 1.0)
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

    # Interpolation layers
    if params_by_type['interpolation']:
        optimizers['interpolation'] = initialize_optimizer(
            params_by_type['interpolation'],
            MAX_LEARNING_RATE_INTERPOLATION,
            MIN_LEARNING_RATE_INTERPOLATION
        )

    return optimizers

def reset_scheduler(scheduler, multiplier):
    """Recursively reset scheduler base LRs and eta_min"""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
        for s in scheduler.schedulers:
            reset_scheduler(s, multiplier)
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [lr / multiplier for lr in scheduler.base_lrs]
    if hasattr(scheduler, 'eta_min'):
        scheduler.eta_min = scheduler.eta_min / multiplier

# ========== Evaluation Function ==========
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, projection_layers: List[torch.nn.Module],
                  align_loss_fn, token_loss_fn, sequence_loss_fn,
                  device: str, autocast_dtype: torch.dtype,
                  student_tokenizer, teacher_tokenizer, teacher_model=None) -> Dict[str, float]:
    """Evaluate model with new loss functions"""
    current_model_state = model.training
    current_layer_states = [layer.training for layer in projection_layers]

    model.eval()
    for layer in projection_layers:
        layer.eval()

    if align_loss_fn is not None:
        align_loss_fn.eval()
    if token_loss_fn is not None:
        token_loss_fn.eval()
    if sequence_loss_fn is not None:
        sequence_loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'align_huber': 0.0,
        'align_cos': 0.0,
        'position_huber': 0.0,
        'position_cos': 0.0,
        'token_huber': 0.0,
        'token_cos': 0.0,
        'sequence_huber': 0.0,
        'sequence_cos': 0.0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                if USE_CACHED_EMBEDDINGS:
                    s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask = batch
                    teacher_hidden = t_embeddings.to(device)
                else:
                    s_input_ids, s_mask, t_input_ids, t_mask = batch
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
                        if isinstance(layer, LearnedInterpolationLayer):
                            projected_student = layer(
                                projected_student,
                                s_mask=s_mask,
                                t_mask=t_mask,
                                target_length=512
                            )
                        else:
                            projected_student = layer(projected_student)

                eval_loss = torch.tensor(0.0, device=device)
                eval_align_huber = torch.tensor(0.0, device=device)
                eval_align_cos = torch.tensor(0.0, device=device)
                eval_position_huber = torch.tensor(0.0, device=device)
                eval_position_cos = torch.tensor(0.0, device=device)
                eval_token_huber = torch.tensor(0.0, device=device)
                eval_token_cos = torch.tensor(0.0, device=device)
                eval_sequence_huber = torch.tensor(0.0, device=device)
                eval_sequence_cos = torch.tensor(0.0, device=device)

                if align_loss_fn is not None and (TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0 or POSITION_MAPPING_COSINE_WEIGHT > 0 or POSITION_MAPPING_HUBER_WEIGHT > 0):
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        total_loss, eval_align_huber, eval_align_cos, eval_position_huber, eval_position_cos, _, _, eval_position_prediction = align_loss_fn(
                            projected_student,
                            teacher_hidden,
                            s_mask,
                            t_mask,
                            s_input_ids,
                            t_input_ids
                        )
                    eval_loss += total_loss

                if sequence_loss_fn is not None and (SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0):
                    sequence_loss, eval_sequence_huber, eval_sequence_cos, _ = sequence_loss_fn(
                        projected_student,
                        teacher_hidden,
                        s_mask,
                        t_mask
                    )
                    eval_loss += sequence_loss

                if token_loss_fn is not None and (TOKEN_HUBER_WEIGHT > 0 or TOKEN_COSINE_WEIGHT > 0):
                    token_loss, eval_token_huber, eval_token_cos, _ = token_loss_fn(
                        projected_student,
                        teacher_hidden,
                        t_mask,
                        student_mask=s_mask
                    )
                    eval_loss += token_loss

                total_losses['total'] += eval_loss.item()
                total_losses['align_huber'] += eval_align_huber.item()
                total_losses['align_cos'] += eval_align_cos.item()
                total_losses['position_huber'] += eval_position_huber.item()
                total_losses['position_cos'] += eval_position_cos.item()
                total_losses['token_huber'] += eval_token_huber.item()
                total_losses['token_cos'] += eval_token_cos.item()
                total_losses['sequence_huber'] += eval_sequence_huber.item()
                total_losses['sequence_cos'] += eval_sequence_cos.item()

                # Cleanup
                del projected_student, student_hidden, student_outputs
                if not USE_CACHED_EMBEDDINGS:
                    del teacher_outputs
                del s_input_ids, s_mask, t_mask, teacher_hidden
                if not USE_CACHED_EMBEDDINGS:
                    del t_input_ids

            except Exception as e:
                logging.exception(f"Error in evaluation batch {batch_idx}: {e}")
                raise e

    # Compute averages
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches

    # Restore training mode
    if current_model_state:
        model.train()
    for layer, state in zip(projection_layers, current_layer_states):
        if state:
            layer.train()

    if align_loss_fn is not None:
        align_loss_fn.train()
    if token_loss_fn is not None:
        token_loss_fn.train()
    if sequence_loss_fn is not None:
        sequence_loss_fn.train()

    return total_losses

# ========== Optimizer Handling ==========
def save_optimizer_states(save_path: str, model_optimizer, projection_optimizers, position_learner_optimizer=None):
    """Save optimizer states to a subfolder"""
    if not SAVE_OPTIMIZER_STATES:
        return

    optimizer_dir = os.path.join(save_path, "optimizers")
    os.makedirs(optimizer_dir, exist_ok=True)

    # Save model optimizer state
    if model_optimizer is not None:
        torch.save(model_optimizer.state_dict(), os.path.join(optimizer_dir, "model_optimizer.pt"))

    # Save projection optimizer states
    for opt_name, (optimizer, scheduler) in projection_optimizers.items():
        torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, f"{opt_name}_optimizer.pt"))

    # Save position learner optimizer state
    if position_learner_optimizer is not None:
        torch.save(position_learner_optimizer.state_dict(), os.path.join(optimizer_dir, "position_learner_optimizer.pt"))

def load_optimizer_states(save_path: str, model_optimizer, scheduler_model, projection_optimizers):
    """Load optimizer states from a subfolder if available"""
    if not REUSE_OPTIMIZER_STATE:
        return False

    optimizer_dir = os.path.join(save_path, "optimizers")
    if not os.path.exists(optimizer_dir):
        return False

    success = True
    # Load model optimizer state
    if model_optimizer is not None:
        model_opt_path = os.path.join(optimizer_dir, "model_optimizer.pt")
        if os.path.exists(model_opt_path):
            try:
                model_optimizer.load_state_dict(torch.load(model_opt_path))
                # Reset learning rate to current scheduler value
                if scheduler_model is not None:
                    for param_group, lr in zip(model_optimizer.param_groups, scheduler_model.get_last_lr()):
                        param_group['lr'] = lr
                print("Loaded model optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load model optimizer state: {e}")
                success = False
        else:
            print("Warning: Model optimizer state file not found.")
            success = False

    # Load projection optimizer states
    for opt_name, (optimizer, scheduler) in projection_optimizers.items():
        opt_path = os.path.join(optimizer_dir, f"{opt_name}_optimizer.pt")
        if os.path.exists(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path))
                # Reset learning rate to current scheduler value
                if scheduler is not None:
                    for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                        param_group['lr'] = lr
                print(f"Loaded {opt_name} optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load {opt_name} optimizer state: {e}")
                success = False
        else:
            print(f"Warning: {opt_name} optimizer state file not found.")
            success = False

    # Load alignment optimizer state
    if position_learner_optimizer is not None:
        position_learner_opt_path = os.path.join(optimizer_dir, "position_learner_optimizer.pt")
        if os.path.exists(position_learner_opt_path):
            try:
                position_learner_optimizer.load_state_dict(torch.load(position_learner_opt_path))
                print("Loaded position learner optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load position learner optimizer state: {e}")
                success = False
        else:
            print("Warning: Position learner optimizer state file not found.")
            success = False

    return success

def initialize_position_learner_optimizer(align_loss_fn):
    """Initialize optimizer for alignment loss parameters"""
    if align_loss_fn is None:
        return None, None

    # Get all parameters including position learner
    position_learner_params = align_loss_fn.get_position_learner_parameters()
    if not position_learner_params:
        return None, None

    # Ensure position learner parameters are included
    position_params = align_loss_fn.get_position_learner_parameters_by_type().get('position_learner', [])
    all_params = list(position_learner_params) + list(position_params)

    # Remove duplicates while preserving order
    seen = set()
    unique_params = []
    for p in all_params:
        if p not in seen:
            seen.add(p)
            unique_params.append(p)

    return initialize_optimizer(
        unique_params,
        MAX_LEARNING_RATE_POSITION,
        MIN_LEARNING_RATE_POSITION
    )

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
                      projection_layers: List[torch.nn.Module], qwen_embedding_dim: int,
                      model_optimizer=None, projection_optimizers=None,
                      align_loss_fn=None, position_learner_optimizer=None) -> None:
    """Save trained model with all components"""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    for layer in projection_layers:
        layer_state = layer.state_dict()
        layer_path = os.path.join(save_path, f"projection_layer_{layer.file_num}.safetensors")
        save_file(layer_state, layer_path)

    # Save alignment loss parameters
    if align_loss_fn is not None:
        # Save all components separately for better debugging
        position_learner_state = align_loss_fn.state_dict()

        # Save position learner separately
        if hasattr(align_loss_fn, 'position_learner'):
            pos_path = os.path.join(save_path, "position_learner.safetensors")
            save_file(align_loss_fn.position_learner.state_dict(), pos_path)

    projection_config_path = os.path.join(save_path, "projection_config.json")
    save_projection_config(projection_config_path, qwen_embedding_dim)

    # Save optimizer states
    save_optimizer_states(save_path, model_optimizer, projection_optimizers, position_learner_optimizer)

def save_projection_config(projection_config_path: str, embedding_dim: int) -> None:
    """Save projection configuration"""
    projection_config = {
        "layers": PROJECTION_LAYERS_CONFIG,
    }
    with open(projection_config_path, "w") as f:
        json.dump(projection_config, f)

def set_training_mode(model, projection_layers, train_model_flag=True, train_projection_flag=True):
    """Set training mode only for layers that are being trained"""
    if train_model_flag:
        model.train()
    else:
        model.eval()

    for layer_idx, layer in enumerate(projection_layers):
        # Check if this layer is excluded from training
        is_excluded = layer_idx + 1 in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS

        # Only set to train mode if projection training is enabled AND layer is not excluded
        if train_projection_flag and not is_excluded:
            layer.train()
        else:
            layer.eval()

def restore_training_mode(model, projection_layers, train_model_flag=True, train_projection_flag=True):
    """Restore training mode after evaluation"""
    set_training_mode(model, projection_layers, train_model_flag, train_projection_flag)

def get_logging(epoch: int, batch_idx: int, global_step: int, total_steps: int,
               current_loss: float,
               current_align_huber: float, current_align_cos: float,
               current_position_huber: float, current_position_cos: float,
               current_token_huber: float, current_token_cos: float,
               current_sequence_huber: float, current_sequence_cos: float,
               grad_norm_model: float, grad_norm_proj: float,
               grad_norm_position_learner: float,
               current_lr_model: float, current_lr_proj: float,
               current_lr_position_learner: float,
               text_aligned_ratio: float,
               position_mapping_ratio: float,
               current_position_prediction_loss: float,
               elapsed: float, eta: float) -> str:
    """Generate logging string with dynamic loss reporting"""
    model_gn_line = ""
    model_lr_line = ""
    proj_gn_line = ""
    proj_lr_line = ""
    vram_line = ""
    batch_line = ""
    step_line = ""
    epoch_line = ""

    if TRAIN_PROJECTION and projection_optimizers:
        # Get average LR across projection optimizers
        current_lr_proj = sum(opt[0].param_groups[0]['lr'] for opt in projection_optimizers.values()) / len(projection_optimizers)
    else:
        current_lr_proj = 0.0

    if position_learner_optimizer is not None:
        current_lr_position_learner = position_learner_optimizer.param_groups[0]['lr']
    else:
        current_lr_position_learner = 0.0

    if TRAIN_MODEL:
        model_gn_line = f"GN Mod: {grad_norm_model:.6f}, "
        model_lr_line = f"Mod LR: {current_lr_model:.6f}, "
    if TRAIN_PROJECTION:
        proj_gn_line = f"GN Proj: {grad_norm_proj:.6f}, "
        proj_lr_line = f"Avg Proj LR: {current_lr_proj:.6f}, "
    if LOG_VRAM_USAGE:
        vram_free, vram_total, vram_used = get_memory_usage()
        vram_line = f"VRAM: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
    if position_learner_optimizer is not None:
        position_learner_gn_line = f"GN Pos: {grad_norm_position_learner:.6f}, "
    else:
        position_learner_gn_line = ""
    epoch_line = f"Epoch [{epoch + 1}/{EPOCHS}], "
    step_line = f"Step: {global_step}/{total_steps}, "
    batch_line = f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "

    # Build loss lines based on enabled losses
    loss_lines = []
    if TOKEN_HUBER_WEIGHT > 0 or TOKEN_COSINE_WEIGHT > 0:
        loss_lines.append(f"Tok Hub: {current_token_huber:.6f}")
        loss_lines.append(f"Tok Cos: {current_token_cos:.6f}")
    if SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0:
        loss_lines.append(f"Seq Hub: {current_sequence_huber:.6f}")
        loss_lines.append(f"Seq Cos: {current_sequence_cos:.6f}")
    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
        loss_lines.append(f"TM Hub: {current_align_huber:.6f}")
        loss_lines.append(f"TM Cos: {current_align_cos:.6f}")
        loss_lines.append(f"TM Cov: {text_aligned_ratio:.2%}")
        if POSITION_MAPPING_HUBER_WEIGHT > 0 or POSITION_MAPPING_COSINE_WEIGHT > 0:
            loss_lines.append(f"PP Hub: {current_position_huber:.6f}")
            loss_lines.append(f"PP Cos: {current_position_cos:.6f}")
            loss_lines.append(f"PP Cov: {position_mapping_ratio:.2%}")
        loss_lines.append(f"Pos Pred: {current_position_prediction_loss:.6f}")

    loss_str = ", ".join(loss_lines)

    log_line = (
        f"{epoch_line}"
        f"{batch_line}"
        f"{step_line}"
        f"Total Loss: {current_loss:.6f}, "
        f"{loss_str}, "
        f"{model_gn_line}"
        f"{proj_gn_line}"
        f"{position_learner_gn_line}"
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
    global total_steps, train_dataloader, train_dataset, eval_dataloader, eval_dataset, projection_optimizers, position_learner_optimizer

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

    # Initialize loss functions
    align_loss_fn = None
    token_loss_fn = None
    sequence_loss_fn = None

    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
        align_loss_fn = AlignmentLoss(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            window_size=ALIGN_WINDOW,
            use_approximate_position=True,
            text_huber_weight=TEXT_MATCH_HUBER_WEIGHT,
            text_cosine_weight=TEXT_MATCH_COSINE_WEIGHT,
            position_huber_weight=POSITION_MAPPING_HUBER_WEIGHT,
            position_cosine_weight=POSITION_MAPPING_COSINE_WEIGHT,
            additional_coverage=POSITION_MAPPING_COVERAGE,
            max_distance_ratio=3.0
        ).to(device, dtype=torch.bfloat16)

        position_learner_path = os.path.join(QWEN3_MODEL_NAME, "position_learner.safetensors")
        if os.path.exists(position_learner_path):
            print(f"Loading existing position learner loss parameters from {position_learner_path}")
            position_learner_state = load_file(position_learner_path)
            align_loss_fn.position_learner.load_state_dict(position_learner_state)

    if TOKEN_HUBER_WEIGHT > 0 or TOKEN_COSINE_WEIGHT > 0:
        token_loss_fn = TokenLoss(
            huber_weight=TOKEN_HUBER_WEIGHT,
            cosine_weight=TOKEN_COSINE_WEIGHT
        ).to(device, dtype=torch.bfloat16)

    if SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0:
        sequence_loss_fn = SequenceLoss(
            huber_weight=SEQUENCE_HUBER_WEIGHT,
            cosine_weight=SEQUENCE_COSINE_WEIGHT
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

            if align_loss_fn is not None:
                position_learner_optimizer, position_learner_scheduler = initialize_position_learner_optimizer(align_loss_fn)

            model_optimizer = None
            scheduler_model = None
            if TRAIN_MODEL:
                model_parameters = [p for p in student_model.parameters() if p.requires_grad]
                model_optimizer, scheduler_model = initialize_optimizer(model_parameters, MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)

            if REUSE_OPTIMIZER_STATE and load_optimizer_states(QWEN3_MODEL_NAME, model_optimizer, scheduler_model, projection_optimizers):
                # Update learning rates to match scheduler current state
                if TRAIN_MODEL and scheduler_model:
                    for param_group, lr in zip(model_optimizer.param_groups, scheduler_model.get_last_lr()):
                        param_group['lr'] = lr

                if TRAIN_PROJECTION:
                    for optimizer, scheduler in projection_optimizers.values():
                        if scheduler:
                            for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                                param_group['lr'] = lr

                if position_learner_optimizer and position_learner_scheduler and align_loss_fn is not None:
                    for param_group, lr in zip(position_learner_optimizer.param_groups, position_learner_scheduler.get_last_lr()):
                        param_group['lr'] = lr

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
            if align_loss_fn is not None:
                align_loss_fn.train()

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
                                if isinstance(layer, LearnedInterpolationLayer):
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

                        total_loss = torch.tensor(0.0, device=device)
                        align_loss_huber = torch.tensor(0.0, device=device)
                        align_loss_cos = torch.tensor(0.0, device=device)
                        position_loss_huber = torch.tensor(0.0, device=device)
                        position_loss_cos = torch.tensor(0.0, device=device)
                        token_loss = torch.tensor(0.0, device=device)
                        token_huber = torch.tensor(0.0, device=device)
                        token_cos = torch.tensor(0.0, device=device)
                        sequence_loss = torch.tensor(0.0, device=device)
                        sequence_huber = torch.tensor(0.0, device=device)
                        sequence_cos = torch.tensor(0.0, device=device)
                        text_aligned_ratio = 0.0
                        position_mapping_ratio = 0.0
                        position_prediction_loss = torch.tensor(0.0, device=device)

                        if align_loss_fn is not None:
                            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                                total_align_loss, align_loss_huber, align_loss_cos, position_loss_huber, position_loss_cos, text_aligned_ratio, position_mapping_ratio, position_prediction_loss = align_loss_fn(
                                    projected_student,
                                    teacher_hidden,
                                    s_mask,
                                    t_mask,
                                    student_input_ids=s_input_ids,
                                    teacher_input_ids=t_input_ids
                                )
                        total_loss += total_align_loss

                        if token_loss_fn is not None:
                            token_loss, token_huber, token_cos, num_token = token_loss_fn(
                                projected_student,
                                teacher_hidden,
                                t_mask,
                                student_mask=s_mask
                            )
                            total_loss += token_loss

                        if sequence_loss_fn is not None:
                            sequence_loss, sequence_huber, sequence_cos, num_sequence = sequence_loss_fn(
                                projected_student,
                                teacher_hidden,
                                s_mask,
                                t_mask
                            )
                            total_loss += sequence_loss

                        # Scale loss for gradient accumulation
                        scaled_loss = total_loss / GRAD_ACCUM_STEPS
                        scaler.scale(scaled_loss).backward()
                        accumulation_step += 1

                        if accumulation_step >= GRAD_ACCUM_STEPS or batch_idx == len(train_dataloader) - 1:
                            # Unscale the optimizers first
                            if TRAIN_MODEL and model_optimizer:
                                scaler.unscale_(model_optimizer)
                            for opt in projection_optimizers.values():
                                scaler.unscale_(opt[0])
                            if position_learner_optimizer is not None:
                                scaler.unscale_(position_learner_optimizer)

                            # Clip gradients before stepping
                            if TRAIN_MODEL and model_optimizer:
                                grad_norm_model = clip_grad_norm_(
                                    [p for p in student_model.parameters() if p.requires_grad],
                                    max_norm=GRAD_CLIP
                                )
                            else:
                                grad_norm_model = 0.0

                            all_proj_params = []
                            for layer in projection_layers:
                                if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
                                    all_proj_params.extend([p for p in layer.parameters() if p.requires_grad])

                            grad_norm_proj = clip_grad_norm_(
                                all_proj_params if all_proj_params else [],
                                max_norm=GRAD_CLIP
                            )

                            if position_learner_optimizer is not None:
                                grad_norm_position_learner = clip_grad_norm_(
                                    position_learner_optimizer.param_groups[0]['params'],
                                    max_norm=GRAD_CLIP
                                )
                            else:
                                grad_norm_position_learner = 0.0

                            # Step optimizers after clipping
                            if TRAIN_MODEL and model_optimizer:
                                scaler.step(model_optimizer)
                            for opt in projection_optimizers.values():
                                scaler.step(opt[0])
                            if position_learner_optimizer is not None:
                                scaler.step(position_learner_optimizer)

                            # Update scaler
                            scaler.update()

                            # Clear gradients after stepping
                            accumulation_step = 0
                            if TRAIN_MODEL and model_optimizer:
                                model_optimizer.zero_grad()
                            for opt in projection_optimizers.values():
                                opt[0].zero_grad()
                            if position_learner_optimizer is not None:
                                position_learner_optimizer.zero_grad()

                            global_step += 1
                            steps_completed_this_epoch += 1

                            # Reset gradients for projection layers
                            for layer in projection_layers:
                                if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
                                    for param in layer.parameters():
                                        if param.grad is not None:
                                            param.grad = None

                            current_loss = total_loss.item()
                            current_align_huber = align_loss_huber.item()
                            current_align_cos = align_loss_cos.item()
                            current_position_huber = position_loss_huber.item()
                            current_position_cos = position_loss_cos.item()
                            current_position_prediction_loss = position_prediction_loss.item()
                            current_token_huber = token_huber.item()
                            current_token_cos = token_cos.item()
                            current_sequence_huber = sequence_huber.item()
                            current_sequence_cos = sequence_cos.item()

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

                            if position_learner_optimizer is not None:
                                current_lr_position_learner = position_learner_optimizer.param_groups[0]['lr']
                            else:
                                current_lr_position_learner = 0.0

                            if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                                print(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss,
                                    current_align_huber, current_align_cos,
                                    current_position_huber, current_position_cos,
                                    current_token_huber, current_token_cos,
                                    current_sequence_huber, current_sequence_cos,
                                    grad_norm_model, grad_norm_proj,
                                    grad_norm_position_learner,
                                    current_lr_model, current_lr_proj,
                                    current_lr_position_learner,
                                    text_aligned_ratio,
                                    position_mapping_ratio,
                                    current_position_prediction_loss,
                                    elapsed, eta
                                ))

                            if ENABLE_LOGGING:
                                log_lines.append(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss,
                                    current_align_huber, current_align_cos,
                                    current_position_huber, current_position_cos,
                                    current_token_huber, current_token_cos,
                                    current_sequence_huber, current_sequence_cos,
                                    grad_norm_model, grad_norm_proj,
                                    grad_norm_position_learner,
                                    current_lr_model, current_lr_proj,
                                    current_lr_position_learner,
                                    text_aligned_ratio,
                                    position_mapping_ratio,
                                    current_position_prediction_loss,
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
                            if position_learner_optimizer is not None and position_learner_scheduler is not None:
                                position_learner_scheduler.step()

                            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                                print(f"\nSaving checkpoint at step {global_step}\n")
                                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizers, align_loss_fn, position_learner_optimizer)

                            # Define cycle length for restart-based training
                            if REPEAT_WARMUP_AFTER_RESTART or (restart_cycle == 1 and WARMUP_STEPS > 0):
                                cycle_length = WARMUP_STEPS + RESTART_CYCLE_STEPS
                            else:
                                cycle_length = RESTART_CYCLE_STEPS
                            warmup_offset = 0
                            if not REPEAT_WARMUP_AFTER_RESTART and restart_cycle > 1:
                                warmup_offset = WARMUP_STEPS
                            adjusted_global_step = global_step - warmup_offset

                            if cycle_length > 0 and adjusted_global_step % cycle_length == 0 and adjusted_global_step > 0:
                                if SAVE_EVERY_X_RESTARTS > 0 and restart_cycle % SAVE_EVERY_X_RESTARTS == 0:
                                    print(f"\nSaving checkpoint at restart {restart_cycle}\n")
                                    save_path = os.path.join(OUTPUT_DIR, f"restart_{restart_cycle}")
                                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizers, align_loss_fn, position_learner_optimizer)

                                restart_cycle += 1

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
                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizers, align_loss_fn, position_learner_optimizer)

                set_training_mode(student_model, projection_layers, TRAIN_MODEL, TRAIN_PROJECTION)

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
                        student_model, eval_dataloader, projection_layers,
                        align_loss_fn, token_loss_fn, sequence_loss_fn,
                        device, autocast_dtype,
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

                    # Only display enabled losses
                    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
                        print(f"  Text Alignment Huber Loss: {eval_metrics['align_huber']:.6f}")
                        print(f"  Text Alignment Cosine Loss: {eval_metrics['align_cos']:.6f}")

                    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
                        print(f"  Position Mapped Huber Loss: {eval_metrics['position_huber']:.6f}")
                        print(f"  Position Mapped Cosine Loss: {eval_metrics['position_cos']:.6f}")

                    if TOKEN_HUBER_WEIGHT > 0 or TOKEN_COSINE_WEIGHT > 0:
                        print(f"  Token Huber Loss: {eval_metrics['token_huber']:.6f}")
                        print(f"  Token Cosine Loss: {eval_metrics['token_cos']:.6f}")

                    if SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0:
                        print(f"  Sequence Huber Loss: {eval_metrics['sequence_huber']:.6f}")
                        print(f"  Sequence Cosine Loss: {eval_metrics['sequence_cos']:.6f}")

                    if SAVE_BEST_MODEL and avg_eval_loss < best_loss:
                        best_loss = avg_eval_loss
                        print(f"\nNew best model at loss {best_loss:.6f}, saving...")
                        best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
                        save_trained_model(best_model_dir, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizers, align_loss_fn, position_learner_optimizer)

                    eval_end_time = time.time()
                    eval_delta_time += (eval_end_time - eval_start_time)
                    restore_training_mode(student_model, projection_layers, TRAIN_MODEL, TRAIN_PROJECTION)

        except Exception as e:
            logging.exception("Exception during training.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Closing dataloaders and exiting")
            sys.exit(1)

    # ========== Save Final Model ==========
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    save_trained_model(OUTPUT_DIR, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizers, align_loss_fn, position_learner_optimizer)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("✅ Training and saving completed successfully!")

if __name__ == "__main__":
    main()
