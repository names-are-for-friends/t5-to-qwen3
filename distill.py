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
QWEN3_MODEL_NAME = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saikou/QT-embedder-v65/restart_1/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saikou/QT-embedder-v66/"

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
MAX_LEARNING_RATE_PROJ = 1e-4
MIN_LEARNING_RATE_PROJ = 1e-5

# Saving
SAVE_EVERY_X_STEPS = 0
SAVE_EVERY_X_RESTARTS = 1
SAVE_EVERY_X_EPOCHS = 1

# Printing
PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_X_EPOCHS = 1
SAVE_BEST_MODEL = True

# Scheduler
WARMUP_STEPS = 150
RESTART_CYCLE_STEPS = 350
REPEAT_WARMUP_AFTER_RESTART = False
'''
--Alignment weights & settings--
This is the main loss type, and the one you should be using normally
We match word-to-word, blending loss for each student token based on its normalised position relative to the teacher tokens in the matching word
Then we match per token by exact text match using an approx position and a window
These matches are more accurate, so they override existing matches where present
Recommend to start with cosine thresholds TOKEN=0.0, WORD=0.7, then increase TOKEN towards 0.7 as alignment progresses
'''
TEXT_MATCH_HUBER_WEIGHT = 0.70
TEXT_MATCH_COSINE_WEIGHT = 0.30
ALIGN_WINDOW = 2 # We look this many tokens around the approximate matching position when text matching
TOKEN_COSINE_THRESHOLD = 0.0  # Minimum cosine similarity for token matches
WORD_COSINE_THRESHOLD = 0.7

# Basic weights
TOKEN_HUBER_WEIGHT = 0.00
TOKEN_COSINE_WEIGHT = 0.00
SEQUENCE_HUBER_WEIGHT = 0.00
SEQUENCE_COSINE_WEIGHT = 0.00

# Dataset
SHUFFLE_DATASET = True

# Optimizer state preservation
REUSE_OPTIMIZER_STATE = True
SAVE_OPTIMIZER_STATES = True

# Debugging
LOG_VRAM_USAGE = True

# Dropout - this could potentially be used to train the sequential interpolation layer to over-project and infer embedding space from context, but is untested
ENABLE_STUDENT_WORD_DROPOUT = False
STUDENT_WORD_DROPOUT_RATIO = 0.10
ENABLE_STUDENT_TOKEN_DROPOUT = False
STUDENT_TOKEN_DROPOUT_RATIO = 0.10
SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True

# Enhanced dataset - experimental option that is likely not useful
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

def get_word_token_mappings(tokens, tokenizer, original_text):
    """Map tokens to words with their positions in the token sequence"""
    words = []
    current_word_tokens = []
    current_word_text = ""
    char_position = 0

    for token_idx, token in enumerate(tokens):
        # Skip special tokens
        if token in [tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token]:
            continue

        # Get the actual text for this token
        if hasattr(tokenizer, 'convert_tokens_to_string'):
            token_text = tokenizer.convert_tokens_to_string([token])
        else:
            token_text = token

        # Remove special prefixes
        if token_text.startswith('Ġ') or token_text.startswith('▁'):
            token_text = token_text[1]

        # If this token starts with a space (or is the first token), it's a new word
        if token.startswith('Ġ') or token.startswith('▁') or token_idx == 0:
            # Save the previous word if it exists
            if current_word_tokens:
                words.append({
                    'text': current_word_text,
                    'tokens': current_word_tokens,
                    'char_start': char_position - len(current_word_text),
                    'char_end': char_position,
                    'first_token_idx': current_word_tokens[0],
                    'last_token_idx': current_word_tokens[-1]
                })

            # Start new word
            current_word_tokens = [token_idx]
            current_word_text = token_text
        else:
            # Continue current word
            current_word_tokens.append(token_idx)
            current_word_text += token_text

        char_position += len(token_text)

    # Don't forget the last word
    if current_word_tokens:
        words.append({
            'text': current_word_text,
            'tokens': current_word_tokens,
            'char_start': char_position - len(current_word_text),
            'char_end': char_position,
            'first_token_idx': current_word_tokens[0],
            'last_token_idx': current_word_tokens[-1]
        })

    return words

def calculate_token_boundaries_in_word(word_info):
    """Calculate the normalized boundaries of each token within a word"""
    num_tokens = len(word_info['tokens'])
    if num_tokens == 1:
        return [(0.0, 1.0)]  # Single token gets the whole word

    boundaries = []
    # Distribute the word evenly among tokens
    token_width = 1.0 / num_tokens

    for i in range(num_tokens):
        start = i * token_width
        end = (i + 1) * token_width
        boundaries.append((start, end))

    return boundaries

def get_token_loss_distribution(predicted_norm_pos, token_boundaries):
    """Calculate loss distribution for tokens based on predicted normalized position"""
    weights = []

    for start, end in token_boundaries:
        # Calculate distance from predicted position to token center
        token_center = (start + end) / 2
        distance = abs(predicted_norm_pos - token_center)

        # Calculate weight using triangular distribution
        # Max weight (1.0) at center, goes to 0 at edges
        max_distance = (end - start) / 2
        if distance <= max_distance:
            weight = 1.0 - (distance / max_distance)
        else:
            weight = 0.0

        weights.append(weight)

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        # If no weights, distribute evenly
        weights = [1.0 / len(weights)] * len(weights)

    return weights

def word_based_position_matching(student_input_ids, teacher_input_ids,
                               student_tokenizer, teacher_tokenizer,
                               window_size=5, student_embeddings=None,
                               teacher_embeddings=None,
                               word_cosine_threshold=0.4):
    """
    Word-based position matching with normalized positions.
    Processes single sequences (not batches).
    Returns: (all_pairs, weighted_pairs)
    """
    # Ensure inputs are 1D (single sequence)
    if student_input_ids.dim() > 1:
        student_input_ids = student_input_ids.squeeze(0)
    if teacher_input_ids.dim() > 1:
        teacher_input_ids = teacher_input_ids.squeeze(0)

    # Get the original texts
    student_text = student_tokenizer.decode(student_input_ids, skip_special_tokens=True)
    teacher_text = teacher_tokenizer.decode(teacher_input_ids, skip_special_tokens=True)

    # Get tokens
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    # Map tokens to words
    student_words = get_word_token_mappings(student_tokens, student_tokenizer, student_text)
    teacher_words = get_word_token_mappings(teacher_tokens, teacher_tokenizer, teacher_text)

    # Check if texts match exactly
    texts_match = student_text == teacher_text

    # Store all pairs and weighted pairs
    all_pairs = []
    weighted_pairs = []  # (s_pos, t_pos, weight)

    if texts_match:
        # Exact text match - use normalized positions
        for word_idx in range(min(len(student_words), len(teacher_words))):
            student_word = student_words[word_idx]
            teacher_word = teacher_words[word_idx]

            # Get token boundaries within each word
            student_boundaries = calculate_token_boundaries_in_word(student_word)
            teacher_boundaries = calculate_token_boundaries_in_word(teacher_word)

            # For each student token, find its normalized position
            for s_idx, s_token_idx in enumerate(student_word['tokens']):
                # Calculate normalized position within the word
                s_start, s_end = student_boundaries[s_idx]
                s_norm_pos = (s_start + s_end) / 2

                # Apply this position to teacher word
                t_weights = get_token_loss_distribution(s_norm_pos, teacher_boundaries)

                # Create weighted pairs for all teacher tokens
                for t_idx, t_token_idx in enumerate(teacher_word['tokens']):
                    weight = t_weights[t_idx]

                    # Apply cosine threshold if embeddings available
                    if student_embeddings is not None and teacher_embeddings is not None:
                        student_emb = student_embeddings[s_token_idx]
                        teacher_emb = teacher_embeddings[t_token_idx]
                        cosine_sim = F.cosine_similarity(student_emb.unsqueeze(0), teacher_emb.unsqueeze(0), dim=-1)

                        if cosine_sim.item() >= word_cosine_threshold:
                            all_pairs.append((s_token_idx, t_token_idx))
                            weighted_pairs.append((s_token_idx, t_token_idx, weight))
                    else:
                        all_pairs.append((s_token_idx, t_token_idx))
                        weighted_pairs.append((s_token_idx, t_token_idx, weight))
    else:
        # Texts don't match - fall back to approximate word matching
        # Create word-to-position mappings
        student_word_positions = {}
        for idx, word in enumerate(student_words):
            student_word_positions[normalize_token(word['text'])] = idx

        teacher_word_positions = {}
        for idx, word in enumerate(teacher_words):
            teacher_word_positions[normalize_token(word['text'])] = idx

        # Match words using approximate positions
        for s_word_text, s_word_idx in student_word_positions.items():
            if s_word_text in teacher_word_positions:
                t_word_idx = teacher_word_positions[s_word_text]
                student_word = student_words[s_word_idx]
                teacher_word = teacher_words[t_word_idx]

                # Create all-to-all token pairs for matching words
                for s_token_idx in student_word['tokens']:
                    for t_token_idx in teacher_word['tokens']:
                        # Apply cosine threshold if embeddings available
                        if student_embeddings is not None and teacher_embeddings is not None:
                            student_emb = student_embeddings[s_token_idx]
                            teacher_emb = teacher_embeddings[t_token_idx]
                            cosine_sim = F.cosine_similarity(student_emb.unsqueeze(0), teacher_emb.unsqueeze(0), dim=-1)

                            if cosine_sim.item() >= word_cosine_threshold:
                                all_pairs.append((s_token_idx, t_token_idx))
                                weighted_pairs.append((s_token_idx, t_token_idx, 1.0))
                        else:
                            all_pairs.append((s_token_idx, t_token_idx))
                            weighted_pairs.append((s_token_idx, t_token_idx, 1.0))

    return all_pairs, weighted_pairs

def token_based_alignment(student_input_ids, teacher_input_ids,
                         student_tokenizer, teacher_tokenizer,
                         window_size=5, existing_pairs=None,
                         student_embeddings=None, teacher_embeddings=None,
                         token_cosine_threshold=0.3):
    """
    Exact token-based alignment that can override existing pairs.
    Processes single sequences (not batches).
    Returns: (exact_pairs, overridden_pairs)
    """
    # Ensure inputs are 1D (single sequence)
    if student_input_ids.dim() > 1:
        student_input_ids = student_input_ids.squeeze(0)
    if teacher_input_ids.dim() > 1:
        teacher_input_ids = teacher_input_ids.squeeze(0)

    # Get tokens
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    # Create sets of already aligned positions
    if existing_pairs:
        aligned_student_positions = {pair[0] for pair in existing_pairs}
        aligned_teacher_positions = {pair[1] for pair in existing_pairs}
    else:
        aligned_student_positions = set()
        aligned_teacher_positions = set()

    # Find exact token matches
    exact_pairs = []
    overridden_pairs = []  # Track what got overridden

    # Create mapping from normalized teacher tokens to positions
    normalized_teacher = [normalize_token(token) for token in teacher_tokens]
    token_to_teacher_positions = {}
    for j, norm_t in enumerate(normalized_teacher):
        if j not in aligned_teacher_positions:
            token_to_teacher_positions.setdefault(norm_t, []).append(j)

    # Iterate through student tokens
    for i, token in enumerate(student_tokens):
        # Skip if already aligned
        if i in aligned_student_positions:
            continue

        # Skip special tokens
        if token in [student_tokenizer.pad_token, student_tokenizer.bos_token, student_tokenizer.eos_token]:
            continue

        norm_s = normalize_token(token)

        # Get approximate position
        t_pos_approx = int(i * len(teacher_tokens) / len(student_tokens))

        # Define search window
        start_j = max(0, t_pos_approx - window_size)
        end_j = min(len(teacher_tokens), t_pos_approx + window_size + 1)

        # Get candidate positions
        candidate_positions = token_to_teacher_positions.get(norm_s, [])
        matches = [j for j in candidate_positions if start_j <= j < end_j]

        if matches and student_embeddings is not None and teacher_embeddings is not None:
            # Filter by cosine similarity
            student_emb = student_embeddings[i]
            valid_matches = []

            for j in matches:
                teacher_emb = teacher_embeddings[j]
                cosine_sim = F.cosine_similarity(student_emb.unsqueeze(0), teacher_emb.unsqueeze(0), dim=-1)

                if cosine_sim.item() >= token_cosine_threshold:
                    valid_matches.append((j, cosine_sim.item()))

            if valid_matches:
                # Use the match with highest cosine similarity
                closest_match = max(valid_matches, key=lambda x: x[1])[0]

                # Check if this overrides any word-based alignment
                if existing_pairs:
                    # Remove any existing pairs with this student token
                    removed_pairs = [pair for pair in existing_pairs if pair[0] == i]
                    if removed_pairs:
                        overridden_pairs.extend(removed_pairs)

                exact_pairs.append((i, closest_match))
        elif matches:
            # If embeddings not available, use the closest match
            closest_match = min(matches, key=lambda x: abs(x - t_pos_approx))

            # Check if this overrides any word-based alignment
            if existing_pairs:
                # Remove any existing pairs with this student token
                removed_pairs = [pair for pair in existing_pairs if pair[0] == i]
                if removed_pairs:
                    overridden_pairs.extend(removed_pairs)

            exact_pairs.append((i, closest_match))

    return exact_pairs, overridden_pairs

def hybrid_alignment_with_weights(student_input_ids, teacher_input_ids,
                                 student_tokenizer, teacher_tokenizer,
                                 window_size=5, student_embeddings=None,
                                 teacher_embeddings=None,
                                 token_cosine_threshold=0.3,
                                 word_cosine_threshold=0.4):
    """
    Hybrid alignment combining word-based position matching and exact token matching.
    Processes single sequences (not batches).
    Returns: (final_pairs, final_weighted_pairs, stats)
    """
    # Ensure inputs are 1D (single sequence)
    if student_input_ids.dim() > 1:
        student_input_ids = student_input_ids.squeeze(0)
    if teacher_input_ids.dim() > 1:
        teacher_input_ids = teacher_input_ids.squeeze(0)

    # Step 1: Word-based position matching
    word_pairs, word_weighted_pairs = word_based_position_matching(
        student_input_ids, teacher_input_ids,
        student_tokenizer, teacher_tokenizer,
        window_size,
        student_embeddings=student_embeddings,
        teacher_embeddings=teacher_embeddings,
        word_cosine_threshold=word_cosine_threshold
    )

    # Step 2: Exact token matching (overrides word matches)
    exact_pairs, overridden_pairs = token_based_alignment(
        student_input_ids, teacher_input_ids,
        student_tokenizer, teacher_tokenizer,
        window_size,
        existing_pairs=word_pairs,
        student_embeddings=student_embeddings,
        teacher_embeddings=teacher_embeddings,
        token_cosine_threshold=token_cosine_threshold
    )

    # Step 3: Create final pairs
    # Remove overridden pairs from word pairs
    remaining_word_pairs = []
    remaining_word_weighted = []

    if overridden_pairs:
        overridden_set = set(overridden_pairs)
        remaining_word_pairs = [pair for pair in word_pairs if pair not in overridden_set]
        # Also remove from weighted pairs
        remaining_word_weighted = [
            (s, t, w) for s, t, w in word_weighted_pairs
            if (s, t) not in overridden_set
        ]
    else:
        remaining_word_pairs = word_pairs
        remaining_word_weighted = word_weighted_pairs

    # Combine exact matches with remaining word matches
    final_pairs = exact_pairs + remaining_word_pairs

    # For weighted pairs, exact matches get weight 1.0
    final_weighted_pairs = [(s, t, 1.0) for s, t in exact_pairs] + remaining_word_weighted

    # Statistics for logging
    stats = {
        'word_pairs': len(word_pairs),
        'exact_pairs': len(exact_pairs),
        'overridden_pairs': len(overridden_pairs),
        'final_pairs': len(final_pairs)
    }

    return final_pairs, final_weighted_pairs, stats

# ========== Loss Functions ==========
class AlignmentLoss(torch.nn.Module):
    def __init__(self,
                 student_tokenizer=None,
                 teacher_tokenizer=None,
                 window_size: int = 3,
                 # Text alignment loss weights
                 text_huber_weight: float = 0.7,
                 text_cosine_weight: float = 0.3,
                 # Coverage parameters
                 additional_coverage: float = 0.25,
                 # Cosine thresholds
                 token_cosine_threshold: float = 0.3,
                 word_cosine_threshold: float = 0.4):
        super().__init__()

        # Text matching parameters
        self.window_size = window_size
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Loss weights for text alignment
        self.text_huber_weight = text_huber_weight
        self.text_cosine_weight = text_cosine_weight

        # Cosine thresholds
        self.token_cosine_threshold = token_cosine_threshold
        self.word_cosine_threshold = word_cosine_threshold

        # Loss functions
        self.huber_loss = torch.nn.HuberLoss(delta=1.0, reduction='mean')
        self.cosine_loss = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                student_mask: torch.Tensor, teacher_mask: torch.Tensor,
                student_input_ids: Optional[torch.Tensor] = None,
                teacher_input_ids: Optional[torch.Tensor] = None):
        device = student_output.device
        batch_size = student_output.size(0)

        # Initialize losses
        total_text_huber_loss = torch.tensor(0.0, device=device)
        total_text_cosine_loss = torch.tensor(0.0, device=device)

        # Initialize counters for logging - PER BATCH ITEM
        total_text_aligned_tokens = 0
        total_student_tokens = 0
        batch_aligned_positions = []  # List of sets, one per batch item

        for i in range(batch_size):
            # Get actual tokens (not padded)
            t_indices = (teacher_mask[i] == 1).nonzero(as_tuple=True)[0]
            s_indices = (student_mask[i] == 1).nonzero(as_tuple=True)[0]

            if len(t_indices) == 0 or len(s_indices) == 0:
                batch_aligned_positions.append(set())
                continue

            # Count total student tokens (excluding special tokens) FOR THIS BATCH ITEM
            valid_student_tokens = 0
            for idx in s_indices:
                if student_input_ids[i][idx].item() not in [
                    self.student_tokenizer.pad_token_id,
                    self.student_tokenizer.bos_token_id,
                    self.student_tokenizer.eos_token_id
                ]:
                    valid_student_tokens += 1
            total_student_tokens += valid_student_tokens

            # Step 1: Use hybrid alignment combining word-based and token-based matching
            final_pairs, final_weighted_pairs, stats = hybrid_alignment_with_weights(
                student_input_ids[i], teacher_input_ids[i],
                self.student_tokenizer, self.teacher_tokenizer,
                window_size=self.window_size,
                student_embeddings=student_output[i],
                teacher_embeddings=teacher_output[i],
                token_cosine_threshold=self.token_cosine_threshold,
                word_cosine_threshold=self.word_cosine_threshold
            )

            # Track aligned positions FOR THIS BATCH ITEM ONLY
            text_aligned_student_positions = {pair[0] for pair in final_pairs}
            batch_aligned_positions.append(text_aligned_student_positions)

            # Count unique text-aligned tokens (excluding special tokens) FOR THIS BATCH ITEM
            text_aligned_tokens_count = sum(
                1 for idx in text_aligned_student_positions
                if student_input_ids[i][idx].item() not in [
                    self.student_tokenizer.pad_token_id,
                    self.student_tokenizer.bos_token_id,
                    self.student_tokenizer.eos_token_id
                ]
            )

            # Step 2: Compute losses using the hybrid alignment results
            if final_pairs:
                # Get student and teacher embeddings for aligned positions
                student_embs = student_output[i, [pair[0] for pair in final_pairs]]
                teacher_embs = teacher_output[i, [pair[1] for pair in final_pairs]]

                # Create a mapping from (s_pos, t_pos) to weight
                weight_map = {(s_pos, t_pos): weight for s_pos, t_pos, weight in final_weighted_pairs}
                weights = torch.tensor([weight_map.get((s_pos, t_pos), 1.0)
                                    for s_pos, t_pos in final_pairs],
                                    device=device, dtype=student_embs.dtype)

                # Apply weights to losses
                huber_loss = F.huber_loss(student_embs, teacher_embs, reduction='none')
                huber_loss = (huber_loss * weights.unsqueeze(-1)).mean()

                cos_sim = F.cosine_similarity(student_embs, teacher_embs, dim=-1)
                cosine_loss = ((1 - cos_sim) * weights).mean()

                # Update text-aligned token count
                total_text_aligned_tokens += text_aligned_tokens_count

                # Add to batch losses
                total_text_huber_loss += huber_loss
                total_text_cosine_loss += cosine_loss

        # Average across batch
        if batch_size > 0:
            total_text_huber_loss = total_text_huber_loss / batch_size
            total_text_cosine_loss = total_text_cosine_loss / batch_size

        # Combine all losses
        total_loss = (
            self.text_huber_weight * total_text_huber_loss +
            self.text_cosine_weight * total_text_cosine_loss
        )

        # Calculate coverage PER BATCH ITEM then average
        item_coverages = []
        for i in range(batch_size):
            # Count valid tokens for this batch item
            valid_tokens = 0
            aligned_count = len(batch_aligned_positions[i])

            if student_input_ids is not None and student_mask is not None:
                for idx in range(len(student_input_ids[i])):
                    if (student_input_ids[i][idx].item() not in [
                        self.student_tokenizer.pad_token_id,
                        self.student_tokenizer.bos_token_id,
                        self.student_tokenizer.eos_token_id
                    ] and student_mask[i][idx] == 1):
                        valid_tokens += 1

            coverage = aligned_count / max(valid_tokens, 1)
            item_coverages.append(coverage)

        text_aligned_ratio = sum(item_coverages) / batch_size if batch_size > 0 else 0.0

        return (
            total_loss,
            total_text_huber_loss, total_text_cosine_loss,
            text_aligned_ratio
        )

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
        device = student_output.device
        batch_size = student_output.size(0)

        total_loss = torch.tensor(0.0, device=device)
        total_huber_loss = torch.tensor(0.0, device=device)
        total_cos_loss = torch.tensor(0.0, device=device)
        total_tokens = 0

        for i in range(batch_size):
            position_mask = teacher_mask[i].bool()

            if not position_mask.any():
                continue

            # Get masked embeddings for this sequence
            student_seq = student_output[i][position_mask]
            teacher_seq = teacher_output[i][position_mask]

            if len(student_seq) == 0:
                continue

            # Compute losses
            huber_loss = self.huber_loss(student_seq, teacher_seq)
            cos_sim = self.cos_loss(student_seq, teacher_seq)
            cos_loss = (1 - cos_sim).mean()

            # Combine losses
            loss = self.huber_weight * huber_loss + self.cosine_weight * cos_loss

            # Accumulate
            num_tokens = len(student_seq)
            total_loss += loss * num_tokens
            total_huber_loss += huber_loss * num_tokens
            total_cos_loss += cos_loss * num_tokens
            total_tokens += num_tokens

        # Average
        if total_tokens > 0:
            total_loss = total_loss / total_tokens
            total_huber_loss = total_huber_loss / total_tokens
            total_cos_loss = total_cos_loss / total_tokens

        return total_loss, total_huber_loss, total_cos_loss, total_tokens

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
        batch_size = student_output.size(0)

        total_loss = torch.tensor(0.0, device=device)
        total_huber_loss = torch.tensor(0.0, device=device)
        total_cos_loss = torch.tensor(0.0, device=device)
        total_positions = 0

        for i in range(batch_size):
            # Get mask for this sequence
            position_mask = teacher_mask[i].bool()

            if not position_mask.any():
                continue

            # Apply position mask to embeddings for this sequence
            masked_student = student_output[i] * position_mask.unsqueeze(-1)
            masked_teacher = teacher_output[i] * position_mask.unsqueeze(-1)

            # Mean pooling across sequence dimension
            # Sum across sequence, divide by number of actual tokens in this sequence
            pool_denominator = position_mask.sum().clamp(min=1)
            student_pooled = masked_student.sum(dim=0) / pool_denominator
            teacher_pooled = masked_teacher.sum(dim=0) / pool_denominator

            # Compute losses
            huber_loss = self.huber_loss(student_pooled.unsqueeze(0), teacher_pooled.unsqueeze(0))
            cos_sim = self.cos_loss(student_pooled.unsqueeze(0), teacher_pooled.unsqueeze(0))
            cos_loss = (1 - cos_sim).mean()

            # Combine losses
            loss = (
                self.huber_weight * huber_loss +
                self.cosine_weight * cos_loss
            )

            # Accumulate losses (weighted by number of positions)
            num_positions = position_mask.sum().item()
            total_loss += loss * num_positions
            total_huber_loss += huber_loss * num_positions
            total_cos_loss += cos_loss * num_positions
            total_positions += num_positions

        # Average across all positions
        if total_positions > 0:
            total_loss = total_loss / total_positions
            total_huber_loss = total_huber_loss / total_positions
            total_cos_loss = total_cos_loss / total_positions

        return total_loss, total_huber_loss, total_cos_loss, total_positions

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

# ========== Optimiser Initialisation ==========
def initialize_optimizer(parameters: List[torch.nn.Parameter], max_lr: float, min_lr: float):
    # Ensure parameters are properly collected before optimizer creation
    parameters = [p for p in parameters if p.requires_grad]

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

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
                eval_token_huber = torch.tensor(0.0, device=device)
                eval_token_cos = torch.tensor(0.0, device=device)
                eval_sequence_huber = torch.tensor(0.0, device=device)
                eval_sequence_cos = torch.tensor(0.0, device=device)

                if align_loss_fn is not None and (TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0):
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        total_loss, eval_align_huber, eval_align_cos, _, = align_loss_fn(
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
                torch.cuda.empty_cache()

            except Exception as e:
                logging.exception(f"Error in evaluation batch {batch_idx}: {e}")
                raise e

    # Compute averages
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches


    if align_loss_fn is not None:
        align_loss_fn.train()
    if token_loss_fn is not None:
        token_loss_fn.train()
    if sequence_loss_fn is not None:
        sequence_loss_fn.train()

    return total_losses

# ========== Optimizer Handling ==========
def save_optimizer_states(save_path: str, model_optimizer, projection_optimizer):
    """Save optimizer states to a subfolder"""
    if not SAVE_OPTIMIZER_STATES:
        return

    optimizer_dir = os.path.join(save_path, "optimizers")
    os.makedirs(optimizer_dir, exist_ok=True)

    # Save model optimizer state
    if model_optimizer is not None:
        torch.save(model_optimizer.state_dict(), os.path.join(optimizer_dir, "model_optimizer.pt"))

    # Save projection optimizer state
    if projection_optimizer is not None:
        torch.save(projection_optimizer.state_dict(), os.path.join(optimizer_dir, "projection_optimizer.pt"))

def load_optimizer_states(save_path: str, model_optimizer, scheduler_model, projection_optimizer, scheduler_projection) -> bool:
    """Load optimizer states from a subfolder if available"""
    if not REUSE_OPTIMIZER_STATE or not projection_optimizer:
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

    # Load projection optimizer state
    opt_path = os.path.join(optimizer_dir, "projection_optimizer.pt")
    if os.path.exists(opt_path):
        try:
            projection_optimizer.load_state_dict(torch.load(opt_path))
            # Reset learning rate to current scheduler value
            if scheduler_projection is not None:
                for param_group, lr in zip(projection_optimizer.param_groups, scheduler_projection.get_last_lr()):
                    param_group['lr'] = lr
            print("Loaded projection optimizer state")
        except Exception as e:
            print(f"Warning: Failed to load projection optimizer state: {e}")
            success = False
    else:
        print("Warning: Projection optimizer state file not found.")
        success = False

    return success

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
                      model_optimizer=None, projection_optimizer=None,
                      align_loss_fn=None) -> None:
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

    # Save optimizer states
    save_optimizer_states(save_path, model_optimizer, projection_optimizer)

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

def get_logging(epoch: int, batch_idx: int, global_step: int, total_steps: int,
               current_loss: float,
               current_align_huber: float, current_align_cos: float,
               current_token_huber: float, current_token_cos: float,
               current_sequence_huber: float, current_sequence_cos: float,
               grad_norm_model: float, grad_norm_proj: float,
               current_lr_model: float, current_lr_proj: float,
               text_aligned_ratio: float,
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

    current_lr_proj = 0.0
    if TRAIN_PROJECTION and projection_optimizer is not None:
        current_lr_proj = projection_optimizer.param_groups[0]['lr']

    if TRAIN_MODEL:
        model_gn_line = f"GN Mod: {grad_norm_model:.6f}, "
        model_lr_line = f"Mod LR: {current_lr_model:.6f}, "
    if TRAIN_PROJECTION:
        proj_gn_line = f"GN Proj: {grad_norm_proj:.6f}, "
        proj_lr_line = f"Avg Proj LR: {current_lr_proj:.6f}, "
    if LOG_VRAM_USAGE:
        vram_free, vram_total, vram_used = get_memory_usage()
        vram_line = f"VRAM: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
    epoch_line = f"Epoch [{epoch + 1}/{EPOCHS}], "
    step_line = f"Step: {global_step}/{total_steps}, "
    batch_line = f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "

    # Build loss lines based on enabled losses
    loss_lines = []
    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
        loss_lines.append(f"Huber: {current_align_huber:.6f}")
        loss_lines.append(f"Cosine: {current_align_cos:.6f}")
        loss_lines.append(f"Coverage: {text_aligned_ratio:.2%}")
    if TOKEN_HUBER_WEIGHT > 0 or TOKEN_COSINE_WEIGHT > 0:
        loss_lines.append(f"Tok Hub: {current_token_huber:.6f}")
        loss_lines.append(f"Tok Cos: {current_token_cos:.6f}")
    if SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0:
        loss_lines.append(f"Seq Hub: {current_sequence_huber:.6f}")
        loss_lines.append(f"Seq Cos: {current_sequence_cos:.6f}")

    loss_str = ", ".join(loss_lines)

    log_line = (
        f"{epoch_line}"
        f"{batch_line}"
        f"{step_line}"
        f"Total Loss: {current_loss:.6f}, "
        f"{loss_str}, "
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
    global total_steps, train_dataloader, train_dataset, eval_dataloader, eval_dataset, projection_optimizer

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
        full_finetuning=True if TRAIN_MODEL else False
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
            text_huber_weight=TEXT_MATCH_HUBER_WEIGHT,
            text_cosine_weight=TEXT_MATCH_COSINE_WEIGHT,
            token_cosine_threshold=TOKEN_COSINE_THRESHOLD,
            word_cosine_threshold=WORD_COSINE_THRESHOLD
        ).to(device, dtype=torch.bfloat16)

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
                persistent_workers=True,
                prefetch_factor=PREFETCH_FACTOR,
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                num_workers=min(4, os.cpu_count()//2) if torch.cuda.is_available() else 0,
                persistent_workers=True,
                prefetch_factor=PREFETCH_FACTOR,
            )

            # Calculate total steps
            total_steps = (len(train_dataloader) // GRAD_ACCUM_STEPS) * EPOCHS

            # ========== Training Setup ==========
            autocast_dtype = torch.bfloat16
            scaler = GradScaler('cuda', enabled=False)

            global_step = 0
            accumulation_step = 0

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
            projection_optimizer = None
            scheduler_projection = None
            projection_parameters = []
            if TRAIN_PROJECTION:
                for projection_layer in projection_layers:
                    for p in projection_layer.parameters():
                        if p.requires_grad:
                            projection_parameters.append(p)
                projection_optimizer, scheduler_projection = initialize_optimizer(projection_parameters, MAX_LEARNING_RATE_PROJ, MIN_LEARNING_RATE_PROJ)

            model_optimizer = None
            scheduler_model = None
            if TRAIN_MODEL:
                model_parameters = [p for p in student_model.parameters() if p.requires_grad]
                model_optimizer, scheduler_model = initialize_optimizer(model_parameters, MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)

            if REUSE_OPTIMIZER_STATE and load_optimizer_states(QWEN3_MODEL_NAME, model_optimizer, scheduler_model, projection_optimizer, scheduler_projection):
                # Update learning rates to match scheduler current state
                if TRAIN_MODEL and scheduler_model:
                    for param_group, lr in zip(model_optimizer.param_groups, scheduler_model.get_last_lr()):
                        param_group['lr'] = lr

                if TRAIN_PROJECTION and scheduler_projection:
                    for param_group, lr in zip(projection_optimizer.param_groups, scheduler_projection.get_last_lr()):
                        param_group['lr'] = lr

            if ENABLE_LOGGING:
                log_dir = os.path.join(OUTPUT_DIR, "logging")
                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                log_filename = f"training_log_{current_time}.txt"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, log_filename)
                log_lines = []

            # ========== Training Loop ==========
            if align_loss_fn is not None:
                align_loss_fn.train()

            start_time = time.time()
            eval_delta_time = 0
            best_loss = float('inf')

            for epoch in range(EPOCHS):
                if model_optimizer: model_optimizer.zero_grad()
                if projection_optimizer: projection_optimizer.zero_grad()
                set_training_mode(student_model, projection_layers, TRAIN_MODEL, TRAIN_PROJECTION)

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
                        token_loss = torch.tensor(0.0, device=device)
                        token_huber = torch.tensor(0.0, device=device)
                        token_cos = torch.tensor(0.0, device=device)
                        sequence_loss = torch.tensor(0.0, device=device)
                        sequence_huber = torch.tensor(0.0, device=device)
                        sequence_cos = torch.tensor(0.0, device=device)
                        text_aligned_ratio = 0.0

                        if align_loss_fn is not None:
                            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                                total_align_loss, align_loss_huber, align_loss_cos, text_aligned_ratio = align_loss_fn(
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
                            if TRAIN_PROJECTION and projection_optimizer:
                                scaler.unscale_(projection_optimizer)

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

                            if TRAIN_PROJECTION and projection_optimizer:
                                grad_norm_proj = clip_grad_norm_(
                                    all_proj_params if all_proj_params else [],
                                    max_norm=GRAD_CLIP
                                )
                            else:
                                grad_norm_proj = 0.0

                            # Step optimizers after clipping
                            if TRAIN_MODEL and model_optimizer:
                                scaler.step(model_optimizer)
                            if projection_optimizer:
                                scaler.step(projection_optimizer)

                            # Update scaler
                            scaler.update()

                            # Clear gradients after stepping
                            accumulation_step = 0
                            if TRAIN_MODEL and model_optimizer:
                                model_optimizer.zero_grad()
                            if projection_optimizer:
                                projection_optimizer.zero_grad()

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

                            if TRAIN_PROJECTION and projection_optimizer:
                                # Get average LR across projection optimizers
                                current_lr_proj = projection_optimizer.param_groups[0]['lr']
                            else:
                                current_lr_proj = 0.0

                            if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                                print(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss,
                                    current_align_huber, current_align_cos,
                                    current_token_huber, current_token_cos,
                                    current_sequence_huber, current_sequence_cos,
                                    grad_norm_model, grad_norm_proj,
                                    current_lr_model, current_lr_proj,
                                    text_aligned_ratio,
                                    elapsed, eta
                                ))

                            if ENABLE_LOGGING:
                                log_lines.append(get_logging(
                                    epoch, batch_idx, global_step, total_steps,
                                    current_loss,
                                    current_align_huber, current_align_cos,
                                    current_token_huber, current_token_cos,
                                    current_sequence_huber, current_sequence_cos,
                                    grad_norm_model, grad_norm_proj,
                                    current_lr_model, current_lr_proj,
                                    text_aligned_ratio,
                                    elapsed, eta
                                ))
                                if global_step % WRITE_TO_LOG_EVERY_X_STEPS == 0:
                                    with open(log_file, "a") as f:
                                        for line in log_lines:
                                            f.write(line + "\n")
                                    log_lines.clear()

                            # Update schedulers
                            if TRAIN_MODEL and scheduler_model:
                                model_optimizer.step()
                                scheduler_model.step()
                            if scheduler_projection:
                                projection_optimizer.step()
                                scheduler_projection.step()

                            if SAVE_EVERY_X_STEPS > 0 and global_step % SAVE_EVERY_X_STEPS == 0:
                                print(f"\nSaving checkpoint at step {global_step}\n")
                                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizer, align_loss_fn)

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
                                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizer, align_loss_fn)

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
                        if projection_optimizer:
                            projection_optimizer.zero_grad()
                        continue

                    finally:
                        # Cleanup batch variables
                        del s_input_ids, s_mask, t_mask
                        if USE_CACHED_EMBEDDINGS:
                            del t_embeddings
                        else:
                            if 't_input_ids' in locals():
                                del t_input_ids
                        torch.cuda.empty_cache()

                print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")
                next_epoch = epoch + 1
                if next_epoch % SAVE_EVERY_X_EPOCHS == 0:
                    print(f"\nSaving checkpoint at epoch {next_epoch}\n")
                    save_path = os.path.join(OUTPUT_DIR, f"epoch_{next_epoch}")
                    save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizer, align_loss_fn)

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
                        save_trained_model(best_model_dir, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizer, align_loss_fn)

                    eval_end_time = time.time()
                    eval_delta_time += (eval_end_time - eval_start_time)

        except Exception as e:
            logging.exception("Exception during training.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Closing dataloaders and exiting")
            sys.exit(1)

    # ========== Save Final Model ==========
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    save_trained_model(OUTPUT_DIR, student_model, student_tokenizer, projection_layers, qwen_embedding_dim, model_optimizer, projection_optimizer, align_loss_fn)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("✅ Training and saving completed successfully!")

if __name__ == "__main__":
    main()
