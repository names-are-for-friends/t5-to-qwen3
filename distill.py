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
import math
from difflib import SequenceMatcher
from typing import List, Tuple, Any, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.DEBUG)

# ========== Configuration ==========
# Paths
DATASET_DIR = "/mnt/f/q5_xxs_training_script/400K_dataset.txt"
T5_MODEL_DIR = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_DIR = "/mnt/f/q5_xxs_training_script/QT-encoder-final/stage3-13/restart_1"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-encoder-final/stage3-14"

# Caching
USE_CACHED_EMBEDDINGS = True
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2"
PREFETCH_FACTOR = 3

# Evaluation
USE_SEPARATE_EVALUATION_DATASET = True
EVALUATION_DATASET_DIR = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

# Logging
ENABLE_LOGGING = True
WRITE_TO_LOG_EVERY_X_STEPS = 10

# Debug output
ENABLE_ALIGNMENT_DEBUG = False

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
RESTART_CYCLE_STEPS = 350 # 0 = flat LR with no restart
REPEAT_WARMUP_AFTER_RESTART = False
'''
--Loss weights--
This is the main loss type, and the one you should be using normally
We index the words, and then match individual tokens by text via normalised position in matched words - this is TEXT_MATCH loss
Then, we attempt to match single or multiple student tokens to single or multiple teacher tokens by text - this is SPLIT_TOKEN loss
Finally, we can use mean pooling loss for extra reinforcement - this is SEQUENCE loss
'''
TEXT_MATCH_HUBER_WEIGHT = 1.00
TEXT_MATCH_COSINE_WEIGHT = 0.10
SPLIT_TOKEN_HUBER_WEIGHT = 1.00
SPLIT_TOKEN_COSINE_WEIGHT = 0.10
SEQUENCE_HUBER_WEIGHT = 0.10
SEQUENCE_COSINE_WEIGHT = 0.01

# Padding options - You can match tokens after the attended sequence. T5 has one EOS then repeated pad, Qwen has just one repeated post-sequence joint pad/EOS token
MIN_EXTRA_MATCHED_PADDING = 1
MAX_EXTRA_MATCHED_PADDING = 1
PADDING_LOSS_WEIGHT = 0.2

# Cosine thresholds - turn this to zero at first, since semantic representations will be unrelated until trained
TEXT_MATCH_COSINE_THRESHOLD = 0.0
SPLIT_TOKEN_COSINE_THRESHOLD = 0.0

# Dataset
SHUFFLE_DATASET = True

# Optimizer state preservation
REUSE_OPTIMIZER_STATE = True
SAVE_OPTIMIZER_STATES = True

# Debugging
LOG_VRAM_USAGE = True

# T5 components - we can extract weights from the T5-xxl model. That said, saved weights in Qwen dir are always prioritised
TAKE_T5_ENCODER = False # Pulling the T5 encoder weights doesn't work well, so leave this false. The RMSNorm works good though!
TAKE_T5_RMSNORM = True # This is the final output RMSNorm from T5-xxl

# Training flags
TRAIN_PROJECTION = True
TRAIN_MODEL = True

# Layer arrangement - We use T5-like blocks to both project the dim and refine the output towards the target. Final output should be 4096
# If you use T5RMSNorm, we'll extract the weights from T5 directly. Recommended for final output to match T5
EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS = []
PROJECTION_LAYERS_CONFIG = [
    {
        "type": "linear",
        "output_dim": 2048,
        "file_num": 1
    },
    {
        "type": "transformer_encoder",
        "num_heads": 64,
        "dropout_rate": 0.1,
        "dim_feedforward": 8192,
        "file_num": 2
    },
    {
        "type": "transformer_encoder",
        "num_heads": 64,
        "dropout_rate": 0.1,
        "dim_feedforward": 8192,
        "file_num": 3
    },
    {
        "type": "transformer_encoder",
        "num_heads": 64,
        "dropout_rate": 0.1,
        "dim_feedforward": 8192,
        "file_num": 4
    },
    {
        "type": "linear",
        "output_dim": 4096,
        "file_num": 5
    },
    {
        "type": "t5_rmsnorm",
        "file_num": 6
    },
]

# ========== Experimental/Unused Configuration ==========
# Dropout - this could potentially be used to train sequential interpolation to over-project and infer embedding space from context, but is untested and not really useful at the moment
ENABLE_STUDENT_WORD_DROPOUT = False
STUDENT_WORD_DROPOUT_RATIO = 0.10
ENABLE_STUDENT_TOKEN_DROPOUT = False
STUDENT_TOKEN_DROPOUT_RATIO = 0.10
SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True

# Enhanced dataset - experimental option that is not really useful at the moment
ENHANCED_DATASET = True
ENHANCED_DATASET_DIR = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"
UNTAMPERED_STUDENT_AND_TEACHER_RATIO = 0.50
ENHANCED_TEACHER_EMBEDDING_RATIO = 0.00
ENHANCED_STUDENT_AND_TEACHER_RATIO = 0.50

# ========== Projection Layers ==========
class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.layer_type = "linear"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear(x)

class TransformerLayer(torch.nn.Module):
    def __init__(self, dim_feedforward: int, num_heads: int = 8, hidden_size: int = 4096, dropout_rate: int = 0.1):
        super().__init__()
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.layer_type = "transformer_encoder"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.transformer(x)
        return x

# ========== T5 Layers ==========
class T5RMSNorm(torch.nn.Module):
    """T5's RMSNorm (no mean subtraction, just scaling)"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: normalize by root mean square, no mean subtraction
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        # Cast back to original dtype if needed
        if self.weight.dtype == torch.bfloat16:
            x = x.to(torch.bfloat16)

        return self.weight * x

class T5RelativePositionBias(torch.nn.Module):
    """T5's relative position bias for attention"""
    def __init__(self, num_heads: int, relative_attention_num_buckets: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_bias = torch.nn.Parameter(
            torch.zeros(num_heads, relative_attention_num_buckets)
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:

        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # Now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half are for larger distances
        relative_position_if_large = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_position_if_large = torch.clamp(relative_position_if_large.round().long(), min=0, max=num_buckets - 1)

        ret += torch.where(is_small, relative_position, relative_position_if_large)
        return ret

    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets
        )

        device = self.relative_attention_bias.device
        relative_position_bucket = relative_position_bucket.to(device)

        bias_expanded = self.relative_attention_bias.unsqueeze(1).expand(-1, query_length, -1)

        values = bias_expanded.gather(
            2,  # Gather along the bucket dimension (dim=2 in expanded bias)
            relative_position_bucket.unsqueeze(0).expand(self.num_heads, -1, -1)
        )
        return values

class T5Attention(torch.nn.Module):
    """T5's multi-head self-attention with relative position biases"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        relative_attention_num_buckets: int = 32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Q, K, V projections (no bias)
        self.q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.o = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Relative position bias
        self.relative_attention_bias = T5RelativePositionBias(
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative position bias
        relative_bias = self.relative_attention_bias(seq_length, seq_length)
        attn_scores = attn_scores + relative_bias.unsqueeze(0)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and dropout
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        attn_output = self.o(attn_output)

        return attn_output


class T5FeedForwardGated(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # Two parallel dense layers for the gating mechanism (no bias)
        self.wi_0 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)  # Gate layer
        self.wi_1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)  # Input layer

        # Output layer (no bias)
        self.wo = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

        # Activation and dropout (applied after gating)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Project through both parallel layers
        hidden_gates = self.wi_0(hidden_states)
        hidden_linear = self.wi_1(hidden_states)

        # Apply activation only to the gate
        hidden_gates = self.activation(hidden_gates)

        # Element-wise multiplication (gating)
        hidden_states = hidden_gates * hidden_linear

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Project to original dimension
        hidden_states = self.wo(hidden_states)

        return hidden_states

class T5EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 64,
        dropout_rate: float = 0.1,
        relative_attention_num_buckets: int = 32,
        dim_feedforward: int = 10240
    ):
        super().__init__()

        # Self-attention layer
        self.layer_norm_self_attention = T5RMSNorm(hidden_size)
        self.self_attention = T5Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            relative_attention_num_buckets=relative_attention_num_buckets
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Gated Feed-forward layer
        self.layer_norm_feed_forward = T5RMSNorm(hidden_size)
        self.feed_forward = T5FeedForwardGated(
            hidden_size=hidden_size,
            intermediate_size=dim_feedforward,
            dropout_rate=dropout_rate
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block with pre-norm and residual
        normed_hidden_states = self.layer_norm_self_attention(hidden_states)
        attn_output = self.self_attention(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask
        )
        attn_output = self.dropout(attn_output)
        hidden_states = hidden_states + attn_output

        # Feed-forward block with pre-norm and residual
        normed_hidden_states = self.layer_norm_feed_forward(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = hidden_states + ff_output

        return hidden_states

# ========== T5 Helper Functions ==========
def extract_t5_layer(t5_model, layer_type: str, layer_config: dict, hidden_size):
    """Extract a layer from the T5-xxl model or create one with same config"""
    try:
        if layer_type == "t5_encoder":
            # Get T5 config
            config = t5_model.config
            t5_hidden_size = config.d_model
            t5_num_heads = config.num_heads
            t5_dropout_rate = config.dropout_rate
            t5_relative_attention_num_buckets = config.relative_attention_num_buckets

            # Get requested config from PROJECTION_LAYERS_CONFIG
            requested_hidden_size = hidden_size
            requested_num_heads = layer_config["num_heads"]
            requested_dropout_rate = layer_config["dropout_rate"]
            requested_relative_attention_num_buckets = layer_config["relative_attention_num_buckets"]

            # Check if requested dimensions match T5-xxl config
            dimensions_match = (
                (requested_hidden_size == t5_hidden_size) and
                (requested_num_heads == t5_num_heads) and
                (requested_dropout_rate == t5_dropout_rate) and
                (requested_relative_attention_num_buckets == t5_relative_attention_num_buckets)
            )

            if dimensions_match:
                print(f"Loading T5 encoder weights from T5-xxl (hidden_size={requested_hidden_size}, heads={requested_num_heads})")
                # Extract the actual last encoder block
                block = t5_model.encoder.block[-1]

                # Create new block with requested dimensions
                new_block = T5EncoderBlock(
                    hidden_size=requested_hidden_size,
                    num_heads=requested_num_heads,
                    dropout_rate=requested_dropout_rate,
                    relative_attention_num_buckets=requested_relative_attention_num_buckets
                )

                # Load weights from T5 block
                try:
                    # Load self-attention weights
                    self_attention = block.layer[0]  # T5LayerSelfAttention
                    attention = self_attention.SelfAttention  # T5Attention

                    # Load Q, K, V, O projections
                    new_block.self_attention.q.weight.data = attention.q.weight.data.clone()
                    new_block.self_attention.k.weight.data = attention.k.weight.data.clone()
                    new_block.self_attention.v.weight.data = attention.v.weight.data.clone()
                    new_block.self_attention.o.weight.data = attention.o.weight.data.clone()

                    # Load relative position bias from FIRST block (not last)
                    rel_bias_loaded = False
                    try:
                        first_block = t5_model.encoder.block[0]
                        if hasattr(first_block.layer[0].SelfAttention, 'relative_attention_bias'):
                            bias_weights = first_block.layer[0].SelfAttention.relative_attention_bias.weight.data
                            # Transpose to [num_heads, num_buckets] for our implementation and make contiguous
                            new_block.self_attention.relative_attention_bias.relative_attention_bias.data = bias_weights.T.contiguous()
                            rel_bias_loaded = True
                    except Exception as e:
                        print(f"Warning: Could not load relative attention bias from first block: {e}")

                    if not rel_bias_loaded:
                        print("Warning: Could not load relative_attention_bias from T5 first block, initializing randomly")

                    # Load layer norm for self-attention
                    new_block.layer_norm_self_attention.weight.data = \
                        self_attention.layer_norm.weight.data.clone()

                    # Load feed-forward weights - handle gated case (now the primary case)
                    feed_forward = block.layer[1]  # T5LayerFF
                    dense = feed_forward.DenseReluDense

                    # We now expect the gated architecture (T5DenseGatedActDense)
                    if hasattr(dense, 'wi_0') and hasattr(dense, 'wi_1'):
                        # Load weights for the gated feed-forward
                        new_block.feed_forward.wi_0.weight.data = dense.wi_0.weight.data.clone()
                        new_block.feed_forward.wi_1.weight.data = dense.wi_1.weight.data.clone()
                        new_block.feed_forward.wo.weight.data = dense.wo.weight.data.clone()
                    elif hasattr(dense, 'wi'):  # Fallback for non-gated T5DenseActDense
                        new_block.feed_forward.wi_0.weight.data = dense.wi.weight.data.clone()
                        new_block.feed_forward.wi_1.weight.data = dense.wi.weight.data.clone()
                        new_block.feed_forward.wo.weight.data = dense.wo.weight.data.clone()
                        print("Warning: Loaded from T5DenseActDense (non-gated) architecture. Duplicated wi layer for wi_0 and wi_1.")
                    else:
                        # Fallback to state_dict
                        state_dict = dense.state_dict()
                        if 'wi.weight' in state_dict:
                            new_block.feed_forward.wi_0.weight.data = state_dict['wi.weight'].clone()
                        if 'wo.weight' in state_dict:
                            new_block.feed_forward.wo.weight.data = state_dict['wo.weight'].clone()
                        print("Warning: Could not identify feedforward type, loaded from state_dict (may not work for gated arch)")

                    # Load layer norm for feed-forward
                    new_block.layer_norm_feed_forward.weight.data = \
                        feed_forward.layer_norm.weight.data.clone()

                    new_block.extracted_from_t5 = True
                    return new_block

                except Exception as e:
                    print(f"Failed to load weights from T5 block: {e}")
                    # Fall back to new initialization
                    pass

            # Dimensions don't match or weight loading failed - create new block
            print(f"Initializing new T5 encoder block with config (hidden_size={requested_hidden_size}, heads={requested_num_heads})")
            new_block = T5EncoderBlock(
                hidden_size=requested_hidden_size,
                num_heads=requested_num_heads,
                dropout_rate=requested_dropout_rate,
                relative_attention_num_buckets=requested_relative_attention_num_buckets
            )
            new_block.extracted_from_t5 = False
            return new_block

        elif layer_type == "t5_rmsnorm":
            # Get the final layer norm
            if hasattr(t5_model, 'encoder') and hasattr(t5_model.encoder, 'final_layer_norm'):
                norm = t5_model.encoder.final_layer_norm
                t5_hidden_size = norm.weight.size(0)

                # For RMSNorm, we need to match the previous layer's output
                requested_hidden_size = hidden_size

                # Check if dimensions match
                if requested_hidden_size == t5_hidden_size:
                    print(f"Loading T5 RMSNorm weights from T5-xxl (hidden_size={requested_hidden_size})")
                    new_norm = T5RMSNorm(
                        hidden_size=requested_hidden_size,
                        eps=norm.variance_epsilon
                    )
                    new_norm.weight.data = norm.weight.data.clone()
                    new_norm.extracted_from_t5 = True
                    return new_norm
                else:
                    print(f"Initializing new T5 RMSNorm (hidden_size={requested_hidden_size})")
                    new_norm = T5RMSNorm(
                        hidden_size=requested_hidden_size,
                        eps=1e-6
                    )
                    new_norm.extracted_from_t5 = False
                    return new_norm
            else:
                return None

        else:
            return None

    except Exception as e:
        print(f"Warning: Failed to extract T5 layer ({layer_type}): {e}")
        return None

# ========== Token Alignment ==========
def normalize_token(token):
    prefixes_to_remove = ['Ġ', '▁', '▔', '▃', '�']
    for prefix in prefixes_to_remove:
        if token.startswith(prefix):
            token = token[1:]

    # Handle common punctuation spacing but preserve most characters
    replacements = {
        ' .': '.', ' ,': ',', ' !': '!', ' ?': '?',
        ' :': ':', ' ;': ';', ' (': '(', ' )': ')'
    }
    for old, new in replacements.items():
        token = token.replace(old, new)

    return token.lower()

def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=3):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.
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

def ids_to_tokens(token_ids, tokenizer):
    """Optimized token conversion"""
    return tokenizer.convert_ids_to_tokens(token_ids)

def get_word_token_mappings(tokens, tokenizer, original_text):
    """Map tokens to words with robust position tracking"""
    words = []
    current_word_tokens = []
    current_word_text = ""
    word_start_char = -1

    # Track position in reconstructed text
    reconstructed_text = ""
    token_positions = []

    for token_idx, token in enumerate(tokens):
        # Store position before adding this token
        token_start = len(reconstructed_text)

        # Normalize token for text reconstruction
        token_text = normalize_token(token)

        # Add the normalized token to reconstructed text
        reconstructed_text += token_text
        token_end = len(reconstructed_text)

        token_positions.append((token_start, token_end))

        # Check if this token starts a new word
        starts_word = False
        if token_idx == 0:
            starts_word = True
        else:
            # Check if original token has space prefix (before normalization)
            if any(token.startswith(prefix) for prefix in ['Ġ', '▁', 'Ċ', 'ĉ', 'ľ', 'ŀ', '­', '﻿', ' ', '\t', '\n']):
                starts_word = True

        if starts_word:
            # Save previous word if exists
            if current_word_tokens:
                word_info = create_word_info(
                    current_word_tokens, current_word_text,
                    token_positions[len(token_positions) - len(current_word_tokens):len(token_positions)],
                    original_text, word_start_char
                )
                if word_info:  # Only add if word was found in original text
                    words.append(word_info)

            # Start new word
            current_word_tokens = [{
                'idx': token_idx,
                'text': token_text,
                'char_start': token_start,
                'char_end': token_end
            }]
            current_word_text = token_text
            word_start_char = token_start
        else:
            # Continue current word
            current_word_tokens.append({
                'idx': token_idx,
                'text': token_text,
                'char_start': token_start,
                'char_end': token_end
            })
            current_word_text += token_text

    # Add last word
    if current_word_tokens:
        word_info = create_word_info(
            current_word_tokens, current_word_text,
            token_positions[len(token_positions) - len(current_word_tokens):len(token_positions)],
            original_text, word_start_char
        )
        if word_info:
            words.append(word_info)

    # Calculate boundaries for each word
    for word_info in words:
        word_info['token_boundaries'] = calculate_token_boundaries_in_word(
            word_info, word_info['text']
        )

    return words

def create_word_info(tokens, reconstructed_text, token_positions, original_text, word_start_char):
    """Create word info with robust text matching"""
    # Use the stored word start position
    word_start = word_start_char
    word_end = tokens[-1]['char_end']

    # Ensure boundaries are within original text
    word_start = max(0, min(word_start, len(original_text)))
    word_end = max(0, min(word_end, len(original_text)))

    if word_start >= word_end:
        return None

    # Get actual text from original
    actual_text = original_text[word_start:word_end]

    return {
        'text': actual_text,
        'tokens': [t['idx'] for t in tokens],
        'token_strings': [t['text'] for t in tokens],
        'char_start': word_start,
        'char_end': word_end,
        'first_token_idx': tokens[0]['idx'],
        'last_token_idx': tokens[-1]['idx'],
        'original_text': original_text,
        'token_char_positions': [t['char_start'] for t in tokens]
    }

def get_word_exterior_sequences(tokens, words):
    """Identify sequences of space tokens between words"""
    if not words:
        return []

    # Get all tokens that are part of words
    word_tokens = set()
    for word in words:
        word_tokens.update(word['tokens'])

    # Find space tokens (tokens that are empty after normalization)
    space_tokens = []
    for idx, token in enumerate(tokens):
        if idx not in word_tokens:
            norm_token = normalize_token(token)
            # Check if this is effectively a space (empty after normalization)
            if norm_token == '' or norm_token == ' ':
                space_tokens.append(idx)

    # Group contiguous space tokens into sequences
    sequences = []
    if not space_tokens:
        return sequences

    current_seq = [space_tokens[0]]
    for i in range(1, len(space_tokens)):
        if space_tokens[i] == space_tokens[i-1] + 1:
            current_seq.append(space_tokens[i])
        else:
            sequences.append(current_seq)
            current_seq = [space_tokens[i]]
    sequences.append(current_seq)

    return sequences

def match_space_sequences(student_spaces, teacher_spaces,
                         student_words, teacher_words,
                         window_size=2):
    """Match space sequences based on surrounding words"""
    if not student_spaces or not teacher_spaces:
        return []

    matches = []

    # For each student space sequence
    for s_seq in student_spaces:
        # Find surrounding words
        s_prev_word = None
        s_next_word = None

        # Find previous word (word ending before this space)
        for word in reversed(student_words):
            if word['last_token_idx'] < s_seq[0]:
                s_prev_word = word
                break

        # Find next word (word starting after this space)
        for word in student_words:
            if word['first_token_idx'] > s_seq[-1]:
                s_next_word = word
                break

        # Find matching teacher space sequence
        best_t_seq = None
        best_score = -1

        for t_seq in teacher_spaces:
            # Find surrounding words for teacher sequence
            t_prev_word = None
            t_next_word = None

            for word in reversed(teacher_words):
                if word['last_token_idx'] < t_seq[0]:
                    t_prev_word = word
                    break

            for word in teacher_words:
                if word['first_token_idx'] > t_seq[-1]:
                    t_next_word = word
                    break

            # Score based on word matches
            score = 0
            if s_prev_word and t_prev_word:
                if normalize_token(s_prev_word['text']) == normalize_token(t_prev_word['text']):
                    score += 1
            if s_next_word and t_next_word:
                if normalize_token(s_next_word['text']) == normalize_token(t_next_word['text']):
                    score += 1

            # Prefer sequences with similar length
            length_diff = abs(len(s_seq) - len(t_seq))
            score = score - (length_diff * 0.1)

            if score > best_score:
                best_score = score
                best_t_seq = t_seq

        # Match tokens within sequences by position
        if best_t_seq and best_score > 0:
            for i, s_idx in enumerate(s_seq):
                # Calculate normalized position
                s_norm_pos = i / max(len(s_seq) - 1, 1)
                t_pos_approx = int(s_norm_pos * len(best_t_seq))

                # Find closest teacher token
                t_idx = best_t_seq[min(t_pos_approx, len(best_t_seq) - 1)]
                matches.append((s_idx, t_idx))

    return matches

def calculate_token_boundaries_in_word(word_info, word_text):
    """Calculate actual normalized boundaries for tokens in a word"""
    word_start = word_info['char_start']
    word_end = word_info['char_end']
    word_length = word_end - word_start

    if word_length == 0 or not word_info['tokens']:
        return [(0.0, 1.0)] if word_info['tokens'] else []

    boundaries = []
    token_strings = word_info['token_strings']
    token_positions = word_info['token_char_positions']

    for i in range(len(token_strings)):
        token_start = max(word_start, token_positions[i])
        if i < len(token_strings) - 1:
            token_end = min(word_end, token_positions[i + 1])
        else:
            token_end = word_end

        # Normalize to [0, 1] relative to word
        norm_start = (token_start - word_start) / word_length if word_length > 0 else 0.0
        norm_end = (token_end - word_start) / word_length if word_length > 0 else 1.0

        # Ensure boundaries are within [0, 1]
        norm_start = max(0.0, min(1.0, norm_start))
        norm_end = max(0.0, min(1.0, norm_end))

        boundaries.append((norm_start, norm_end))

    return boundaries

def token_based_alignment(student_input_ids, teacher_input_ids,
                         student_tokenizer, teacher_tokenizer,
                         window_size=2, existing_pairs=None,
                         student_embeddings=None, teacher_embeddings=None,
                         token_cosine_threshold=0.7):
    """Process single sequences for exact token matches"""
    # Ensure inputs are 1D
    if student_input_ids.dim() > 1:
        student_input_ids = student_input_ids.squeeze(0)
    if teacher_input_ids.dim() > 1:
        teacher_input_ids = teacher_input_ids.squeeze(0)

    # Get tokens
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    # Find exact token matches
    exact_pairs = []
    overridden_pairs = []

    # Create mapping from normalized teacher tokens to positions
    normalized_teacher = [normalize_token(token) for token in teacher_tokens]
    token_to_teacher_positions = {}
    for j, norm_t in enumerate(normalized_teacher):
        token_to_teacher_positions.setdefault(norm_t, []).append(j)

    # Iterate through student tokens
    for i, token in enumerate(student_tokens):
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
                exact_pairs.append((i, closest_match))
        elif matches:
            # If embeddings not available, use the closest match
            closest_match = min(matches, key=lambda x: abs(x - t_pos_approx))
            exact_pairs.append((i, closest_match))

    return exact_pairs, overridden_pairs

def text_based_token_matching(student_word, teacher_word, student_embeddings, teacher_embeddings,
                            window_size=2, student_tokenizer=None, teacher_tokenizer=None):
    """Match tokens within matched words"""
    student_tokens = student_word['tokens']
    teacher_tokens = teacher_word['tokens']

    if not student_tokens or not teacher_tokens:
        return [], []

    # Use normalized token strings
    student_token_strings = [normalize_token(t) for t in student_word['token_strings']]
    teacher_token_strings = [normalize_token(t) for t in teacher_word['token_strings']]

    # Get normalized positions for tokens
    student_boundaries = calculate_token_boundaries_in_word(student_word, student_word['text'])
    teacher_boundaries = calculate_token_boundaries_in_word(teacher_word, teacher_word['text'])

    text_matches = []
    aligned_positions = set()

    # For each student token
    for s_local_idx in range(len(student_tokens)):
        s_token_text = student_token_strings[s_local_idx]

        # Find matching teacher tokens
        matching_teacher_indices = []
        for t_local_idx, t_text in enumerate(teacher_token_strings):
            if s_token_text == t_text:
                matching_teacher_indices.append(t_local_idx)

        if not matching_teacher_indices:
            continue

        # Get positions
        s_start, s_end = student_boundaries[s_local_idx]
        s_norm_pos = (s_start + s_end) / 2

        # Find closest match by position
        best_t_idx = None
        min_distance = float('inf')

        for t_local_idx in matching_teacher_indices:
            if t_local_idx in aligned_positions:
                continue

            t_start, t_end = teacher_boundaries[t_local_idx]
            t_norm_pos = (t_start + t_end) / 2
            distance = abs(s_norm_pos - t_norm_pos)

            if distance < min_distance:
                min_distance = distance
                best_t_idx = t_local_idx

        if best_t_idx is not None:
            text_matches.append((s_local_idx, best_t_idx))
            aligned_positions.add(best_t_idx)

    return text_matches, []

def word_level_mean_pooling_loss(student_word, teacher_word, student_embeddings, teacher_embeddings,
                                aligned_tokens):
    """Calculate word-level loss using mean pooling of unmatched tokens"""
    student_tokens = student_word['tokens']
    teacher_tokens = teacher_word['tokens']

    # Get unmatched tokens
    aligned_student = {pair[0] for pair in aligned_tokens}
    aligned_teacher = {pair[1] for pair in aligned_tokens}

    unmatched_student = [idx for idx in student_tokens if idx not in aligned_student]
    unmatched_teacher = [idx for idx in teacher_tokens if idx not in aligned_teacher]

    if not unmatched_student or not unmatched_teacher:
        return None, None, 0, 0

    # Create mapping to word-local indices
    student_to_local = {idx: i for i, idx in enumerate(student_tokens)}
    teacher_to_local = {idx: i for i, idx in enumerate(teacher_tokens)}

    # Convert to word-local indices
    unmatched_student_local = [student_to_local[idx] for idx in unmatched_student if idx in student_to_local]
    unmatched_teacher_local = [teacher_to_local[idx] for idx in unmatched_teacher if idx in teacher_to_local]

    if not unmatched_student_local or not unmatched_teacher_local:
        return None, None, 0, 0

    # Get embeddings
    student_embs = student_embeddings[unmatched_student_local]
    teacher_embs = teacher_embeddings[unmatched_teacher_local]

    # Mean pooling
    student_pooled = student_embs.mean(dim=0)
    teacher_pooled = teacher_embs.mean(dim=0)

    return student_pooled.unsqueeze(0), teacher_pooled.unsqueeze(0), len(unmatched_student_local), len(unmatched_teacher_local)

def find_word_matches(student_words, teacher_words, texts_match=True, word_window_size=2):
    """Find word matches between student and teacher"""
    matches = []

    if texts_match:
        # For identical texts, match by position
        min_len = min(len(student_words), len(teacher_words))
        for idx in range(min_len):
            matches.append((idx, idx))
    else:
        # Position-based approximate matching
        for s_idx, s_word in enumerate(student_words):
            if not s_word:  # Skip empty words
                continue

            s_norm_text = normalize_token(s_word['text'])

            # Calculate approximate position
            s_pos_norm = s_idx / max(len(student_words) - 1, 1)
            t_pos_approx = int(s_pos_norm * len(teacher_words))

            # Define search window
            start_idx = max(0, t_pos_approx - word_window_size)
            end_idx = min(len(teacher_words), t_pos_approx + word_window_size + 1)

            # Find best match in window
            best_match = None
            best_score = -1

            for t_idx in range(start_idx, end_idx):
                t_word = teacher_words[t_idx]
                if not t_word:  # Skip empty words
                    continue

                t_norm_text = normalize_token(t_word['text'])

                # Check if words match
                if s_norm_text == t_norm_text:
                    distance = abs(t_idx - t_pos_approx)
                    score = 100 - distance
                    if score > best_score:
                        best_score = score
                        best_match = t_idx

            if best_match is not None:
                matches.append((s_idx, best_match))

    return matches

def log_alignment_details(student_input_ids, teacher_input_ids, student_tokenizer, teacher_tokenizer,
                         token_matches, split_token_matches,
                         student_words, teacher_words, exclude_tokens):
    """Log detailed alignment information for diagnosis"""
    # Get original texts
    student_text = student_tokenizer.decode(student_input_ids, skip_special_tokens=True)
    teacher_text = teacher_tokenizer.decode(teacher_input_ids, skip_special_tokens=True)

    # Find actual sequence boundaries
    student_pad_id = student_tokenizer.pad_token_id
    teacher_eos_id = student_tokenizer.eos_token_id  # Usually same for both tokenizers

    # Find first padding in student sequence
    student_seq_end = len(student_input_ids)
    for i, token_id in enumerate(student_input_ids):
        if token_id == student_pad_id:
            student_seq_end = i
            break

    # Find first EOS in teacher sequence
    teacher_seq_end = len(teacher_input_ids)
    for i, token_id in enumerate(teacher_input_ids):
        if token_id == teacher_eos_id:
            teacher_seq_end = i
            break

    # Count valid tokens (excluding padding and special tokens)
    exclude_set = set(exclude_tokens) if exclude_tokens else set()
    valid_student_tokens = sum(1 for i, tid in enumerate(student_input_ids)
                             if i < student_seq_end and tid.item() not in exclude_set)
    valid_teacher_tokens = sum(1 for i, tid in enumerate(teacher_input_ids)
                             if i < teacher_seq_end and tid.item() not in exclude_set)

    # Count all matches (categorized)
    all_matches = set()
    direct_text_matches = set()
    split_matches = set()  # Now stores tuples: (s_tuple, t_tuple, weight)

    # Direct text matches
    for s, t in token_matches:
        if s < student_seq_end and t < teacher_seq_end and s not in exclude_set and t not in exclude_set:
            all_matches.add((s, t))
            direct_text_matches.add((s, t))

    # Split matches - handle multiple student tokens
    for match in split_token_matches:
        # Unpack the enhanced match structure
        if len(match) == 4:  # New format: (s_tuple, t_tuple, weight, s_primary)
            s_tuple, t_tuple, weight, s_primary = match
        else:  # Legacy format: (s_primary, t_tuple, weight)
            s_primary, t_tuple, weight = match
            s_tuple = (s_primary,)

        # Validate match
        if (s_primary < student_seq_end and
            all(t < teacher_seq_end for t in t_tuple) and
            s_primary not in exclude_set and
            all(t not in exclude_set for t in t_tuple)):

            # Add primary match for counting
            all_matches.add((s_primary, t_tuple[0]))
            # Store full match structure
            split_matches.add((s_tuple, tuple(t_tuple), weight))

    total_matches = len(all_matches)
    matched_student_tokens = len({s for s, t in all_matches})

    print("\n=== ALIGNMENT LOG ===")
    print(f"Student text: {student_text}")
    print(f"Teacher text: {teacher_text}")
    print(f"Student sequence end: {student_seq_end}, Teacher sequence end: {teacher_seq_end}")
    print(f"Valid student tokens: {valid_student_tokens}, Valid teacher tokens: {valid_teacher_tokens}")
    print(f"Total matched student tokens: {matched_student_tokens}")
    print(f"Match ratio: {matched_student_tokens / max(valid_student_tokens, 1):.2%}")

    print(f"\nMatch breakdown:")
    print(f"  Direct text matches: {len(direct_text_matches)}")
    print(f"  Split token matches: {len(split_matches)}")

    # Log each match pair with categorization
    print("\nMatched token pairs:")
    student_tokens = student_tokenizer.convert_ids_to_tokens(student_input_ids)
    teacher_tokens = teacher_tokenizer.convert_ids_to_tokens(teacher_input_ids)

    # Direct text matches
    if direct_text_matches:
        print("\n  Direct text matches:")
        for s, t in sorted(direct_text_matches):
            s_text = student_tokens[s] if s < len(student_tokens) else "OUT_OF_BOUNDS"
            t_text = teacher_tokens[t] if t < len(teacher_tokens) else "OUT_OF_BOUNDS"
            print(f"    Student[{s}]|{normalize_token(s_text)}| -> Teacher[{t}]|{normalize_token(t_text)}|")

    # Split token matches
    if split_matches:
        print("\n  Split token matches:")
        for s_tuple, t_tuple, weight in sorted(split_matches):
            # Get all student token texts
            s_texts = [student_tokens[i] if i < len(student_tokens) else "OUT_OF_BOUNDS"
                       for i in s_tuple]
            s_normalized = [normalize_token(t) for t in s_texts]

            # Get all teacher token texts
            t_texts = [teacher_tokens[i] if i < len(teacher_tokens) else "OUT_OF_BOUNDS"
                       for i in t_tuple]
            t_normalized = [normalize_token(t) for t in t_texts]

            # Format student side (handle multiple tokens)
            if len(s_tuple) > 1:
                student_str = f"[{','.join(map(str, s_tuple))}]|{' + '.join(s_normalized)}|"
            else:
                student_str = f"[{s_tuple[0]}]|{s_normalized[0]}|"

            # Format teacher side (handle multiple tokens)
            if len(t_tuple) > 1:
                teacher_str = f"[{','.join(map(str, t_tuple))}]|{' + '.join(t_normalized)}|"
            else:
                teacher_str = f"[{t_tuple[0]}]|{t_normalized[0]}|"

            print(f"    Student{student_str} -> Teacher{teacher_str}")

    print("=== END ALIGNMENT LOG ===\n")

def split_token_matching(
    student_word: Dict[str, Any],
    teacher_word: Dict[str, Any],
    student_embeddings: Any,
    teacher_embeddings: Any,
    aligned_student_tokens: set = None,
    aligned_teacher_tokens: set = None,
) -> List[Tuple[int, List[int], float]]:
    """Match tokens using bidirectional split token matching within the same word"""
    if aligned_student_tokens is None:
        aligned_student_tokens = set()
    if aligned_teacher_tokens is None:
        aligned_teacher_tokens = set()

    student_tokens = student_word.get('tokens', [])
    teacher_tokens = teacher_word.get('tokens', [])

    if not student_tokens or not teacher_tokens:
        return []

    # Verify words are the same (normalized)
    student_word_text = normalize_token(student_word['text'])
    teacher_word_text = normalize_token(teacher_word['text'])
    if student_word_text != teacher_word_text:
        return []

    # Get token boundaries and normalized strings
    teacher_boundaries = calculate_token_boundaries_in_word(teacher_word, teacher_word['text'])
    teacher_positions = [(b[0] + b[1]) / 2 for b in teacher_boundaries]
    teacher_norm_strings = [normalize_token(t) for t in teacher_word.get('token_strings', [])]

    student_boundaries = calculate_token_boundaries_in_word(student_word, student_word['text'])
    student_positions = [(b[0] + b[1]) / 2 for b in student_boundaries]
    student_norm_strings = [normalize_token(t) for t in student_word.get('token_strings', [])]

    split_matches = []
    used_student_tokens = set(aligned_student_tokens)
    used_teacher_tokens = set(aligned_teacher_tokens)

    # Direction 1: Single student token -> Multiple teacher tokens
    for s_local_idx in range(len(student_tokens)):
        s_global_idx = student_tokens[s_local_idx]

        if s_global_idx in used_student_tokens:
            continue

        s_token_text = student_norm_strings[s_local_idx]
        if s_token_text == '':
            continue

        s_pos = student_positions[s_local_idx]

        # Find closest teacher token by position
        min_distance = float('inf')
        anchor_teacher_idx = None
        for t_local_idx in range(len(teacher_positions)):
            t_global_idx = teacher_tokens[t_local_idx]
            if t_global_idx not in used_teacher_tokens:
                distance = abs(s_pos - teacher_positions[t_local_idx])
                if distance < min_distance:
                    min_distance = distance
                    anchor_teacher_idx = t_local_idx

        if anchor_teacher_idx is None:
            continue

        # Look in window around anchor
        window_start = max(0, anchor_teacher_idx - 2)
        window_end = min(len(teacher_tokens), anchor_teacher_idx + 3)

        # Check contiguous sequences
        best_match = None
        best_text_similarity = 0.0

        for start in range(window_start, window_end):
            for end in range(start + 1, window_end + 1):
                if not (start <= anchor_teacher_idx < end):
                    continue

                teacher_sequence_indices = list(range(start, end))
                if any(teacher_tokens[idx] in used_teacher_tokens for idx in teacher_sequence_indices):
                    continue

                teacher_sequence_texts = [teacher_norm_strings[i] for i in teacher_sequence_indices]
                if any(text == '' for text in teacher_sequence_texts):
                    continue

                concatenated_teacher_text = ''.join(teacher_sequence_texts)

                # Check text similarity
                text_similarity = SequenceMatcher(None, s_token_text, concatenated_teacher_text).ratio()
                if text_similarity < 0.8:  # High threshold for split matching
                    continue

                if text_similarity > best_text_similarity:
                    best_match = teacher_sequence_indices
                    best_text_similarity = text_similarity

        if best_match and best_text_similarity >= 0.8:
            teacher_global_indices = [teacher_tokens[i] for i in best_match]
            split_matches.append((s_global_idx, teacher_global_indices, 1.0))
            # Mark tokens as used
            used_student_tokens.add(s_global_idx)
            used_teacher_tokens.update(teacher_global_indices)

    # Direction 2: Multiple student tokens -> Single teacher token
    for t_local_idx in range(len(teacher_tokens)):
        t_global_idx = teacher_tokens[t_local_idx]

        if t_global_idx in used_teacher_tokens:
            continue

        t_token_text = teacher_norm_strings[t_local_idx]
        if t_token_text == '':
            continue

        t_pos = teacher_positions[t_local_idx]

        # Find closest student token
        min_distance = float('inf')
        anchor_student_idx = None
        for s_local_idx in range(len(student_positions)):
            s_global_idx = student_tokens[s_local_idx]
            if s_global_idx not in used_student_tokens:
                distance = abs(t_pos - student_positions[s_local_idx])
                if distance < min_distance:
                    min_distance = distance
                    anchor_student_idx = s_local_idx

        if anchor_student_idx is None:
            continue

        # Look in window around anchor
        window_start = max(0, anchor_student_idx - 2)
        window_end = min(len(student_tokens), anchor_student_idx + 3)

        # Check contiguous sequences
        best_match = None
        best_text_similarity = 0.0

        for start in range(window_start, window_end):
            for end in range(start + 1, window_end + 1):
                if not (start <= anchor_student_idx < end):
                    continue

                student_sequence_indices = list(range(start, end))
                if any(student_tokens[idx] in used_student_tokens for idx in student_sequence_indices):
                    continue

                student_sequence_texts = [student_norm_strings[i] for i in student_sequence_indices]
                if any(text == '' for text in student_sequence_texts):
                    continue

                concatenated_student_text = ''.join(student_sequence_texts)

                # Check text similarity
                text_similarity = SequenceMatcher(None, concatenated_student_text, t_token_text).ratio()
                if text_similarity < 0.8:
                    continue

                if text_similarity > best_text_similarity:
                    best_match = student_sequence_indices
                    best_text_similarity = text_similarity

        if best_match and best_text_similarity >= 0.8:
            # Convert to student-first format
            primary_student = student_tokens[best_match[0]]
            split_matches.append((primary_student, [t_global_idx], 1.0))
            # Mark tokens as used
            used_student_tokens.update(student_tokens[idx] for idx in best_match)
            used_teacher_tokens.add(t_global_idx)

    return split_matches

def find_direct_token_matches(
    student_sequence: List[int],
    teacher_sequence: List[int],
    student_texts: List[str],
    teacher_texts: List[str],
    student_positions: List[float],
    teacher_positions: List[float],
    threshold: float
) -> List[Tuple[int, int]]:
    """
    Find direct token-to-token matches within sequences based on text similarity.
    Returns list of (student_idx, teacher_idx) tuples.
    """
    if not student_sequence or not teacher_sequence:
        return []

    # Calculate normalized positions for each sequence (0 to 1)
    if len(student_positions) > 1:
        s_pos_min = min(student_positions[i] for i in student_sequence)
        s_pos_max = max(student_positions[i] for i in student_sequence)
        s_pos_range = s_pos_max - s_pos_min if s_pos_max > s_pos_min else 1.0
        s_norm_positions = [(student_positions[i] - s_pos_min) / s_pos_range for i in student_sequence]
    else:
        s_norm_positions = [0.5]  # Center position for single token

    if len(teacher_positions) > 1:
        t_pos_min = min(teacher_positions[i] for i in teacher_sequence)
        t_pos_max = max(teacher_positions[i] for i in teacher_sequence)
        t_pos_range = t_pos_max - t_pos_min if t_pos_max > t_pos_min else 1.0
        t_norm_positions = [(teacher_positions[i] - t_pos_min) / t_pos_range for i in teacher_sequence]
    else:
        t_norm_positions = [0.5]  # Center position for single token

    # Find best matches for each student token
    matches = []
    used_s_indices = set()
    used_t_indices = set()

    # Create a list of potential matches with their similarity scores
    potential_matches = []

    for i, s_idx in enumerate(student_sequence):
        if s_idx in used_s_indices:
            continue

        s_text = student_texts[i]
        s_norm_pos = s_norm_positions[i]

        # Find the position in teacher sequence that corresponds to this student's position
        target_t_pos = s_norm_pos

        # Check neighboring teacher tokens
        best_match = None
        best_similarity = 0.0

        # Find closest teacher position
        closest_t_idx = None
        min_dist = float('inf')
        for j, t_norm_pos in enumerate(t_norm_positions):
            dist = abs(t_norm_pos - target_t_pos)
            if dist < min_dist:
                min_dist = dist
                closest_t_idx = j

        if closest_t_idx is not None:
            # Check the closest and its neighbors
            candidates = [closest_t_idx]
            if closest_t_idx > 0:
                candidates.append(closest_t_idx - 1)
            if closest_t_idx < len(teacher_sequence) - 1:
                candidates.append(closest_t_idx + 1)

            for j in candidates:
                if j in used_t_indices:
                    continue

                t_text = teacher_texts[j]
                similarity = SequenceMatcher(None, s_text, t_text).ratio()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = j

        if best_match is not None and best_similarity >= threshold:
            potential_matches.append((s_idx, teacher_sequence[best_match], best_similarity))

    # Sort by similarity descending and greedily select matches
    potential_matches.sort(key=lambda x: x[2], reverse=True)

    for s_idx, t_idx, similarity in potential_matches:
        if s_idx not in used_s_indices and t_idx not in used_t_indices:
            matches.append((s_idx, t_idx))
            used_s_indices.add(s_idx)
            used_t_indices.add(t_idx)

    return matches

def hybrid_alignment(student_input_ids, teacher_input_ids,
                            student_tokenizer, teacher_tokenizer,
                            student_embeddings, teacher_embeddings,
                            exclude_tokens=None, enable_logging=False):
    """Three-stage alignment: text tokens, space tokens, split token matching"""
    # Get tokens and words
    student_text = student_tokenizer.decode(student_input_ids, skip_special_tokens=True)
    teacher_text = teacher_tokenizer.decode(teacher_input_ids, skip_special_tokens=True)

    # Check if texts match exactly
    texts_match = student_text == teacher_text

    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    # Filter out special tokens only (NOT spaces or padding tokens)
    if exclude_tokens is not None:
        exclude_set = set(exclude_tokens)

        # Create mappings for student
        original_to_filtered = {}
        filtered_to_original = {}

        student_tokens_filtered = []
        student_embeddings_filtered = []
        student_input_ids_filtered = []

        filtered_idx = 0
        for original_idx, token in enumerate(student_tokens):
            # Only exclude actual special tokens, not spaces or padding
            token_id = student_input_ids[original_idx].item()
            if original_idx not in exclude_set and token_id not in exclude_set:
                original_to_filtered[original_idx] = filtered_idx
                filtered_to_original[filtered_idx] = original_idx

                student_tokens_filtered.append(token)
                student_embeddings_filtered.append(student_embeddings[original_idx])
                student_input_ids_filtered.append(student_input_ids[original_idx])
                filtered_idx += 1

        # Convert to tensors
        if student_embeddings_filtered:
            student_embeddings = torch.stack(student_embeddings_filtered)
        else:
            student_embeddings = torch.empty(0, device=student_embeddings.device, dtype=student_embeddings.dtype)
        student_input_ids = torch.tensor(student_input_ids_filtered, device=student_input_ids.device)
        student_tokens = student_tokens_filtered

        # Create mappings for teacher
        teacher_original_to_filtered = {}
        teacher_filtered_to_original = {}

        teacher_tokens_filtered = []
        teacher_embeddings_filtered = []
        teacher_input_ids_filtered = []

        filtered_idx = 0
        for original_idx, token in enumerate(teacher_tokens):
            token_id = teacher_input_ids[original_idx].item()
            if original_idx not in exclude_set and token_id not in exclude_set:
                teacher_original_to_filtered[original_idx] = filtered_idx
                teacher_filtered_to_original[filtered_idx] = original_idx

                teacher_tokens_filtered.append(token)
                teacher_embeddings_filtered.append(teacher_embeddings[original_idx])
                teacher_input_ids_filtered.append(teacher_input_ids[original_idx])
                filtered_idx += 1

        if teacher_embeddings_filtered:
            teacher_embeddings = torch.stack(teacher_embeddings_filtered)
        else:
            teacher_embeddings = torch.empty(0, device=teacher_embeddings.device, dtype=teacher_embeddings.dtype)
        teacher_input_ids = torch.tensor(teacher_input_ids_filtered, device=teacher_input_ids.device)
        teacher_tokens = teacher_tokens_filtered

    # Get word mappings
    student_words = get_word_token_mappings(student_tokens, student_tokenizer, student_text)
    teacher_words = get_word_token_mappings(teacher_tokens, teacher_tokenizer, teacher_text)

    if len(student_words) == 0 or len(teacher_words) == 0:
        return [], [], [], [], []

    # Find word matches
    word_matches = find_word_matches(student_words, teacher_words, texts_match=texts_match, word_window_size=2)

    # Track all matches
    all_token_matches = []
    all_weighted_matches = []
    split_token_matches = []

    # Track aligned tokens
    aligned_student_tokens = set()
    aligned_teacher_tokens = set()

    # Stage 1: Text-based token matching within matched words
    for s_word_idx, t_word_idx in word_matches:
        if s_word_idx >= len(student_words) or t_word_idx >= len(teacher_words):
            continue

        student_word = student_words[s_word_idx]
        teacher_word = teacher_words[t_word_idx]

        # Bounds checking
        if any(idx < 0 or idx >= len(student_embeddings) for idx in student_word['tokens']):
            continue
        if any(idx < 0 or idx >= len(teacher_embeddings) for idx in teacher_word['tokens']):
            continue

        s_word_embs = student_embeddings[student_word['tokens']]
        t_word_embs = teacher_embeddings[teacher_word['tokens']]

        # Get text matches
        text_matches, weighted_matches = text_based_token_matching(
            student_word, teacher_word, s_word_embs, t_word_embs,
            window_size=2 if texts_match else 3,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer
        )

        # Convert to global indices
        for s_local, t_local in text_matches:
            s_global = student_word['tokens'][s_local]
            t_global = teacher_word['tokens'][t_local]
            all_token_matches.append((s_global, t_global))
            aligned_student_tokens.add(s_global)
            aligned_teacher_tokens.add(t_global)

        for s_local, t_local, weight in weighted_matches:
            s_global = student_word['tokens'][s_local]
            t_global = teacher_word['tokens'][t_local]
            all_weighted_matches.append((s_global, t_global, weight))

    # Stage 2: Space token matching
    student_spaces = get_word_exterior_sequences(student_tokens, student_words)
    teacher_spaces = get_word_exterior_sequences(teacher_tokens, teacher_words)

    if student_spaces and teacher_spaces:
        space_matches = match_space_sequences(
            student_spaces, teacher_spaces,
            student_words, teacher_words,
            window_size=2
        )

        for s_idx, t_idx in space_matches:
            all_token_matches.append((s_idx, t_idx))
            aligned_student_tokens.add(s_idx)
            aligned_teacher_tokens.add(t_idx)

    # Stage 3: Split token matching for remaining tokens in matched words
    for s_word_idx, t_word_idx in word_matches:
        if s_word_idx >= len(student_words) or t_word_idx >= len(teacher_words):
            continue

        student_word = student_words[s_word_idx]
        teacher_word = teacher_words[t_word_idx]

        # Get embeddings for this word
        s_word_embs = student_embeddings[student_word['tokens']]
        t_word_embs = teacher_embeddings[teacher_word['tokens']]

        # Perform split token matching
        word_split_matches = split_token_matching(
            student_word, teacher_word, s_word_embs, t_word_embs,
            aligned_student_tokens, aligned_teacher_tokens
        )

        # Update split token matches and aligned tokens
        for s_global, teacher_global_indices, weight in word_split_matches:
            split_token_matches.append((s_global, teacher_global_indices, weight))
            aligned_student_tokens.add(s_global)
            aligned_teacher_tokens.update(teacher_global_indices)

    # Stage 4: Fallback matching for unmatched tokens by position (within matched words)
    for s_word_idx, t_word_idx in word_matches:
        if s_word_idx >= len(student_words) or t_word_idx >= len(teacher_words):
            continue

        student_word = student_words[s_word_idx]
        teacher_word = teacher_words[t_word_idx]

        # Get unmatched tokens in this word
        unmatched_student = [idx for idx in student_word['tokens'] if idx not in aligned_student_tokens]
        unmatched_teacher = [idx for idx in teacher_word['tokens'] if idx not in aligned_teacher_tokens]

        if not unmatched_student or not unmatched_teacher:
            continue

        # Create position mappings within the word
        student_boundaries = calculate_token_boundaries_in_word(student_word, student_word['text'])
        teacher_boundaries = calculate_token_boundaries_in_word(teacher_word, teacher_word['text'])

        student_positions = [(b[0] + b[1]) / 2 for b in student_boundaries]
        teacher_positions = [(b[0] + b[1]) / 2 for b in teacher_boundaries]

        # Map to local indices
        student_to_local = {idx: i for i, idx in enumerate(student_word['tokens'])}
        teacher_to_local = {idx: i for i, idx in enumerate(teacher_word['tokens'])}

        # Convert unmatched to local indices and positions
        unmatched_student_local = [(student_to_local[idx], student_positions[student_to_local[idx]]) for idx in unmatched_student]
        unmatched_teacher_local = [(teacher_to_local[idx], teacher_positions[teacher_to_local[idx]]) for idx in unmatched_teacher]

        # Match by closest position
        for s_local_idx, s_pos in unmatched_student_local:
            if not unmatched_teacher_local:
                break
            # Find closest teacher token
            best_t_local_idx, best_t_pos, min_dist = None, None, float('inf')
            for t_local_idx, t_pos in unmatched_teacher_local:
                dist = abs(s_pos - t_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_t_local_idx = t_local_idx
                    best_t_pos = t_pos

            if best_t_local_idx is not None:
                s_global_idx = student_word['tokens'][s_local_idx]
                t_global_idx = teacher_word['tokens'][best_t_local_idx]
                all_token_matches.append((s_global_idx, t_global_idx))
                aligned_student_tokens.add(s_global_idx)
                aligned_teacher_tokens.add(t_global_idx)
                # Remove matched teacher token
                unmatched_teacher_local = [(t_idx, t_pos) for t_idx, t_pos in unmatched_teacher_local if t_idx != best_t_local_idx]

    # Convert back to original indices if filtering was done
    if exclude_tokens is not None:
        # Convert token matches back
        original_token_matches = []
        for s_filtered_idx, t_filtered_idx in all_token_matches:
            if s_filtered_idx in filtered_to_original and t_filtered_idx in teacher_filtered_to_original:
                s_original_idx = filtered_to_original[s_filtered_idx]
                t_original_idx = teacher_filtered_to_original[t_filtered_idx]
                original_token_matches.append((s_original_idx, t_original_idx))
        all_token_matches = original_token_matches

        # Convert weighted matches
        original_weighted_matches = []
        for s_filtered_idx, t_filtered_idx, weight in all_weighted_matches:
            if s_filtered_idx in filtered_to_original and t_filtered_idx in teacher_filtered_to_original:
                s_original_idx = filtered_to_original[s_filtered_idx]
                t_original_idx = teacher_filtered_to_original[t_filtered_idx]
                original_weighted_matches.append((s_original_idx, t_original_idx, weight))
        all_weighted_matches = original_weighted_matches

        # Convert split matches
        original_split_matches = []
        for s_filtered_idx, t_indices_list, weight in split_token_matches:
            if s_filtered_idx in filtered_to_original:
                s_original_idx = filtered_to_original[s_filtered_idx]
                t_original_indices = []
                for t_filtered_idx in t_indices_list:
                    if t_filtered_idx in teacher_filtered_to_original:
                        t_original_indices.append(teacher_filtered_to_original[t_filtered_idx])
                if t_original_indices:
                    original_split_matches.append((s_original_idx, t_original_indices, weight))
        split_token_matches = original_split_matches

    # Enable detailed logging if requested
    if enable_logging:
        log_alignment_details(
            student_input_ids, teacher_input_ids,
            student_tokenizer, teacher_tokenizer,
            all_token_matches, split_token_matches,
            student_words, teacher_words, exclude_tokens
        )

    return all_token_matches, all_weighted_matches, split_token_matches, student_words, teacher_words

# ========== Loss Functions ==========
class AlignmentLoss(torch.nn.Module):
    def __init__(self,
                 student_tokenizer=None,
                 teacher_tokenizer=None,
                 # Text alignment loss weights
                 text_huber_weight: float = 0.70,
                 text_cosine_weight: float = 0.30,
                 # Split token alignment loss weights
                 split_huber_weight: float = 0.50,
                 split_cosine_weight: float = 0.10):
        super().__init__()

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Loss weights
        self.text_huber_weight = text_huber_weight
        self.text_cosine_weight = text_cosine_weight
        self.split_huber_weight = split_huber_weight
        self.split_cosine_weight = split_cosine_weight
        # Weight for padding token losses
        self.padding_loss_weight = PADDING_LOSS_WEIGHT

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
        total_split_huber_loss = torch.tensor(0.0, device=device)
        total_split_cosine_loss = torch.tensor(0.0, device=device)
        total_padding_huber_loss = torch.tensor(0.0, device=device)
        total_padding_cosine_loss = torch.tensor(0.0, device=device)

        # Initialize counters
        total_text_aligned_tokens = 0
        total_split_aligned_tokens = 0
        total_padding_aligned_tokens = 0
        total_attended_tokens = 0  # Will include padding tokens that are attended

        # Get special token IDs
        student_pad_id = self.student_tokenizer.pad_token_id if self.student_tokenizer else None
        teacher_pad_id = self.teacher_tokenizer.pad_token_id if self.teacher_tokenizer else None
        teacher_eos_id = self.teacher_tokenizer.eos_token_id if self.teacher_tokenizer else None

        # Define tokens to exclude (special tokens only, not spaces)
        exclude_tokens = {
            getattr(self.student_tokenizer, 'pad_token_id', None),
            getattr(self.student_tokenizer, 'bos_token_id', None),
            getattr(self.student_tokenizer, 'eos_token_id', None),
        }
        exclude_tokens = {t for t in exclude_tokens if t is not None}

        # Enable logging for first few batches
        enable_logging = ENABLE_ALIGNMENT_DEBUG

        for i in range(batch_size):
            # Get actual tokens based on masks
            t_indices = (teacher_mask[i] == 1).nonzero(as_tuple=True)[0]
            s_indices = (student_mask[i] == 1).nonzero(as_tuple=True)[0]

            if len(t_indices) == 0 or len(s_indices) == 0:
                continue

            # Count ALL attended tokens (including padding tokens that are attended)
            attended_tokens = []
            for idx in s_indices:
                token_id = student_input_ids[i][idx].item()
                attended_tokens.append(idx)
            total_attended_tokens += len(attended_tokens)

            # Count valid student tokens (non-special tokens within mask) for alignment
            valid_student_indices = []
            for idx in s_indices:
                token_id = student_input_ids[i][idx].item()
                if token_id not in exclude_tokens:
                    valid_student_indices.append(idx)

            actual_student_tokens = len(valid_student_indices)

            # Get embeddings
            student_embs = student_output[i]
            teacher_embs = teacher_output[i]

            # Use hybrid alignment
            token_matches, weighted_matches, split_token_matches, student_words, teacher_words = hybrid_alignment(
                student_input_ids[i], teacher_input_ids[i],
                self.student_tokenizer, self.teacher_tokenizer,
                student_embs, teacher_embs,
                exclude_tokens=exclude_tokens,
                enable_logging=enable_logging
            )

            # Text-based token matches
            if token_matches:
                student_token_embs = student_embs[[pair[0] for pair in token_matches]]
                teacher_token_embs = teacher_embs[[pair[1] for pair in token_matches]]

                weight_map = {(s, t): w for s, t, w in weighted_matches}
                token_weights = []
                for s, t in token_matches:
                    token_weights.append(weight_map.get((s, t), 1.0))
                token_weights = torch.tensor(token_weights, device=device, dtype=student_token_embs.dtype)

                token_huber_loss = F.huber_loss(
                    student_token_embs, teacher_token_embs, reduction='none'
                )
                token_huber_loss = (token_huber_loss * token_weights.unsqueeze(-1)).mean()

                token_cosine_sim = F.cosine_similarity(
                    student_token_embs, teacher_token_embs, dim=-1
                )
                token_cosine_loss = ((1 - token_cosine_sim) * token_weights).mean()

                total_text_huber_loss += token_huber_loss
                total_text_cosine_loss += token_cosine_loss
                total_text_aligned_tokens += len(token_matches)

            # Split token matches
            if split_token_matches:
                split_student_embs = []
                split_teacher_embs = []
                split_weights = []

                for s_idx, t_indices_list, weight in split_token_matches:
                    # Skip if teacher indices list is empty
                    if not t_indices_list:
                        continue
                    student_emb = student_embs[s_idx]

                    # Average the teacher embeddings
                    teacher_embs_list = [teacher_embs[t_idx] for t_idx in t_indices_list]
                    teacher_avg_emb = torch.stack(teacher_embs_list).mean(dim=0)

                    split_student_embs.append(student_emb)
                    split_teacher_embs.append(teacher_avg_emb)
                    split_weights.append(weight)

                # Only process if we have at least one match
                if split_student_embs:
                    split_student_embs = torch.stack(split_student_embs)
                    split_teacher_embs = torch.stack(split_teacher_embs)
                    split_weights = torch.tensor(split_weights, device=device, dtype=split_student_embs.dtype)

                    split_huber_loss = F.huber_loss(
                        split_student_embs, split_teacher_embs, reduction='none'
                    )
                    split_huber_loss = (split_huber_loss * split_weights.unsqueeze(-1)).mean()

                    split_cosine_sim = F.cosine_similarity(
                        split_student_embs, split_teacher_embs, dim=-1
                    )
                    split_cosine_loss = ((1 - split_cosine_sim) * split_weights).mean()

                    total_split_huber_loss += split_huber_loss
                    total_split_cosine_loss += split_cosine_loss
                    total_split_aligned_tokens += len(split_token_matches)

            # Stage 4: Padding token alignment
            # Find the actual sequence end for student (first pad token)
            student_seq_end = 0
            for idx in range(len(student_input_ids[i])):
                if student_input_ids[i][idx].item() == student_pad_id:
                    student_seq_end = idx
                    break
            if student_seq_end == 0:
                student_seq_end = len(student_input_ids[i])  # No padding found

            # Find the actual sequence end for teacher (first EOS token, then pad)
            teacher_seq_end = 0
            for idx in range(len(teacher_input_ids[i])):
                if teacher_input_ids[i][idx].item() == teacher_eos_id:
                    teacher_seq_end = idx + 1  # Include EOS
                    break
            if teacher_seq_end == 0:
                # No EOS found, find first pad
                for idx in range(len(teacher_input_ids[i])):
                    if teacher_input_ids[i][idx].item() == teacher_pad_id:
                        teacher_seq_end = idx
                        break
            if teacher_seq_end == 0:
                teacher_seq_end = len(teacher_input_ids[i])  # No padding found

            # Count attended padding tokens (mask == 1 but after sequence end)
            student_pad_tokens = []
            teacher_pad_tokens = []

            # Get all pad tokens that are attended (mask == 1)
            for idx in range(len(student_input_ids[i])):
                if idx >= student_seq_end and student_mask[i][idx].item() == 1:
                    token_id = student_input_ids[i][idx].item()
                    if token_id == student_pad_id:
                        student_pad_tokens.append(idx)

            for idx in range(len(teacher_input_ids[i])):
                if idx >= teacher_seq_end and teacher_mask[i][idx].item() == 1:
                    token_id = teacher_input_ids[i][idx].item()
                    if token_id == teacher_pad_id:
                        teacher_pad_tokens.append(idx)

            # Match padding tokens one-to-one by position
            if student_pad_tokens and teacher_pad_tokens:
                num_pad_matches = min(len(student_pad_tokens), len(teacher_pad_tokens))

                for j in range(num_pad_matches):
                    s_idx = student_pad_tokens[j]
                    t_idx = teacher_pad_tokens[j]

                    # Ensure indices are within bounds
                    if s_idx < len(student_embs) and t_idx < len(teacher_embs):
                        # Calculate loss for each padding pair
                        pad_student_emb = student_embs[s_idx].unsqueeze(0)
                        pad_teacher_emb = teacher_embs[t_idx].unsqueeze(0)

                        pad_huber_loss = F.huber_loss(pad_student_emb, pad_teacher_emb)
                        pad_cosine_sim = F.cosine_similarity(pad_student_emb, pad_teacher_emb, dim=-1)
                        pad_cosine_loss = (1 - pad_cosine_sim).mean()

                        # Apply padding loss weight of 0.5
                        total_padding_huber_loss += self.padding_loss_weight * pad_huber_loss
                        total_padding_cosine_loss += self.padding_loss_weight * pad_cosine_loss
                        total_padding_aligned_tokens += 1

        # Average across batch
        if batch_size > 0:
            total_text_huber_loss = total_text_huber_loss / batch_size
            total_text_cosine_loss = total_text_cosine_loss / batch_size
            total_split_huber_loss = total_split_huber_loss / batch_size
            total_split_cosine_loss = total_split_cosine_loss / batch_size
            total_padding_huber_loss = total_padding_huber_loss / batch_size
            total_padding_cosine_loss = total_padding_cosine_loss / batch_size

        # Combine all losses
        total_loss = (
            self.text_huber_weight * total_text_huber_loss +
            self.text_cosine_weight * total_text_cosine_loss +
            self.split_huber_weight * total_split_huber_loss +
            self.split_cosine_weight * total_split_cosine_loss +
            self.text_huber_weight * total_padding_huber_loss +  # Use text weights for padding
            self.text_cosine_weight * total_padding_cosine_loss
        )

        # Calculate token match ratio properly (include attended padding tokens in denominator)
        total_aligned_tokens = total_text_aligned_tokens + total_split_aligned_tokens + total_padding_aligned_tokens
        text_aligned_ratio = total_aligned_tokens / max(total_attended_tokens, 1)

        total_huber_loss = total_text_huber_loss + total_split_huber_loss + total_padding_huber_loss
        total_cosine_loss = total_text_cosine_loss + total_split_cosine_loss + total_padding_cosine_loss

        return (
            total_loss, total_huber_loss, total_cosine_loss, text_aligned_ratio
        )

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
            file_path = EVALUATION_DATASET_DIR
            sample_rate = None

        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

        if is_eval and sample_rate is not None:
            self.lines = random.sample(self.lines, min(int(len(self.lines) * sample_rate), len(self.lines)))

        self.enhanced_lines = []
        if ENHANCED_DATASET:
            with open(ENHANCED_DATASET_DIR, "r", encoding="utf-8") as f:
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
                    enhanced_base_name = os.path.basename(ENHANCED_DATASET_DIR)
                    enhanced_cache_folder = os.path.join(cache_path, enhanced_base_name)
                    enhanced_validation_file = os.path.join(enhanced_cache_folder, f"{enhanced_base_name}.validation")

                    if os.path.exists(enhanced_validation_file):
                        self.enhanced_cache_folder = enhanced_cache_folder
                        self.enhanced_embedding_files = [os.path.join(enhanced_cache_folder, f"{i}.pt") for i in range(len(self.enhanced_lines))]
                        self.enhanced_mask_files = [os.path.join(enhanced_cache_folder, f"{i}_mask.pt") for i in range(len(self.enhanced_lines))]
                    else:
                        print(f"Generating and caching enhanced embeddings for {ENHANCED_DATASET_DIR}")
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
    """Get projection layers with T5-xxl layer extraction and fallback"""
    projection_layers = []

    output_dim_prev = 1024
    layers_to_load = len(PROJECTION_LAYERS_CONFIG)

    # Load T5 model once if we need to extract layers
    t5_model = None
    needs_t5_model = any(
        layer_config["type"] in ["t5_encoder", "t5_rmsnorm"]
        and not os.path.exists(os.path.join(QWEN3_MODEL_DIR, f"projection_layer_{layer_config['file_num']}.safetensors"))
        for layer_config in PROJECTION_LAYERS_CONFIG
    )

    if needs_t5_model:
        print("Loading T5-xxl model for layer extraction (CPU-only)...")
        t5_model = T5EncoderModel.from_pretrained(
            T5_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Keep on CPU
            low_cpu_mem_usage=True
        )
        t5_model.eval()  # Ensure it's in eval mode

    for i in range(1, layers_to_load + 1):
        layer_config = PROJECTION_LAYERS_CONFIG[i-1]
        layer_num = layer_config["file_num"]
        layer_path = os.path.join(QWEN3_MODEL_DIR, f"projection_layer_{layer_num}.safetensors")
        input_dim = output_dim_prev
        file_num = layer_config["file_num"]

        if layer_config["type"] == "linear":
            output_dim = layer_config["output_dim"]
            input_dim = output_dim_prev
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = LinearLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing linear layer {file_num}")
                projection_layer.is_new = False
            else:
                projection_layer = LinearLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                print(f"Initialising new linear layer {file_num}")
                projection_layer.is_new = True
            output_dim_prev = output_dim

        elif layer_config["type"] == "transformer_encoder":
            hidden_size = output_dim_prev
            num_heads = layer_config["num_heads"]
            dropout_rate = layer_config["dropout_rate"]
            dim_feedforward = layer_config["dim_feedforward"]

            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = TransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    dim_feedforward=dim_feedforward
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing standard transformer encoder block {file_num}")
                projection_layer.is_new = False
            else:
                projection_layer = TransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    dim_feedforward=dim_feedforward
                )
                print(f"Initialising new standard transformer encoder block {file_num}")
                projection_layer.is_new = True

        elif layer_config["type"] == "t5_encoder":
            hidden_size = output_dim_prev
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = T5EncoderBlock(
                    hidden_size=hidden_size,
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing T5 encoder layer {file_num}")
                projection_layer.is_new = False
            else:
                if TAKE_T5_ENCODER:
                    if t5_model is None:
                        print("Loading T5-xxl model for layer extraction (CPU-only)...")
                        t5_model = T5EncoderModel.from_pretrained(
                            T5_MODEL_DIR,
                            torch_dtype=torch.bfloat16,
                            device_map="cpu",
                            low_cpu_mem_usage=True
                        )
                        t5_model.eval()

                    extracted_layer = extract_t5_layer(t5_model, "t5_encoder", layer_config, hidden_size)
                    if extracted_layer is not None:
                        projection_layer = extracted_layer
                        print(f"Extracting T5 encoder block from T5-xxl for layer {file_num}")
                        projection_layer.is_new = True
                        projection_layer.extracted_from_t5 = True
                    else:
                        # Fallback to new initialization
                        projection_layer = T5EncoderBlock(
                            hidden_size=hidden_size,
                        )
                        print(f"Failed to extract T5 encoder block, initializing new layer {file_num}")
                        projection_layer.is_new = True
                        projection_layer.extracted_from_t5 = False
                else:
                    projection_layer = T5EncoderBlock(
                        hidden_size=hidden_size,
                    )
                    print(f"Initialising new T5 Encoder layer {file_num}")
                    projection_layer.is_new = True
                    projection_layer.extracted_from_t5 = False

        elif layer_config["type"] == "t5_rmsnorm":
            hidden_size = output_dim_prev
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = T5RMSNorm(
                    hidden_size=hidden_size,
                )
                projection_layer.load_state_dict(state_dict)
                print(f"Loading existing T5 RMSNorm layer {file_num}")
                projection_layer.is_new = False
            else:
                if TAKE_T5_RMSNORM:
                    if t5_model is None:
                        print("Loading T5-xxl model for layer extraction (CPU-only)...")
                        t5_model = T5EncoderModel.from_pretrained(
                            T5_MODEL_DIR,
                            torch_dtype=torch.bfloat16,
                            device_map="cpu",
                            low_cpu_mem_usage=True
                        )
                        t5_model.eval()

                    extracted_layer = extract_t5_layer(t5_model, "t5_rmsnorm", {}, hidden_size)
                    if extracted_layer is not None and TAKE_T5_RMSNORM:
                        projection_layer = extracted_layer
                        print(f"Extracting T5 RMSNorm from T5-xxl for layer {file_num}")
                        projection_layer.is_new = True
                        projection_layer.extracted_from_t5 = True
                    else:
                        projection_layer = T5RMSNorm(
                            hidden_size=output_dim_prev,
                        )
                        print(f"Failed to extract T5 RMSNorm, initializing new layer {file_num}")
                        projection_layer.is_new = True
                        projection_layer.extracted_from_t5 = False
                else:
                    projection_layer = T5RMSNorm(
                        hidden_size=output_dim_prev,
                    )
                    print(f"Initialising new T5 RMSNorm layer {file_num}")
                    projection_layer.is_new = True
                    projection_layer.extracted_from_t5 = False

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
                  align_loss_fn, sequence_loss_fn,
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
    if sequence_loss_fn is not None:
        sequence_loss_fn.eval()

    total_losses = {
        'total': 0.0,
        'align_huber': 0.0,
        'align_cos': 0.0,
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

                if TRAIN_MODEL:
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        student_outputs = model(
                            input_ids=s_input_ids,
                            attention_mask=s_mask,
                            output_hidden_states=True
                        )
                        student_hidden = student_outputs.hidden_states[-1]
                        projected_student = student_hidden

                        for layer in projection_layers:
                            projected_student = layer(projected_student)
                else:
                    # Use no_grad and only get last hidden state when not training model
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        student_outputs = model(
                            input_ids=s_input_ids,
                            attention_mask=s_mask,
                            output_hidden_states=True
                        )
                        student_hidden = student_outputs.hidden_states[-1]

                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        projected_student = student_hidden
                        for layer in projection_layers:
                            projected_student = layer(projected_student)

                extra_padding = 1
                s_mask = modify_mask_to_attend_padding(s_mask, 512, num_extra_padding=extra_padding)
                t_mask = modify_mask_to_attend_padding(t_mask, 512, num_extra_padding=extra_padding)

                eval_loss = torch.tensor(0.0, device=device)
                eval_align_huber = torch.tensor(0.0, device=device)
                eval_align_cos = torch.tensor(0.0, device=device)
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

                total_losses['total'] += eval_loss.item()
                total_losses['align_huber'] += eval_align_huber.item()
                total_losses['align_cos'] += eval_align_cos.item()
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

def load_optimizer_states(save_path: str, model_optimizer, scheduler_model, projection_optimizer, scheduler_projection, new_layer_exists) -> bool:
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
            # Discard projection optimizer state if new proj layer loaded
            if not new_layer_exists:
                projection_optimizer.load_state_dict(torch.load(opt_path))
                # Reset learning rate to current scheduler value
                if scheduler_projection is not None:
                    for param_group, lr in zip(projection_optimizer.param_groups, scheduler_projection.get_last_lr()):
                        param_group['lr'] = lr
                print("Loaded projection optimizer state")
            else:
                print("Discarded projection optimizer state due to new layer initialisation")
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
        proj_lr_line = f"Proj LR: {current_lr_proj:.6f}, "
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
        loss_lines.append(f"Token Match: {text_aligned_ratio:.2%}")
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
        QWEN3_MODEL_DIR,
        max_seq_length=512,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        local_files_only=True,
        revision="main",
        full_finetuning=True if TRAIN_MODEL else False
    )

    student_tokenizer.padding_side = "right"
    student_tokenizer.truncation_side = "right"

    teacher_tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_DIR)
    teacher_tokenizer.padding_side = "right"
    teacher_tokenizer.truncation_side = "right"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)

    # ========== Load T5 Teacher Model ==========
    teacher_model = None
    if not USE_CACHED_EMBEDDINGS:
        print("Loading T5-xxl model...")
        teacher_model = T5EncoderModel.from_pretrained(
            T5_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        teacher_model.eval()
    else:
        base_name = os.path.basename(DATASET_DIR)
        cache_folder = os.path.join(CACHE_PATH, base_name)
        validation_file = os.path.join(cache_folder, f"{base_name}.validation")
        enhanced_base_name = os.path.basename(ENHANCED_DATASET_DIR)
        enhanced_cache_folder = os.path.join(CACHE_PATH, enhanced_base_name)
        enhanced_validation_file = os.path.join(enhanced_cache_folder, f"{enhanced_base_name}.validation")
        if not os.path.exists(validation_file) or (ENHANCED_DATASET and not os.path.exists(enhanced_validation_file)):
            teacher_model = T5EncoderModel.from_pretrained(
                T5_MODEL_DIR,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            teacher_model.eval()

    # Initialize loss functions
    align_loss_fn = None
    sequence_loss_fn = None

    if TEXT_MATCH_HUBER_WEIGHT > 0 or TEXT_MATCH_COSINE_WEIGHT > 0:
        align_loss_fn = AlignmentLoss(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            text_huber_weight=TEXT_MATCH_HUBER_WEIGHT,
            text_cosine_weight=TEXT_MATCH_COSINE_WEIGHT,
            split_huber_weight=SPLIT_TOKEN_HUBER_WEIGHT,
            split_cosine_weight=SPLIT_TOKEN_COSINE_WEIGHT
        ).to(device, dtype=torch.bfloat16)

    if SEQUENCE_HUBER_WEIGHT > 0 or SEQUENCE_COSINE_WEIGHT > 0:
        sequence_loss_fn = SequenceLoss(
            huber_weight=SEQUENCE_HUBER_WEIGHT,
            cosine_weight=SEQUENCE_COSINE_WEIGHT
        ).to(device, dtype=torch.bfloat16)

    # ========== Dataset and Dataloader ==========
    train_dataset = PreTokenizedDataset(
        DATASET_DIR,
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
        EVALUATION_DATASET_DIR if USE_SEPARATE_EVALUATION_DATASET else DATASET_DIR,
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
                with open(os.path.join(QWEN3_MODEL_DIR, "config.json"), "r") as f:
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
            new_layer_exists = False
            if TRAIN_PROJECTION:
                for projection_layer in projection_layers:
                    for p in projection_layer.parameters():
                        if p.requires_grad:
                            projection_parameters.append(p)
                    if layer.is_new == True:
                        new_layer_exists = True
                projection_optimizer, scheduler_projection = initialize_optimizer(projection_parameters, MAX_LEARNING_RATE_PROJ, MIN_LEARNING_RATE_PROJ)

            model_optimizer = None
            scheduler_model = None
            if TRAIN_MODEL:
                model_parameters = [p for p in student_model.parameters() if p.requires_grad]
                model_optimizer, scheduler_model = initialize_optimizer(model_parameters, MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)

            if REUSE_OPTIMIZER_STATE and load_optimizer_states(QWEN3_MODEL_DIR, model_optimizer, scheduler_model, projection_optimizer, scheduler_projection, new_layer_exists):
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

                        if TRAIN_MODEL:
                            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                                student_outputs = student_model(
                                    input_ids=s_input_ids,
                                    attention_mask=s_mask,
                                    output_hidden_states=True
                                )
                                student_hidden = student_outputs.hidden_states[-1]
                                projected_student = student_hidden

                                for layer in projection_layers:
                                    projected_student = layer(projected_student)
                        else:
                            # Use no_grad and only get last hidden state when not training model
                            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                                student_outputs = student_model(
                                    input_ids=s_input_ids,
                                    attention_mask=s_mask,
                                    output_hidden_states=True
                                )
                                student_hidden = student_outputs.hidden_states[-1]

                            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                                projected_student = student_hidden
                                for layer in projection_layers:
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

                        num_extra_padding = random.randint(MIN_EXTRA_MATCHED_PADDING, MAX_EXTRA_MATCHED_PADDING)

                        # Modify masks to randomly attend to extra padding tokens
                        s_mask = modify_mask_to_attend_padding(s_mask, 512, num_extra_padding=num_extra_padding)
                        t_mask = modify_mask_to_attend_padding(t_mask, 512, num_extra_padding=num_extra_padding)

                        total_loss = torch.tensor(0.0, device=device)
                        align_loss_huber = torch.tensor(0.0, device=device)
                        align_loss_cos = torch.tensor(0.0, device=device)
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

                        if sequence_loss_fn is not None:
                            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
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
                            T5_MODEL_DIR,
                            torch_dtype=torch.bfloat16,
                            device_map="auto"
                        )
                        teacher_model.eval()

                    eval_metrics = evaluate_model(
                        student_model, eval_dataloader, projection_layers,
                        align_loss_fn, sequence_loss_fn,
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
