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
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.DEBUG,
)
# ========== Configuration ==========
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt"
T5_MODEL_NAME = "/home/naff/q3-xxs_script/t5-xxl"
QWEN3_MODEL_NAME = "/mnt/f/models/Qwen3-Embedding-0.6B/"
OUTPUT_DIR = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/kibou/QT-embedder-initialize/"

USE_CACHED_EMBEDDINGS = True # 4MB per cached T5-xxl embedding, so be mindful of available capacity when using this. It's recommended though given the prohibitive size of the T5-xxl model
CACHE_PATH = "/mnt/f/q5_xxs_training_script/cache2"
PREFETCH_FACTOR = 16

USE_SEPARATE_EVALUATION_DATASET = True
EVALUATION_DATASET_PATH = "/mnt/f/q5_xxs_training_script/eval_prompts.txt"

ENABLE_LOGGING = True
WRITE_TO_LOG_EVERY_X_STEPS = 10

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
GRAD_CLIP = 5.0

EPOCHS = 1

MAX_LEARNING_RATE_MODEL = 1e-5
MIN_LEARNING_RATE_MODEL = 1e-6
MAX_LEARNING_RATE_PROJECTION = 10e-5
MIN_LEARNING_RATE_PROJECTION = 10e-6

SAVE_EVERY_X_STEPS = 0
SAVE_EVERY_X_RESTARTS = 1
SAVE_EVERY_X_EPOCHS = 1

PRINT_EVERY_X_STEPS = 1
EVAL_EVERY_X_EPOCHS = 1
SAVE_BEST_MODEL = True
'''
Token loss compares tokens 1:1 (or with token alignment), sequence loss uses mean pooling
Token loss is most useful for strict prompt following, so I use it as the main loss condition
Sequence loss seems to help as a secondary loss condition with aesthetic quality and some aspects of prompt following
'''
TOKEN_HUBER_LOSS = 0.50
TOKEN_COSINE_LOSS = 0.20
SEQUENCE_HUBER_LOSS = 0.20
SEQUENCE_COSINE_LOSS = 0.10

WARMUP_STEPS = 500 # Warmup steps occur prior to the restart cycle
RESTART_CYCLE_STEPS = 2000 #
ALIGNMENT_STEPS = 250 # This is not additive to the step count, unlike the above two. I'd set it to half of your warmup steps, or something like that. This is necessary to prevent degradation and I recommend repeating it, and warmup, for every restart using the options below. I need to test more to see if other options would work better, but for now this seems to work well enough. Note: this setting does nothing with purely sequence loss, and isn't necessary, because sequence loss does not cause degradation in the way that 1:1 token loss does

REPEAT_WARMUP_AFTER_RESTART = True
REPEAT_ALIGNMENT_AFTER_RESTART = True

SHUFFLE_DATASET = True # Random order is better but may introduce random read bottlenecking, especially if you have a crappy SMR drive (lol)

TRAIN_PROJECTION = True
TRAIN_MODEL = True

LOG_VRAM_USAGE = False

AUTO_LAYER_INIT_TRAINING = True  # If enabled, trains layer-by-layer, iterating over restarts in an entirely restart-based, epoch-agnostic training regime. This is recommended for initializing the layer array, because training multiple layers from scratch is unstable and will lead to bad results. Training will end when all layers have been trained together in a final restart cycle. Note that your settings for restart step, alignment steps, warmup steps are used here. Only run this once, and then disable for subsequent training of the now-initialized projection layers
AUTO_LAYER_INIT_TRAINING_LR_SCALER = 3.0  # Scale the max LR for projection training higher for earlier layers when using automatic training, with linear degradation of the scaling rate towards 1.0 at the end of the run
'''
Most of these are self explanatory. "auto" for input aligns to previous output dim, "auto" for output aligns to previous transformer dim
Using file_num you can bolt on new layers anywhere in the chain
Using omit_output_mlp removes the activation and final linear from the transformer
Using omit_output linear similarly just removes the final linear
Transformer architecture is linear -> transformer_encoder -> linear -> activation -> linear
Each layer starts with a linear, so when daisy-chaining we can omit the redundant linear or full MLP (the lin->act->lin sandwich)
If the output of the final layer is not 4096, or omit_output_mlp is True for the final layer, an additional MLP is bolted on and trained, to make the output usable. This extra MLP will be removed whenever a new layer is to be added when using automatic training, unless its dimensions still match
Oh, and if you omit the output MLP, output_dim will be overridden to "auto" behaviour ie. matching the transformer dim
As well as transformer type, you can bolt on MLP or linear
MLP takes input_dim, activation_dim, output_dim and omit_output_linear
Linear takes input_dim and output_dim
'''
PROJECTION_LAYERS_CONFIG = [
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 1024,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 1,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 1024,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 2,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 1024,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 3,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 2048,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 4,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 2048,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 5,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 2048,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": True,
        "omit_output_linear": True,
        "file_num": 6,
    },
    {
        "type": "transformer",
        "input_dim": "auto",
        "transformer_dim": 4096,
        "output_dim": "auto",
        "num_layers": 1,
        "dim_feedforward": 2048,
        "omit_output_mlp": False,
        "omit_output_linear": False,
        "file_num": 7,
    },
]

# ========== Advanced Configuration ==========
EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS = [] # Use the file_num, not the index

ENHANCED_DATASET = True # You can use a second dataset file and swap embeddings. Of dubious use, but whatever
ENHANCED_DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"

UNTAMPERED_STUDENT_AND_TEACHER_RATIO = 0.50 # Both use normal prompt/embedding
ENHANCED_TEACHER_EMBEDDING_RATIO = 0.00 # Teacher embedding is swapped for enhanced, but student is normal
ENHANCED_STUDENT_AND_TEACHER_RATIO = 0.50 # Both use enhanced prompt/embedding

ENABLE_STUDENT_WORD_DROPOUT = False # Probably token dropout is a better option
STUDENT_WORD_DROPOUT_RATIO = 0.10

ENABLE_STUDENT_TOKEN_DROPOUT = False # Teachers the model to generally infer complete sequence from incomplete sequence; also encourages over-projection
STUDENT_TOKEN_DROPOUT_RATIO = 0.10

SKIP_DROPOUT_IF_NORMAL_STUDENT_ENHANCED_TEACHER = True # No reason to dropout in this event

TOKEN_ALIGNMENT_WINDOW = 5

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
                self.activation_dim = 4096

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
                self.activation_dim = teacher_model.config.hidden_size

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
            pass

        self.use_cached_embeddings = use_cached_embeddings
        self.line_index = list(range(len(self.lines)))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        def apply_dropout(line, word_dropout_ratio, token_dropout_ratio, tokenizer=None):
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

            if choice == 0:  # PRIMARY_DATASET_RATIO
                student_line = self.student_raw_lines[idx]
                teacher_line = self.teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                student_dropout_token = STUDENT_TOKEN_DROPOUT_RATIO if ENABLE_STUDENT_TOKEN_DROPOUT else 0
                teacher_type = "original"

            elif choice == 1:  # ENHANCED_EMBEDDING_RATIO
                student_line = self.student_raw_lines[idx]
                teacher_line = self.enhanced_teacher_raw_lines[idx]
                student_dropout_word = STUDENT_WORD_DROPOUT_RATIO if ENABLE_STUDENT_WORD_DROPOUT else 0
                student_dropout_token = STUDENT_TOKEN_DROPOUT_RATIO if ENABLE_STUDENT_TOKEN_DROPOUT else 0
                teacher_type = "enhanced"

            elif choice == 2:  # ENHANCED_PROMPT_AND_EMBEDDING_RATIO
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

# ========== Projection Layers ==========
class LinearProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        return self.linear(x)

class MLPProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, activation_dim, output_dim, omit_output_linear=False):
        super().__init__()
        self.linear_in = torch.nn.Linear(input_dim, activation_dim)
        self.activation = torch.nn.GELU()
        if omit_output_linear:
            self.linear_out = None
        else:
            self.linear_out = torch.nn.Linear(activation_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.activation(x)
        if self.linear_out is not None:
            x = self.linear_out(x)
        return x

class TransformerProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_dim, output_dim, num_layers, dim_feedforward, omit_output_mlp=False, omit_output_linear=False):
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
        if omit_output_mlp:
            self.linear = None
        else:
            self.linear = torch.nn.Linear(transformer_dim, output_dim)
            self.activation = torch.nn.GELU()
            if omit_output_linear:
                self.linear_out = None
            else:
                self.linear_out = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.transformer(x)
        if self.linear is not None:
            x = self.linear(x)
            x = self.activation(x)
            if self.linear_out is not None:
                x = self.linear_out(x)
        return x

# ========== Token Alignment ==========
def ids_to_tokens(token_ids, tokenizer):
    return tokenizer.convert_ids_to_tokens(token_ids)

def normalize_token(token):
    token = token.replace('Ġ', '')
    token = token.replace('▁', '')

    token = token.replace('▔', '')
    token = token.replace('▃', '')
    token = token.replace('�', '')

    token = token.replace(' .', '.')
    token = token.replace(' ', ',')
    token = token.replace(' !', '!')
    token = token.replace(' ?', '?')
    token = token.replace(' :', ':')
    token = token.replace(' ;', ';')

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
    student_tokens = ids_to_tokens(student_input_ids.cpu().numpy(), student_tokenizer)
    teacher_tokens = ids_to_tokens(teacher_input_ids.cpu().numpy(), teacher_tokenizer)

    aligned_pairs = []

    normalized_student = [normalize_token(token) for token in student_tokens]
    normalized_teacher = [normalize_token(token) for token in teacher_tokens]

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
def position_based_alignment(teacher_mask):
    if teacher_mask.dtype != torch.bool:
        teacher_mask = teacher_mask.bool()

    attended_positions = torch.where(teacher_mask)[0]
    aligned_pairs = [(pos.item(), pos.item()) for pos in attended_positions]
    return aligned_pairs

class HybridLoss(torch.nn.Module):
    def __init__(self, huber_weight=0.7, cosine_weight=0.3, seq_huber_weight=0.0,
                 seq_cosine_weight=0.0, huber_delta=1.0,
                 student_tokenizer=None, teacher_tokenizer=None,
                 alignment_steps=10, warmup_steps=500):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='none')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.alignment_steps = alignment_steps
        self.seq_huber_weight = seq_huber_weight
        self.seq_cosine_weight = seq_cosine_weight

    def reset_step(self):
        self.global_step = 0

    def update_global_step(self, step):
        self.global_step = step

    def forward(self, student_output, teacher_output, student_mask, teacher_mask,
                student_input_ids=None, teacher_input_ids=None):
        device = student_output.device
        batch_size = student_output.size(0)

        token_huber = torch.tensor(0.0, device=device)
        token_cos_loss = torch.tensor(0.0, device=device)
        num_aligned_tokens = 0
        num_aligned_seqs = 0
        num_valid_seqs = 0

        for i in range(batch_size):
            if self.global_step < self.alignment_steps and (self.huber_weight > 0 or self.cosine_weight > 0):
                # Token-based alignment
                aligned_pairs = token_based_alignment(
                    student_input_ids[i], teacher_input_ids[i],
                    self.student_tokenizer, self.teacher_tokenizer,
                    window_size=TOKEN_ALIGNMENT_WINDOW
                )
            else:
                # Position-based alignment
                aligned_pairs = position_based_alignment(teacher_mask[i])

            if len(aligned_pairs) == 0:
                continue

            num_aligned_seqs += 1

            student_aligned = []
            teacher_aligned = []
            for pair in aligned_pairs:
                stu_idx, tea_idx = pair
                student_aligned.append(student_output[i, stu_idx])
                teacher_aligned.append(teacher_output[i, tea_idx])

            student_aligned = torch.stack(student_aligned)
            teacher_aligned = torch.stack(teacher_aligned)

            token_huber += self.huber_loss(student_aligned, teacher_aligned).mean()
            token_cos_sim = self.cos_loss(student_aligned, teacher_aligned)
            token_cos_loss += (1 - token_cos_sim).mean()
            num_aligned_tokens += len(aligned_pairs)

        if num_aligned_seqs > 0:
            token_huber /= num_aligned_seqs
            token_cos_loss /= num_aligned_seqs
        else:
            token_huber = torch.tensor(0.0, device=device)
            token_cos_loss = torch.tensor(0.0, device=device)

        # Initialize sequence losses to zero
        seq_huber = torch.tensor(0.0, device=device)
        seq_cos_loss = torch.tensor(0.0, device=device)

        # Only calculate sequence losses if weights are positive
        if self.seq_huber_weight > 0 or self.seq_cosine_weight > 0:
            student_means = []
            teacher_means = []

            for i in range(batch_size):
                stu_attended = student_output[i, teacher_mask[i].bool()]
                tea_attended = teacher_output[i, teacher_mask[i].bool()]

                if len(stu_attended) == 0 or len(tea_attended) == 0:
                    continue

                if stu_attended.dim() != 2 or tea_attended.dim() != 2:
                    continue

                stu_mean = stu_attended.mean(dim=0)
                tea_mean = tea_attended.mean(dim=0)

                student_means.append(stu_mean)
                teacher_means.append(tea_mean)
                num_valid_seqs += 1

            if num_valid_seqs > 0:
                student_means = torch.stack(student_means)
                teacher_means = torch.stack(teacher_means)

                seq_huber = self.huber_loss(student_means, teacher_means).mean()
                seq_cos_sim = self.cos_loss(student_means, teacher_means)
                seq_cos_loss = (1 - seq_cos_sim).mean()

        total_loss = (
            self.huber_weight * token_huber +
            self.cosine_weight * token_cos_loss +
            self.seq_huber_weight * seq_huber +
            self.seq_cosine_weight * seq_cos_loss
        )

        return total_loss, token_huber, token_cos_loss, seq_huber, seq_cos_loss, num_aligned_tokens

class TokenLoss(torch.nn.Module):
    def __init__(self, huber_weight=0.7, cosine_weight=0.3, huber_delta=1.0,
                 student_tokenizer=None, teacher_tokenizer=None,
                 alignment_steps=10, warmup_steps=501):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='none')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.alignment_steps = alignment_steps

    def reset_step(self):
        self.global_step = 0

    def update_global_step(self, step):
        self.global_step = step

    def forward(self, student_output, teacher_output, student_mask, teacher_mask,
                student_input_ids=None, teacher_input_ids=None):
        device = student_output.device
        batch_size = student_output.size(0)

        token_huber = torch.tensor(0.0, device=device)
        token_cos_loss = torch.tensor(0.0, device=device)
        num_aligned_tokens = 0
        num_aligned_seqs = 0

        for i in range(batch_size):
            if self.global_step < self.alignment_steps:
                aligned_pairs = token_based_alignment(
                    student_input_ids[i], teacher_input_ids[i],
                    self.student_tokenizer, self.teacher_tokenizer,
                    window_size=TOKEN_ALIGNMENT_WINDOW
                )
            else:
                aligned_pairs = position_based_alignment(teacher_mask[i])

            if len(aligned_pairs) == 0:
                continue

            num_aligned_seqs += 1

            student_aligned = []
            teacher_aligned = []
            for pair in aligned_pairs:
                stu_idx, tea_idx = pair
                student_aligned.append(student_output[i, stu_idx])
                teacher_aligned.append(teacher_output[i, tea_idx])

            student_aligned = torch.stack(student_aligned)
            teacher_aligned = torch.stack(teacher_aligned)

            token_huber += self.huber_loss(student_aligned, teacher_aligned).mean()
            token_cos_sim = self.cos_loss(student_aligned, teacher_aligned)
            token_cos_loss += (1 - token_cos_sim).mean()
            num_aligned_tokens += len(aligned_pairs)

        if num_aligned_seqs > 0:
            token_huber /= num_aligned_seqs
            token_cos_loss /= num_aligned_seqs
        else:
            token_huber = torch.tensor(0.0, device=device)
            token_cos_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.huber_weight * token_huber +
            self.cosine_weight * token_cos_loss
        )

        return total_loss, token_huber, token_cos_loss, num_aligned_tokens

class SequenceLoss(torch.nn.Module):
    def __init__(self, huber_weight=0.7, cosine_weight=0.3, huber_delta=1.0,
                 student_tokenizer=None, teacher_tokenizer=None,
                 alignment_steps=10, warmup_steps=501):
        super().__init__()
        self.huber_weight = huber_weight
        self.cosine_weight = cosine_weight
        self.huber_loss = torch.nn.HuberLoss(delta=huber_delta, reduction='none')
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.alignment_steps = alignment_steps

    def reset_step(self):
        self.global_step = 0

    def update_global_step(self, step):
        self.global_step = step

    def forward(self, student_output, teacher_output, student_mask, teacher_mask,
                student_input_ids=None, teacher_input_ids=None):
        device = student_output.device
        batch_size = student_output.size(0)

        seq_huber = torch.tensor(0.0, device=device)
        seq_cos_loss = torch.tensor(0.0, device=device)
        num_valid_seqs = 0

        student_means = []
        teacher_means = []

        for i in range(batch_size):
            stu_attended = student_output[i, teacher_mask[i].bool()]
            tea_attended = teacher_output[i, teacher_mask[i].bool()]

            if len(stu_attended) == 0 or len(tea_attended) == 0:
                continue

            if stu_attended.dim() != 2 or tea_attended.dim() != 2:
                continue

            stu_mean = stu_attended.mean(dim=0)
            tea_mean = tea_attended.mean(dim=0)

            student_means.append(stu_mean)
            teacher_means.append(tea_mean)
            num_valid_seqs += 1

        if num_valid_seqs > 0:
            student_means = torch.stack(student_means)
            teacher_means = torch.stack(teacher_means)

            seq_huber = self.huber_loss(student_means, teacher_means).mean()
            seq_cos_sim = self.cos_loss(student_means, teacher_means)
            seq_cos_loss = (1 - seq_cos_sim).mean()
        else:
            seq_huber = torch.tensor(0.0, device=device)
            seq_cos_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.huber_weight * seq_huber +
            self.cosine_weight * seq_cos_loss
        )

        return total_loss, seq_huber, seq_cos_loss, num_valid_seqs


# ========== Projection Function ==========
def get_projection_layers(restart_cycle, layers_to_load):
    projection_layers = []
    file_num = None
    top_file_num = None
    if not AUTO_LAYER_INIT_TRAINING:
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
            input_dim = output_dim_prev

        # Override if specified in config
        if layer_config.get("input_dim", "auto") != "auto":
            input_dim = layer_config["input_dim"]

        # Handle output_dim="auto"
        if layer_config.get("output_dim", "auto") == "auto" or layer_config["omit_output_mlp"]:
            if layer_config["type"] == "mlp":
                output_dim = layer_config["activation_dim"]
            elif layer_config["type"] == "transformer":
                output_dim = layer_config["transformer_dim"]
            else:
                if "output_dim" not in layer_config or layer_config["output_dim"] == "auto":
                    raise ValueError("Linear layer must specify output_dim")
                output_dim = layer_config["output_dim"]
        else:
            output_dim = layer_config["output_dim"]

        output_dim_prev = output_dim

        if file_num is not None:
            if layer_config["file_num"] > file_num:
                top_file_num = layer_config["file_num"]
        else:
            top_file_num = layer_config["file_num"]
        file_num = layer_config["file_num"]

        # Load or initialize layer based on type
        if layer_config["type"] == "linear":
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = LinearProjectionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                projection_layer.load_state_dict(state_dict)
            else:
                projection_layer = LinearProjectionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim
                )
        elif layer_config["type"] == "mlp":
            activation_dim = layer_config["activation_dim"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = MLPProjectionLayer(
                    input_dim=input_dim,
                    activation_dim=activation_dim,
                    output_dim=output_dim,
                    omit_output_linear=layer_config.get("omit_output_linear", False)
                )
                projection_layer.load_state_dict(state_dict)
            else:
                projection_layer = MLPProjectionLayer(
                    input_dim=input_dim,
                    activation_dim=activation_dim,
                    output_dim=output_dim,
                    omit_output_linear=layer_config.get("omit_output_linear", False)
                )
        elif layer_config["type"] == "transformer":
            transformer_dim = layer_config["transformer_dim"]
            num_layers = layer_config["num_layers"]
            dim_feedforward = layer_config["dim_feedforward"]
            if os.path.exists(layer_path):
                state_dict = load_file(layer_path)
                projection_layer = TransformerProjectionLayer(
                    input_dim=input_dim,
                    transformer_dim=transformer_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    omit_output_mlp=layer_config.get("omit_output_mlp", False),
                    omit_output_linear=layer_config.get("omit_output_linear", False)
                )
                projection_layer.load_state_dict(state_dict)
            else:
                projection_layer = TransformerProjectionLayer(
                    input_dim=input_dim,
                    transformer_dim=transformer_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    omit_output_mlp=layer_config.get("omit_output_mlp", False),
                    omit_output_linear=layer_config.get("omit_output_linear", False)
                )
        projection_layer.file_num = layer_config["file_num"]
        projection_layer.output_dim = output_dim
        projection_layers.append(projection_layer)

    # Add final linear layer if output dim is not 4096
    if output_dim != 4096:
        print("Adding final MLP layer to match teacher dimension")
        final_mlp = MLPProjectionLayer(
            input_dim=output_dim,
            activation_dim=4096,
            output_dim=4096
        )
        final_mlp.is_extra = True
        final_mlp.input_dim = output_dim
        final_mlp.file_num = top_file_num+1
        projection_layers.append(final_mlp)

    return projection_layers, layers_to_load

def update_projection_layers(restart_cycle, layers_to_load):
    if restart_cycle % 2 == 0:
        layers_to_load += 1
    len_offset = 0
    if len(projection_layers) > 0 and hasattr(projection_layers[-1], 'is_extra'):
        len_offset = 1
    if layers_to_load + len_offset > len(projection_layers):
        if len(projection_layers) > 0 and hasattr(projection_layers[-1], 'is_extra') and projection_layers[-1].is_extra:
            dropped_layer = projection_layers[-1]
            projection_layers.pop(-1)
        else:
            dropped_layer = None

        output_dim_prev = None
        if len(projection_layers) > 0:
            idx = len(projection_layers) - 1
            while idx >= 0 and hasattr(projection_layers[idx], 'is_extra'):
                idx -= 1
            if idx >= 0:
                output_dim_prev = projection_layers[idx].output_dim
            else:
                output_dim_prev = qwen_embedding_dim
        else:
            output_dim_prev = qwen_embedding_dim

        file_num = None
        top_file_num = None
        for i in range(len(projection_layers), layers_to_load + 1):
            layer_config = PROJECTION_LAYERS_CONFIG[i-1]
            layer_num = layer_config["file_num"]
            layer_path = os.path.join(QWEN3_MODEL_NAME, f"projection_layer_{layer_num}.safetensors")

            if i == 1:
                input_dim = qwen_embedding_dim
            else:
                input_dim = output_dim_prev

            if layer_config.get("input_dim", "auto") != "auto":
                input_dim = layer_config["input_dim"]

            if layer_config.get("output_dim", "auto") == "auto" or layer_config["omit_output_mlp"]:
                if layer_config["type"] == "mlp":
                    output_dim = layer_config["activation_dim"]
                elif layer_config["type"] == "transformer":
                    output_dim = layer_config["transformer_dim"]
                else:
                    if "output_dim" not in layer_config or layer_config["output_dim"] == "auto":
                        raise ValueError("Linear layer must specify output_dim")
                    output_dim = layer_config["output_dim"]
            else:
                output_dim = layer_config["output_dim"]

            output_dim_prev = output_dim

            if file_num is not None:
                if layer_config["file_num"] > file_num:
                    top_file_num = layer_config["file_num"]
            file_num = layer_config["file_num"]

            if layer_config["type"] == "linear":
                if os.path.exists(layer_path):
                    state_dict = load_file(layer_path)
                    projection_layer = LinearProjectionLayer(
                        input_dim=input_dim,
                        output_dim=output_dim
                    )
                    projection_layer.load_state_dict(state_dict)
                else:
                    projection_layer = LinearProjectionLayer(
                        input_dim=input_dim,
                        output_dim=output_dim
                    )
            elif layer_config["type"] == "mlp":
                activation_dim = layer_config["activation_dim"]
                if os.path.exists(layer_path):
                    state_dict = load_file(layer_path)
                    projection_layer = MLPProjectionLayer(
                        input_dim=input_dim,
                        activation_dim=activation_dim,
                        output_dim=output_dim,
                        omit_output_linear=layer_config.get("omit_output_linear", False)
                    )
                    projection_layer.load_state_dict(state_dict)
                else:
                    projection_layer = MLPProjectionLayer(
                        input_dim=input_dim,
                        activation_dim=activation_dim,
                        output_dim=output_dim,
                        omit_output_linear=layer_config.get("omit_output_linear", False)
                    )
            elif layer_config["type"] == "transformer":
                transformer_dim = layer_config["transformer_dim"]
                num_layers = layer_config["num_layers"]
                dim_feedforward = layer_config["dim_feedforward"]
                if os.path.exists(layer_path):
                    state_dict = load_file(layer_path)
                    projection_layer = TransformerProjectionLayer(
                        input_dim=input_dim,
                        transformer_dim=transformer_dim,
                        output_dim=output_dim,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        omit_output_mlp=layer_config.get("omit_output_mlp", False),
                        omit_output_linear=layer_config.get("omit_output_linear", False)
                    )
                    projection_layer.load_state_dict(state_dict)
                else:
                    projection_layer = TransformerProjectionLayer(
                        input_dim=input_dim,
                        transformer_dim=transformer_dim,
                        output_dim=output_dim,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        omit_output_mlp=layer_config.get("omit_output_mlp", False),
                        omit_output_linear=layer_config.get("omit_output_linear", False)
                    )
            projection_layer.file_num = layer_config["file_num"]
            projection_layer.output_dim = output_dim
            projection_layers.append(projection_layer)
            projection_layer.to(device, dtype=torch.bfloat16)

        if output_dim != 4096:
            if dropped_layer is not None and dropped_layer.input_dim == output_dim:
                print(f"Re-using previous final MLP layer due to matching dimension\n")
                final_mlp = dropped_layer
            else:
                print(f"Adding final MLP layer to match teacher dimension\n")
                final_mlp = MLPProjectionLayer(
                    input_dim=output_dim,
                    activation_dim=4096,
                    output_dim=4096
                )
                # Move new MLP to device and dtype
                final_mlp.to(device, dtype=torch.bfloat16)
            final_mlp.is_extra = True
            final_mlp.input_dim = output_dim
            final_mlp.file_num = top_file_num+1
            projection_layers.append(final_mlp)

    return projection_layers, layers_to_load

def get_projection_parameters(projection_layers, restart_cycle):
    if AUTO_LAYER_INIT_TRAINING:
        if restart_cycle % 2 == 0:  # Even restart: train only last layer
            # Start from last non-extra layer
            start_idx = len(projection_layers) - 1
            while start_idx >= 0 and hasattr(projection_layers[start_idx], 'is_extra'):
                start_idx -= 1

            if start_idx < 0:
                start_idx = 0

            return [p for layer in projection_layers[start_idx:]
                    for p in layer.parameters()
                    if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS]
        else:  # Odd restart: train all layers
            return [p for layer in projection_layers
                    for p in layer.parameters()
                    if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS]
    else:
        return [p for layer in projection_layers
                for p in layer.parameters()
                if layer.file_num not in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS]

# ========== Optimiser Initialisation ==========
def initialize_optimizer(max_lr, min_lr):
    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
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
            total_iters=total_steps - WARMUP_STEPS
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
def evaluate_model(model, dataloader, projection_layers, loss_fn, device, autocast_dtype):
    model.eval()
    for layer in projection_layers:
        layer.eval()
    loss_fn.eval()

    if SEQUENCE_COSINE_LOSS <= 0 and SEQUENCE_HUBER_LOSS <= 0:
        total_losses = {
            'total': 0.0,
            'seq_huber': 0.0,
            'seq_cos': 0.0
        }
    elif TOKEN_COSINE_LOSS <= 0 and TOKEN_HUBER_LOSS <= 0:
        total_losses = {
            'total': 0.0,
            'huber': 0.0,
            'cos': 0.0
        }
    else:
        total_losses = {
            'total': 0.0,
            'huber': 0.0,
            'cos': 0.0,
            'seq_huber': 0.0,
            'seq_cos': 0.0
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                if USE_CACHED_EMBEDDINGS:
                    s_input_ids, s_mask, t_input_ids, t_embeddings, t_mask = batch
                else:
                    s_input_ids, s_mask, t_input_ids, t_mask = batch

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
                if SEQUENCE_COSINE_LOSS <= 0 and SEQUENCE_HUBER_LOSS <= 0:
                    loss, huber_loss, cos_loss, num_aligned_tokens = loss_fn(
                        projected_student,
                        teacher_hidden,
                        s_mask,
                        t_mask,
                        student_input_ids=s_input_ids,
                        teacher_input_ids=t_input_ids
                    )
                elif TOKEN_COSINE_LOSS <= 0 and TOKEN_HUBER_LOSS <= 0:
                    loss, seq_huber_loss, seq_cos_loss, num_aligned_tokens = loss_fn(
                        projected_student,
                        teacher_hidden,
                        s_mask,
                        t_mask,
                        student_input_ids=s_input_ids,
                        teacher_input_ids=t_input_ids
                    )
                else:
                    loss, huber_loss, cos_loss, seq_huber_loss, seq_cos_loss, num_aligned_tokens = loss_fn(
                        projected_student,
                        teacher_hidden,
                        s_mask,
                        t_mask,
                        student_input_ids=s_input_ids,
                        teacher_input_ids=t_input_ids
                    )

                total_losses['total'] += loss.item()
                if TOKEN_COSINE_LOSS > 0 or TOKEN_HUBER_LOSS > 0:
                    total_losses['huber'] += huber_loss.item()
                    total_losses['cos'] += cos_loss.item()
                if SEQUENCE_COSINE_LOSS > 0 or SEQUENCE_HUBER_LOSS > 0:
                    total_losses['seq_huber'] += seq_huber_loss.item()
                    total_losses['seq_cos'] += seq_cos_loss.item()

                del projected_student, student_hidden, student_outputs
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
    for layer in projection_layers:
        layer.train()
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

def save_trained_model(save_path, model, tokenizer, projection_layers, qwen_embedding_dim):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    for layer in projection_layers:
        layer_state = layer.state_dict()
        layer_path = os.path.join(save_path, f"projection_layer_{layer.file_num}.safetensors")
        save_file(layer_state, layer_path)

    projection_config_path = os.path.join(save_path, "projection_config.json")
    save_projection_config(projection_config_path, qwen_embedding_dim)

def save_projection_config(projection_config_path, embedding_dim):
    if AUTO_LAYER_INIT_TRAINING and restart_cycle <= total_restarts:
        projection_config = {
            "layers": PROJECTION_LAYERS_CONFIG,
            "restart_cycle": restart_cycle,
        }
    else:
        projection_config = {
            "layers": PROJECTION_LAYERS_CONFIG,
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
    model_gn_line = ""
    model_lr_line = ""
    proj_gn_line = ""
    proj_lr_line = ""
    seq_huber_line = ""
    seq_cos_line = ""
    huber_line = ""
    cos_line = ""
    vram_line = ""
    batch_line = ""
    step_line = ""
    epoch_line = ""
    layers_line = ""
    if TRAIN_MODEL:
        model_gn_line = f"Grad Norm Model: {grad_norm_model:.6f}, "
        model_lr_line = f"Model LR: {current_lr_model:.6f}, "
    if TRAIN_PROJECTION:
        proj_gn_line = f"Grad Norm Projection: {grad_norm_proj:.6f}, "
        proj_lr_line = f"Projection LR: {current_lr_proj:.6f}, "
    if SEQUENCE_HUBER_LOSS > 0 and current_seq_huber > 0:
        seq_huber_line = f"SQ Huber: {current_seq_huber:.6f}, "
    if SEQUENCE_COSINE_LOSS > 0 and current_seq_cos > 0:
        seq_cos_line = f"SQ Cosine: {current_seq_cos:.6f}, "
    if TOKEN_HUBER_LOSS > 0 and current_huber > 0:
        huber_line = f"PT Huber: {current_huber:.6f}, "
    if TOKEN_COSINE_LOSS > 0 and current_cos > 0:
        cos_line = f"PT Cosine: {current_cos:.6f}, "
    if LOG_VRAM_USAGE:
        vram_line = f"VRAM Usage: {vram_used:.0f}MiB / {vram_total:.0f}MiB, "
    if AUTO_LAYER_INIT_TRAINING:
        epoch_line = f"Restart Cycle: {restart_cycle}/{total_restarts}, "
        step_line = f"Step: {global_step}/{total_steps}, "
        layers_line = f"Layers Training: {layers_to_load}/{len(PROJECTION_LAYERS_CONFIG)}, "
    else:
        epoch_line = f"Epoch [{epoch + 1}/{EPOCHS}], "
        step_line = f"Step: {global_step}/{total_steps}, "
        batch_line = f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
    log_line = (
        f"{epoch_line}"
        f"{batch_line}"
        f"{layers_line}"
        f"{step_line}"
        f"Total Loss: {current_loss:.6f}, "
        f"{huber_line}"
        f"{cos_line}"
        f"{seq_huber_line}"
        f"{seq_cos_line}"
        f"{model_gn_line}"
        f"{proj_gn_line}"
        f"{model_lr_line}"
        f"{proj_lr_line}"
        f"{vram_line}"
        f"Elapsed: {elapsed/60:.1f} min, "
        f"ETA: {eta/60:.1f} min"
    )
    return log_line

# ========== Auto Init Training Setup ==========
# Store original max learning rates for automatic training scaling
if AUTO_LAYER_INIT_TRAINING:
    ORIGINAL_MAX_LEARNING_RATE_PROJECTION = MAX_LEARNING_RATE_PROJECTION
    MAX_LEARNING_RATE_PROJECTION = ORIGINAL_MAX_LEARNING_RATE_PROJECTION * AUTO_LAYER_INIT_TRAINING_LR_SCALER
    # Hacky method for making the epoch number irrevelant for restart-based training, lol
    EPOCHS = 999999999

# ========== Load Qwen3 Model ==========
gc.collect()
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

losses = [TOKEN_HUBER_LOSS, TOKEN_COSINE_LOSS, SEQUENCE_HUBER_LOSS, SEQUENCE_COSINE_LOSS]
sum_loss = sum(losses)
normalised_losses = [loss / sum_loss for loss in losses]

if SEQUENCE_HUBER_LOSS <= 0 and SEQUENCE_COSINE_LOSS <= 0:
    if TOKEN_HUBER_LOSS <= 0 and TOKEN_COSINE_LOSS <= 0:
        loss_fn = SequenceLoss(
            huber_weight=normalised_losses[2],
            cosine_weight=normalised_losses[3],
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            alignment_steps=ALIGNMENT_STEPS,
            warmup_steps=WARMUP_STEPS
        ).to(device, dtype=torch.bfloat16)
    else:
        loss_fn = TokenLoss(
            huber_weight=normalised_losses[0],
            cosine_weight=normalised_losses[1],
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            alignment_steps=ALIGNMENT_STEPS,
            warmup_steps=WARMUP_STEPS
        ).to(device, dtype=torch.bfloat16)
elif TOKEN_COSINE_LOSS <= 0 and TOKEN_HUBER_LOSS <= 0:
    loss_fn = SequenceLoss(
        huber_weight=normalised_losses[2],
        cosine_weight=normalised_losses[3],
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        alignment_steps=ALIGNMENT_STEPS,
        warmup_steps=WARMUP_STEPS
    ).to(device, dtype=torch.bfloat16)
else:
    loss_fn = HybridLoss(
        huber_weight=normalised_losses[0],
        cosine_weight=normalised_losses[1],
        seq_huber_weight=normalised_losses[2],
        seq_cosine_weight=normalised_losses[3],
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        alignment_steps=ALIGNMENT_STEPS,
        warmup_steps=WARMUP_STEPS
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

        # Calculate total restarts for automatic training scaling
        if AUTO_LAYER_INIT_TRAINING:
            total_restarts = 1
            for i in range(2, len(PROJECTION_LAYERS_CONFIG)+1):
                total_restarts += 2
        else:
            total_restarts = 0

        # Define cycle length for restart-based training
        if REPEAT_WARMUP_AFTER_RESTART:
            cycle_length = WARMUP_STEPS + RESTART_CYCLE_STEPS
        else:
            cycle_length = RESTART_CYCLE_STEPS
        if AUTO_LAYER_INIT_TRAINING:
            total_steps = total_restarts * cycle_length
        else:
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
        projection_layers, layers_to_load = get_projection_layers(restart_cycle, layers_to_load)
        for layer in projection_layers:
            layer.to(device, dtype=torch.bfloat16)

        # Set excluded layers to not require grad
        for layer_idx in EXCLUDE_TRAINING_PROJECTION_LAYER_NUMS:
            if layer_idx <= len(projection_layers):
                for param in projection_layers[layer_idx-1].parameters():
                    param.requires_grad = False

        projection_parameters = get_projection_parameters(projection_layers, restart_cycle)

        if ENABLE_LOGGING:
            log_dir = os.path.join(OUTPUT_DIR, "logging")
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            log_filename = f"training_log_{current_time}.txt"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, log_filename)
            log_lines = []

        if TRAIN_MODEL:
            model_optimizer, scheduler_model = initialize_optimizer(MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)

        if TRAIN_PROJECTION:
            projection_optimizer, scheduler_proj = initialize_optimizer(MAX_LEARNING_RATE_PROJECTION, MIN_LEARNING_RATE_PROJECTION)

        # ========== Training Loop ==========
        student_model.train()

        start_time = time.time()
        eval_delta_time = 0
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            if TRAIN_MODEL: model_optimizer.zero_grad()
            if TRAIN_PROJECTION: projection_optimizer.zero_grad()

            steps_completed_this_epoch = 0
            accumulation_step = 0
            if AUTO_LAYER_INIT_TRAINING:
                print(f"Starting automatic layer initialization training, total restarts: {total_restarts}")
            else:
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
                            del teacher_outputs

                    if SEQUENCE_COSINE_LOSS <= 0 and SEQUENCE_HUBER_LOSS <= 0:
                        loss, huber_loss, cos_loss, num_aligned_tokens = loss_fn(
                            projected_student,
                            teacher_hidden,
                            s_mask,
                            t_mask,
                            student_input_ids=s_input_ids,
                            teacher_input_ids=t_input_ids
                        )
                    elif TOKEN_COSINE_LOSS <= 0 and TOKEN_HUBER_LOSS <= 0:
                        loss, seq_huber_loss, seq_cos_loss, num_aligned_tokens = loss_fn(
                            projected_student,
                            teacher_hidden,
                            s_mask,
                            t_mask,
                            student_input_ids=s_input_ids,
                            teacher_input_ids=t_input_ids
                        )
                    else:
                        loss, huber_loss, cos_loss, seq_huber_loss, seq_cos_loss, num_aligned_tokens = loss_fn(
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
                        grad_norm_model = clip_grad_norm_(
                            [p for p in student_model.parameters() if p.requires_grad],
                            max_norm=GRAD_CLIP
                        )
                        grad_norm_proj = clip_grad_norm_(
                            projection_parameters if TRAIN_PROJECTION else [],
                            max_norm=GRAD_CLIP
                        )

                        if TRAIN_MODEL: scaler.step(model_optimizer)
                        if TRAIN_PROJECTION: scaler.step(projection_optimizer)
                        scaler.update()

                        global_step += 1
                        steps_completed_this_epoch += 1

                        accumulation_step = 0
                        if TRAIN_MODEL: model_optimizer.zero_grad()
                        if TRAIN_PROJECTION: projection_optimizer.zero_grad()

                        current_loss = loss.item()
                        if TOKEN_COSINE_LOSS <= 0 and TOKEN_HUBER_LOSS <= 0:
                            current_huber = 0
                            current_cos = 0
                        else:
                            current_huber = huber_loss.item()
                            current_cos = cos_loss.item()
                        if SEQUENCE_COSINE_LOSS <= 0 and SEQUENCE_HUBER_LOSS <= 0:
                            current_seq_huber = 0
                            current_seq_cos = 0
                        else:
                            current_seq_huber = seq_huber_loss.item()
                            current_seq_cos = seq_cos_loss.item()

                        elapsed = time.time() - start_time - eval_delta_time
                        remaining_steps = total_steps - global_step
                        eta = (elapsed / global_step) * remaining_steps if global_step > 0 else 0
                        vram_free, vram_total, vram_used = get_memory_usage()
                        if TRAIN_MODEL: current_lr_model = model_optimizer.param_groups[0]['lr']
                        if TRAIN_PROJECTION: current_lr_proj = projection_optimizer.param_groups[0]['lr']

                        if PRINT_EVERY_X_STEPS > 0 and global_step % PRINT_EVERY_X_STEPS == 0:
                            print(get_logging())

                        if ENABLE_LOGGING:
                            log_lines.append(get_logging())
                            if global_step % WRITE_TO_LOG_EVERY_X_STEPS == 0:
                                with open(log_file, "a") as f:
                                    for line in log_lines:
                                        f.write(line + "\n")
                                log_lines.clear()

                        if TRAIN_MODEL:
                            scheduler_model.step()
                            model_optimizer.zero_grad()
                        if TRAIN_PROJECTION:
                            scheduler_proj.step()
                            projection_optimizer.zero_grad()

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
                            if REPEAT_ALIGNMENT_AFTER_RESTART:
                                loss_fn.reset_step()
                            if SAVE_EVERY_X_RESTARTS > 0 and restart_cycle % SAVE_EVERY_X_RESTARTS == 0:
                                print(f"\nSaving checkpoint at restart {restart_cycle}\n")
                                save_path = os.path.join(OUTPUT_DIR, f"restart_{restart_cycle}")
                                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

                            restart_cycle += 1
                            if restart_cycle > total_restarts and AUTO_LAYER_INIT_TRAINING:
                                print(f"\nSaving final checkpoint with initialized layers\n")
                                save_path = os.path.join(OUTPUT_DIR, f"initialized_model")
                                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)
                                break
                            if AUTO_LAYER_INIT_TRAINING:
                                projection_layers, layers_to_load = update_projection_layers(restart_cycle, layers_to_load)
                                projection_parameters = get_projection_parameters(projection_layers, restart_cycle)

                                if TRAIN_PROJECTION:
                                    projection_optimizer.param_groups[0]['params'] = projection_parameters

                            # Automatic training scaling: reset learning rate and scheduler
                            if AUTO_LAYER_INIT_TRAINING and total_restarts >= 2:
                                k = restart_cycle  # Current cycle index after increment
                                multiplier = AUTO_LAYER_INIT_TRAINING_LR_SCALER - (k - 1) * (AUTO_LAYER_INIT_TRAINING_LR_SCALER - 1) / (total_restarts - 1)
                                new_max_lr_proj = ORIGINAL_MAX_LEARNING_RATE_PROJECTION * multiplier

                                if TRAIN_PROJECTION:
                                    for param_group in projection_optimizer.param_groups:
                                        param_group['lr'] = new_max_lr_proj
                                    MAX_LEARNING_RATE_PROJECTION = new_max_lr_proj

                            if REPEAT_WARMUP_AFTER_RESTART or AUTO_LAYER_INIT_TRAINING:
                                # Reset schedulers if repeating warmup or for auto init LR scaler warmup recalculation
                                if not REPEAT_WARMUP_AFTER_RESTART:
                                    WARMUP_STEPS = 0
                                if TRAIN_MODEL:
                                    model_optimizer, scheduler_model = initialize_optimizer(MAX_LEARNING_RATE_MODEL, MIN_LEARNING_RATE_MODEL)
                                if TRAIN_PROJECTION:
                                    projection_optimizer, scheduler_proj = initialize_optimizer(MAX_LEARNING_RATE_PROJECTION, MIN_LEARNING_RATE_PROJECTION)

                        del student_outputs, student_hidden, projected_student, teacher_hidden
                        if 't_input_ids' in locals():
                            del t_input_ids

                    del s_input_ids, s_mask, t_mask
                    if USE_CACHED_EMBEDDINGS:
                        del t_embeddings
                    else:
                        del t_input_ids

                except Exception as e:
                    logging.exception(f"Error in batch {batch_idx}: {e}")
                    if 'student_outputs' in locals():
                        del student_outputs
                    if 'student_hidden' in locals():
                        del student_hidden
                    if 'projected_student' in locals():
                        del projected_student
                    if 'teacher_hidden' in locals():
                        del teacher_hidden
                    if 't_input_ids' in locals():
                        del t_input_ids
                    del s_input_ids, s_mask, t_mask
                    if USE_CACHED_EMBEDDINGS:
                        del t_embeddings
                    else:
                        del t_input_ids

                    accumulation_step = 0
                    if TRAIN_MODEL: model_optimizer.zero_grad()
                    if TRAIN_PROJECTION: projection_optimizer.zero_grad()
                    continue

            if AUTO_LAYER_INIT_TRAINING:
                break
            print(f"Completed epoch {epoch + 1}/{EPOCHS} with {steps_completed_this_epoch} steps")
            next_epoch = epoch + 1
            if next_epoch % SAVE_EVERY_X_EPOCHS == 0:
                print(f"\nSaving checkpoint at epoch {next_epoch}\n")
                save_path = os.path.join(OUTPUT_DIR, f"epoch_{next_epoch}")
                save_trained_model(save_path, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

            if next_epoch % EVAL_EVERY_X_EPOCHS == 0:
                eval_start_time = time.time()

                eval_metrics = evaluate_model(
                    student_model, eval_dataloader, projection_layers, loss_fn, device, autocast_dtype)

                avg_eval_loss = eval_metrics['total']
                print(f"\n[Validation] Epoch {epoch + 1}")
                print(f"  Average Total Loss: {avg_eval_loss:.6f}")
                if TOKEN_HUBER_LOSS > 0 and TOKEN_COSINE_LOSS > 0:
                    print(f"  PT Huber Loss: {eval_metrics['huber']:.6f}")
                    print(f"  PT Cosine Loss: {eval_metrics['cos']:.6f}")
                if SEQUENCE_HUBER_LOSS > 0 and SEQUENCE_COSINE_LOSS > 0:
                    print(f"  SQ Huber Loss: {eval_metrics['seq_huber']:.6f}")
                    print(f"  SQ Cosine Loss: {eval_metrics['seq_cos']:.6f}")

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
if not AUTO_LAYER_INIT_TRAINING:
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    save_trained_model(OUTPUT_DIR, student_model, student_tokenizer, projection_layers, qwen_embedding_dim)

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("✅ Training and saving completed successfully!")
