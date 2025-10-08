import json
import math
import torch
from torch import Tensor, nn
from torch.nn import Module, Embedding
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, T5EncoderModel, T5TokenizerFast, AutoModel
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import requests
from PIL import Image
import numpy as np
import argparse
from torchvision.utils import save_image
from torchvision.utils import make_grid
import os
from safetensors.torch import load_file
import logging
from tqdm import tqdm
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CHROMA_FILE = "chroma/chroma-unlocked-v41.safetensors"
DEFAULT_VAE_FILE = "ae/ae.safetensors"
DEFAULT_QWEN3_FOLDER = "/mnt/f/q5_xxs_training_script/QT-embedder-ALL/saigo/QT-embedder-v1/restart_1"
DEFAULT_T5_FOLDER = "t5-xxl/"
DEFAULT_POSITIVE_PROMPT = "Hatsune Miku, depicted in anime style, holding up a sign that reads 'Qwen3'. In the background there is an anthroporphic muscular wolf, rendered like a high-resolution 3D model, wearing a t-shirt that reads 'Chroma'. They're stood on the moon."
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SEED = 42
DEFAULT_STEPS = 30
DEFAULT_CFG = 4
DEFAULT_RESOLUTION = [512,512]
DEFAULT_OUTPUT_FILE = "output/q3"
APPEND_DATETIME = True

T5_ADDITIONAL_PADDING_ATTENTION = 0 # Unmask specified padding amount when using T5-xxl
USE_T5_MASK_WITH_QWEN = False # It's recommended to use the T5 mask with our trained model, so leave this on
QWEN_WITH_T5_MASK_ADDITIONAL_PADDING_ATTENTION = 0

# === Configuration Dataclasses ===
@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: List[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    _use_compiled: bool

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: List[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

# === Helper Functions ===
def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # mask should have shape [B, H, L, D]
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

def distribute_modulations(tensor: torch.Tensor, depth_single_blocks, depth_double_blocks):
    batch_size, vectors, dim = tensor.shape
    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(depth_single_blocks):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            block_dict[key] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            double_block = []
            for _ in range(2):
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3
            block_dict[key] = double_block

        elif "txt_mod" in key:
            double_block = []
            for _ in range(2):
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3
            block_dict[key] = double_block

        elif "final_layer" in key:
            block_dict[key] = [
                tensor[:, idx : idx + 1, :],
                tensor[:, idx + 1 : idx + 2, :],
            ]
            idx += 2

    return block_dict

def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
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

# === Chroma Model Components ===
class EmbedND(Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

class MLPEmbedder(Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = torch.nn.SiLU()
        self.out_layer = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.use_compiled = use_compiled

    def _forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

    def forward(self, x: Tensor):
        return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)

class Approximator(Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = torch.nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)]
        )
        self.norms = torch.nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        x = self.out_proj(x)
        return x

class QKNorm(Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.key_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.use_compiled = use_compiled

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class SelfAttention(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.proj = torch.nn.Linear(dim, dim)
        self.use_compiled = use_compiled

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x

def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift

def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params

class DoubleStreamBlock(Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.img_norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.txt_norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_compiled = use_compiled

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        img_modulated = self.img_norm1(img)
        img_modulated = self.modulation_shift_scale_fn(img_modulated, img_mod1.scale, img_mod1.shift)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = self.modulation_shift_scale_fn(txt_modulated, txt_mod1.scale, txt_mod1.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = self.modulation_gate_fn(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = self.modulation_gate_fn(
            img,
            img_mod2.gate,
            self.img_mlp(
                self.modulation_shift_scale_fn(
                    self.img_norm2(img), img_mod2.scale, img_mod2.shift
                )
            ),
            )

        txt = self.modulation_gate_fn(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = self.modulation_gate_fn(
            txt,
            txt_mod2.gate,
            self.txt_mlp(
                self.modulation_shift_scale_fn(
                    self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift
                )
            ),
        )

        return img, txt

class SingleStreamBlock(Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.hidden_size = hidden_size
        self.pre_norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = torch.nn.GELU(approximate="tanh")
        self.use_compiled = use_compiled

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor
    ) -> Tensor:
        mod = distill_vec
        x_mod = self.modulation_shift_scale_fn(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe, mask=mask)
        output = self.linear2(torch.cat([attn, self.mlp_act(mlp)], dim=-1))
        return self.modulation_gate_fn(x, mod.gate, output)

class LastLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = torch.nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.use_compiled = use_compiled

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = self.modulation_shift_scale_fn(
            self.norm_final(x), scale[:, None, :], shift[:, None, :]
        )
        x = self.linear(x)
        return x

class Chroma(Module):
    def __init__(self, params: ChromaParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size / params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = torch.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth
        )
        self.txt_in = torch.nn.Linear(params.context_in_dim, self.hidden_size)
        self.double_blocks = torch.nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth)
        ])
        self.single_blocks = torch.nn.ModuleList([
            SingleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth_single_blocks)
        ])
        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            use_compiled=params._use_compiled,
        )
        self.mod_index_length = 3 * params.depth_single_blocks + 2 * 6 * params.depth + 2
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth
        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length)), device="cpu"),
            persistent=False,
        )
        self.approximator_in_dim = params.approximator_in_dim

    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, txt_mask: Tensor, timesteps: Tensor, guidance: Tensor, attn_padding: int = 1, use_padding_modification: bool = False) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Project inputs to hidden size
        img = self.img_in(img)
        txt = self.txt_in(txt)

        with torch.no_grad():
            # Move mod_index to current device and dtype
            mod_index = self.mod_index.to(device=img.device, dtype=img.dtype)
            distill_timestep = timestep_embedding(timesteps, self.params.approximator_in_dim // 4)
            distil_guidance = timestep_embedding(guidance, self.params.approximator_in_dim // 4)
            modulation_index = timestep_embedding(self.mod_index, self.params.approximator_in_dim // 2)
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            # Cast input_vec to the model's dtype
            input_vec = input_vec.to(next(self.parameters()).dtype)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))
        mod_vectors_dict = distribute_modulations(mod_vectors, self.depth_single_blocks, self.depth_double_blocks)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        max_len = txt.shape[1]
        with torch.no_grad():
            if use_padding_modification:
                txt_mask_w_padding = modify_mask_to_attend_padding(
                    txt_mask, max_len, T5_ADDITIONAL_PADDING_ATTENTION
                )
                txt_mask_w_padding = txt_mask_w_padding.to(img.device)
            else:
                txt_mask_w_padding = txt_mask.to(img.device)

            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img_ids.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
        txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
        txt_img_mask = txt_img_mask[None, None, ...].repeat(txt.shape[0], self.num_heads, 1, 1).int().bool()

        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]
            img, txt = block(img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask)

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)
        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, distill_vec=final_mod)
        return img

# === Detailed AutoEncoder Implementation (from autoencoder.py) ===
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != (self.num_resolutions - 1):
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != (self.num_resolutions - 1):
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def encode_for_train(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z, _ = torch.chunk(z, 2, dim=1)  # grab the mean only
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

# ========== Projection Layers ==========
class LinearProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear(x)

class MLPProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.GELU()
        self.output_dim = hidden_dim

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

        return output

# === Utility Functions ===
def get_noise(num_samples: int, height: int, width: int, device: torch.device, dtype: torch.dtype, seed: int):
    return torch.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed)
    )

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def prepare_latent_image_ids(batch_size, height, width):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        torch.arange(height // 2)[:, None].expand(height // 2, width // 2)
    )
    latent_image_ids[..., 2] = (
        torch.arange(width // 2)[None, :].expand(height // 2, width // 2)
    )
    latent_image_ids = latent_image_ids.reshape(height // 2 * width // 2, 3)
    latent_image_ids = latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_image_ids

def vae_flatten(latents: Tensor) -> Tuple[Tensor, Tuple]:
    return rearrange(latents, "n c (h dh) (w dw) -> n (h w) (c dh dw)", dh=2, dw=2), latents.shape

def vae_unflatten(latents, shape):
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=2,
        dw=2,
        c=c,
        h=h // 2,
        w=w // 2
    )

def cast_linear(module, dtype, name=''):
    for child_name, child in module.named_children():
        child_full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear):
            if any(keyword in child_full_name for keyword in KEEP_IN_HIGH_PRECISION):
                continue  # Skip casting for these layers
            else:
                child.to(dtype)
        else:
            cast_linear(child, dtype, child_full_name)

# === Inference Function (Fixed) ===
def denoise_cfg(
    model: Chroma,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    neg_txt: Tensor,
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    timesteps: list[float],
    cfg: float,
    first_n_steps_wo_cfg: int,
    image_dim: Tuple[int, int],
    use_t5: bool = False
) -> Tensor:
    logger.info("Starting denoising with CFG")
    # Set guidance to zero as in training
    guidance_vec = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)
    step_count = 0
    pbar = tqdm(total=len(timesteps)-1, desc="Denoising steps")

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        step_size = t_curr - t_prev

        # Positive prediction (guidance=0)
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, txt_mask=txt_mask, timesteps=t_vec, guidance=guidance_vec, use_padding_modification=use_t5)

        if step_count < first_n_steps_wo_cfg or first_n_steps_wo_cfg == -1:
            img = img - step_size * pred
        else:
            # Negative prediction (guidance=0)
            pred_neg = model(img=img, img_ids=img_ids, txt=neg_txt, txt_ids=neg_txt_ids, txt_mask=neg_txt_mask, timesteps=t_vec, guidance=guidance_vec, use_padding_modification=use_t5)
            # CFG scaling by blending predictions
            pred_cfg = pred_neg + cfg * (pred - pred_neg)
            img = img - step_size * pred_cfg

        step_count += 1
        pbar.update(1)

    pbar.close()
    logger.info("Denoising complete")
    return img

def inference_chroma(
    model: Chroma,
    ae: AutoEncoder,
    embed: Tensor,
    embed_neg: Tensor,
    text_ids: Tensor,
    neg_text_ids: Tensor,
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    seed: int,
    steps: int,
    cfg: float,
    first_n_steps_wo_cfg: int,
    image_dim: Tuple[int, int] = (512, 512),
    use_t5: bool = False
) -> Tensor:
    logger.info("Starting inference")
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            device = next(model.parameters()).device
            logger.info("Generating noise")
            noise = get_noise(embed.shape[0], HEIGHT, WIDTH, device, torch.bfloat16, seed)
            noise, shape = vae_flatten(noise)
            noise = noise.to(device)
            n, c, h, w = shape
            image_pos_id = prepare_latent_image_ids(embed.shape[0], h, w).to(device)

            embed = embed.to(device)
            embed_neg = embed_neg.to(device)
            text_ids = text_ids.to(device)
            neg_text_ids = neg_text_ids.to(device)
            txt_mask = txt_mask.to(device)
            neg_txt_mask = neg_txt_mask.to(device)

            logger.info("Generating timesteps schedule")
            timesteps = get_schedule(STEPS, noise.shape[1])

            logger.info("Denoising with CFG")
            latent_cfg = denoise_cfg(
                model,
                noise,
                image_pos_id,
                embed,
                embed_neg,
                text_ids,
                neg_text_ids,
                txt_mask,
                neg_txt_mask,
                timesteps,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
                image_dim,
                use_t5
            )

            logger.info("Decoding latent image")
            output_image = ae.decode(vae_unflatten(latent_cfg, shape))

    logger.info("Inference complete")
    return output_image

# === Model Loading ===
def load_chroma_model(chroma_file: str) -> Chroma:
    logger.info(f"Loading Chroma model from {chroma_file}")
    chroma_params = ChromaParams(
        in_channels=64,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
        approximator_in_dim=64,
        approximator_depth=5,
        approximator_hidden_size=5120,
        _use_compiled=False
    )
    model = Chroma(chroma_params)
    if chroma_file.endswith('.safetensors'):
        state_dict = load_file(chroma_file)
    else:
        state_dict = torch.load(chroma_file)
    model.load_state_dict(state_dict, assign=True)
    return model

def load_autoencoder(vae_file: str) -> AutoEncoder:
    logger.info(f"Loading Autoencoder from {vae_file}")
    ae_params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159
    )
    ae = AutoEncoder(ae_params)
    if vae_file.endswith('.safetensors'):
        state_dict = load_file(vae_file)
    else:
        state_dict = torch.load(vae_file)
    ae.load_state_dict(state_dict, assign=True)
    ae.to(torch.bfloat16)
    return ae

def load_qwen3_model(qwen3_folder: str) -> Tuple[Module, Module, Module]:
    logger.info(f"Loading Qwen3 model from {qwen3_folder}")

    # Load model config
    model_config_path = os.path.join(qwen3_folder, "config.json")
    if not os.path.exists(model_config_path):
        logger.error(f"Model config not found in {qwen3_folder}")
        raise FileNotFoundError(f"Model config not found in {qwen3_folder}")

    with open(model_config_path, "r") as file:
        model_config = json.load(file)
    qwen_embedding_dim = model_config.get("hidden_size", 4096)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(qwen3_folder)

    # Load projection config
    projection_config_path = os.path.join(qwen3_folder, "projection_config.json")
    if not os.path.exists(projection_config_path):
        logger.error(f"Projection config not found in {qwen3_folder}")
        raise FileNotFoundError(f"Projection config not found in {qwen3_folder}")

    with open(projection_config_path, "r") as file:
        projection_config = json.load(file)

    projection_layers = []

    for layer_config in projection_config["layers"]:
        if layer_config["type"] == "linear":
            layer = LinearProjectionLayer(
                layer_config["input_dim"],
                layer_config["output_dim"],
            )
        elif layer_config["type"] == "mlp":
            layer = MLPProjectionLayer(
                layer_config["input_dim"],
                layer_config["hidden_dim"],
            )
        elif layer_config["type"] == "transformer":
            layer = TransformerProjectionLayer(
                layer_config["input_dim"],
                layer_config["hidden_dim"],
                layer_config["dim_feedforward"],
            )
        elif layer_config["type"] == "interpolation":
            layer = LearnedInterpolationLayer(
                layer_config["input_dim"],
                layer_config["output_dim"]
            )

        file_num = layer_config["file_num"]
        layer_path = os.path.join(qwen3_folder, f"projection_layer_{file_num}.safetensors")
        if os.path.exists(layer_path):
            state_dict = load_file(layer_path)
            layer.load_state_dict(state_dict)

        layer.to(device, dtype=torch.bfloat16).eval()
        projection_layers.append(layer)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        qwen3_folder,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to(device)
    model.eval()

    return model, projection_layers, tokenizer

def load_t5_model(t5_folder: str) -> Tuple[Module, Module]:
    logger.info(f"Loading T5 model from {t5_folder}")

    # Look for the model file
    model_file = os.path.join(t5_folder, "model.safetensors")
    if not os.path.exists(model_file):
        logger.error(f"Model file not found in {t5_folder}. Expected file: model.safetensors")
        raise FileNotFoundError(f"Model file not found in {t5_folder}. Expected file: model.safetensors")

    # Use flash attention for speed if available
    model = T5EncoderModel.from_pretrained(t5_folder).to(device)
    model.eval()

    return model

def load_t5_tokenizer(t5_folder: str) -> Tuple[Module, Module]:
    # Look for the tokenizer file
    tokenizer_file = os.path.join(t5_folder, "tokenizer.json")
    if not os.path.exists(tokenizer_file):
        logger.error(f"Tokenizer file not found in {t5_folder}. Expected file: tokenizer")
        raise FileNotFoundError(f"Tokenizer file not found in {t5_folder}. Expected file: tokenizer")

    # Load tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(t5_folder)

    return tokenizer

# === Image Saving ===
def save_images(images: torch.Tensor, filename: str, format: str = 'png', quality: int = 95):
    images = (images + 1) / 2  # normalize to [0,1]
    grid = make_grid(images, nrow=min(4, images.shape[0]))
    if format == 'png':
        save_image(grid, filename)
    else:
        grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(grid)
        pil_image.save(filename, quality=quality, format='JPEG')

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chroma Inference Script')
    parser.add_argument('--chroma_file', type=str, default=DEFAULT_CHROMA_FILE, help='Path to Chroma model file')
    parser.add_argument('--vae_file', type=str, default=DEFAULT_VAE_FILE, help='Path to VAE model file')
    parser.add_argument('--qwen3_folder', type=str, default=DEFAULT_QWEN3_FOLDER, help='Path to Qwen3 folder containing model, tokenizer, and projection layer')
    parser.add_argument('--t5_folder', type=str, default=DEFAULT_T5_FOLDER, help='Path to T5 folder containing model and tokenizer')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help='Number of inference steps')
    parser.add_argument('--cfg', type=float, default=DEFAULT_CFG, help='CFG scale')
    parser.add_argument('--positive_prompt', type=str, default=DEFAULT_POSITIVE_PROMPT, help='Positive prompt for generation')
    parser.add_argument('--negative_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT, help='Negative prompt for generation')
    parser.add_argument('--first_n_steps_wo_cfg', type=int, default=4, help='First n steps without CFG')
    parser.add_argument('--output_resolution', nargs=2, type=int, default=DEFAULT_RESOLUTION, help='Output resolution (width height)')
    parser.add_argument('--max_length', type=int, default=512, help='Max token length')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE, help='Output filename (without extension)')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png', help='Output format')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality')
    parser.add_argument('--t5', action='store_true', help='Use T5 instead of Qwen for text embeddings')
    args = parser.parse_args()

    logger.info("Parsing arguments complete")

    if args.t5:
        use_t5 = True
    else:
        use_t5 = False

    device = 'cuda'
    dtype = torch.bfloat16

    # Choose text embedder based on flag
    logger.info("Using T5-xxl for text embeddings")

    tokenizer = load_t5_tokenizer(args.t5_folder)

    if use_t5:
        if not args.positive_prompt.strip():
            args.positive_prompt = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
        if not args.negative_prompt.strip():
            args.negative_prompt = tokenizer.pad_token if tokenizer.pad_token else "<pad>"

    # Tokenize and create embeddings
    text_inputs = tokenizer(
        [args.positive_prompt],
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    text_inputs_neg = tokenizer(
        [args.negative_prompt],
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    if use_t5:
        t5_model = load_t5_model(args.t5_folder)
        attention_mask = text_inputs["attention_mask"].to(device).to(device)
        attention_mask_neg = text_inputs_neg["attention_mask"].to(device)

        with torch.no_grad():
            embed = t5_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=attention_mask,
            ).last_hidden_state.to(device, dtype=dtype)

            embed_neg = t5_model(
                input_ids=text_inputs_neg["input_ids"],
                attention_mask=attention_mask_neg,
            ).last_hidden_state.to(device, dtype=dtype)

        text_ids = torch.zeros((1, args.max_length, 3), device=t5_model.device)
        neg_text_ids = torch.zeros((1, args.max_length, 3), device=t5_model.device)

    else:
        t5_attention_mask = text_inputs["attention_mask"].to(device).to(device)
        t5_attention_mask_neg = text_inputs_neg["attention_mask"].to(device)
        logger.info("Using Qwen3 for text embeddings")
        # Load Qwen3 model and projection to CUDA first
        qwen3_model, projection_layers, tokenizer = load_qwen3_model(args.qwen3_folder)

        if not args.positive_prompt.strip():
            args.positive_prompt = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
        if not args.negative_prompt.strip():
            args.negative_prompt = tokenizer.pad_token if tokenizer.pad_token else "<pad>"

        # Tokenize and create embeddings
        text_inputs = tokenizer(
            [args.positive_prompt],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(qwen3_model.device)

        text_inputs_neg = tokenizer(
            [args.negative_prompt],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(qwen3_model.device)

        attention_mask = text_inputs["attention_mask"].to(device)
        attention_mask_neg = text_inputs_neg["attention_mask"].to(device)

        with torch.no_grad():
            output = qwen3_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            embed = output.hidden_states[-1].to(device, dtype=dtype)  # Last hidden state

        with torch.no_grad():
            output_neg = qwen3_model(
                input_ids=text_inputs_neg["input_ids"],
                attention_mask=attention_mask_neg,
                output_hidden_states=True
            )
            embed_neg = output_neg.hidden_states[-1].to(device, dtype=dtype)  # Last hidden state

        for layer in projection_layers:
            if isinstance(layer, (LearnedInterpolationLayer)):
                embed = layer(
                    embed,
                    s_mask=attention_mask,
                    t_mask=t5_attention_mask,
                    target_length=512
                )
                embed_neg = layer(
                    embed_neg,
                    s_mask=attention_mask_neg,
                    t_mask=t5_attention_mask_neg,
                    target_length=512
                )
            else:
                embed = layer(embed)
                embed_neg = layer(embed_neg)

        if USE_T5_MASK_WITH_QWEN:
            attention_mask = t5_attention_mask
            attention_mask_neg = t5_attention_mask_neg

            use_t5 = True

            T5_ADDITIONAL_PADDING_ATTENTION = QWEN_WITH_T5_MASK_ADDITIONAL_PADDING_ATTENTION

        text_ids = torch.zeros((1, args.max_length, 3), device=qwen3_model.device)
        neg_text_ids = torch.zeros((1, args.max_length, 3), device=qwen3_model.device)

    # Clear text models from CUDA memory
    if 'qwen3_model' in locals():
        del qwen3_model, projection_layers
    else:
        del model
    torch.cuda.empty_cache()

    # Load Chroma model with memory optimization
    model = load_chroma_model(args.chroma_file)
    model = model.to("cuda")

    # Load VAE model
    ae = load_autoencoder(args.vae_file)
    ae = ae.to("cuda")

    # Run inference with precomputed embeddings
    images = inference_chroma(
        model,
        ae,
        embed,
        embed_neg,
        text_ids,
        neg_text_ids,
        attention_mask,
        attention_mask_neg,
        args.seed,
        args.steps,
        args.cfg,
        args.first_n_steps_wo_cfg,
        tuple(args.output_resolution),
        use_t5
    )

    # Adjust output filename if requested
    if APPEND_DATETIME:
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_name = args.output_file
        output_filename = f"{base_name}_{current_time}.{args.format}"
    else:
        output_filename = f"{args.output_file}.{args.format}"

    logger.info(f"Saving images to {output_filename}")
    save_images(images, output_filename, args.format, args.quality)
    print(f"Generated images saved to {output_filename}")
