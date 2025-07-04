# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch YuLanMini model."""
import json
import math
import re
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (AttentionMaskConverter,
                                                   _prepare_4d_attention_mask)
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (ALL_LAYERNORM_LAYERS,
                                        is_torch_greater_or_equal_than_1_13)
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                is_flash_attn_2_available,
                                is_flash_attn_greater_or_equal_2_10, logging,
                                replace_return_docstrings)

try:
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=4096, KV_LEN=4096)
except ImportError:
    pass
import os
import sys

sys.path.append('/home/u20140041/pretrain-mini/model')
from configuration_yulanmini import YuLanMiniConfig

# from unsloth.models.llama import CausalLM_fast_forward, LlamaModel_fast_forward_inference, LlamaAttention_fast_forward, LlamaModel_fast_forward, LlamaDecoderLayer_fast_forward

if is_flash_attn_2_available():
    from modeling_flash_attention_utils import _flash_attention_forward

# from liger_kernel.transformers.experimental.embedding import LigerEmbedding
import wandb
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.fused_linear_cross_entropy import \
    LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from rope import liger_rotary_pos_emb as liger_rotary_pos_emb_flash
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

FLASH_ROPE = True

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "YuLanMiniConfig"


# https://github.com/unslothai/unsloth/blob/4e570be9ae4ced8cdc64e498125708e34942befc/unsloth/models/llama.py#L276
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    hidden_fp32 = hidden.to(torch.float32)
    variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
    hidden = (hidden_fp32 * (variance + eps).rsqrt()).to(old_dtype)
    hidden *= weight
    return hidden


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length),
                                 fill_value=min_dtype,
                                 dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length,
                                    device=device) > cache_position.reshape(
                                        -1, 1)
        causal_mask = causal_mask[None,
                                  None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone(
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :
                                       mask_length] + attention_mask[:, None,
                                                                     None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :
                        mask_length] = causal_mask[:, :, :, :
                                                   mask_length].masked_fill(
                                                       padding_mask, min_dtype)

    return causal_mask


class YuLanMiniRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6, casting_mode="llama", offset=0, init_fn="ones"):
        """
        YuLanMiniRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if init_fn == "ones":
            self.weight = nn.Parameter(torch.ones(hidden_size))
        elif init_fn == "zeros":
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            raise ValueError(f"Invalid init_fn: {init_fn}")
        self.variance_epsilon = eps
        self.offset = offset
        self.casting_mode = casting_mode

    def forward(self, hidden_states):
        old_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
        if self.casting_mode == "gemma":
            hidden = (hidden_fp32 * (variance + self.variance_epsilon).rsqrt()).to(old_dtype)
            hidden *= (self.weight + self.offset)
        elif self.casting_mode == "llama":
            hidden = (hidden_fp32 * (variance + self.variance_epsilon).rsqrt())
            hidden *= (self.weight.float() + self.offset)
            hidden = hidden.to(old_dtype)
        else:
            raise ValueError(f"Invalid casting_mode: {self.casting_mode}")
        return hidden

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(YuLanMiniRMSNorm)
ALL_LAYERNORM_LAYERS.append(LigerRMSNorm)


class YuLanMiniRotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=4096,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device="cuda" if device is None else device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):

        self.max_seq_len_cached = seq_len
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64, device="cpu").float() /
                                      self.dim))
        t = torch.arange(self.max_seq_len_cached,
                         device="cpu",
                         dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype=dtype,
                                          device=device,
                                          non_blocking=True),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype=dtype,
                                          device=device,
                                          non_blocking=True),
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def get_cached(self, seq_len=None):
        return self.cos_cached, self.sin_cached


class YuLanMiniLinearScalingRotaryEmbedding(YuLanMiniRotaryEmbedding):
    """YuLanMiniRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)


class YuLanMiniDynamicNTKScalingRotaryEmbedding(YuLanMiniRotaryEmbedding):
    """YuLanMiniRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len /
                                 self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(self.dim /
                                                             (self.dim - 2))
            inv_freq = 1.0 / (base**(
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q,
                         k,
                         cos,
                         sin,
                         position_ids,
                         unsqueeze_dim=None,
                         fast=False):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if FLASH_ROPE:
        if unsqueeze_dim is None:
            unsqueeze_dim = 2
        assert unsqueeze_dim == 2
    else:
        if unsqueeze_dim is None:
            unsqueeze_dim = 1
        assert unsqueeze_dim == 1

    if fast:
        # print(f"position_ids: {position_ids}")
        if FLASH_ROPE:
            return liger_rotary_pos_emb_flash(q, k, cos, sin, position_ids,
                                    unsqueeze_dim)
        else:
            return liger_rotary_pos_emb(q, k, cos, sin, position_ids,
                                    unsqueeze_dim)

    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    # return q_embed, k_embed
    # weired, its faster to run in float32
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class YuLanMiniMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj.__do_scale_tager__ = True

        self.gate_proj.__do_scale_tager_mu_dim_model__  = True
        self.up_proj.__do_scale_tager_mu_dim_model__ = True
        self.down_proj.__do_scale_tager_mu_ffn__ = True

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def repeat_kv_flash(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch,
                                                    slen,
                                                    num_key_value_heads,  n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep,
                                 head_dim)

def get_hidden_states_logger(layer_idx, num_hidden_layers=None):
    if num_hidden_layers is None:
        log_interval = None
    else:
        log_interval = 1

    @torch.no_grad()
    def log_hidden_states_decoder_layers(name, hidden_states):
        if layer_idx % log_interval == 0 and wandb.run is not None and wandb.config.get("global_step", 0) % 23 == 0:
            layer = layer_idx
            try:
                wandb.log({f"hidden_states_var/{layer}_{name}": torch.var(hidden_states, dim=-1).mean().item()}, commit=False)
                wandb.log({f"hidden_states_mean/{layer}_{name}": torch.mean(hidden_states, dim=-1).mean().item()}, commit=False)
                wandb.log({f"hidden_states_rms/{layer}_{name}": torch.sqrt(torch.mean(hidden_states**2, dim=-1)).mean().item()}, commit=False)
            except Exception as e:
                print(f"Error logging hidden states: {e}")

    @torch.no_grad()
    def log_hidden_states_transformers(layer_idx, name, hidden_states):
        if wandb.run is not None and  wandb.config.get("global_step", 0) % 23 == 0:
            try:
                wandb.log({f"hidden_states_var/{layer_idx}_{name}": torch.var(hidden_states, dim=-1).mean().item()}, commit=False)
                wandb.log({f"hidden_states_mean/{layer_idx}_{name}": torch.mean(hidden_states, dim=-1).mean().item()}, commit=False)
                wandb.log({f"hidden_states_rms/{layer_idx}_{name}": torch.sqrt(torch.mean(hidden_states**2, dim=-1)).mean().item()}, commit=False)
            except Exception as e:
                print(f"Error logging hidden states: {e}")

    if num_hidden_layers is None:
        return log_hidden_states_transformers
    else:
        return log_hidden_states_decoder_layers

def get_od_weight_logger(layer_idx, num_hidden_layers=None):
    if num_hidden_layers is None:
        log_interval = None
    else:
        log_interval = 1

    @torch.no_grad()
    def log_od_weight(name, weight_matrix):
        if layer_idx % log_interval == 0 and wandb.run is not None and wandb.config.get("global_step", 0) % 23 == 0:
            layer = layer_idx
            try:
                wandb.log({f"weight_var/{layer}_{name}": torch.var(weight_matrix).item()}, commit=False)
                wandb.log({f"weight_mean/{layer}_{name}": torch.mean(weight_matrix).item()}, commit=False)
                wandb.log({f"weight_rms/{layer}_{name}": torch.sqrt(torch.mean(weight_matrix**2)).item()}, commit=False)
            except Exception as e:
                print(f"Error logging weight: {e}")

    return log_od_weight


class StableLmLayerNormPerHead(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-5, bias=False, use_liger=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if use_liger:
            self.norms = nn.ModuleList([LigerLayerNorm(dim, eps=eps, bias=bias) for _ in range(self.num_heads)])
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=eps, bias=bias) for _ in range(self.num_heads)])

    def forward(self, hidden_states: torch.Tensor):
        # Split along the num_heads axis to get per-head inputs
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, 1, seq_len, head_dim] * num_heads
        states_per_heads = torch.split(hidden_states, 1, dim=1)
        # Normalize and merge the heads back together
        return torch.cat([norm(hidden_states) for norm, hidden_states in zip(self.norms, states_per_heads)], dim=1)


class YuLanMiniAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self,
                 config: YuLanMiniConfig,
                 layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                bias=False)
        self.o_proj.__do_scale_tager__ = True
        self.q_proj.__do_scale_tager_mu_dim_model__=True
        self.k_proj.__do_scale_tager_mu_dim_model__=True
        self.v_proj.__do_scale_tager_mu_dim_model__=True
        self.o_proj.__do_scale_tager_mu_o__=True
        if self.config.wesar_weights:
            self.q_proj_alpha = nn.Parameter(torch.ones(1) * self.config.q_proj_alpha)
            self.k_proj_alpha = nn.Parameter(torch.ones(1) * self.config.k_proj_alpha)
            self.v_proj_alpha = nn.Parameter(torch.ones(1) * self.config.v_proj_alpha)
            self.o_proj_alpha = nn.Parameter(torch.ones(1) * self.config.o_proj_alpha)
        else:
            self.q_proj_alpha=1
            self.k_proj_alpha=1
            self.v_proj_alpha=1
            self.o_proj_alpha=1


        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = StableLmLayerNormPerHead(
                self.head_dim, self.num_heads, eps=config.layer_norm_eps, use_liger=config.use_liger,
            )
            self.k_layernorm = StableLmLayerNormPerHead(
                self.head_dim, self.num_key_value_heads, eps=config.layer_norm_eps, use_liger=config.use_liger,
            )

        self.log_hidden_states = get_hidden_states_logger(self.layer_idx, self.config.num_hidden_layers)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states * self.q_proj_alpha
        key_states = self.k_proj(hidden_states)
        key_states = key_states * self.k_proj_alpha
        value_states = self.v_proj(hidden_states)
        value_states = value_states * self.v_proj_alpha


        if FLASH_ROPE:
            query_states = query_states.view(bsz, q_len, self.num_heads,
                                            self.head_dim)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2)

        if self.qk_layernorm:
            raise NotImplementedError("qk_layernorm is not implemented")
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        kv_seq_len = key_states.shape[-3 if FLASH_ROPE else -2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        if FLASH_ROPE:
            key_states = repeat_kv_flash(key_states, self.num_key_value_groups)
            value_states = repeat_kv_flash(value_states, self.num_key_value_groups)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) * math.sqrt(self.config.dim_model_base_attn) / self.head_dim

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        self.log_hidden_states("1_attn_weights", attn_weights)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attention_dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        if FLASH_ROPE:
            attn_output = attn_output.contiguous()
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj_alpha * attn_output

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None

class YuLanMiniFlashAttention2(YuLanMiniAttention):
    """
    YuLanMini flash attention module. This module inherits from `YuLanMiniAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10(
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states * self.q_proj_alpha
        key_states = self.k_proj(hidden_states)
        key_states = key_states * self.k_proj_alpha
        value_states = self.v_proj(hidden_states)
        value_states = value_states * self.v_proj_alpha
        self.log_hidden_states("query_states", query_states)
        self.log_hidden_states("key_states", key_states)
        self.log_hidden_states("value_states", value_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        if FLASH_ROPE:
            query_states = query_states.view(bsz, q_len, self.num_heads,
                                            self.head_dim)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2)

        if self.qk_layernorm:
            raise NotImplementedError("qk_layernorm is not implemented")
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        kv_seq_len = key_states.shape[-3 if FLASH_ROPE else -2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states,
                                                        cos,
                                                        sin,
                                                        position_ids=position_ids,
                                                        fast=True)
        self.log_hidden_states("query_states_after_rope", query_states)
        self.log_hidden_states("key_states_after_rope", key_states)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(
                self.layer_idx) > 0
            if (getattr(self.config, "sliding_window", None) is not None
                    and kv_seq_len > self.config.sliding_window
                    and cache_has_contents):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}")

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(attention_mask[:, -1:])
                    ],
                                               dim=-1)

            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # todo: check if we need to repeat_kv
        # repeat k/v heads if n_kv_heads < n_heads
        if FLASH_ROPE:
            key_states = repeat_kv_flash(key_states, self.num_key_value_groups)
            value_states = repeat_kv_flash(value_states, self.num_key_value_groups)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (YuLanMiniRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}.")

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        if not FLASH_ROPE:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        if (self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output, softmax_lse, _ = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            softmax_scale = math.sqrt(self.config.dim_model_base_attn) / self.head_dim,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            return_attn_probs=True,
        )
        self.log_hidden_states("1_attn_weights", softmax_lse)

        attn_output = attn_output.reshape(bsz, q_len,
                                          self.hidden_size).contiguous()

        attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj_alpha * attn_output

        return attn_output, None, past_key_value


YULANMINI_ATTENTION_CLASSES = {
    "eager": YuLanMiniAttention,
    "flash_attention_2": YuLanMiniFlashAttention2,
}


class YuLanMiniDecoderLayer(nn.Module):

    def __init__(self, config: YuLanMiniConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered.")
        self.self_attn = YULANMINI_ATTENTION_CLASSES[
            config._attn_implementation](config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx

        mlp_class = LigerSwiGLUMLP if config.use_liger else YuLanMiniMLP
        self.mlp = mlp_class(config)
        if self.config.wesar_weights:
            self.gate_up_proj_alpha = nn.Parameter(torch.tensor(1) * self.config.gate_up_proj_alpha)
            self.down_proj_alpha = nn.Parameter(torch.tensor(1) * self.config.down_proj_alpha)
        else:
            self.gate_up_proj_alpha=1
            self.down_proj_alpha=1

        rms_class = LigerRMSNorm if config.use_liger else YuLanMiniRMSNorm
        if config.rms_type == "llama":
            rms_kwargs = {"offset": 0, "init_fn": "ones", "casting_mode": "llama"}
        elif config.rms_type == "gemma":
            rms_kwargs = {"offset": 1, "init_fn": "zeros", "casting_mode": "gemma"}
        self.input_layernorm = rms_class(config.hidden_size, eps=config.rms_norm_eps, **rms_kwargs)
        if self.config.wesar_weights and self.config.use_norm_alpha:
            self.input_layernorm_alpha = nn.Parameter(torch.tensor(1) * self.config.input_layernorm_alpha)
        else:
            # print("哈哈，没有 use  input_layernorm_alpha！！！！！！！！")
            self.input_layernorm_alpha = 1
        self.post_attention_layernorm = rms_class(config.hidden_size, eps=config.rms_norm_eps, **rms_kwargs)
        if self.config.wesar_weights and self.config.use_norm_alpha :
            self.post_attention_layernorm_alpha = nn.Parameter(torch.tensor(1) * self.config.post_attention_layernorm_alpha)
        else:
            # print("哈哈，没有 use  post_attention_layernorm_alpha！！！！！！！！")
            self.post_attention_layernorm_alpha = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        log_hidden_states = get_hidden_states_logger(self.layer_idx, self.config.num_hidden_layers)
        log_weights = get_od_weight_logger(self.layer_idx, self.config.num_hidden_layers)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states) * self.config.ln_scale * self.input_layernorm_alpha
        log_hidden_states("0_input_ln", hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # log_hidden_states("2_attn", hidden_states)
        shrink = self.config.hidden_states_shrink
        if 0 <= shrink < 1:
            # hidden_states = hidden_states * shrink + hidden_states.detach() * (1 - shrink)
            hidden_states = hidden_states * shrink
        hidden_states = residual + hidden_states
        # log_hidden_states("3_attn_res", hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) * self.config.ln_scale * self.post_attention_layernorm_alpha
        log_hidden_states("4_post_ln", hidden_states)
        hidden_states = hidden_states * self.gate_up_proj_alpha
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states * self.down_proj_alpha
        log_hidden_states("5_mlp", hidden_states)

        if 0 <= shrink < 1:
            # hidden_states = hidden_states * shrink + hidden_states.detach() * (1 - shrink)
            hidden_states = hidden_states * shrink
        hidden_states = residual + hidden_states
        log_hidden_states("6_mlp_res", hidden_states)

        outputs = (hidden_states, )
        # log_weights("down_weight", self.mlp.down_proj.weight)
        # log_weights("up_weight", self.mlp.up_proj.weight)
        # log_weights("gate_weight", self.mlp.up_proj.weight)
        # log_weights("o_proj_weight", self.self_attn.o_proj.weight)
        # log_weights("q_proj_weight", self.self_attn.q_proj.weight)
        # log_weights("k_proj_weight", self.self_attn.q_proj.weight)
        # log_weights("v_proj_weight", self.self_attn.q_proj.weight)
        # if output_attentions:
        #     outputs += (self_attn_weights, )

        return outputs


YULANMINI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`YuLanMiniConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare YuLanMini Model outputting raw hidden-states without any specific head on top.",
    YULANMINI_START_DOCSTRING,
)
class YuLanMiniPreTrainedModel(PreTrainedModel):
    config_class = YuLanMiniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["YuLanMiniDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # reproduce版本
            module_std = std
            if not self.config.model_reproduce == "transformer":
                if getattr(module, "__do_scale_tager__", False):
                    module_std = module_std / self.config.init_scale_o

                # muparam版本
                if getattr(module, "__do_scale_tager_mu_original__", False):
                    module_std = module_std
                elif getattr(module, "__do_scale_tager_mu_o__", False):
                    if self.config.model_reproduce == "cerebras":
                    # module_std = module_std / math.sqrt(self.config.hidden_size / self.config.dim_model_base_init)
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt(2*(self.config.hidden_size / self.config.dim_model_base_init)*self.config.num_hidden_layers)
                        else:
                            module_std = module_std
                    elif self.config.model_reproduce == "minicpm":
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt((self.config.hidden_size / self.config.dim_model_base_init))
                        else:
                            module_std = module_std
                    else:
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt((self.config.hidden_size / self.config.dim_model_base_init))
                        else:
                            module_std = module_std
                elif getattr(module, "__do_scale_tager_mu_ffn__", False):
                    # module_std = std / math.sqrt(self.config.intermediate_size / self.config.dim_ffn_base_init)
                    if self.config.model_reproduce == "cerebras":
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt(2*(self.config.hidden_size / self.config.dim_model_base_init)*self.config.num_hidden_layers)
                        else:
                            module_std = module_std

                    elif self.config.model_reproduce == "minicpm":
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt((self.config.hidden_size / self.config.dim_model_base_init))
                        else:
                            module_std = module_std
                    else:
                        if self.config.dim_model_base_init is not None:
                            module_std = module_std / math.sqrt((self.config.hidden_size / self.config.dim_model_base_init))
                        else:
                            module_std = module_std
                elif getattr(module, "__do_scale_tager_mu_dim_model__", False):
                    if self.config.dim_model_base_init is not None:
                        module_std = module_std / math.sqrt(self.config.hidden_size / self.config.dim_model_base_init)
                    else:
                        module_std = module_std
                elif getattr(module, "__do_scale_tager_mu_dim_base_model__", False):
                    module_std = module_std / math.sqrt(self.config.dim_model_base_lmh)
                else:
                    module_std = module_std

            print(f"init {module} with std {module_std} ({module.__class__.__name__})")
            module.weight.data.normal_(mean=0.0, std=module_std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module_std = getattr(module, "__std__", std)
            print(f"init {module} with std {module_std} ({module.__class__.__name__})")
            module.weight.data.normal_(mean=0.0, std=module_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


YULANMINI_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare YuLanMini Model outputting raw hidden-states without any specific head on top.",
    YULANMINI_START_DOCSTRING,
)
class YuLanMiniModel(YuLanMiniPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`YuLanMiniDecoderLayer`]
    Args:
        config: YuLanMiniConfig
    """

    def __init__(self, config: YuLanMiniConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        # if self.config.wesar_weights and self.config.use_emb_alpha:
        #     # self.embed_tokens_alpha = nn.Parameter(torch.tensor(1.0) * self.config.embed_tokens_alpha)
        #     self.embed_tokens_alpha = 1
        # else:
        self.embed_tokens_alpha = 1
        if not self.config.tie_word_embeddings:
            self.embed_tokens.__std__ = 1.0

        rms_class = LigerRMSNorm if config.use_liger else YuLanMiniRMSNorm
        if config.rms_type == "llama":
            rms_kwargs = {"offset": 0, "init_fn": "ones", "casting_mode": "llama"}
        elif config.rms_type == "gemma":
            rms_kwargs = {"offset": 1, "init_fn": "zeros", "casting_mode": "gemma"}
        if self.config.embedding_ln:
            ln_class = LigerLayerNorm if config.use_liger else nn.LayerNorm
            self.embedding_layernorm = ln_class(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        elif self.config.embedding_rmsln:
            self.embedding_layernorm = rms_class(config.hidden_size, eps=config.rms_norm_eps, **rms_kwargs)

        self.layers = nn.ModuleList([
            YuLanMiniDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self._attn_implementation = config._attn_implementation

        self.norm = rms_class(config.hidden_size,
                                     eps=config.rms_norm_eps, **rms_kwargs)
        if self.config.wesar_weights  and self.config.use_norm_alpha :
            self.norm_alpha = nn.Parameter(torch.tensor(1) * self.config.norm_alpha)
        else:
            # print("哈哈，没有 use  norm_alpha！！！！！！！！")
            self.norm_alpha = 1
        self._init_rope()

        self.gradient_checkpointing = True
        if self.config.wesar_weights:
            self.shrink_alpha = config.shrink_alpha
        else:
            self.shrink_alpha = 1
        self.scale_emb = config.scale_emb
        self.log_hidden_states = get_hidden_states_logger(0, None)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.config.rope_scaling is None:
            self.rotary_emb = YuLanMiniRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # https://huggingface.co/docs/text-generation-inference/basic_tutorials/preparing_model#rope-scaling
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = YuLanMiniLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = YuLanMiniDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(YULANMINI_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = True

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values,
                                        Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.scale_emb
            inputs_embeds = inputs_embeds * self.embed_tokens_alpha
            self.log_hidden_states(0, "0_embed", inputs_embeds)

            if 0 <= self.shrink_alpha < 1:
                shrink_alpha = self.shrink_alpha
                inputs_embeds = inputs_embeds * shrink_alpha + inputs_embeds.detach() * (1 - shrink_alpha)
                self.log_hidden_states(0, "1_shrink", inputs_embeds)

        if self.config.embedding_ln:
            inputs_embeds = self.embedding_layernorm(inputs_embeds)
            self.log_hidden_states(0, "2_embln", inputs_embeds)
        elif self.config.embedding_rmsln:
            inputs_embeds = self.embedding_layernorm(inputs_embeds) * self.config.ln_scale
            self.log_hidden_states(0, "2_embln", inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens +
                                          inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                               cache_position, past_key_values,
                                               output_attentions)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, hidden_states.shape[1])  # Warning: ignore the position_ids

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            if self.gradient_checkpointing and self.training and idx % self.config.gradient_checkpointing_step != 0:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        old_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.norm(hidden_states) * self.config.ln_scale * self.norm_alpha
        hidden_states = hidden_states.to(old_dtype)
        self.log_hidden_states(56, "0_norm", hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length(
        ) if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (attention_mask.shape[-1] if isinstance(
                attention_mask, torch.Tensor) else past_seen_tokens +
                             sequence_length + 1)

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype)

        return causal_mask


class YuLanMiniModelForCausalLM(YuLanMiniPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = YuLanMiniModel(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        if self.config.wesar_weights:
            self.lm_head_alpha = nn.Parameter(torch.tensor(1) * self.config.lm_head_alpha)
        else:
            self.lm_head_alpha = 1
        # Initialize weights and apply final processing
        self.lm_head.__do_scale_tager_mu_dim_base_model__ = not self.config.tie_word_embeddings
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(YULANMINI_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        teacher_logits: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        subset: Optional[List[str]] = None,
        idx: Optional[List[int]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, YuLanMiniForCausalLM
        >>> model = YuLanMiniForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = None
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            if self.config.dim_model_base_logits is not None and self.config.hidden_size != self.config.dim_model_base_logits:
                hidden_states = hidden_states / (self.config.hidden_size / self.config.dim_model_base_logits)

            liger = True
            if not liger:
                logits = self.lm_head(hidden_states).float()

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                hidden_states = hidden_states * self.lm_head_alpha
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
                shift_labels = shift_labels.view(-1)

                lce = LigerFusedLinearCrossEntropyLoss(lse_square_scale=self.config.z_loss)
                loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
            
            # print(loss)
            # check nan
            if torch.isnan(loss):
                if not liger:
                    print(shift_logits[:10,:10], shift_labels[:10])
                    print(self.lm_head.weight.dtype)
                    print(hidden_states.dtype)
                    print(shift_labels.dtype)
                else:
                    print(shift_hidden_states[:10,:10], shift_labels[:10])
                    print(self.lm_head.weight.dtype)
                    print(hidden_states.dtype)
                    print(shift_labels.dtype)
                print(loss)
                exit()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
