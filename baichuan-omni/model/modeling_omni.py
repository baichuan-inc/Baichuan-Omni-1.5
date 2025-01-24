# Copyright 2023 Baichuan Inc. All Rights Reserved.
#
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
""" PyTorch omni model."""
import os
import time
import json
import math
import numpy as np
from typing import List, Optional, Tuple, Union, Any
from threading import Thread
from easydict import EasyDict

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from dataclasses import dataclass
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.generation.utils import GenerationConfig
from transformers.utils import logging
# import for dynamic import not used in this file
from .vector_quantize import VectorQuantize, EuclideanCodebook
from .matcha_components import (
    SinusoidalPosEmb,
    Block1D,
    ResnetBlock1D,
    Downsample1D,
    TimestepEmbedding,
    Upsample1D,
)
from .matcha_transformer import BasicTransformerBlock
from .flow_matching import ConditionalDecoder, ConditionalCFM

from .configuration_omni import OmniConfig
from .audio_modeling_omni import (RMSNorm,
                                      OmniAudioEncoder, 
                                      OmniAudioDecoder,
                                      OmniAudioVQBridgeTokenizer, 
                                      OmniAudioFlowMatchingDecoder)
from .visual_modeling_omni import OmniVisualEncoder, OmniVisualBridge
from .processor_omni import OmniMMProcessor

# support model path contain point(.)
try:
    # step1: copy relative imports to transformers_modules
    from .generation_utils import build_chat_input, TextIterStreamer
    from .sequence_parallel_utils import (
        create_attention_layer,
        get_sequence_parallel_size,
        get_sequence_parallel_chunk,
    )
except ModuleNotFoundError:
    # step2: direct import from transformers_modules
    try:  # bypass check_imports failure
        import sys
        sys.path.append(os.path.dirname(__file__))
        from generation_utils import build_chat_input, TextIterStreamer
        from sequence_parallel_utils import (
            create_attention_layer,
            get_sequence_parallel_size,
            get_sequence_parallel_chunk,
        )
    except Exception:
        raise

logger = logging.get_logger(__name__)

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n)  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n - closest_power_of_2]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=5e6, device=None):
        super().__init__()
        # 修复RePE初始化精度问题 https://zhuanlan.zhihu.com/p/678963442
        # DeepSpeed 会 Hack torch.arange 强制在 GPU 上运行，这里使用原生的 torch.arange
        try:
            import deepspeed
            self.arange = deepspeed.runtime.zero.partition_parameters._orig_torch_arange
        except:
            self.arange = torch.arange

        self.inv_freq = 1.0 / (base ** (self.arange(0, dim, 2).float().to(device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32).to(x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(torch.float32).to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(torch.float32).to(x.device),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: OmniConfig, is_sparse=False):
        super().__init__()
        self.config = config
        self.position_embedding_type = config.position_embedding_type.lower()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.num_attention_heads * self.head_dim
        self.hidden_kv_size = self.num_kv_heads * self.head_dim

        if is_sparse:
            self.num_heads = config.sparse_attention_heads
            assert self.num_kv_heads == config.num_attention_heads
            self.W_pack = nn.Linear(self.hidden_size, 3 * self.num_heads * self.head_dim, bias=config.attention_qkv_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        else:
            self.num_heads = config.num_attention_heads
            if self.config.attention_qkv_pack:
                self.W_pack = nn.Linear(config.hidden_size, self.hidden_size + self.hidden_kv_size * 2, bias=config.attention_qkv_bias)
            else:
                self.q_proj = nn.Linear(config.hidden_size, self.hidden_size, bias=config.attention_qkv_bias)
                self.k_proj = nn.Linear(config.hidden_size, self.hidden_kv_size, bias=config.attention_qkv_bias)
                self.v_proj = nn.Linear(config.hidden_size, self.hidden_kv_size, bias=config.attention_qkv_bias)

            self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if self.position_embedding_type == 'rope':
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.get_rotary_base()
            )
        elif self.position_embedding_type == 'alibi':
            self.alibi_slopes = get_slopes(self.num_heads)
        self.attention = create_attention_layer(self.hidden_size, self.num_heads, self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _repeat_kv(self, hidden_states: torch.Tensor, num_heads: int) -> torch.Tensor:
        assert hidden_states.size(1) <= num_heads and num_heads % hidden_states.size(1) == 0
        return repeat_kv(hidden_states, num_heads // hidden_states.size(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len = hidden_states.shape[:2]

        if self.config.attention_qkv_pack:
            proj = self.W_pack(hidden_states)
            query_states, key_states, value_states = proj.split([self.hidden_size, self.hidden_kv_size, self.hidden_kv_size], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # (B, S, hidden_size) -> (B, num_heads, S, head_size)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        # (B, S, hidden_size) -> (B, num_kv_heads, S, head_size)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if self.position_embedding_type == 'rope':
            max_position = position_ids.max().item()+1 if position_ids is not None else kv_seq_len * get_sequence_parallel_size()
            cos, sin = self.rotary_emb(value_states, seq_len=max_position)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
                get_sequence_parallel_chunk(position_ids)
            )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, query_states.size(1))
        value_states = self._repeat_kv(value_states, query_states.size(1))

        if seqlens is not None:
            seqlens = seqlens.to(dtype=torch.int32)
            max_seqlen = (seqlens[1:] - seqlens[:-1]).max().item()
            if self.position_embedding_type == 'alibi':
                alibi_slopes = torch.tensor(self.alibi_slopes, dtype=torch.float32).to(query_states.device)
            else:
                alibi_slopes = None
            attn_output = self.attention(
                query_states, key_states, value_states, seqlens, seqlens,
                max_seqlen, max_seqlen, causal=True, alibi_slopes=alibi_slopes, use_flash=True)
        else:
            attn_output = self.attention(
                query_states, key_states, value_states, attn_mask=attention_mask, use_flash=False)

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: OmniConfig, is_sparse=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, is_sparse=is_sparse)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        group_index=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seqlens=seqlens,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OmniPreTrainedModel(PreTrainedModel):
    config_class = OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OmniModel):
            module.gradient_checkpointing = value

@dataclass
class OmniModelOutputWithPast(BaseModelOutputWithPast):
    audio_encoder_ret: Optional[Any] = None
    audio_decoder_ret: Optional[Any] = None

class OmniModel(OmniPreTrainedModel):
    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if config.visual_config.enable:
            self.visual_model = OmniVisualEncoder(config.visual_config)
            self.visual_bridge_model = OmniVisualBridge(config.visual_config)
        if config.video_config.enable and not config.visual_config.enable: # in case 没有visual_config而只有video_config
            self.visual_model = OmniVisualEncoder(config.video_config)
            self.visual_bridge_model = OmniVisualBridge(config.video_config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DecoderLayer(config, is_sparse=layer_idx in config.sparse_attention_layers)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.audio_embed_layers = nn.ModuleList([
            nn.Embedding(codedim + 1, config.hidden_size)
                for i, codedim in enumerate(config.audio_config.vq_config.codebook_sizes)
        ])

        self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def get_multimodal_mask(self, input_ids, pad_token_id, special_token_list):
        '''
        获取任意模态的特殊mask，包含以下
        1. pad mask 表示文本中图像/语音/视频模态提前留出的token位置
        2. special token mask 特殊token 例如对理解模型<start> <end> 不需要next token prediction
        3. embedding mask / lm_head mask 标记出特殊token在embedding中的mask
        '''
        pad_mask = torch.eq(input_ids, pad_token_id)
        sp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        lm_head_mask = torch.zeros([self.config.vocab_size, 1], dtype=torch.bool)
        for sp_id in special_token_list:
            sp_mask = torch.logical_or(sp_mask, torch.eq(input_ids, sp_id))
            lm_head_mask[sp_id, 0] = True
        return pad_mask, sp_mask, lm_head_mask

    def get_multimodal_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            multimodal_embed,
            pad_token_id,
            fake_input,
            group_index=None,  # 某种模态的编号
        ):
        pad_mask, sp_mask, _ = self.get_multimodal_mask(input_ids, pad_token_id, self.config.multimodal_special_token_list)
        if not self.training:  # 推理支持auto map 把多模态模块输出和input_ids 统一到一个device
            multimodal_embed = multimodal_embed.to(input_ids.device)
        if not fake_input:  # 检查多模态token 和 pad mask数量一致 （不正确的截断会导致该问题）
            assert pad_mask.sum() == multimodal_embed.shape[0]
        else:
            assert pad_mask.sum() <= 0

        # 合并 当前模态embeddings 和text embeddings
        input_ids = torch.where(pad_mask, torch.cumsum(pad_mask.view(-1).to(input_ids), dim=0).view(input_ids.shape)-1, input_ids)
        text_embedding = (1 - pad_mask.to(text_embedding)).unsqueeze(-1) * text_embedding  # pad token位置填0
        multimodal_embedding = torch.embedding(multimodal_embed, input_ids * pad_mask)  # 非 pad token 位置填idx=0位置结果
        multimodal_embedding = pad_mask.to(multimodal_embedding).unsqueeze(-1) * multimodal_embedding  # 非pad token 位置填0
        final_embedding = multimodal_embedding.to(text_embedding) + text_embedding

        if group_index is None:
            group_index = pad_mask.to(torch.int32)
        else:
            current_index = torch.max(group_index) + 1
            group_index += pad_mask.to(torch.int32) * current_index  # 假设模态无重叠

        return final_embedding, group_index

    def get_visual_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            images = None,
            patch_nums = None, 
            images_grid = None,
            videos = None,
            videos_patch_nums = None, 
            videos_grid = None,
            group_index = None,  # 某种模态的编号
        ): 
        if images is None or len(images) <= 0:
            images, images_grid, patch_nums = self.visual_model.fake_input(input_ids.device)
            image_fake_input = True
        else:
            image_fake_input = False
            
        if videos is None or len(videos) <= 0 :
            videos, videos_grid, videos_patch_nums = self.visual_model.fake_input(input_ids.device)
            video_fake_input = True
        else:
            video_fake_input = False
        
        visual_input = images + videos
        visual_grid = images_grid + videos_grid
        
        visual_input = torch.cat(visual_input, dim=0)
        visual_grid = torch.tensor(np.array(visual_grid))
        
        visual_embed = self.visual_model(visual_input, grid_thw=visual_grid)
        visual_embed = self.visual_bridge_model(visual_embed)

        assert sum(patch_nums) + sum(videos_patch_nums) == visual_embed.shape[0]
        images_embed = visual_embed[:sum(patch_nums)]
        videos_embed = visual_embed[sum(patch_nums):]
        
        final_embedding, group_index = self.get_multimodal_embed(input_ids, text_embedding, images_embed, self.config.visual_config.image_pad_token_id, image_fake_input, group_index=group_index)
        final_embedding, group_index = self.get_multimodal_embed(input_ids, final_embedding, videos_embed, self.config.video_config.video_place_token_id, video_fake_input, group_index=group_index)
        return final_embedding, group_index
    

    @torch.no_grad()
    def audio_fake_input(self, device):
        return torch.zeros(5, len(self.config.audio_config.vq_config.codebook_sizes), dtype=torch.int32, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audios_tokens: Optional[List|torch.Tensor] = None, # 音频token bs*seqlen*vq_num
        images: Optional[List|torch.Tensor] = None,
        patch_nums: Optional[torch.Tensor] = None, 
        images_grid: Optional[List|torch.Tensor] = None,
        videos: Optional[List|torch.Tensor] = None,
        videos_patch_nums: Optional[torch.Tensor] = None, 
        videos_grid: Optional[List|torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OmniModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = True if (return_dict is not None or self.training) else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        group_index, audio_decoder_ret = None, None
        if inputs_embeds is None:
            sp_input_ids = get_sequence_parallel_chunk(input_ids)
            inputs_embeds = self.embed_tokens(sp_input_ids)
            if audios_tokens is None or len(audios_tokens) <= 0 :
                audios_tokens = torch.zeros(5, len(self.config.audio_config.vq_config.codebook_sizes), dtype=torch.int32, device=input_ids.device)  # a fake input
                fake_input = True
            else:
                fake_input = False
            for i, audio_emb_layer in enumerate(self.audio_embed_layers):
                if i==0:
                    audio_embs = audio_emb_layer(audios_tokens[..., i]) 
                else:
                    audio_embs += audio_emb_layer(audios_tokens[..., i]) 
            inputs_embeds, group_index = self.get_multimodal_embed(sp_input_ids, inputs_embeds, audio_embs, self.config.audio_config.audio_pad_token_id, fake_input, group_index=group_index)
                
            if self.config.visual_config.enable or self.config.video_config.enable:
                inputs_embeds, group_index = self.get_visual_embed(sp_input_ids, inputs_embeds, images, patch_nums, images_grid, videos, videos_patch_nums, videos_grid, group_index=group_index)  # 注意更新group index

        if seqlens is not None and seqlens.ndim == 2:
            cu_seqlens = []
            offset, seqlen = 0, seqlens.size(1)
            for lens in seqlens:
                cu_seqlens.append(offset)
                cu_seqlens.extend((lens[(lens > 0) & (lens < seqlen)] + offset).tolist())
                offset += seqlen
            cu_seqlens.append(offset)
            seqlens = torch.tensor(cu_seqlens, dtype=seqlens.dtype, device=seqlens.device)
        elif seqlens is None and self.training:
            seqlens = torch.arange(
                end=input_ids.size(0) + 1,
                dtype=torch.int32,
                device=input_ids.device
            ) * input_ids.size(1)
        if seqlens is not None:
            attention_mask = None  # unset attention_mask to save memory

        if seqlens is None and attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, False, group_index)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    seqlens,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seqlens=seqlens,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    group_index=group_index,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, mask=None):
        norm_weight = nn.functional.normalize(self.weight)
        if mask is not None:
            mask = mask.to(norm_weight)
            norm_weight = norm_weight * mask + (1 - mask) * norm_weight.detach()
        return nn.functional.linear(hidden_states, norm_weight)


    def extra_repr(self) -> str:
        return f'in_features={self.hidden_size}, out_features={self.vocab_size}'

@dataclass
class OmniMMCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    audios_emb_for_infer: Optional[torch.FloatTensor] = None # 用于audio head 推理的 embeddings


class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config
        embed_size = config.hidden_size
        assert embed_size % 128 == 0
        num_heads = embed_size // 128
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,batch_first=True)
        self.layernorm1 = RMSNorm(embed_size)
        self.layernorm2 = RMSNorm(embed_size)
        self.linear1 = nn.Linear(embed_size * depth, 2 * embed_size)
        self.linear2 = nn.Linear(2 * embed_size * depth, embed_size)

    def forward(self, x):
        seq_len = x.size(1)
        res = x
        x = self.layernorm1(x)
        src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        _x, _ = self.self_attention(x, x, x,  is_causal=True, attn_mask=src_mask)
        res = _x + res  # (bs, sl, d)
        res = self.layernorm2(res)
        x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.config.hidden_size, -1, self.config.hidden_size)))
        x = torch.nn.functional.gelu(x)
        x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.config.hidden_size, -1, 2 * self.config.hidden_size)))
        return res + x

class OmniAudioHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.transformer_layers = nn.ModuleList([
            CasualDepthTransformerLayer(config, len(config.audio_config.vq_config.codebook_sizes)) 
            for _ in range(config.audio_config.audio_head_transformer_layers)
        ])
        self.headnorm = RMSNorm(hidden_size) 
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vq_size+1)
            for vq_size in config.audio_config.vq_config.codebook_sizes
        ])
        self.gradient_checkpointing = True

    def forward(self, x, audios_tokens, audio_emb_layers):
        cumsum_audio_embed = torch.stack([
            audio_emb_layers[i](audios_tokens[..., i]) 
            for i, vq_size in enumerate(self.config.audio_config.vq_config.codebook_sizes[:-1])
            ], dim=1)
        cumsum_audio_embed = torch.cumsum(cumsum_audio_embed, dim=1)  # (bs, depth-1, d)
        hidden_states = torch.concat([x.reshape(-1, 1, self.config.hidden_size), cumsum_audio_embed], dim=1)  # (bs, depth, d)
        assert hidden_states.size(1) == len(self.config.audio_config.vq_config.codebook_sizes)
        for i, tlayer in enumerate(self.transformer_layers):
            hidden_states  = tlayer(hidden_states,)
        hidden_states = self.headnorm(hidden_states)
        logits = [head(hidden_states[:,i]) for i, head in enumerate(self.heads)]
        return logits


class OmniForCausalLM(OmniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = OmniModel(config)
        self.audio_tokenizer = OmniAudioTokenizer(config)
        self.audio_head = OmniAudioHead(config)
        if config.use_norm_head:
            self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def main_device(self):
        return self.lm_head.weight.device

    def bind_processor(self, tokenizer, **kwargs):
        self.processor = OmniMMProcessor(
                tokenizer=tokenizer,
                config=self.config,
                **kwargs,
                )
        return self.processor

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
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audios: Optional[List|torch.Tensor] = None,
        audios_tokens: Optional[List|torch.Tensor] = None,
        encoder_length: Optional[torch.Tensor] = None,
        bridge_length: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None, 
        patch_nums: Optional[torch.Tensor] = None, 
        images_grid: Optional[torch.Tensor] = None, 
        videos: Optional[torch.Tensor] = None, 
        videos_patch_nums: Optional[torch.Tensor] = None, 
        videos_grid: Optional[torch.Tensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if audios_tokens is not None:
            assert isinstance(audios_tokens, torch.Tensor)
        else:
            if audios is None or len(audios) == 0:
                audios_tokens = None
            else:
                audios_tokens = self.audio_tokenizer(audios,encoder_length,bridge_length)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seqlens=seqlens,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            audios_tokens=audios_tokens,
            images=images,
            patch_nums=patch_nums, 
            images_grid=images_grid,
            videos=videos,
            videos_patch_nums=videos_patch_nums, 
            videos_grid=videos_grid,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        audios_emb_for_infer = hidden_states[:,-1,:]
        logits = self.lm_head(hidden_states)

        return OmniMMCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audios_emb_for_infer=audios_emb_for_infer
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, past_key_values[0][0].shape[-2]:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1)
            # position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, past_key_values[0][0].shape[-2]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        elif past_key_values is not None:
            model_inputs = {"input_ids": input_ids}
        else:
            model_inputs = {"input_ids": input_ids, 
                            "audios": kwargs.get("audios", None), "encoder_length": kwargs.get("encoder_length", None), "bridge_length": kwargs.get("bridge_length", None),
                            "audios_tokens": kwargs.get("audios_tokens", None),
                            "images": kwargs.get("images", None),
                            "videos": kwargs.get("videos", None)
                            }

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images_grid":  kwargs.get("images_grid"),
                "videos_grid":  kwargs.get("videos_grid"),
                "patch_nums":  kwargs.get("patch_nums"),
                "videos_patch_nums":  kwargs.get("videos_patch_nums"),
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response


class OmniAudioTokenizer(OmniPreTrainedModel):
    """
    Construct an audio tokenizer and decoder.
    """
    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.training = False
        self.eval()
        self.audio_model = OmniAudioEncoder(config.audio_config)
        self.audio_bridge_model = OmniAudioVQBridgeTokenizer(config)
        if config.vocoder_config.enable:
            self.audio_decoder = OmniAudioDecoder(config)
            if config.flow_matching_config.enable:
                self.audio_flow_matching_decoder = OmniAudioFlowMatchingDecoder(config)

    def encode(self, x, encoder_length: Optional[torch.Tensor] = None,
        bridge_length: Optional[torch.Tensor] = None):
        audio_emb = self.audio_model(x, encoder_length)
        audios_tokens = self.audio_bridge_model(audio_emb, bridge_length)
        return audios_tokens
    
    def decode(self, audio_code_ids, bridge_length: Optional[torch.Tensor] = None):
        assert self.config.vocoder_config.enable, "Vocoder is not enabled in config."
        audio_emb = self.audio_bridge_model.decode(audio_code_ids)
        audio_dec = self.audio_decoder(
            audio_emb.to(next(self.audio_decoder.parameters())), bridge_length
        )
        if self.config.flow_matching_config.enable:
            if self.config.flow_matching_config.use_hidden_states_before_dconv2:
                hidden_states, hidden_states_length = (
                    self.audio_flow_matching_decoder.unpack_hidden_states(
                        audio_dec.hidden_states_before_dconv2,
                        audio_dec.output_length_before_dconv2,
                    )
                )
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    hidden_states, hidden_states_length
                )

            else:
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    audio_dec.refined_mel, audio_dec.mel_length
                )
            return audio_flow_matching_decoder_ret
        else:
            return audio_dec
    
    @torch.no_grad()    
    def forward(self, audios, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        self.eval()
        audios_tokens = self.encode(audios, encoder_length, bridge_length)
        return audios_tokens
