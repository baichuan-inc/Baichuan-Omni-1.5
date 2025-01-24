import torch, fire
from typing import Optional
import torch.distributed
from torch.nn import functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn
import numpy as np
import deepspeed
from transformers.activations import ACT2FN
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
try:
    from .vector_quantize import VectorQuantize
except:
    from vector_quantize import VectorQuantize

from .flow_matching import (
    ConditionalDecoder,
    ConditionalCFM,
)

import math
import copy

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1  # 转成下标
    return sequence_mask, unpacking_index

def unpack_hidden_states(hidden_states, lengths):
    bsz = lengths.shape[0]
    sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    hidden_states = torch.where(
        sequence_mask, hidden_states, 0
    )  # 3d (bsz, max_input_len, d)
    return hidden_states


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    

class OmniWhisperAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class OmniWhisperTransformerLayer(nn.Module):
    def __init__(
        self,
        act,
        d_model,
        encoder_attention_heads,
        encoder_ffn_dim,
        causal,
        ln_type="LayerNorm",
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = OmniWhisperAttention(
            self.embed_dim, encoder_attention_heads, causal
        )

        if ln_type == "LayerNorm":
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.self_attn_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        if ln_type == "LayerNorm":
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.final_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

    def forward(
        self, hidden_states: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, seq_len)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (
            hidden_states.dtype == torch.float16
            or hidden_states.dtype == torch.bfloat16
        ) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


class OmniAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config._attn_implementation = 'flash_attention_2'  #
        self.config = config
        self.max_source_positions = (config.max_audio_seconds * config.sampling_rate // config.hop_length) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=config.kernel_size,
                               stride=config.stride_size, padding=1)
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, config.d_model))  # 1500 * d

        self.layers = nn.ModuleList([OmniWhisperTransformerLayer(
            ACT2FN[config.activation_function],
            config.d_model,
            config.encoder_attention_heads,
            config.encoder_ffn_dim,
            False) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    @torch.no_grad()
    def fake_input(self, device):
        input_features = torch.rand([2, self.config.num_mel_bins, 10], dtype=torch.float32, device=device)
        encoder_length = torch.ones([2], dtype=torch.int32, device=device) * 3
        bridge_length = torch.ones([2], dtype=torch.int32, device=device)
        return input_features, encoder_length, bridge_length

    def forward(
            self,
            input_features,
            output_length,
    ):
        input_features = input_features.to(self.conv1.weight.dtype)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (bs, channels, frames)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (bs, channels, frames // 2)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (bs, frams, channels)
        bsz, tgt_len, _ = inputs_embeds.size()
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]
        else:
            current_positional_embedding = self.positional_embedding
        hidden_states = (inputs_embeds.to(torch.float32) + current_positional_embedding).to(inputs_embeds.dtype)

        # packing hidden states
        attention_mask, unpacking_index = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(torch.sum(output_length),
                                                                                self.config.d_model)
        
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, output_length)
        hidden_states = self.layer_norm(hidden_states)
        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, tgt_len, self.config.d_model)
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states


class CasualConvTranspose1d(nn.Module):  # 反卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, hidden_states, input_length, output_dim=None):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        bsz = input_length.shape[0]

        if output_dim is None:
            output_dim = hidden_states.dim()
        if hidden_states.dim() <= 2:  # unpack sequence to 3d
            sequence_mask, unpacking_index = get_sequence_mask(hidden_states, input_length)
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, torch.max(input_length),
                                                                                       self.in_channels)
            hidden_states = torch.where(sequence_mask, hidden_states, 0)  # 3d (bsz, max_input_len, d)

        hidden_states = hidden_states.transpose(2, 1)  # (N, L, C) -> (N, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (N, C, L) -> (N, L, C)

        casual_padding_right = max(0, kernel_size - stride)
        hidden_states = hidden_states[:, :hidden_states.shape[1] - casual_padding_right,
                        :]
        output_length = (input_length - 1) * stride + kernel_size - casual_padding_right
        sequence_mask, _ = get_sequence_mask(hidden_states, output_length)
        if output_dim <= 2:
            hidden_states = torch.masked_select(hidden_states, sequence_mask).view(-1, self.out_channels)
        else:
            hidden_states = torch.where(sequence_mask, hidden_states, 0)
            hidden_states = hidden_states[:, :torch.max(output_length), :]  # 截断到最大有效长度
        return hidden_states, output_length


class MelSpecRefineNet(nn.Module):
    """
    # post net, coarse to refined mel-spectrogram frames
    # ref1: Autoregressive Speech Synthesis without Vector Quantization
    # ref2: CosyVoice length_regulator.py
    # ref3: Neural Speech Synthesis with Transformer Network https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
    """

    def __init__(self, encoder_config, vocoder_config):
        super().__init__()
        self.encoder_config = encoder_config
        self.vocoder_config = vocoder_config

        layers = nn.ModuleList([])
        in_channels = self.vocoder_config.num_mel_bins
        for i, out_channels in enumerate(self.vocoder_config.channels[:-1]):
            module = nn.Conv1d(in_channels, out_channels, 5, 1, 2)  # cosyvoice kernel=3, stride=1, pad=1
            in_channels = out_channels
            norm = nn.GroupNorm(1, out_channels)
            act = nn.Mish()
            layers.extend([module, norm, act])
        layers.append(nn.Conv1d(in_channels, self.vocoder_config.num_mel_bins, 1, 1))  # projector
        self.layers = nn.Sequential(*layers)

    def compute_output_length(self, input_length):
        output_length = input_length.to(
            torch.float32) * self.encoder_config.hop_length / self.encoder_config.sampling_rate
        output_length = output_length * self.vocoder_config.sampling_rate / self.vocoder_config.hop_length
        return output_length.to(torch.int64)

    def forward(self, coarse_mel, input_length, output_length=None):
        bsz, _, d = coarse_mel.shape
        assert (d == self.vocoder_config.num_mel_bins)
        if output_length is None or not self.training:
            output_length = self.compute_output_length(input_length)
        coarse_mel, default_dtype = coarse_mel[:, :torch.max(input_length), :], coarse_mel.dtype
        coarse_mel = F.interpolate(coarse_mel.to(torch.float32).transpose(1, 2).contiguous(), size=output_length.max(),
                                   mode='nearest').to(default_dtype)
        refined_mel = self.layers(coarse_mel).transpose(1, 2).contiguous()  # (bs, t, d)
        coarse_mel = coarse_mel.transpose(1, 2)  # (bs, max(output_length), d)
        refined_mel += coarse_mel  # residual conntection
        sequence_mask, _ = get_sequence_mask(refined_mel, output_length)
        coarse_mel = torch.where(sequence_mask, coarse_mel, 0)
        refined_mel = torch.where(sequence_mask, refined_mel, 0)
        return refined_mel, coarse_mel, output_length


@dataclass
class OmniAudioDecoderOutput(ModelOutput):
    refined_mel: Optional[torch.FloatTensor] = None
    coarse_mel: Optional[torch.FloatTensor] = None
    mel_length: Optional[torch.Tensor] = None
    hidden_states_before_dconv2: Optional[torch.FloatTensor] = None
    output_length_before_dconv2: Optional[torch.Tensor] = None


class OmniAudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.audio_config
        self.vocoder_config = config.vocoder_config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length

        self.dconv1 = CasualConvTranspose1d(
            self.config.d_model,
            self.config.d_model,
            self.config.decoder_kernel_size,
            self.config.avg_pooler,
        )
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, self.config.d_model))
        # causal transformer layers
        self.layers = nn.ModuleList(
            [OmniWhisperTransformerLayer(
                ACT2FN[self.config.activation_function],
                self.config.d_model,
                self.config.decoder_attention_heads,
                self.config.decoder_ffn_dim,
                True  # causal
            ) for _ in range(self.config.decoder_layers)
            ])
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.dconv2 = CasualConvTranspose1d(
            self.config.d_model,
            self.vocoder_config.num_mel_bins,
            self.config.decoder_kernel_size,
            self.config.decoder_stride_size
        )
        self.post_net = MelSpecRefineNet(config.audio_config, config.vocoder_config)
        self.gradient_checkpointing = True

    @torch.no_grad()
    def fake_input(self, device):
        audio_embed = torch.rand([1, 10, self.config.d_model], dtype=torch.float32, device=device)
        input_length = torch.ones([1], dtype=torch.int32, device=device) * 10
        mel_labels_length = self.post_net.compute_output_length(input_length)
        return audio_embed, input_length, None, mel_labels_length

    def forward(self,
                audio_embed,
                input_length,
                mel_labels=None,
                mel_labels_length=None,
                fake_input=False,
                ):
        if fake_input:
            audio_embed, input_length, mel_labels, mel_labels_length = self.fake_input(self.layer_norm.weight.device)

        assert (audio_embed.shape[-1] == self.config.d_model)
        audio_embed = audio_embed.to(self.layer_norm.weight)  # device and type
        audio_embed, output_length = self.dconv1(audio_embed, input_length, output_dim=3)  # (b, l*2, d_model)
        _, tgt_len, _ = audio_embed.size()
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]
        else:
            current_positional_embedding = self.positional_embedding
        hidden_states = (audio_embed.to(torch.float32) + current_positional_embedding).to(audio_embed.dtype)

        # packing hidden states
        attention_mask, _ = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(torch.sum(output_length), self.config.d_model)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, output_length)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states_before_dconv2 = hidden_states
        output_length_before_dconv2 = output_length

        coarse_mel, output_length = self.dconv2(hidden_states, output_length, output_dim=3)
        refined_mel, coarse_mel, mel_labels_length = self.post_net(coarse_mel, output_length, mel_labels_length)

        return OmniAudioDecoderOutput(
            refined_mel=refined_mel,
            coarse_mel=coarse_mel,
            mel_length=mel_labels_length,
            hidden_states_before_dconv2=hidden_states_before_dconv2,
            output_length_before_dconv2=output_length_before_dconv2,
        )


class OmniAudioVQBridgeTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.audio_config
        self.gradient_checkpointing = False
        self.intermediate_dim = self.config.d_model * self.config.avg_pooler 
        self.gate_proj = nn.Conv1d(self.config.d_model, self.intermediate_dim, self.config.avg_pooler, self.config.avg_pooler, bias=False)
        self.up_proj = nn.Conv1d(self.config.d_model, self.intermediate_dim, self.config.avg_pooler, self.config.avg_pooler, bias=False)

        self.down_proj = nn.Linear(self.intermediate_dim, self.intermediate_dim, bias=False)
        self.act_fn = ACT2FN['silu']
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.proj_decoder = nn.Linear(self.intermediate_dim, self.config.d_model)

        self.vq_list = nn.ModuleList([])
        for idx, codebook_size in enumerate(self.config.vq_config.codebook_sizes):
            vq_config = copy.deepcopy(self.config.vq_config)
            vq_config.dim = self.intermediate_dim
            vq_config.codebook_size = codebook_size
            self.vq_list.append(VectorQuantize(vq_config))
        for vq_layer in self.vq_list:
            deepspeed.zero.register_external_parameter(self, vq_layer.codebook.embed)

    def rvq_op(self, inputs, output_length):
        def rvq_layer_op(vq_layer, residual_encoding, output_length):
            q_v_i, code_ids_i = vq_layer(residual_encoding, output_length)
            residual_encoding = residual_encoding.float() - q_v_i.float()
            residual_encoding = residual_encoding.to(inputs.dtype)
            return residual_encoding, code_ids_i
            
        cmt_loss, residual_encoding = 0, inputs
        code_ids_list = []
        for i, vq_layer in enumerate(self.vq_list):
            residual_encoding, code_ids_i = rvq_layer_op(vq_layer, residual_encoding, output_length)
            code_ids_list.append(code_ids_i)
        return torch.stack(code_ids_list, -1)
    
    def forward(self, x, output_length):
        batch_size, _, _ = x.shape
        output_length = output_length.to(x.device)
    
        if x.shape[1] % self.config.avg_pooler != 0:
            x = F.pad(x, (0, 0, 0, self.config.avg_pooler - x.shape[1] % self.config.avg_pooler), "constant", 0)
        xt = x.permute(0, 2, 1)
        g = self.gate_proj(xt).permute(0, 2, 1)  # (bs, sl//poolersizre+1, d*2)
        u = self.up_proj(xt).permute(0, 2, 1)
        x = x.reshape(batch_size, -1, self.intermediate_dim)  # (bs, sl//poolersizre+1, d*2)

        c = self.down_proj(self.act_fn(g) * u)
        res = self.layer_norm(c + x)
        valid_mask, _ = get_sequence_mask(res, output_length)
        code_ids = self.rvq_op(res, output_length)
        code_ids = torch.masked_select(code_ids, valid_mask).reshape(-1, len(self.vq_list))  # (sum(valid_sequence_length), vq_num)
        return code_ids

    @torch.no_grad()
    def decode(self, code_ids):
        vq_num = code_ids.shape[-1]
        res = sum(self.vq_list[i].get_output_from_indices(code_ids[:, i]).float() for i in range(vq_num-1,-1,-1)).to(self.proj_decoder.weight)
        decoder_emb = self.proj_decoder(res.to(self.proj_decoder.weight))
        return decoder_emb
    
    @torch.no_grad()
    def recover(self, code_ids):
        vq_num = code_ids.shape[-1]
        res = sum(self.vq_list[i].get_output_from_indices(code_ids[:, i]).float() for i in range(vq_num-1,-1,-1)).to(self.proj_decoder.weight)
        return res


class FlowmatchingPrenet(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        out_feat_dim,
        d_model,
        attention_heads,
        ffn_dim,
        nlayers,
        activation_function,
        max_source_positions,
        target_mel_length_scale_ratio,
    ):
        super().__init__()

        self.d_model = d_model
        self.target_mel_length_scale_ratio = target_mel_length_scale_ratio
        self.gradient_checkpointing = False

        self.register_buffer(
            "positional_embedding", sinusoids(max_source_positions, d_model)
        )

        self.in_mlp = nn.Sequential(
            nn.Linear(input_feat_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.transformer_layers = nn.ModuleList(
            [
                OmniWhisperTransformerLayer(
                    act=ACT2FN[activation_function],
                    d_model=d_model,
                    encoder_attention_heads=attention_heads,
                    encoder_ffn_dim=ffn_dim,
                    causal=True,  # causal
                    ln_type="RMSNorm",
                )
                for _ in range(nlayers)
            ]
        )

        self.final_norm = RMSNorm(self.d_model)
        self.out_proj = nn.Linear(d_model, out_feat_dim, bias=False)

    def compute_output_length(self, input_length):
        output_length = input_length.float() * self.target_mel_length_scale_ratio
        return output_length.to(torch.int64)

    def forward(self, input_feat, input_length, output_length=None):
        """
        Args:
            input_feat: [B, T, input_feat_dim]
            input_length: [B]
            output_length: [B]

        """
        if output_length is None or not self.training:
            output_length = self.compute_output_length(input_length)

        input_feat = input_feat[:, : input_length.max(), :]  # [B, T, D]
        orig_dtype = input_feat.dtype

        input_feat = F.interpolate(
            input=input_feat.to(torch.float32).transpose(1, 2).contiguous(),
            size=output_length.max(),
            mode="nearest",
        ).to(orig_dtype)
        input_feat = input_feat.transpose(1, 2).contiguous()  # [B, T, D]
        hidden_states = self.in_mlp(input_feat)

        # packing hidden states
        bsz, tgt_len, d_model = hidden_states.shape
        attention_mask, unpacking_index = get_sequence_mask(
            hidden_states, output_length
        )
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.d_model
        )

        for idx, encoder_layer in enumerate(self.transformer_layers):
            hidden_states = encoder_layer(hidden_states, output_length)

        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
            bsz, tgt_len, d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)

        hidden_states = self.final_norm(hidden_states)
        output = self.out_proj(hidden_states)
        return output, output_length


@dataclass
class OmniAudioFlowMatchingDecoderOutput(ModelOutput):
    flow_matching_mel: Optional[torch.FloatTensor] = None
    flow_matching_mel_lengths: Optional[torch.FloatTensor] = None


class OmniAudioFlowMatchingDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.flow_matching_config
        self.in_channels = self.config.in_channels
        self.spk_emb_dim = self.config.spk_emb_dim
        self.diffusion_steps = self.config.diffusion_steps
        self.cal_mel_mae = self.config.cal_mel_mae
        self.forward_step = -1

        self.prenet = FlowmatchingPrenet(
            input_feat_dim=self.config.prenet_in_dim,
            out_feat_dim=self.config.prenet_out_dim,
            d_model=self.config.prenet_d_model,
            attention_heads=self.config.prenet_attention_heads,
            ffn_dim=self.config.prenet_ffn_dim,
            nlayers=self.config.prenet_nlayers,
            activation_function=self.config.prenet_activation_function,
            max_source_positions=self.config.prenet_max_source_positions,
            target_mel_length_scale_ratio=self.config.prenet_target_mel_length_scale_ratio,
        )

        self.conditional_decoder = ConditionalDecoder(
            in_channels=self.in_channels * 2 + self.spk_emb_dim,
            out_channels=self.in_channels,
            causal=True,
            channels=self.config.channels,
            dropout=self.config.dropout,
            attention_head_dim=self.config.attention_head_dim,
            n_blocks=self.config.n_blocks,
            num_mid_blocks=self.config.num_mid_blocks,
            num_heads=self.config.num_heads,
            act_fn=self.config.act_fn,
        )

        self.cfm = ConditionalCFM(
            in_channels=self.in_channels,
            cfm_params=self.config.cfm_params,
            n_spks=0,
            spk_emb_dim=self.spk_emb_dim,
        )
        

    def unpack_hidden_states(self, hidden_states, output_length):
        unpacked = unpack_hidden_states(hidden_states, output_length)
        return unpacked, output_length

    def forward(
        self, refined_mel, input_length, mel_labels=None, mel_labels_length=None
    ):
        """
        :param refined_mel: [bs,  max_input_len, mel_bin]
        :param input_length:  [batch_size]
        :param refined_mel: [bs, mel_bin, max_input_len]
        :return:
        """
        self.forward_step += 1

        orig_dtype = refined_mel.dtype
        prenet_mae_metric = torch.tensor(0.0).to(refined_mel.device)
        prenet_regression_loss = torch.tensor(0.0).to(refined_mel.device)

        if self.prenet is not None:
            refined_mel = refined_mel[:, : torch.max(input_length), :]
            if mel_labels_length is None:
                mel_labels_length = self.prenet.compute_output_length(input_length)
            refined_mel, input_length = self.prenet(
                refined_mel, input_length, mel_labels_length
            )

        float_dtype = refined_mel.dtype
        refined_mel = refined_mel.float()
        input_length = input_length.long()

        refined_mel = refined_mel[:, : torch.max(input_length), :]
        sequence_mask, unpacking_index = get_sequence_mask(refined_mel, input_length)
        refined_mel = refined_mel.transpose(1, 2)  # (bs, mel_bin, max_input_len)
        sequence_mask = sequence_mask.transpose(2, 1)  # (bs, 1, sl)

        fm_mel = self.cfm.forward(
            estimator=self.conditional_decoder,
            mu=refined_mel.to(float_dtype),
            mask=sequence_mask.float(),
            n_timesteps=self.diffusion_steps,
        )
        return OmniAudioFlowMatchingDecoderOutput(
            flow_matching_mel=fm_mel.transpose(1, 2),
            flow_matching_mel_lengths=mel_labels_length,
        )
