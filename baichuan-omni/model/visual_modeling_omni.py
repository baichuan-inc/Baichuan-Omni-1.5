
from typing import List, Optional, Tuple, Union
import torch, math
import torch.utils.checkpoint
from torch import nn
import transformers
from flash_attn import flash_attn_varlen_func
from transformers.activations import ACT2FN
from PIL import Image
import io, fire
from torch.nn import functional as F

class OmniVisualEncoder(transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config_attn_implementation = 'flash_attention_2'
        self.gradient_checkpointing = True  # 强制开启
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.merge_size = config.merge_size if hasattr(config, 'merge_size') else 2
        del self.merger
        
    def forward(
        self,
        pixel_values: torch.Tensor, 
        grid_thw: torch.Tensor,
    ):
        hidden_states = pixel_values.to(self.get_dtype())
        grid_thw = grid_thw.to(pixel_values.device)
        
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return hidden_states
    
    @torch.no_grad()
    def fake_input(self, device):
        merge_size = max(self.merge_size, self.config.spatial_merge_size)
        fake_image = torch.zeros([
            1,
            self.config.temporal_patch_size,
            3,
            merge_size // self.config.spatial_merge_size,
            self.config.spatial_merge_size,
            self.config.patch_size,
            merge_size // self.config.spatial_merge_size,
            self.config.spatial_merge_size,
            self.config.patch_size,
        ], dtype=torch.float32, device=device)
        patches = fake_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            merge_size * merge_size, 3 * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size
        )
        return [flatten_patches], [(1, merge_size, merge_size)], [1]


class OmniVisualBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.hidden_size = config.embed_dim * (self.merge_size**2)
        self.ln_q = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


if __name__ == '__main__':
    fire.Fire()
    