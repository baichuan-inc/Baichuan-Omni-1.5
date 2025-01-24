import torch, random
from torch.nn import functional as F
from torch import nn
import numpy as np
from torch.cuda.amp import autocast

def uniform_init(*shape):
    t = torch.zeros(shape)
    nn.init.kaiming_uniform_(t)
    return t

def cdist(x, y):
    x2 = torch.sum(x ** 2, dim=-1, keepdims=True)  # (b, 1)
    y2 = torch.sum(y ** 2, dim=-1).reshape(1, -1)  # (1, c)
    xy = torch.einsum('bd,cd->bc', x, y) * -2
    return (x2 + y2 + xy).clamp(min=0).sqrt()  #  (b, c)

def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1  # 转成下标
    return sequence_mask, unpacking_index


class EuclideanCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            init_std=0.02,
    ):
        super().__init__()
        self.init_std = init_std
        self.dim = dim
        self.codebook_size = codebook_size

        embed = uniform_init(codebook_size, dim).to(torch.float32)
        self.cluster_size = nn.Parameter(torch.ones(codebook_size))
        self.embed_avg = nn.Parameter(embed.clone())
        self.embed = nn.Parameter(embed)
        del embed

    @autocast(enabled=True, dtype=torch.float32)
    @torch.no_grad()
    def forward(self, x):
        assert(len(x.shape) == 2)
        assert(x.dtype == torch.float32)
        embed = self.embed.detach().to(x.device)
        dist = -cdist(x, embed)  # dist((bs*sl, d), (c, d)) --> (bs*sl, c)
        embed_ind = dist.argmax(dim=-1)
        quantize = embed[embed_ind]  # (bs*sl, d)
        return quantize, embed_ind, dist

class VectorQuantize(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.codebook = EuclideanCodebook(dim=config.dim, codebook_size=config.codebook_size)

    def forward(self, x, input_length):
        batch_size, seq_len, _ = x.shape
        mask, unpacking_index = get_sequence_mask(x, input_length)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.masked_select(x, mask).reshape(-1, self.config.dim)  # (bs*sl?, d)
        quantize, embed_ind, _ = self.codebook(x)
        quantize = torch.index_select(quantize, 0, unpacking_index).view(batch_size, seq_len, self.config.dim)
        quantize = torch.where(mask, quantize, 0) 
        embed_ind = torch.index_select(embed_ind.reshape(-1, 1), 0, unpacking_index).view(batch_size, seq_len, 1)
        embed_ind = torch.where(mask, embed_ind, -1).squeeze()
        return quantize, embed_ind

    def get_output_from_indices(self, indices):
        return self.codebook.embed[indices]