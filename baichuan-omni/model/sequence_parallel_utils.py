from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flash_attn import flash_attn_varlen_func
try:
    import deepspeed.comm as dist
except:
    dist = None


try:
    from utils import (
        get_sequence_parallel_group,
        get_sequence_parallel_size,
        get_sequence_parallel_rank
    )
except (ModuleNotFoundError, ImportError):
    # 从 utils 获取seq parallel设置，import不成功默认为不开启
    get_sequence_parallel_group = lambda : None
    get_sequence_parallel_size = lambda : 1
    get_sequence_parallel_rank = lambda : 0


def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    # [sp_size, seq_len//sp_size, batch_size, head_num // sp_size, head_dim] -->
    # [seq_len//sp_size,batch_size, sp_size, head_num // sp_size, head_dim]
    if scatter_idx < 2:
        output = output.transpose(0, 1).transpose(1, 2).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: 'dist.ProcessGroup', input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


# import from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
# but fix some bugs for 符合训练的维度设置
class DistributedAttention(nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: nn.Module,
        sequence_process_group: 'dist.ProcessGroup',
        scatter_idx: int = 2,
        gather_idx: int = 0,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
    
    def pad_attention_head(self, query: Tensor, key: Tensor, value: Tensor):
        # 将输入的head 维度pad到sp_size的倍数
        sp_size = torch.distributed.get_world_size(self.spg)
        pad_size = (sp_size - query.size(1) % sp_size) % sp_size
        if pad_size > 0:
            # [bs, num_head, seq_len, head_dim] -> [bs, num_head+pad_size, seq_len, head_dim]
            query = torch.nn.functional.pad(query, (0,0,0,0,0,pad_size), value = 0.01)
            key = torch.nn.functional.pad(key, (0,0,0,0,0,pad_size), value = 0.01)
            value = torch.nn.functional.pad(value, (0,0,0,0,0,pad_size),value=0.0)
        return query, key, value
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer [batch_size, num_head, seq_len, head_dim]
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # [batch_size,num_head,seq_len, head_dim ]trans to [seq_len,batch_size,num_head,head_dim]
        origin_num_head = query.size(1)
        query, key, value = self.pad_attention_head(query,key,value)

        query = query.transpose(1,2).transpose(0,1)
        key = key.transpose(1,2).transpose(0,1)
        value = value.transpose(1,2).transpose(0,1)
        #in shape : e.g.,  [s/p,bs,h,head_dim]
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx).transpose(0,1).transpose(1,2).contiguous()
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx).transpose(0,1).transpose(1,2).contiguous()
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx).transpose(0,1).transpose(1,2).contiguous()

        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        context_layer = context_layer.transpose(0,1).contiguous()
        # [seq_len, batch_size, num_head, head_dim]
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        return output.transpose(0,1)[:,:,:origin_num_head,:]


class LocalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, *args, use_flash=True, **kwargs):
        # input q,k,v [batch_size, num_head, seq_len, head_dim]
        # output [batch_size, seq_len, num_head, head_dim]
        if use_flash:
            q_len, num_heads = q.shape[2], q.shape[1]
            q = q.transpose(1,2).reshape(-1, num_heads, self.head_dim)
            k = k.transpose(1,2).reshape(-1, num_heads, self.head_dim)
            v = v.transpose(1,2).reshape(-1, num_heads, self.head_dim)
            return flash_attn_varlen_func(q,k,v,*args, **kwargs).reshape(-1,q_len, num_heads, self.head_dim)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                attn_output = F.scaled_dot_product_attention(
                    q,k,v, *args, **kwargs)
            attn_output = attn_output.transpose(1, 2)
            return attn_output


def create_attention_layer(hidden_size, num_heads, head_dim):
    if get_sequence_parallel_group() is None:
        return LocalAttention(hidden_size, num_heads, head_dim)
    else:
        return DistributedAttention(
            local_attention=LocalAttention(hidden_size, num_heads, head_dim),
            sequence_process_group=get_sequence_parallel_group()
        )


def get_sequence_parallel_chunk(tensor, dim=1, shift=0):
    assert tensor.size(dim) % get_sequence_parallel_size() == 0
    original_size = tensor.size(dim)
    if shift:
        tensor = tensor.split([shift, tensor.size(dim) - shift], dim=dim)[1]
    if get_sequence_parallel_group() is None:
        return tensor
    else:
        chunk_size = original_size // get_sequence_parallel_size()
        return tensor.split(chunk_size, dim=dim)[get_sequence_parallel_rank()]
