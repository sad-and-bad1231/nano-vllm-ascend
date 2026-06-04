import torch
from torch import nn

from nanovllm.utils.context import get_context


def store_kvcache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    assert slot_mapping.numel() == key.size(0)
    valid = slot_mapping >= 0
    if not valid.any():
        return
    slots = slot_mapping[valid].long()
    flat_k_cache = k_cache.view(-1, key.size(1), key.size(2))
    flat_v_cache = v_cache.view(-1, value.size(1), value.size(2))
    flat_k_cache.index_copy_(0, slots, key[valid])
    flat_v_cache.index_copy_(0, slots, value[valid])


def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    assert num_heads % x.size(1) == 0
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def _attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    query_start: int = 0,
) -> torch.Tensor:
    k = _repeat_kv(k, q.size(1))
    v = _repeat_kv(v, q.size(1))
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if causal:
        q_pos = torch.arange(query_start, query_start + q.size(0), device=q.device)
        k_pos = torch.arange(k.size(0), device=q.device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("hqk,khd->qhd", probs, v)


def _gather_cache(cache: torch.Tensor, block_table: torch.Tensor, length: int) -> torch.Tensor:
    block_size = cache.size(1)
    slots = []
    for block_id in block_table.tolist():
        if block_id == -1 or len(slots) >= length:
            break
        start = block_id * block_size
        end = min(start + block_size, start + length - len(slots))
        slots.extend(range(start, end))
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=cache.device)
    return cache.view(-1, cache.size(-2), cache.size(-1)).index_select(0, slot_tensor)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache_torch(k, v, k_cache, v_cache, context.slot_mapping)

        outputs = []
        if context.is_prefill:
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
            for i in range(len(cu_q) - 1):
                qs, qe = cu_q[i], cu_q[i + 1]
                ks, ke = cu_k[i], cu_k[i + 1]
                if context.block_tables is None:
                    seq_k, seq_v = k[ks:ke], v[ks:ke]
                else:
                    seq_k = _gather_cache(k_cache, context.block_tables[i], ke - ks)
                    seq_v = _gather_cache(v_cache, context.block_tables[i], ke - ks)
                query_start = (ke - ks) - (qe - qs)
                outputs.append(_attention_torch(q[qs:qe], seq_k, seq_v, self.scale, True, query_start))
        else:
            block_tables = context.block_tables
            for i in range(q.size(0)):
                seq_len = int(context.context_lens[i].item())
                seq_k = _gather_cache(k_cache, block_tables[i], seq_len)
                seq_v = _gather_cache(v_cache, block_tables[i], seq_len)
                outputs.append(_attention_torch(q[i:i + 1], seq_k, seq_v, self.scale, False))
        return torch.cat(outputs, dim=0)
