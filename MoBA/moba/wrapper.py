import torch
import torch.nn as nn
import transformers
from typing import Callable, Tuple, Optional
from flash_attn import flash_attn_func
from .config import MoBAConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    apply_rotary_pos_emb, 
    Qwen2RotaryEmbedding  # <--- 新增导入这个！
)
def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def fa_to_hf(x: torch.Tensor, batch: int):
    """
    Args:
        x (torch.Tensor): [batch * seqlen, heads, head_dim]

    Returns:
        torch.Tensor: [batch, heads, seqlen, head_dim]
    """
    return x.view(batch, -1, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)


def moba_layer(
    moba_impl: Callable,
    moba_config: MoBAConfig,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # q_test = query.transpose(1, 2)
    # k_test = key.transpose(1, 2)
    # v_test = value.transpose(1, 2)
    
    # # 2. 打印形状看看是否符合预期 (调试完可删除)
    # # print(f"DEBUG: q={q_test.shape}, k={k_test.shape}")
    
    # # 3. 调用官方算子
    # out = flash_attn_func(q_test, k_test, v_test, dropout, scaling, causal=True)
    
    # # 4. 直接返回，不走下面的 MoBA 逻辑
    # return out, None
    """
    Args:
        query (torch.Tensor): [batch, q_heads, q_len, head_dim]
        key (torch.Tensor): [batch, kv_heads, kv_len, head_dim]
        value (torch.Tensor): [batch, kv_heads, kv_len, head_dim]

    Returns:
        attn_output (torch.Tensor): [batch, q_len, q_heads, head_dim]
        attn_weights (None): not needed
    """
    assert module.is_causal
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape
    if q_len == kv_len:
        # prefill phase
        query = hf_to_fa(query)
        key = hf_to_fa(key)
        value = hf_to_fa(value)
        kv_replicas = q_heads // kv_heads
        # key = torch.repeat_interleave(key, kv_replicas, dim=1)
        # value = torch.repeat_interleave(value, kv_replicas, dim=1)
        # cu_seqlens_k = torch.cumsum(
        #     torch.tensor([0] + [kv_len] * batch, device=query.device),
        #     dim=0,
        #     dtype=torch.int32,
        # )
        if kv_replicas > 1:
            key = key.repeat_interleave(kv_replicas, dim=1)
            value = value.repeat_interleave(kv_replicas, dim=1)
            
        cu_seqlens_k = torch.arange(
            0, (batch + 1) * kv_len, step=kv_len, dtype=torch.int32, device=query.device
        )
        out = moba_impl(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=kv_len,
            moba_chunk_size=moba_config.moba_chunk_size,
            moba_topk=moba_config.moba_topk,
        )
        out = fa_to_hf(out, batch) 

    else:
        # decode phase
        # TODO release paged attn implementation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = flash_attn_func(query, key, value, dropout, scaling, True)
        out = out.transpose(1, 2)
    # out = fa_to_hf(out, batch)
    return out, None
class Qwen2MoBAAdaptor(Qwen2Attention):
    """
    MoBA 适配器：严格匹配 Transformers 4.48+ 的 Qwen2 接口
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 依赖类属性注入，不覆盖 self.moba_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # 新版签名
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. 投影
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        
        # 获取维度信息
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads

        # 投影并转为 [B, H, S, D]
        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # 2. 应用 RoPE (直接使用传入的 position_embeddings)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. 更新 KV Cache
        if past_key_values is not None:
            # 必须按照源码传递 cache_kwargs
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 4. MoBA 计算
        scaling = head_dim ** -0.5
        
        # 传入 [B, H, S, D]
        attn_output, _ = moba_layer(
            moba_impl=self.moba_impl,
            moba_config=self.moba_config,
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=scaling
        )

        # moba_layer 返回的是 [B, H, S, D]，需要转回 [B, S, H * D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        
        # 5. 输出投影
        attn_output = self.o_proj(attn_output)

        # 6. 返回值 (严格遵循源码：只返回2个值)
        # 注意：源码中没有返回 past_key_values，因为它是原地更新的引用
        return attn_output, None