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
def moba_layer(
    moba_impl: Callable,
    moba_config: MoBAConfig,
    module: torch.nn.Module,
    query: torch.Tensor,   # [batch, q_heads, q_len, head_dim]
    key: torch.Tensor,     # [batch, kv_heads, kv_len, head_dim]
    value: torch.Tensor,   # [batch, kv_heads, kv_len, head_dim]
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape
    kv_replicas = q_heads // kv_heads

    # 1. 区分阶段：Prefill (q_len > 1) vs Decode (q_len == 1)
    # 注意：在训练或长文本 Prefill 时，q_len == kv_len
    if q_len > 1:
        # === Prefill 阶段 (MoBA 逻辑) ===
        
        # 将形状转为 Flash-Attn 要求的 VarLen 格式: [Total_Seq, Heads, Dim]
        # 使用 transpose(1, 2) 避免不连续导致的显存拷贝，尽量 contiguous
        query_fa = query.transpose(1, 2).reshape(-1, q_heads, head_dim)
        
        # 核心优化：只在必要时才对 KV 进行扩展
        # 既然 moba_efficient 内部需要 heads 对齐，我们在这里进行扩展
        # 使用 expand + reshape 往往比 repeat_interleave 在某些版本下更节省临时显存
        key_fa = key.transpose(1, 2).reshape(-1, kv_heads, head_dim)
        val_fa = value.transpose(1, 2).reshape(-1, kv_heads, head_dim)
        
        if kv_replicas > 1:
            # 扩展 KV 头数以匹配 Q
            key_fa = key_fa.unsqueeze(2).expand(-1, -1, kv_replicas, -1).reshape(-1, q_heads, head_dim)
            val_fa = val_fa.unsqueeze(2).expand(-1, -1, kv_replicas, -1).reshape(-1, q_heads, head_dim)

        # 构建 cu_seqlens
        cu_seqlens = torch.arange(
            0, (batch + 1) * kv_len, step=kv_len, dtype=torch.int32, device=query.device
        )

        # 调用 MoBA 算子
        out = moba_impl(
            q=query_fa,
            k=key_fa,
            v=val_fa,
            cu_seqlens=cu_seqlens,
            max_seqlen=kv_len,
            moba_chunk_size=moba_config.moba_chunk_size,
            moba_topk=moba_config.moba_topk,
        )
        
        # 还原回 [B, H, S, D]
        out = out.view(batch, q_len, q_heads, head_dim).transpose(1, 2)

    else:
        # === Decode 阶段 (原生 Flash-Attn 逻辑) ===
        # 优化点：Flash Attention 原生支持 GQA，不需要 repeat KV！
        # 输入要求: [batch, seqlen, heads, dim]
        query_dec = query.transpose(1, 2)
        key_dec = key.transpose(1, 2)
        value_dec = value.transpose(1, 2)
        
        # 直接调用，Flash-Attn 会自动处理 q_heads=28, kv_heads=4 的情况
        out = flash_attn_func(
            query_dec, key_dec, value_dec, 
            dropout_p=dropout, 
            softmax_scale=scaling, 
            causal=True
        )
        out = out.transpose(1, 2) # 转回 [B, H, S, D]

    return out, None

class Qwen2MoBAAdaptor(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        
        # 1. 投影
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        # 转换为 [B, H, S, D]
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads

        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        # 2. RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. KV Cache 更新 (维持 GQA 结构)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 4. 调用 MoBA Layer
        scaling = self.head_dim ** -0.5
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

        # 5. 后处理 [B, H, S, D] -> [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        attn_output = self.o_proj(attn_output)

        return attn_output, None