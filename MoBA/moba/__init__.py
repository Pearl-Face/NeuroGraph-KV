from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import transformers
from .wrapper import moba_layer, Qwen2MoBAAdaptor 
from .moba_naive import moba_attn_varlen_naive
from .moba_efficient import moba_attn_varlen
from .config import MoBAConfig


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)

    print(f"⚡ [MoBA] Injecting Qwen2FlashAttention2 with Chunk={cfg.moba_chunk_size}, TopK={cfg.moba_topk}")
    
    # 注入全局配置给 Adaptor 类 (简单粗暴但有效)
    Qwen2MoBAAdaptor.moba_config = cfg
    Qwen2MoBAAdaptor.moba_impl = staticmethod(moba_attn_varlen) # 使用高效版 Triton 算子
    
    # 替换 Transformers 库中的定义
    # 这样当你调用 AutoModel 加载 Qwen 时，它以为它在用 FlashAttn2，实际上是在用 MoBA
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention = Qwen2MoBAAdaptor
