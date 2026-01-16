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
    
    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
    
    print(f"🚀 [MoBA] Injecting MoBA into Qwen2...")

    # 1. 注入配置和实现函数
    # 确保 Qwen2MoBAAdaptor 能够接收来自 FlashAttention2 路径的参数
    Qwen2MoBAAdaptor.moba_config = cfg
    Qwen2MoBAAdaptor.moba_impl = staticmethod(moba_attn_varlen)
    
    # 2. 全量劫持：无论 transformers 选哪个类，都强制跳转到我们的 Adaptor
    # 这一步非常关键，它保证了即使在 flash_attention_2 路径下也运行 MoBA
    qwen2_module.Qwen2FlashAttention2 = Qwen2MoBAAdaptor
    qwen2_module.Qwen2Attention = Qwen2MoBAAdaptor
    qwen2_module.Qwen2SdpaAttention = Qwen2MoBAAdaptor

    # 3. 兼容性补丁：有些版本的 transformers 会维护一个内部类映射表
    if hasattr(qwen2_module, "QWEN2_ATTENTION_CLASSES"):
        qwen2_module.QWEN2_ATTENTION_CLASSES = {
            "eager": Qwen2MoBAAdaptor,
            "flash_attention_2": Qwen2MoBAAdaptor,
            "sdpa": Qwen2MoBAAdaptor,
        }
    
    print(f"✅ MoBA is now shielding both Eager and Flash-Attn paths.")
