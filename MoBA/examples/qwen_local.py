import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改 1: 这里的 default 换成你服务器上 Qwen2.5-7B 的绝对路径
    # 例如: "/data/models/Qwen2.5-7B-Instruct"
    parser.add_argument("--model", type=str, default="/home/broken_eclipse/models/Qwen2.5-7B", help="Path to local Qwen model")
    
    # Qwen2.5 支持 32k 甚至 128k，我们可以设大一点的 chunk
    parser.add_argument("--moba-chunk-size", type=int, default=2048) 
    parser.add_argument("--moba-topk", type=int, default=4)
    parser.add_argument(
        "--attn",
        default="moba", # 默认使用 moba (高效版)
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Attention Implementation: {args.attn}")

    # 注册 MoBA (会触发我们在 wrapper.py 里写的替换逻辑)
    if "moba" in args.attn:
        register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True, # Qwen有时候需要这个
        device_map="auto",
        torch_dtype=torch.bfloat16, # A100 建议用 bfloat16
        attn_implementation="eager", 
        # 注意：如果用了 moba，这里通常传 eager 让 transformer 使用我们替换后的 class
        # 如果 moba 是基于 flash-attn 实现的，wrapper 内部会处理，这里传 eager 比较稳妥
    )

    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 构造一个长一点的 Prompt 来测试
    context_text = """
    DeepSeek-V3 is a groundbreaking large language model that represents a significant leap forward in open-source artificial intelligence. Built upon the success of its predecessors, V3 introduces a series of architectural innovations designed to balance computational efficiency with high-level reasoning capabilities. The core philosophy behind DeepSeek-V3 is the "Mixture-of-Experts" (MoE) architecture, combined with a novel attention mechanism known as Multi-Head Latent Attention (MLA).

    Unlike traditional dense models that activate all parameters for every token, DeepSeek-V3 utilizes a fine-grained MoE structure. This allows the model to scale up to 671 billion total parameters while keeping the active parameters per token relatively low, around 37 billion. This sparse activation strategy ensures that the model can learn a vast amount of world knowledge without incurring prohibitive inference costs. The routing mechanism in V3 has been optimized to ensure load balancing across experts, preventing the "expert collapse" problem often seen in earlier MoE models.

    Another key innovation is the Multi-Head Latent Attention (MLA). Standard Key-Value (KV) caching has become a bottleneck for long-context inference, consuming massive amounts of GPU memory. MLA addresses this by compressing the Key and Value heads into a low-rank latent vector. This significantly reduces the memory footprint of the KV cache, allowing DeepSeek-V3 to support context windows of up to 128k tokens on consumer-grade hardware, making it highly accessible for researchers and developers.

    In terms of training data, DeepSeek-V3 was trained on a massive corpus of 14.8 trillion tokens. The data pipeline included rigorous filtering and deduplication processes to ensure high quality. The dataset covers a diverse range of domains, including mathematics, coding, literature, and scientific papers. Special attention was given to synthetic data generation, where smaller, high-quality models were used to generate reasoning traces, which were then used to fine-tune V3. This "distillation" process has greatly enhanced the model's ability to perform complex logical deduction and mathematical problem-solving.

    Furthermore, DeepSeek-V3 employs a unique auxiliary loss function during training to maintain stability. Large-scale MoE training is notoriously unstable, often suffering from gradient explosions. The researchers introduced a router z-loss to stabilize the gating network, ensuring that token distribution among experts remains smooth throughout the training process.

    Benchmarks show that DeepSeek-V3 outperforms Llama-3.1-405B on several key metrics, including HumanEval for coding and GSM8K for math, while being significantly faster to serve. Its open weights release has sparked a new wave of innovation in the community, allowing for fine-tuning and adaptation to specific vertical domains such as healthcare and finance.
    """

    # 构造 Prompt：先给文章，再提问
    prompt = f"""{context_text}

    Based on the text above, please summarize the two main architectural innovations of DeepSeek-V3 and explain why they are important.
    Answer:"""
    
    input_tokens = tknz(prompt, return_tensors="pt").to(model.device)
    
    print("Start Generating...")
    tokens = model.generate(
        **input_tokens, 
        max_new_tokens=512, 
        do_sample=False
    )
    
    print("-" * 20)
    print(tknz.decode(tokens[0], skip_special_tokens=True))
    print("-" * 20)
    print("✅ Qwen2.5 with MoBA Inference Finished!")