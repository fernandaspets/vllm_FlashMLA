🛠️ Instructions for vLLM Users with SM120 GPUs (e.g., RTX 50-series, Pro 6000)
Prerequisites

    An SM120 GPU (Compute Capability 12.0)
    CUDA 12.8 or higher
    Basic Python/pip environment (eg. uv venv)

Step 1: Clone and Prepare the Code

Clone the sm120 ready vLLM repository:

git clone https://github.com/fernandaspets/vllm_sm120.git
cd vllm_sm120 

You can also apply the patches to mainline vllm youself using the commands below. They make three key changes:

    Allow attention backends to activate on SM120.
    Update the build script (flashmla.cmake) to compile the sm120 kernels from the forked FlashMLA.
    Fix the runtime capability check.

Edit: the below sed commands might not be sufficient. clone the above vllm_sm120 fork or check the changes in there for full changelog Check all my sm120 related changes here: https://github.com/vllm-project/vllm/compare/main...fernandaspets:vllm_sm120:main

# 1. Allow FlashMLA backends to recognize SM120
sed -i 's/capability.major in \[9, 10\]/capability.major in [9, 10, 12]/' vllm/v1/attention/backends/mla/flashmla_sparse.py
sed -i 's/capability.major in \[9, 10\]/capability.major in [9, 10, 12]/' vllm/v1/attention/backends/mla/flashmla.py

# 2. Tell vLLM to build the SM120 kernels from our fork
sed -i 's|sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu|sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu|' cmake/external_projects/flashmla.cmake
sed -i 's|sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu|sm120/prefill/dense/fmha_cutlass_bwd_sm100.cu|' cmake/external_projects/flashmla.cmake

# 3. Update the runtime check function
sed -i 's/current_platform.get_device_capability()\[0\] not in (9, 10)/current_platform.get_device_capability()[0] not in (9, 10, 12)/' vllm/attention/ops/flashmla.py

Step 2: Install vLLM with Custom FlashMLA

Set an environment variable to tell vLLM's build system to use this forked FlashMLA repository instead of downloading the official one:

export FLASH_MLA_SRC_DIR=/path/to/your/vllm_FlashMLA

    Note: Replace /path/to/your/vllm_FlashMLA with the actual absolute path to the directory where you cloned this vllm_FlashMLA repository.

Install vLLM in development mode. The build process will automatically fetch and compile the custom FlashMLA from the path you specified.

pip install -e . --no-build-isolation

Step 3: Verify the Installation

After installation, you can verify that vLLM recognizes your SM120 GPU and the FlashMLA backend:

import torch
from vllm.platforms import current_platform
from vllm.attention.ops.flashmla import is_flashmla_sparse_supported

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {current_platform.get_device_capability()}")
print(f"FlashMLA Sparse Supported: {is_flashmla_sparse_supported()}")

Expected output for an RTX Pro 6000:

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
Compute Capability: DeviceCapability(major=12, minor=0)
FlashMLA Sparse Supported: (True, None)

⚠️ Important Limitations (SM120 vs. SM100)

The port enables kernels to run on SM120, testing is required to find out what works and doesn't yet. Key hardware differences affect the design:

    Shared Memory: 99 KB on SM120 vs. 228 KB on SM100. This severely limits kernel tile sizes and register usage.
    No Tensor Memory (TMEM): Requires using Global Memory for some operations, which is slower.
    Different MMA Units: Uses warp-level MMA instructions instead of the larger WGMMA available on SM100.


# FlashMLA

## Introduction

FlashMLA is DeepSeek's library of optimized attention kernels, powering the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) and [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) models. This repository contains the following implementations:

**Sparse Attention Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Token-level sparse attention for the prefill stage
- Token-level sparse attention for the decoding stage, with FP8 KV cache

**Dense Attention Kernels**

- Dense attention for the prefill stage
- Dense attention for the decoding stage

## News

- **2025.09.29 Release of Sparse Attention Kernels**: With the launch of [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), we are releasing the corresponding token-level sparse attention kernels. These kernels power the model's DeepSeek Sparse Attention (DSA) and achieve up to 640 TFlops during prefilling and 410 TFlops during decoding.
- **2025.08.01 Kernels for MHA on Blackwell**: Thanks to [NVIDIA's PR](https://github.com/deepseek-ai/FlashMLA/pull/76) for MHA forward / backward kernels on Blackwell!
- **2025.04.22 Deep-Dive Blog**: We'd love to share the technical details behind the new FlashMLA kernel! Check out our deep-dive write-up [here](docs/20250422-new-kernel-deep-dive.md).
- **2025.04.22 Performance Update**: We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Simply upgrade to the new version for an immediate performance boost! 🚀🚀🚀

## Performance

#### Test & benchmark MLA decoding (Sparse & Dense):

```bash
python tests/test_flash_mla_decoding.py
```

The dense MLA decoding kernel can achieve up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5, using CUDA 12.8. For token-level sparse MLA decoding kernel (which uses an FP8 KV cache while performing the matrix multiplication in bfloat16), it can achieve 410 TFLOPS in compute-bound configuration on H800 SXM5, CUDA 12.8.

For Blackwell GPUs, the token-level sparse MLA decoding kernel can achieve up to 350 TFlops (on B200) which is not really optimized yet.

#### Test & benchmark MHA prefill (Dense):

```bash
python tests/test_fmha_sm100.py
```

It achieves up to 1460 TFlops in forward and 1000 TFlops in backward computation on B200, as reported by NVIDIA.

#### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test_flash_mla_prefill.py
```

It achieves up to 640 TFlops in forward computation on H800 SXM5, CUDA 12.8, and achieves up to 1450 TFlops on B200, CUDA 12.9.

## Requirements

- Hopper / Blackwell GPUs (See the support matrix below)
- CUDA 12.8 and above (CUDA 12.9+ is required for Blackwell kernels)
- PyTorch 2.0 and above

Support matrix:

| Kernel | GPU Architecture | MLA Mode [2] | KVCache Format |
| :---: | :---: | :---: | :---: |
| Dense Decoding | Hopper | MQA | BF16 |
| Sparse Decoding | Hopper & Blackwell | MQA | FP8 [1] |
| Dense Prefill | Blackwell | MHA |  |
| Sparse Prefill | Hopper & Blackwell | MQA |  |

[1]: For more details on using FP8 KV cache, see documents below.

[2]: Here "MLA Mode" refers to the mode used for MLA calculation. MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` =  576 with `head_dim_v` = 512), while MHA stands for Multi-Head Attention mode (i.e. `head_dim_k` = 192 / 128 with `head_dim_v` = 128). For a detailed explanation of these modes, please refer to the appendix of [DeepSeek V3.2's Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

## Installation

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```

## Usage

### MLA Decoding

To use the MLA decoding kernels, call get_mla_metadata once before the decoding loop to get the tile scheduler metadata. Then, call flash_mla_with_kvcache in each decoding step. For example:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv,
    h_q,
    is_fp8,
    topk,
)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices,
    )
    ...
```

Where

- `s_q` is the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1.
- `h_kv` is the number of key-value heads.
- `h_q` is the number of query heads.

**FP8 KV Cache:**
If `is_fp8_kvcache` is set to `True`, the kernel reads the KV cache in the "FP8 with scale" format (described below). It dequantizes the cache to bfloat16 and performs attention computation in bfloat16. The output is also in bfloat16.

In the "FP8 with scale" format, each token's KV cache is 656 Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

See `tests/quant.py` for quantization and dequantization details.

**Sparse Attention (`indices` tensor):**
The `indices` tensor (if provided) enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

-   **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
-   **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not require the `block_table` parameter.
-   **Invalid entries:** Set invalid indices to `-1`.

**Return Values:**
The kernel returns `(out, lse)`, where:
-   `out` is the attention result.
-   `lse` is the log-sum-exp value of the attention scores for each query head.

See `tests/test_flash_mla_decoding.py` for a complete example.

### Sparse MLA Prefill

For the sparse MLA prefill kernel, call `flash_mla_sparse_fwd` directly with the following parameters:
-   `q`: Query tensor of shape `[s_q, h_q, d_qk]`
-   `kv`: Key-Value tensor of shape `[s_kv, h_kv, d_qk]`
-   `indices`: Indices tensor of shape `[s_q, h_kv, topk]`
-   `sm_scale`: A scalar value

**Note on batching:** This kernel does not support a batch dimension. For multi-batch inference, reshape the input tensors and adjust the `indices` parameter to simulate batch processing.

**Invalid indices:** Set invalid entries in `indices` to `-1` or any number `>= s_kv`.

**Return Values and Equivalent PyTorch Code:**
The kernel returns `(out, max_logits, lse)`. This is equivalent to the following PyTorch operations:

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv must be 1
indices = indices.squeeze(1)    # [s_q, topk]
focused_kv = kv[indices]    # For the i-th sequence (s_q), the corresponding KV tokens are selected from the KV cache based on indices[i, :]. This operation results in a tensor of shape [s_q, topk, d_qk].

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * math.log2(math.e)    # [s_q, h_q, topk]
max_logits = P.max(dim=-1) # [s_q, h_q]
lse = log2sumexp2(P, dim=-1, base=2)   # [s_q, h_q]，"log2sumexp2" means that the exponentiation and logarithm are base-2
S = exp2(P - lse)      # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

See `tests/test_flash_mla_prefill.py` for a complete example.

### Dense MHA Prefill

This kernel implements the standard dense Multi-Head Attention (MHA) forward and backward operations. It can be called using:
-   `flash_attn_varlen_func`
-   `flash_attn_varlen_qkvpacked_func`
-   `flash_attn_varlen_kvpacked_func`

The usage is similar to the `flash_attn` package. See `tests/test_fmha_sm100.py` for a complete example.

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Community Support

### MetaX
For MetaX GPUs, visit the official website: [MetaX](https://www.metax-tech.com).

The corresponding FlashMLA version can be found at: [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)


### Moore Threads
For the Moore Threads GPU, visit the official website: [Moore Threads](https://www.mthreads.com/).

The corresponding FlashMLA version is available on GitHub: [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA).


### Hygon DCU
For the Hygon DCU, visit the official website: [Hygon Developer](https://developer.sourcefind.cn/).

The corresponding FlashMLA version is available here: [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention).


### Intellifusion
For the Intellifusion NNP, visit the official website: [Intellifusion](https://www.intellif.com).

The corresponding FlashMLA version is available on Gitee: [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py).


### Iluvatar Corex
For Iluvatar Corex GPUs, visit the official website: [Iluvatar Corex](https://www.iluvatar.com).

The corresponding FlashMLA version is available on GitHub: [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)


### AMD Instinct
For AMD Instinct GPUs, visit the official website: [AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html).

The corresponding FlashMLA version can be found at: [AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
