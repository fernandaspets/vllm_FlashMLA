#!/usr/bin/env python3
import torch, math, vllm.attention.ops.flashmla as flashmla_ops

torch.manual_seed(42); torch.cuda.manual_seed(42)
sparse_op = torch.ops._flashmla_C.sparse_prefill_fwd

s_q, s_kv, h_q, h_kv, d_qk, d_v, topk = 1, 1000, 128, 1, 576, 32, 32   # d_v = 32 → should show 50 % zeros if bug still present
sm_scale = 1.0 / math.sqrt(d_qk)

q  = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda')
kv = torch.randn(s_kv, h_kv, d_qk, dtype=torch.bfloat16, device='cuda')
idx = torch.randint(0, s_kv, (s_q, h_kv, topk), dtype=torch.int32, device='cuda')

out, max_logits, _ = sparse_op(q, kv, idx, sm_scale, d_v)

pct_zeros = (out == 0).float().mean().item() * 100
has_inf   = torch.isinf(max_logits).any().item()

print(f"d_v={d_v}:  {pct_zeros:5.1f}% zeros   inf_in_max_logits={has_inf}")
