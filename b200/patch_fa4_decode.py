#!/usr/bin/env python3
"""Enable static split-KV for the Pochi TP2 full-attention decode path."""
import ast
import hashlib
import json
from pathlib import Path
import sys

venv = Path(sys.argv[1])
file = venv/'lib/python3.12/site-packages/sglang/srt/layers/attention/flashattention_backend.py'
marker = 'FM_POCHI_FA4_DECODE_SPLITKV_V1'
source = file.read_text()
if marker not in source:
    begin = source.index('    def forward_decode(')
    end = source.index('    def init_cuda_graph_state(', begin)
    part = source[begin:end]
    anchor = '                result = flash_attn_with_kvcache(\n                    q=q_reshaped,'
    assert part.count(anchor) == 1, 'Unexpected FA4 decode source; refusing to patch'
    replacement = '''                # FM_POCHI_FA4_DECODE_SPLITKV_V1
                # Static splits keep CUDA graph capture valid. Partition only
                # Pochi TP2 BF16 full-attention decode; prefill/SWA stay unchanged.
                decode_num_splits = self.num_splits
                if (
                    self.fa_impl_ver == 4
                    and not is_swa_layer
                    and not use_cascade_attn
                    and max_seqlen_q == 1
                    and layer.tp_q_head_num == 20
                    and layer.tp_k_head_num == 4
                    and layer.head_dim == 128
                    and q_reshaped.dtype == torch.bfloat16
                    and not get_global_server_args().enable_deterministic_inference
                ):
                    split_budget = max(1, 32 // max(1, forward_batch.batch_size))
                    decode_num_splits = 1 << (split_budget.bit_length() - 1)
                result = flash_attn_with_kvcache(
                    q=q_reshaped,'''
    part = part.replace(anchor, replacement, 1)
    old_arg = '                    num_splits=self.num_splits,\n                    out=_fa_out,'
    assert part.count(old_arg) == 1, 'Unexpected decode call arguments'
    part = part.replace(old_arg, '                    num_splits=decode_num_splits,\n                    out=_fa_out,', 1)
    patched = source[:begin] + part + source[end:]
    ast.parse(patched)
    backup = file.with_name(file.name + '.pochi-splitkv.orig')
    if not backup.exists():
        backup.write_text(source)
    file.write_text(patched)
    print('Patched', file)
else:
    print('Already patched', file)

print(json.dumps({'marker':marker,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}))
