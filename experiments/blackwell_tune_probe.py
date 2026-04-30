import gc
import math
import os

import torch
import torch.nn.functional as F
import tilelang
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_fwd

from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
from flash_qla.ops.gated_delta_rule.chunk.cp_context import _calc_cp_seqs
from flash_qla.utils import l2norm


HEAD_DIM = 128


CASES = [
    ("tp8_1x32768", [32768], 2, 8),
    ("tp8_8192x4", [8192] * 4, 2, 8),
    ("tp4_1x32768", [32768], 4, 16),
    ("tp4_4096x8", [4096] * 8, 4, 16),
    ("tp2_1x32768", [32768], 8, 32),
    ("tp2_4096x8", [4096] * 8, 8, 32),
    ("tp1_1x32768", [32768], 16, 64),
    ("tp1_4096x8", [4096] * 8, 16, 64),
    ("h48_1x32768", [32768], 16, 48),
    ("h16_1x32768", [32768], 16, 16),
]


def cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def prepare_tensors(seqlens, h_qk, h_v):
    total_tokens = sum(seqlens)
    offsets = [0]
    for seqlen in seqlens:
        offsets.append(offsets[-1] + seqlen)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    q = l2norm(
        torch.randn(1, total_tokens, h_qk, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    )
    k = l2norm(
        torch.randn(1, total_tokens, h_qk, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    )
    v = torch.randn(1, total_tokens, h_v, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    g = (
        F.logsigmoid(torch.randn(1, total_tokens, h_v, device="cuda", dtype=torch.float32))
        / 16
    )
    beta = torch.randn(1, total_tokens, h_v, device="cuda", dtype=torch.float32).sigmoid()
    h0 = torch.randn(len(seqlens), h_v, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)

    swa_mask = torch.zeros(h_v, dtype=torch.bool, device="cuda")
    swa_mask[: math.ceil(0.75 * h_v)] = True
    swa_mask = swa_mask[torch.randperm(h_v, device="cuda")]
    g[:, :, ~swa_mask] = 0.0

    return q, k, v, g, beta, h0, HEAD_DIM**-0.5, cu_seqlens


def cp_batch_size(cu_seqlens, h_v):
    use_cp, cp_cu_seqlens, *_ = _calc_cp_seqs(cu_seqlens, 64, h_v)
    if not use_cp:
        return len(cu_seqlens) - 1
    return len(cp_cu_seqlens) - 1


def bench(fn, warmup, repeats):
    cleanup()
    return tilelang.profiler.do_bench(fn, warmup=warmup, rep=repeats)


def main():
    warmup = int(os.getenv("FLASHQLA_PROBE_WARMUP", "5"))
    repeats = int(os.getenv("FLASHQLA_PROBE_REPEATS", "20"))
    cp_scale = os.getenv("FLASHQLA_BLACKWELL_CP_SCALE", "default")
    block_dv = os.getenv("FLASHQLA_BLOCK_DV", "auto")

    print(
        "meta,"
        f"torch={torch.__version__},"
        f"tilelang={getattr(tilelang, '__version__', 'unknown')},"
        f"gpu={torch.cuda.get_device_name(0)},"
        f"capability={torch.cuda.get_device_capability(0)},"
        f"cp_scale={cp_scale},"
        f"block_dv={block_dv},"
        f"warmup={warmup},"
        f"repeats={repeats}",
        flush=True,
    )
    print("case,h_qk,h_v,cp_batch,qla_ms,fla_ms,speedup_fla_over_qla", flush=True)

    for label, seqlens, h_qk, h_v in CASES:
        q, k, v, g, beta, h0, scale, cu_seqlens = prepare_tensors(seqlens, h_qk, h_v)
        cp_batch = cp_batch_size(cu_seqlens, h_v)

        qla_ms = bench(
            lambda: qla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                output_h=False,
                cu_seqlens=cu_seqlens,
                auto_cp=True,
            ),
            warmup,
            repeats,
        )
        fla_ms = bench(
            lambda: fla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            ),
            warmup,
            repeats,
        )
        print(
            f"{label},{h_qk},{h_v},{cp_batch},{qla_ms:.6f},{fla_ms:.6f},{fla_ms / qla_ms:.6f}",
            flush=True,
        )
        cleanup()


if __name__ == "__main__":
    main()
