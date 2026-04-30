#!/usr/bin/env python3
"""Component-level forward correctness probe for FlashQLA on Blackwell GPUs."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from ref_gdr import (  # noqa: E402
    chunk_gated_delta_rule_fwd as ref_fwd,
    torch_kkt_fwd,
    torch_solve,
)

from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.cp_context import (  # noqa: E402
    intra_card_cp_preprocess,
)
from flash_qla.ops.gated_delta_rule.chunk.hopper import kkt_solve  # noqa: E402
from flash_qla.ops.utils import chunk_local_cumsum  # noqa: E402
from flash_qla.utils import l2norm  # noqa: E402


def tensor_report(name: str, got: torch.Tensor | None, ref: torch.Tensor | None = None):
    if got is None:
        print(f"{name:28s} none")
        return

    got_f = got.float()
    finite = torch.isfinite(got_f)
    finite_count = finite.sum().item()
    total = got.numel()
    max_abs = got_f[finite].abs().max().item() if finite_count else float("nan")
    msg = f"{name:28s} finite={finite_count}/{total} max_abs={max_abs:.6g}"

    if ref is not None:
        ref_f = ref.float()
        diff = (got_f - ref_f).abs()
        diff_finite = torch.isfinite(diff)
        diff_max = diff[diff_finite].max().item() if diff_finite.any() else float("nan")
        ref_max = ref_f.abs().max().item()
        rel = diff_max / ref_max if ref_max else diff_max
        msg += f" diff={diff_max:.6g} ref={ref_max:.6g} rel={rel:.6g}"

    print(msg)


def make_inputs(args: argparse.Namespace):
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    q = l2norm(
        torch.randn(
            (args.batch_size, args.num_tokens, args.num_k_heads, args.head_dim),
            device="cuda",
            dtype=dtype,
        )
    )
    k = l2norm(torch.randn_like(q))
    v = torch.randn(
        (args.batch_size, args.num_tokens, args.num_heads, args.head_dim),
        device="cuda",
        dtype=dtype,
    )
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn(
                (args.batch_size, args.num_tokens, args.num_heads),
                device="cuda",
                dtype=torch.float32,
            )
        )
        / 16
    )
    beta = torch.randn(
        (args.batch_size, args.num_tokens, args.num_heads),
        device="cuda",
        dtype=torch.float32,
    ).sigmoid()

    swa_mask = torch.zeros((args.num_heads), dtype=torch.bool, device="cuda")
    swa_mask[: math.ceil(args.swa_ratio * args.num_heads)] = 1
    swa_mask = swa_mask[torch.randperm(args.num_heads, device="cuda")]
    g[:, :, ~swa_mask] = 0.0

    h0 = None
    if args.use_h0:
        h0 = torch.randn(
            (args.batch_size, args.num_heads, args.head_dim, args.head_dim),
            device="cuda",
            dtype=torch.float32,
        )

    cu_seqlens = None
    if args.cu_seqlens:
        cu_seqlens = torch.tensor(
            [int(x) for x in args.cu_seqlens.split("-")],
            device="cuda",
            dtype=torch.int32,
        )
        if h0 is not None:
            h0 = torch.randn(
                (cu_seqlens.numel() - 1, args.num_heads, args.head_dim, args.head_dim),
                device="cuda",
                dtype=torch.float32,
            )

    return q, k, v, g, beta, h0, cu_seqlens


def run_probe(args: argparse.Namespace, experimental: bool):
    if experimental:
        os.environ["FLASHQLA_BLACKWELL_EXPERIMENTAL_HOPPER_FWD"] = "1"
        label = "experimental Hopper-derived fused output/CP"
    else:
        os.environ.pop("FLASHQLA_BLACKWELL_EXPERIMENTAL_HOPPER_FWD", None)
        label = "safe Blackwell output fallback"

    q, k, v, g, beta, h0, cu_seqlens = make_inputs(args)
    scale = q.shape[-1] ** (-0.5)
    print(f"\n=== {label} ===")
    print(
        f"shape B={args.batch_size} T={args.num_tokens} Hk={args.num_k_heads} "
        f"Hv={args.num_heads} varlen={cu_seqlens is not None}"
    )

    ref_g, ref_o, _, ref_h, ref_s = ref_fwd(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )

    qla_g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    qla_a = kkt_solve(k=k, b=beta, cu_seqlens=cu_seqlens)
    ref_a_from_parts = torch_solve(
        torch_kkt_fwd(k=k.float(), g=ref_g, beta=beta.float(), cu_seqlens=cu_seqlens),
        cu_seqlens=cu_seqlens,
    )
    tensor_report("g_cumsum", qla_g, ref_g)
    tensor_report("A", qla_a, ref_a_from_parts)

    if args.inspect_cp:
        cp_h0, cp_cu_seqlens, cp_seq_map, raw_cu = intra_card_cp_preprocess(
            k=k,
            v=v,
            a=qla_a,
            g=qla_g,
            b=beta,
            raw_h0=h0,
            raw_cu_seqlens=cu_seqlens,
        )
        tensor_report("CP h0", cp_h0)
        tensor_report("CP cu_seqlens", cp_cu_seqlens)
        tensor_report("CP seq_map", cp_seq_map)
        tensor_report("CP raw_cu_seqlens", raw_cu)

    qla_g, qla_a, qla_o, qla_h, qla_s = qla_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=True,
        auto_cp=args.auto_cp,
    )
    tensor_report("h", qla_h, ref_h)
    tensor_report("final_state", qla_s, ref_s)
    tensor_report("output", qla_o, ref_o)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--cu-seqlens", default=None)
    parser.add_argument("--no-h0", action="store_true")
    parser.add_argument("--no-auto-cp", action="store_true")
    parser.add_argument("--inspect-cp", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("safe", "experimental", "both"),
        default="both",
    )
    args = parser.parse_args()
    args.use_h0 = not args.no_h0
    args.auto_cp = not args.no_auto_cp
    if args.num_k_heads <= 0:
        args.num_k_heads = args.num_heads
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode in ("safe", "both"):
        run_probe(parsed, experimental=False)
    if parsed.mode in ("experimental", "both"):
        run_probe(parsed, experimental=True)
