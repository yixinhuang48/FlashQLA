# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang

from flash_qla.utils import (
    l2norm,
    pack,
    pad_and_reshape,
    prepare_chunk_offsets,
    unpack,
)
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

if float(tilelang.contrib.nvcc.get_target_compute_version()) >= 9.0:
    from .hopper import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
else:
    raise ValueError("FlashQLA now supports sm90 or above only.")
from .cp_context import intra_card_cp_preprocess


def _is_blackwell_or_newer(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    return torch.cuda.get_device_properties(x.device).major >= 10


def _use_blackwell_experimental_hopper_fwd() -> bool:
    import os

    return os.getenv("FLASHQLA_BLACKWELL_EXPERIMENTAL_HOPPER_FWD") in (
        "1",
        "true",
        "True",
        "yes",
        "on",
    )


def _use_blackwell_tilelang_output() -> bool:
    import os

    return os.getenv("FLASHQLA_BLACKWELL_TILELANG_OUTPUT") in (
        "1",
        "true",
        "True",
        "yes",
        "on",
    )


def _is_single_full_sequence(
    cu_seqlens: torch.LongTensor | None,
    num_tokens: int,
) -> bool:
    return (
        cu_seqlens is not None
        and cu_seqlens.numel() == 2
        and cu_seqlens[0].item() == 0
        and cu_seqlens[1].item() == num_tokens
    )


def _is_uniform_varlen(cu_seqlens: torch.LongTensor | None) -> bool:
    if cu_seqlens is None or cu_seqlens.numel() <= 2:
        return False
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return bool(torch.all(lengths == lengths[0]).item())


def _uniform_varlen_shape(cu_seqlens: torch.LongTensor) -> tuple[int, int]:
    num_seqs = cu_seqlens.numel() - 1
    seq_len = (cu_seqlens[1] - cu_seqlens[0]).item()
    return num_seqs, seq_len


def _unpack_uniform_varlen(x: torch.Tensor, cu_seqlens: torch.LongTensor) -> torch.Tensor:
    num_seqs, seq_len = _uniform_varlen_shape(cu_seqlens)
    return x.reshape(num_seqs, seq_len, *x.shape[2:])


def _pack_uniform_varlen(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(1, x.shape[0] * x.shape[1], *x.shape[2:])


def _fla_output_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
) -> torch.Tensor:
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_fwd as fla_chunk_gated_delta_rule_fwd,
    )

    _, o, _, _, _, _ = fla_chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o


def _fla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
):
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_fwd as fla_chunk_gated_delta_rule_fwd,
    )

    return fla_chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


def _fla_chunk_output_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None,
) -> torch.Tensor:
    from fla.ops.common.chunk_o import chunk_fwd_o

    return chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )


def _tilelang_chunk_output_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    from .hopper.output_fwd import chunk_gdr_output

    return chunk_gdr_output(q=q, k=k, v=v_new, h=h, g=g, scale=scale)


def _torch_output_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        g = unpack(g, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        h = unpack(h, prepare_chunk_offsets(cu_seqlens, chunk_size))

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape

    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k_chunk = pad_and_reshape(k.float(), dim=1, chunk_size=chunk_size)
    g_chunk = pad_and_reshape(g.float(), dim=1, chunk_size=chunk_size)
    beta_chunk = pad_and_reshape(beta.float(), dim=1, chunk_size=chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device)
    )
    decay_mask = torch.exp(g_chunk[:, :, :, None, :] - g_chunk[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)
    a = torch.einsum(
        "bnchk,bndhk->bnchd",
        k_chunk * beta_chunk.unsqueeze(-1),
        k_chunk,
    ) * decay_mask.swapaxes(-2, -1)
    a = -a.swapaxes(2, 3)
    for i in range(1, chunk_size):
        row = a[..., i, :i].clone()
        sub = a[..., :i, :i].clone()
        a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    a += torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    a = a.swapaxes(2, 3)

    k_beta = pad_and_reshape(
        k.float() * beta.float().unsqueeze(-1) * g.float().exp().unsqueeze(-1),
        dim=1,
        chunk_size=chunk_size,
    )
    v_beta = pad_and_reshape(
        v.float() * beta.float().unsqueeze(-1),
        dim=1,
        chunk_size=chunk_size,
    )
    w = torch.einsum("bnchd,bndhk->bnchk", a, k_beta)
    u = torch.einsum("bnchd,bndhk->bnchk", a, v_beta)

    q = pad_and_reshape(q.float(), dim=1, chunk_size=chunk_size)
    k = pad_and_reshape(k.float(), dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g.float(), dim=1, chunk_size=chunk_size)
    h = h.float()
    v = u - torch.einsum("bnchk,bnhkv->bnchv", w, h)

    q = q * scale
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)

    attn = torch.einsum("bnchk,bndhk->bncdh", q, k) * decay_mask
    attn_inter = torch.einsum("bnchk,bnhkv->bnchv", q * g.exp().unsqueeze(-1), h)
    o = attn_inter + torch.einsum("bncdh,bndhv->bnchv", attn, v)

    o = o.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        o = pack(o, cu_seqlens)
    return o.to(v.dtype)


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
):
    raw_g = g
    scale = scale or q.shape[-1] ** (-0.5)
    is_blackwell = _is_blackwell_or_newer(q)
    use_experimental_hopper_fwd = (
        not is_blackwell or _use_blackwell_experimental_hopper_fwd()
    )
    normalized_single_full = (
        is_blackwell
        and not use_experimental_hopper_fwd
        and _is_single_full_sequence(cu_seqlens, q.shape[1])
    )
    if normalized_single_full:
        cu_seqlens = None
    if (
        is_blackwell
        and not use_experimental_hopper_fwd
        and not output_h
    ):
        if cu_seqlens is None:
            fla_g, fla_o, fla_A, fla_final_state, _, _ = _fla_fwd(
                q=q,
                k=k,
                v=v,
                g=raw_g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=None,
            )
            return fla_g, fla_A, fla_o, None, fla_final_state
        if _is_uniform_varlen(cu_seqlens):
            dense_q = _unpack_uniform_varlen(q, cu_seqlens)
            dense_k = _unpack_uniform_varlen(k, cu_seqlens)
            dense_v = _unpack_uniform_varlen(v, cu_seqlens)
            dense_g = _unpack_uniform_varlen(raw_g, cu_seqlens)
            dense_beta = _unpack_uniform_varlen(beta, cu_seqlens)
            fla_g, fla_o, fla_A, fla_final_state, _, _ = _fla_fwd(
                q=dense_q,
                k=dense_k,
                v=dense_v,
                g=dense_g,
                beta=dense_beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=None,
            )
            return (
                _pack_uniform_varlen(fla_g),
                _pack_uniform_varlen(fla_A),
                _pack_uniform_varlen(fla_o),
                None,
                fla_final_state,
            )

    g = chunk_local_cumsum(raw_g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )
    cp_seq_map = None
    raw_cu_seqlens = None
    if auto_cp and use_experimental_hopper_fwd:
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
    blackwell_safe_o = None
    use_qla_v_new_output = (
        is_blackwell
        and not use_experimental_hopper_fwd
        and output_h
        and cu_seqlens is None
    )
    if not use_experimental_hopper_fwd and cu_seqlens is None:
        blackwell_safe_o = None if use_qla_v_new_output else _fla_output_fwd(
            q=q,
            k=k,
            v=v,
            g=raw_g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    internal_output_h = output_h or (
        not use_experimental_hopper_fwd and cu_seqlens is not None
    )
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=internal_output_h,
        output_o=use_experimental_hopper_fwd,
        output_v_new=use_qla_v_new_output,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
    )
    if not use_experimental_hopper_fwd:
        if cu_seqlens is None:
            if use_qla_v_new_output:
                if _use_blackwell_tilelang_output():
                    o = _tilelang_chunk_output_fwd(
                        q=q,
                        k=k,
                        v_new=o,
                        h=h,
                        g=g,
                        scale=scale,
                    )
                else:
                    o = _fla_chunk_output_fwd(
                        q=q,
                        k=k,
                        v_new=o,
                        h=h,
                        g=g,
                        scale=scale,
                        cu_seqlens=None,
                    )
            else:
                o = blackwell_safe_o
        else:
            o = _torch_output_fwd(
                q=q,
                k=k,
                v=v,
                h=h,
                g=g,
                beta=beta,
                a=A,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
            if not output_h:
                h = None
    return g, A, o, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        q_orig = q
        k_orig = k

        g, A, o, _, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=False,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q_orig,
            k=k_orig,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq.to(q_orig),
            dk.to(k_orig),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
    )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )

    return o, final_state
