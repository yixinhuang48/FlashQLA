import os

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_chunk_gdr_output(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    o_dtype,
    block_DV=128,
    num_threads=256,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    g_shape = (batch_size, num_tokens, H)
    h_shape = (batch_size, num_chunks, H, DK, DV)

    @T.prim_func
    def tilelang_chunk_gdr_output_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        h: T.Tensor(h_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        o: T.Tensor(v_shape, dtype=o_dtype),
    ):
        with T.Kernel(
            batch_size * num_chunks * H * T.ceildiv(DV, block_DV), threads=num_threads
        ) as (pid,):
            bv = pid % T.ceildiv(DV, block_DV)
            pidh = pid // T.ceildiv(DV, block_DV)
            bh = pidh % H
            pidbc = pidh // H
            bc = pidbc % num_chunks
            bb = pidbc // num_chunks
            bhg = bh // (H // Hg)

            left = bc * block_S
            right = left + block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            if right <= num_tokens:
                T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
                T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
                T.copy(
                    v[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV],
                    v_shared,
                )
                for j_s in T.Parallel(block_S):
                    g_shared[j_s] = g[bb, left + j_s, bh]
            else:
                for j_s, j_k in T.Parallel(block_S, DK):
                    if left + j_s < num_tokens:
                        q_shared[j_s, j_k] = q[bb, left + j_s, bhg, j_k]
                        k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                    else:
                        q_shared[j_s, j_k] = 0
                        k_shared[j_s, j_k] = 0
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    if left + j_s < num_tokens:
                        v_shared[j_s, j_v] = v[
                            bb, left + j_s, bh, bv * block_DV + j_v
                        ]
                    else:
                        v_shared[j_s, j_v] = 0
                for j_s in T.Parallel(block_S):
                    if left + j_s < num_tokens:
                        g_shared[j_s] = g[bb, left + j_s, bh]
                    else:
                        g_shared[j_s] = g[bb, num_tokens - 1, bh]

            T.copy(
                h[bb, bc, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                h_shared,
            )

            T.gemm_v1(q_shared, h_shared, o_fragment, clear_accum=True)
            T.gemm_v1(
                q_shared,
                k_shared,
                a_fragment,
                transpose_B=True,
                clear_accum=True,
            )

            for j_s, j_t in T.Parallel(block_S, block_S):
                if j_s >= j_t:
                    a_fragment[j_s, j_t] *= T.exp2(
                        (g_shared[j_s] - g_shared[j_t]) * 1.442695
                    )
                else:
                    a_fragment[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, block_DV):
                o_fragment[j_s, j_v] *= T.exp2(g_shared[j_s] * 1.442695)

            T.copy(a_fragment, a_shared)
            T.gemm_v1(a_shared, v_shared, o_fragment, clear_accum=False)

            for j_s, j_v in T.Parallel(block_S, block_DV):
                o_fragment[j_s, j_v] *= scale

            if right <= num_tokens:
                T.copy(
                    o_fragment,
                    o[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV],
                )
            else:
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    if left + j_s < num_tokens:
                        o[bb, left + j_s, bh, bv * block_DV + j_v] = o_fragment[
                            j_s, j_v
                        ]

    return tilelang_chunk_gdr_output_kernel


def chunk_gdr_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    batch_size, num_tokens, Hg, DK = q.shape
    _, _, H, DV = v.shape
    assert DK == DV == 128
    assert chunk_size == 64

    o = torch.empty_like(v)
    block_DV = int(os.getenv("FLASHQLA_BLACKWELL_OUTPUT_BLOCK_DV", "128"))
    num_threads = int(os.getenv("FLASHQLA_BLACKWELL_OUTPUT_THREADS", "128"))
    kernel = tilelang_chunk_gdr_output(
        H,
        Hg,
        DK,
        DV,
        chunk_size,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        o_dtype=o.dtype,
        block_DV=block_DV,
        num_threads=num_threads,
    )
    kernel(q, k, v, h, g, o)
    return o
