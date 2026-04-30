# Blackwell Experiment Summary

Compared focused B200/sm100 tuning variants for FlashQLA forward. Lower `qla_ms` is better.

## Best Variant Per Case

| Case | Best variant | CP batch | QLA ms | FLA ms | FLA/QLA |
| --- | --- | ---: | ---: | ---: | ---: |
| h16_1x32768 | force CP scale 1.0 | 64 | 1.380 | 1.324 | 0.959 |
| h48_1x32768 | force CP scale 1.0 | 32 | 3.602 | 2.571 | 0.714 |
| tp1_1x32768 | CP scale 1.0 | 1 | 4.344 | 3.142 | 0.723 |
| tp1_4096x8 | block_DV 64 | 8 | 3.888 | 3.047 | 0.784 |
| tp2_1x32768 | CP scale 2.0 | 32 | 2.388 | 1.815 | 0.760 |
| tp2_4096x8 | CP scale 3.0 | 8 | 1.992 | 1.737 | 0.872 |
| tp4_1x32768 | force CP scale 1.0 | 64 | 1.374 | 1.292 | 0.940 |
| tp4_4096x8 | CP scale 3.0 | 8 | 1.015 | 0.925 | 0.911 |
| tp8_1x32768 | force CP scale 3.0 | 32 | 0.767 | 1.029 | 1.341 |
| tp8_8192x4 | CP scale 3.0 | 32 | 0.735 | 0.479 | 0.651 |

## Main Findings

- TileLang is already compiling FlashQLA for `sm_100a`; the issue is not that it is targeting sm90 machine code.
- Smaller `block_DV` values (`64` and especially `32`) generally hurt. The existing `128` tile is best or close to best.
- Forcing CP on for high-head cases did not help; it worsened `H=64`.
- CP scale is workload dependent: TP8 prefers the Hopper scale around `3.0`; TP2 long sequence prefers around `2.0`; `H=48`/`H=16` often prefer `1.0`. There is no single scalar that fixes B200.
- Even the best variants usually remain slower than FLA on B200 for medium/high head counts, so a real Blackwell port likely needs kernel-body changes in the fused CP forward path, not just host-side scheduling knobs.

## Correctness Caveat

The original Hopper-derived sm100 path failed a targeted FlashQLA correctness
smoke test after the tuning probes:

```bash
cd /home/yih119/FlashQLA/tests
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 python test_gdr.py --set blackwell_smoke --num-heads 16 --skip-bwd --hide-lat --ref-dtype float32
```

Results:

- Auto-CP path produced `nan` in `h_qla`, `s_qla`, and `o_qla` for `B=1, T=8192, Hk=16, Hv=16`.
- Non-CP path avoided NaNs but still failed output accuracy: `o_qla` max error was about `0.316 / 0.360`, far above the 2% threshold.

This means the original B200/sm100 FlashQLA path should not be treated as
numerically valid. The latency experiments are useful for diagnosis, but they
are not publishable performance results.

## Correctness-First Blackwell Route

The `blackwell-experiments` branch now routes sm100 forward calls through an
explicit safe path by default:

- Hopper/sm90 keeps the original fused QLA behavior.
- Blackwell/sm100 disables the broken auto-CP preprocessing by default.
- Fixed-length and uniform-varlen benchmark-style calls with `output_h=False`
  use FLA's forward path directly and pack results back when needed. This keeps
  these common paths correctness-valid without duplicating the broken sm100
  Hopper-derived output kernel.
- Calls that need `output_h=True` still use QLA `chunk_local_cumsum`,
  `kkt_solve`, and state generation, while replacing the broken output
  projection with a safe fallback.
- Fragmented varlen uses a conservative PyTorch/fp32 output fallback from QLA's
  state and a recomputed fp32 `A`. This is a correctness path, not a speed path.
- Set `FLASHQLA_BLACKWELL_EXPERIMENTAL_HOPPER_FWD=1` to re-enable the original
  Hopper-derived fused output and CP path for diagnosis.

Validated on B200:

```bash
cd /home/yih119/FlashQLA
source .venv/bin/activate
TMPDIR=/home/yih119/FlashQLA/.tmp TILELANG_CLEANUP_TEMP_FILES=1 \
CUDA_VISIBLE_DEVICES=0 python tests/test_gdr.py \
  --set blackwell_smoke --num-heads 16 --skip-bwd --hide-lat --ref-dtype float32
```

Result: fixed-length forward smoke passes. The representative output error
changed from roughly `0.311 / 0.360` to `0.0015 / 0.360`.

Additional B200 correctness gates now pass:

```bash
TMPDIR=/home/yih119/FlashQLA/.tmp TILELANG_CLEANUP_TEMP_FILES=1 \
CUDA_VISIBLE_DEVICES=0 python tests/test_gdr.py \
  --set blackwell_varlen_smoke --num-heads 16 --skip-bwd --hide-lat --ref-dtype float32

TMPDIR=/home/yih119/FlashQLA/.tmp TILELANG_CLEANUP_TEMP_FILES=1 \
CUDA_VISIBLE_DEVICES=0 python tests/test_gdr.py \
  --set blackwell_cp_long_smoke --num-heads 16 --skip-bwd --hide-lat --ref-dtype float32
```

Representative results:

- Fragmented varlen: `o_qla: 0.0013 / 0.4253`
- Long fixed-length: `o_qla: 0.0016 / 0.3649`

The new diagnostic harness:

```bash
TMPDIR=/home/yih119/FlashQLA/.tmp TILELANG_CLEANUP_TEMP_FILES=1 \
CUDA_VISIBLE_DEVICES=0 python experiments/blackwell_correctness_probe.py \
  --num-tokens 8192 --num-heads 16 --mode both --inspect-cp
```

Key probe finding: CP preprocessing is the first non-finite source. `CP h0`
contains non-finite values on B200, and the experimental Hopper-derived fused
path then propagates those values into `h` and `output`. The safe path remains
finite and within the existing fixed-length smoke thresholds.

Remaining correctness/performance work:

- The original Hopper-derived sm100 output and CP path remains unsafe and is
  still diagnostic-only.
- Fragmented varlen is correctness-valid through the conservative fallback but
  slow. It still needs a real Blackwell output kernel.

## Valid Safe-Path Performance Snapshot

After adding single-full-sequence normalization, uniform-varlen densification,
and view-based reshape/pack for uniform varlen, the correctness-valid safe path
recovered most of the fallback overhead. Focused B200 probe command:

```bash
TMPDIR=/home/yih119/FlashQLA/.tmp TILELANG_CLEANUP_TEMP_FILES=1 \
FLASHQLA_PROBE_WARMUP=5 FLASHQLA_PROBE_REPEATS=30 \
CUDA_VISIBLE_DEVICES=0 python experiments/blackwell_tune_probe.py
```

| Case | QLA safe ms | FLA ms | FLA/QLA |
| --- | ---: | ---: | ---: |
| tp8_1x32768 | 1.121 | 1.031 | 0.920 |
| tp8_8192x4 | 0.683 | 0.479 | 0.701 |
| tp4_1x32768 | 1.345 | 1.289 | 0.959 |
| tp4_4096x8 | 1.097 | 0.924 | 0.842 |
| tp2_1x32768 | 1.842 | 1.814 | 0.985 |
| tp2_4096x8 | 1.868 | 1.738 | 0.931 |
| tp1_1x32768 | 3.141 | 3.132 | 0.997 |
| tp1_4096x8 | 3.120 | 3.034 | 0.973 |
| h48_1x32768 | 2.626 | 2.561 | 0.975 |
| h16_1x32768 | 1.392 | 1.323 | 0.951 |

Compared with the first correctness-safe probe, uniform multi-sequence cases are
now much faster; for example `tp1_4096x8` improved from about `56.8 ms` to
`3.12 ms`, and `tp2_4096x8` improved from about `30.5 ms` to `1.87 ms`. These
are valid B200 numbers, but they are still near-parity rather than speedups over
FLA or the repo H200 reference.

## Latest CP/Experimental Kernel Finding

Rechecking the CP path showed that CP preprocessing itself can now produce
finite `CP h0` on B200, but enabling the Hopper-derived CP fused forward still
creates non-finite `h` and `output`. This localizes the remaining CP correctness
problem to the CP-enabled `fused_gdr_fwd` kernel body, not the host-side CP split
or `correct_initial_states` preprocessing.

## Output-Kernel Boundary Experiment

The sm100 safe path now has an intermediate kernel-level route for fixed-length
calls that request `output_h=True`: QLA's TileLang fused forward can emit its
computed `v_new`, and FLA's output-only `chunk_fwd_o` consumes QLA `h` + `v_new`
instead of recomputing the entire FLA forward. This keeps the known-good QLA
state path and avoids using the numerically broken Hopper-derived output body.

Representative B200 microbenchmark for `B=1, T=8192, H=16, K=V=128`:

| Path | ms |
| --- | ---: |
| QLA `output_h=True` with QLA `v_new` + output-only kernel | 0.827 |
| QLA/FLA fast `output_h=False` path | 0.362 |
| FLA full forward | 0.362 |

This is not a final speedup yet, but it is a cleaner decomposition for the next
kernel port: replace only the output-only step with a Blackwell/TileLang kernel,
while preserving the already-correct QLA state generation.

An experimental fixed-length TileLang output-only kernel has been added behind
`FLASHQLA_BLACKWELL_TILELANG_OUTPUT=1`. It consumes QLA `h + v_new` directly and
matches the correctness gate (`o_qla: 0.0015 / 0.3597` on `blackwell_smoke`),
but it is not enabled by default because the first simple implementation is
still slower than FLA's Triton output-only kernel:

| Output-only route for QLA `output_h=True` | ms |
| --- | ---: |
| FLA `chunk_fwd_o` default | 0.829 |
| Experimental TileLang output kernel | 0.833 |

The first TileLang output kernel is correctness-valid and near the FLA
output-only fallback for this case. Tuning `block_DV` showed `128` remains best
(`block_DV=64` was about `0.866 ms`; `block_DV=32` was about `0.935 ms` for the
full `output_h=True` path). Varying CTA threads between 128/256/512 had little
impact.

Component timing for `B=1, T=8192, H=16, K=V=128`:

| Component | ms |
| --- | ---: |
| QLA cumsum | 0.008 |
| QLA KKT solve | 0.035 |
| QLA state + `v_new` | 0.729 |
| TileLang output-only | 0.075 |
| FLA output-only | 0.072 |

This shows the next major kernel target is not output-only anymore; it is the
QLA state/`v_new` section of `fused_gdr_fwd`. A direct attempt to conditionally
skip the unused output math inside that fused kernel hit TileLang layout
inference limits, so the cleaner next step is a dedicated state+`v_new` kernel
for the Blackwell safe path.

## Blackwell Architecture Direction

TileLang 0.1.8 is already targeting `sm_100a`, and it exposes Blackwell `tcgen05` primitives. FlashQLA, however, is written around `T.gemm_v1` calls in the Hopper backend. The likely path to significant B200 speedup over the H200 reference is not another host-side heuristic; it is a real forward-kernel port that replaces the Hopper WGMMA-style GEMM use with a Blackwell-aware `tcgen05`/new TileLang GEMM path and revalidates numerics from the reference tests.
