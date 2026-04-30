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

A targeted FlashQLA correctness smoke test was run after the tuning probes:

```bash
cd /home/yih119/FlashQLA/tests
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 python test_gdr.py --set blackwell_smoke --num-heads 16 --skip-bwd --hide-lat --ref-dtype float32
```

Results:

- Auto-CP path produced `nan` in `h_qla`, `s_qla`, and `o_qla` for `B=1, T=8192, Hk=16, Hv=16`.
- Non-CP path avoided NaNs but still failed output accuracy: `o_qla` max error was about `0.316 / 0.360`, far above the 2% threshold.

This means the current B200/sm100 FlashQLA path should not be treated as numerically valid. The latency experiments are useful for diagnosis, but they are not publishable performance results.

## Blackwell Architecture Direction

TileLang 0.1.8 is already targeting `sm_100a`, and it exposes Blackwell `tcgen05` primitives. FlashQLA, however, is written around `T.gemm_v1` calls in the Hopper backend. The likely path to significant B200 speedup over the H200 reference is not another host-side heuristic; it is a real forward-kernel port that replaces the Hopper WGMMA-style GEMM use with a Blackwell-aware `tcgen05`/new TileLang GEMM path and revalidates numerics from the reference tests.
