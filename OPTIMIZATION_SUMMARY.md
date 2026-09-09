# Mokuro Optimization Summary

This document describes the performance work in this fork: what was changed,
why, and how it was verified. The goal was to keep mokuro's **exact output
format, CLI and OCR results** while making processing several times faster on
NVIDIA / AMD GPUs, Apple Silicon and plain CPUs. The per-change details,
origins and knobs are in `CHANGES.md`; measured numbers are filled in from the
benchmark results (README, "Performance").

## Where the time went (upstream / the previous fork)

Profiling the previous version of this fork on a 177-page volume showed that
on a fast GPU **more than half of the wall time was single-threaded CPU
work** serialised with the GPU: text-mask refinement (47-56%), JPEG decode
(11-17%), detector post-processing and OCR preprocessing. The GPU itself was
idle most of the time. On CPU-only machines the OCR transformer's beam search
dominated (~70%), and most of that was transformers' per-step Python glue
rather than the model's arithmetic.

## What changed

### 1. Worker-process page pipeline (`mokuro/pipeline.py`, `mokuro/page_ops.py`, `mokuro/mokuro_generator.py`)

The per-page work is split into pure functions (`page_ops`) that run in
spawned worker processes: image decode + letterbox, DB polygon extraction,
`group_output`, mask refinement, text-line crop extraction and OCR
preprocessing. The main process only runs the detector forward + NMS and the
OCR beam search. Tensors cross the process boundary through shared memory.
OCR batches are fixed-size and formed strictly in page order, so the result is
independent of scheduling and of the worker count. Failures are handled per
page.

### 2. Lazy, faster text-mask refinement (`page_ops`, `comic_text_detector/utils/textmask.py`)

The refined mask is only consumed when a text line is longer than
`max_ratio` and has to be split into chunks - a small fraction of pages. It is
now computed lazily at page level (identical output: the block list is built
before refinement and nothing else reads the refined mask). The refinement
code itself was rewritten with vectorised numpy (`np.bincount` component
merging, closed-form colour selection), verified bit-identical on every page
of the test volume.

### 3. Hand-rolled beam search (`mokuro/beam.py`)

A beam search that replicates transformers' `_beam_search` token for token for
the model's own generation config (4 beams, no-repeat-ngram 3, length penalty
2.0, early stopping), calling the model's own modules with the same shapes -
so the numerics are the same kernels - but with a static, in-place KV cache
and cross-attention K/V projected once per crop and shared across beams. On
CPU this removes most of the decoder's overhead. Non-default decoding
settings automatically fall back to `generate()`, where the (also exact)
cross-attention cache-reorder skip applies.

### 4. CPU-only sharding (`mokuro/cpu_shards.py`)

One model process per L3 cache domain (CCD), pinned to its cores, pulling
one-page jobs from a queue. Each shard runs the same pipeline driver as the
GPU path. Measured ~1.5-1.6x over one process with all cores on a 7950X and a
9960X; fewer, fatter processes aligned to the cache domains beat every other
split. The detector additionally runs in `channels_last` on CPU (~2x on its
forward pass).

### 5. Batched OCR with larger batches, single-plane preprocessing, cv2 decode, transfer micro-wins

- OCR batch size 128 on CUDA/ROCm (64 MPS, 32 CPU); the batch composition is
  page-ordered and continuous across the volume (no ragged batches).
- OCR crops are grayscale: the single luma plane is resized with the same
  torchvision call the stock processor uses and expanded on the device
  (`torch.equal` to the stock pixel values).
- Plain JPEGs are decoded by cv2 (pixel-identical to PIL).
- The detector's redundant BGR<->RGB conversions are dropped, the input is
  uploaded as uint8 and normalised on the GPU with a correctly rounded
  division, and the mask is converted to uint8 before the copy back.
- OCR precision: fp32 by default (byte-identical to upstream). `--fp16`
  (`USE_FP16`) is the only opt-in precision mode: 1.6x (RTX 4090) to 4.9x
  (RX 9070 XT) faster but not exact: on 140 volumes / 2.52 M characters it
  changes 0.19% of the characters on 26 pages per 1000, boxes never
  (README, "Precision policy").

### 6. Correctness fixes over the previous fork

- `torch.compile` (crashed on current torch, and slower than eager when
  fixed) and conv+bn fusion (changed detector boxes on CUDA, no speed gain)
  are off; cuDNN TF32 is off on CUDA so GPU and CPU results agree.
- `--force_cpu` is honoured by the automatic defaults.
- `--ignore_errors` skips exactly the failing page (no empty cache files for
  the rest of a chunk); a broken image no longer aborts the volume.
- Models are not loaded for fully cached volumes.

## What was tried and rejected

- `torch.compile` in every mode (default, reduce-overhead, max-autotune) on an
  RTX 4090 and an RX 9070 XT: slower than eager.
- ROCm runtime knobs (TunableOp, hipBLASLt, MIOpen find modes): within noise.
- Detector fp16 autocast: 4-7% of boxes drift on CUDA/ROCm.
- Greedy / 2-beam decoding (3-5% CER), on-device OCR preprocessing
  (0.3-0.4% CER), the manga-ocr-2025 model (net accuracy loss): all excluded
  from the defaults because they change the OCR text.

## Verification

Every change was benchmarked with an external benchmark harness (not part of
this repository) on four machines (RTX 4090 / Threadripper 9960X,
RX 9070 XT / Ryzen 9 7950X, RX 6900 XT / Ryzen 7 5800X, MacBook Pro M2 Pro)
against upstream 0.2.5 reference outputs produced on the same machine, with a
box-drift / character-error comparison over a 24-page subset and the full
177-page volume; the numbers are reported in the README and in `CHANGES.md`.
The unit tests in `tests/` additionally assert the bit-exactness of the
rewritten preprocessing and mask-refinement code without running the models.
