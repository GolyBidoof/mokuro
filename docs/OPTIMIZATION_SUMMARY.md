# Mokuro Optimization Summary

This document summarises the performance work in this branch: what changed,
why, and how it was verified. The goal is upstream mokuro's **exact output
format, CLI and OCR results** while making processing several times faster on
NVIDIA / AMD GPUs, Apple Silicon and plain CPUs. Per-change details and knobs
are in [CHANGES.md](CHANGES.md).

This branch is the mokuro part of a three-PR split: the detector changes live
in a companion PR against kha-white/comic-text-detector and the OCR changes
(batched recognition, single-plane preprocessing, beam search, fp16) in a
companion PR against kha-white/manga-ocr. mokuro keeps the detector as a
submodule and calls manga-ocr's public API. The numbers below are for the
complete stack.

## Where the time went (upstream / the previous fork)

Profiling the starting point of this work on a 177-page volume showed that
on a fast GPU **more than half of the wall time was single-threaded CPU
work** serialised with the GPU: text-mask refinement (47-56%), JPEG decode
(11-17%), detector post-processing and OCR preprocessing. The GPU itself was
idle most of the time. On CPU-only machines the OCR transformer's beam search
dominated (~70%), and most of that was transformers' per-step Python glue
rather than the model's arithmetic.

## What this PR does

### 1. Worker-process page pipeline (`mokuro/pipeline.py`, `mokuro/page_ops.py`, `mokuro/mokuro_generator.py`)

The per-page work is split into pure functions (`page_ops`) that run in
spawned worker processes: image decode + letterbox, DB polygon extraction,
`group_output`, mask refinement, text-line crop extraction and OCR
preprocessing. The main process only runs the detector forward + NMS and the
batched OCR. Tensors cross the process boundary through shared memory. OCR
batches are fixed-size and formed strictly in page order, so the result is
independent of scheduling and of the worker count. Failures are handled per
page.

### 2. Lazy text-mask refinement (`page_ops`)

The refined mask is only consumed when a text line is longer than `max_ratio`
and has to be split into chunks, a small fraction of pages. It is now computed
lazily at page level (identical output: the block list is built before
refinement and nothing else reads the refined mask). The refinement code
itself, vectorised in the detector PR, is now bit-identical and several times
faster.

### 3. CPU-only sharding (`mokuro/cpu_shards.py`)

One model process per L3 cache domain (CCD), pinned to its cores, pulling
one-page jobs from a queue. Each shard runs the same pipeline driver as the
GPU path. Measured ~1.5-1.6x over one process with all cores on a 7950X and a
9960X; fewer, fatter processes aligned to the cache domains beat every other
split. The detector additionally runs in `channels_last` on CPU in that mode.

### 4. Batched OCR, cv2 decode, larger batches

- OCR batch size 128 on CUDA/ROCm (64 MPS, 32 CPU); the batch composition is
  page-ordered and continuous across the volume (no ragged batches).
- Plain JPEGs are decoded by cv2 (pixel-identical to PIL).
- The OCR side itself (batched recognition, single-plane preprocessing, beam
  search, fp16) is manga-ocr's, so other manga-ocr users get it too.

### 5. Correctness fixes over the previous fork

- cuDNN TF32 is off on CUDA so GPU and CPU results agree (`ALLOW_CUDNN_TF32`).
- conv+bn fusion stays off: it changed detector boxes on CUDA with no gain.
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
177-page volume. The numbers are in [CHANGES.md](CHANGES.md).
`tests/test_mokuro.py` additionally runs the real detector and OCR on the
repository's test volumes and checks boxes, coordinates, font sizes and text
against upstream-produced expected results.

## AI contributions

This performance work was developed and refined with the help of
**DeepSeek V4 Flash**, an AI coding agent, under the direction of
**GolyBidoof** (this fork's maintainer). The agent assisted with profiling,
designing and implementing the optimisations (worker pipeline, CPU sharding,
batching, correctness fixes), running the benchmark harness and the
bit-exactness comparisons, and iterating on correctness fixes. The human
maintainer owns the direction, the final review, and every design decision
that trades speed against accuracy; claims that could not be reproduced were
removed rather than kept.
