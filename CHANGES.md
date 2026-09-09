# Changes in this fork

This file describes what the `perf/worker-pipeline` series changes relative to
the previous version of this fork (v0.3.0b, itself rebased on upstream mokuro
v0.2.5), why, and what was measured. All numbers were measured with an external
benchmark harness (not part of this repository) on the machines named below,
against upstream mokuro 0.2.5 run on the same machine with the same
dependencies.

**Parity contract of the defaults:** text-block boxes, `lines_coords` and
`font_size` are identical to upstream mokuro 0.2.5 on the same device, and the
page text is byte-identical with the fp32 default (GPU and CPU). The opt-in
`--fp16` OCR precision is faster but not exact (see "Precision policy"). Every
knob whose default would trade accuracy for speed is **off** (`NUM_BEAMS`,
`FUSE_CONV_BN`, `ALLOW_CUDNN_TF32`, `USE_FP16`).

The CLI and the `.mokuro` / `_ocr/*.json` / legacy HTML output formats are
unchanged.

## Architecture

```
GPU mode (CUDA / ROCm / MPS)                 CPU-only mode (--force-cpu or no GPU)
---------------------------------------      ----------------------------------------
main process: models, single GPU context     parent: no models, hands 1-page jobs to
  detector forward + NMS                       NUM_WORKERS shard processes
  OCR beam search (mokuro/beam.py)           each shard: own models, own core block
NUM_WORKERS CPU workers (mokuro/pipeline.py)   (one L3 domain / CCD), runs the SAME
  decode (cv2 fast path) + letterbox          driver loop (process_pages) with an
  DB polygons, group_output, lazy refine,     InlinePool (no extra processes)
  crops, single-plane OCR preprocess
```

Both modes run `mokuro.mokuro_generator.process_pages()`, so the per-page code
path is one and the same; the only difference is where the CPU-side stages
execute. OCR batches are fixed-size and formed strictly in page order (batch k
= crops `[k*bs, (k+1)*bs)` of the page-ordered crop sequence), so the output
does not depend on scheduling or on the worker count. Per-page JSON files are
written as soon as a page's crops all have text; `.mokuro` assembly is
unchanged (natsorted cache files).

## Performance (full 177-page volume, s/page, fp32 default)

| Machine | Device | Upstream 0.2.5 | This fork | Speedup | `--fp16` |
|---|---|---|---|---|---|
| RTX 4090 + Threadripper 9960X | CUDA | 0.416 | **0.039** | 10.7x | 0.024 |
| Threadripper 9960X (CPU only) | CPU | 1.241 | **0.332** | 3.7x | - |
| RX 9070 XT + Ryzen 9 7950X | ROCm | 0.973 | **0.241** | 4.0x | 0.049 |
| Ryzen 9 7950X (CPU only) | CPU | 2.164 | **0.667** | 3.2x | - |
| MacBook Pro M2 Pro 16 GB | MPS | 1.207 | **0.339** | 3.6x | 0.318 |
| M2 Pro (CPU only) | CPU | 2.351 | **0.609** | 3.9x | - |
| RX 6900 XT + Ryzen 7 5800X | ROCm | 0.718 | **0.118** | 6.1x | 0.091 |
| Ryzen 7 5800X (CPU only) | CPU | 2.625 | **1.267** | 2.1x | - |

fp32 output is byte-identical to upstream on every row. The RTX 4090 upstream
figure uses PyTorch's default cuDNN TF32 setting; with TF32 off (the parity
setting this fork uses) upstream runs at 0.393 s/page.

## Optimisations

Each change was first measured in isolation (on a 24-page subset, s/page,
unless noted) and verified for identical output before being combined.

| # | change | measured in isolation | files |
|---|---|---|---|
| 1 | **Worker-process page pipeline**: decode / detector post-processing / crop extraction / OCR preprocessing run in `NUM_WORKERS` spawned processes, the models stay in the main process; tensors cross the queue through shared memory; page-ordered fixed-size OCR batches; per-page failure handling; the pool starts with `init_models()` | RTX 4090 0.111 -> 0.034 (3.3x), RX 9070 XT 0.184 -> 0.077 (2.4x); exact | `mokuro/pipeline.py`, `mokuro/page_ops.py`, `mokuro/mokuro_generator.py`, `mokuro/manga_page_ocr.py` |
| 2 | **Page-level lazy mask refinement**: `refine_mask` / `refine_undetected_mask` run only for pages with a line whose crop ratio exceeds `max_ratio` (their only consumer is `split_into_chunks`) | RTX 4090 2.1x, RX 9070 XT 1.8-2.0x, 7950X 1.14x; exact on all 177 pages | `mokuro/page_ops.py` (knob `LAZY_MASK_REFINE`) |
| 3 | **Bit-identical rewrites of the mask refinement**: `np.bincount` component merge, closed-form `get_topk_color`, boolean-mask indexing, vectorised `refine_undetected_mask` block overlap | RTX 4090 1.25x, RX 9070 XT 1.41x, CPUs 1.04-1.12x; `np.array_equal` on all 177 pages | `comic_text_detector/utils/textmask.py`, `tests/test_textmask.py` |
| 4 | **CPU-only multi-process sharding**: one model process per L3 domain (CCD), pinned, 1-page jobs; the parent loads no models; the config snapshot is re-applied in the children | 9960X 0.729 -> 0.459 (1.6x), 7950X 1.405 -> 0.912 (1.5x); byte-identical | `mokuro/cpu_shards.py`, `mokuro/config.py` (topology helpers), `mokuro/mokuro_generator.py` |
| 5 | **Hand-rolled beam search** replicating transformers' `_beam_search` for the model's own generation config (static in-place KV cache, cross-attention K/V projected once per crop and shared by the beams, `sync_lag` on CUDA/ROCm); falls back to `generate()` when the decoding arguments differ | whole pipeline: 9960X 1.36x, 7950X 1.34-1.44x, RX 9070 XT 1.07-1.14x, RTX 4090 neutral; exact | `mokuro/beam.py`, `mokuro/manga_page_ocr.py` (knobs `USE_CUSTOM_BEAM`, `BEAM_SYNC_LAG`) |
| 6 | **`generate()` fallback**: an instance-scoped `_reorder_cache` that skips the no-op cross-attention KV gather. Only reachable with `USE_CUSTOM_BEAM = False` or a non-default `num_beams` | CPU 1.19-1.23x, ROCm 1.03-1.10x when `generate()` is used; identical | `mokuro/hf_patches.py` (knob `SKIP_CROSS_ATTN_CACHE_REORDER`) |
| 7 | **Text detector in `channels_last`** memory format when it runs on the CPU | 7950X 1.17x, 9960X 1.09x; identical | `mokuro/manga_page_ocr.py`, `comic_text_detector/inference.py` (knob `DETECTOR_CPU_CHANNELS_LAST`) |
| 8 | **OCR batch-size defaults**: 128 on CUDA/ROCm, 64 on MPS, 32 on CPU; `force_cpu` honoured by the defaults. Batches are continuous across the volume (no ragged chunk-end batches) | RTX 4090 1.07-1.16x, ROCm 1.05x, CPU neutral; boxes identical | `mokuro/config.py` |
| 9 | **cv2 fast path** for plain RGB/grayscale JPEGs in `imread` (`IMREAD_IGNORE_ORIENTATION`; PIL for everything else) | ROCm 1.10x, RTX 4090 ~1.07x; pixel-identical to the PIL path | `mokuro/utils.py` (knob `IMAGE_DECODER`) |
| 10 | **Single-plane OCR preprocessing**: the luma plane is resized once with the same call the installed `ViTImageProcessor` uses (torchvision on transformers 5.x, PIL on 4.x - the two differ), normalised with that version's arithmetic and expanded to 3 channels on the device: 1/3 of the work and of the host->device bytes. Runs in the workers (#1) / shards (#4) | ROCm 1.09x, 7950X 1.07x, RTX 4090 neutral; `torch.equal` to the stock `pixel_values` on every batch | `mokuro/page_ops.py` (knob `OCR_PREPROCESS_SINGLE_PLANE`), `tests/test_ocr_preprocess.py` |
| 11 | **Detector transfer micro-wins**: the cancelling BGR->RGB->BGR conversions dropped; uint8 upload + on-device `/ tensor(255.)` normalisation on CUDA/ROCm; the mask scaled to uint8 on the device before the D2H copy (1 MB instead of 4 MB). CPU and MPS keep the numpy path | 1-2% (noise level); input tensor and uint8 mask `np.array_equal` on CUDA/ROCm/CPU | `comic_text_detector/inference.py` (`letterbox_input`, `normalize_input`, `mask_to_uint8`) |

## Correctness fixes

* `USE_TORCH_COMPILE = False`: the previous default (`True`) crashed on every
  CUDA/ROCm machine tested, and once fixed it was slower than eager in every
  mode (default, reduce-overhead, max-autotune). The knob is kept, off, and
  now uses the inductor default mode so that it at least runs; this
  off-by-default path is not benchmarked.
* `FUSE_CONV_BN = False`: no measurable speed-up and it changes the detector
  output on CUDA. Opt-in; logs a warning when on.
* cuDNN TF32 off by default on CUDA (upstream's CUDA boxes differ from its
  CPU boxes by a few pixels because of TF32 convolutions); knob
  `ALLOW_CUDNN_TF32` to re-enable.
* `config.get_device(force_cpu)` / `get_default_num_workers(force_cpu)` /
  `get_default_ocr_batch_size(force_cpu)` honour `--force_cpu` (the previous
  version picked CUDA defaults under `--force_cpu`).
* `--ignore_errors`: decode / detection / post-processing failures are caught
  **per page** (previously `InvalidImage` escaped the decode thread pool and
  aborted the volume); a failing OCR batch is retried page by page and only
  the failing pages are skipped; **no JSON is written for a failed page**
  (previously every page of the chunk got an all-empty JSON, which poisoned
  the cache). Failed pages are retried on the next run.
* Models and worker processes are loaded only when a volume has uncached
  pages (previously they were initialised for fully cached volumes too).
* `--num_workers` now means worker *processes* (previously it only sized the
  8-page chunks and parallelised nothing).
* `MokuroGenerator.close()` shuts the pools down; `run()` calls it in a
  `finally`, and `atexit` covers library use.
* **Small `/dev/shm` fallback** (`mokuro/pipeline.py`): Docker's default
  64 MB shm made every page fail with "unable to allocate shared memory"; the
  pool now detects a small shm (or `MOKURO_FORCE_PIPE_TRANSFER=1`) and passes
  page tensors as numpy arrays through the queue pipes instead (slower,
  identical output, one warning). Pass `--shm-size=8g` to containers for full
  speed.
* **Worker start-up**: a worker that dies while importing is reported
  immediately (was: after a 300 s queue timeout).
* **CPU shards**: `except BaseException` -> `except Exception` so Ctrl-C
  propagates; single-L3 CPUs (Ryzen 7 5800X, Apple M-series) now default to
  one shard per ~2 physical cores, capped by RAM (about 2.5 GB per shard):
  5800X full volume 1.442 -> 1.267 s/page.

## Housekeeping

* `mokuro/__init__.py` exposes `MangaPageOcr` / `MokuroGenerator` lazily
  (PEP 562) so the pipeline workers, which only need `mokuro.page_ops`, do not
  import transformers / manga-ocr. `from mokuro import MokuroGenerator` still
  works.
* `mokuro/config.py` gained `snapshot()` / `apply_snapshot()` so knobs patched
  at runtime by library callers reach the spawned worker / shard processes.
* `comic_text_detector/` is vendored (it was a git submodule) because the
  pipeline needs edits inside it (`inference.py`, `utils/textmask.py`).
* README / OPTIMIZATION_SUMMARY rewritten: claims of the previous version
  that could not be reproduced (the M4 Pro 2.09x table, "byte-identical"
  fp16, `torch.compile` / fusion "wins") are removed; the performance numbers
  are the measured ones above.
* `mokuro/benchmark_mokuro.py` removed: it never ran upstream and forced
  greedy decoding, so it could not produce the numbers it was cited for.
* The code is `ruff check` / `ruff format` clean under the repository's ruff
  configuration.

## Knobs (`mokuro/config.py`)

| knob | default | meaning |
|---|---|---|
| `NUM_WORKERS` / `--num_workers N` | auto: GPU 4 (<= cores-1); CPU one per L3 domain (7950X 2, 9960X 4), ~2 physical cores per shard on single-L3 CPUs, 1 below 4 cores | GPU: pipeline workers (0 = in-process). CPU: shard processes (0/1 = single process) |
| `PIPELINE_MAX_INFLIGHT` / `max_inflight=` | `2*workers+2` | GPU mode: pages in flight (~40 MB each) |
| `CPU_THREADS_PER_PROCESS` / `cpu_threads=` | physical cores of the shard's block | CPU mode |
| `CPU_CHUNK_PAGES` / `cpu_chunk_pages=` | 1 | CPU mode: pages per shard job |
| `CPU_PIN_CORES` / `cpu_pin=` | True | CPU mode: pin shards to core blocks (Linux) |
| `OCR_BATCH_SIZE` / `--ocr_batch_size N` | auto: 128 CUDA/ROCm, 64 MPS, 32 CPU | crops per beam-search call |
| `IMAGE_DECODER` | `"auto"` | `"pil"` forces PIL for every file |
| `NUM_BEAMS` / `--num_beams N` | None (model default 4) | **trade-off** when != 4 |
| `USE_FP16` / `--fp16` | False | fp16 OCR on GPUs: faster, **not exact** (see "Precision policy") |
| `FUSE_CONV_BN` | False | **trade-off** on CUDA when True |
| `USE_TORCH_COMPILE` | False | experiment only (slower) |
| `ALLOW_CUDNN_TF32` | False | **trade-off** on CUDA when True |
| `LAZY_MASK_REFINE` | True | False = eager refinement (same output) |
| `DETECTOR_CPU_CHANNELS_LAST` | True | CPU detector memory format |
| `OCR_PREPROCESS_SINGLE_PLANE` | True | False = stock `ViTImageProcessor` path |
| `USE_CUSTOM_BEAM`, `BEAM_SYNC_LAG` | True, 1 | `mokuro/beam.py` vs `generate()`; host/GPU overlap |
| `SKIP_CROSS_ATTN_CACHE_REORDER` | True | `generate()` fallback path only |

Removed knobs of the previous version: `OCR_CHUNK_SIZE`, `IMAGE_LOAD_THREADS`
(there is no chunking and no decode thread pool any more).

## Tried and not adopted

| experiment | reason |
|---|---|
| greedy / 2-beam decoding | trade-off: 4.9% / 3.3% of the characters change. `NUM_BEAMS` still exists, documented as a trade-off |
| OCR preprocessing on the device | trade-off: 0.3-0.4% CER on the full volume |
| detector fp16 autocast | 4-7% of the boxes drift on CUDA/ROCm |
| next-chunk decode prefetch | superseded by the worker pipeline (only the cv2 decoder is kept) |
| 32-page OCR chunking | superseded by continuous page-ordered batching |
| de-duplicated cross-attention K/V in the `generate()` fallback | only empirical (ULP-level) parity, -2.5..3% on CPU |
| `torch.compile` of the CPU detector | 19-23 s of compile time for ~0.5-1 s per 24 pages |
| worker pipeline in CPU-only mode | the shards are faster there (1.5x vs 1.16x); with `NUM_WORKERS <= 1` on CPU the pipeline runs in-process |
| `torch.compile` on GPUs, ROCm runtime knobs (TunableOp, hipBLASLt, MIOpen find modes), the manga-ocr-2025 model | slower / within noise / net accuracy loss |

## Tests

`pytest -q tests` runs 74 tests on both supported transformers majors
(4.57 and 5.16):

* `tests/test_mokuro.py`, `tests/test_cache.py`: the existing suites,
  unchanged (`--force_cpu` inside, i.e. the CPU shard path).
* `tests/test_ocr_preprocess.py` (new): single-plane preprocessing is
  `torch.equal` to the stock `ViTImageProcessor` of the installed transformers
  on random crops (fp32 and fp16), plus the version -> backend switch.
* `tests/test_beam_compat.py` (new): the beam search's attention-scale
  fallback for transformers 4.x equals SDPA's default scale bit-for-bit.
* `tests/test_textmask.py` (new): the vectorised refinement helpers vs the
  original per-component loops on random masks / histograms (binary,
  hole-fill and float-weight variants).

## transformers 4.x compatibility

Tools such as mokuro-bunko install the OCR engine with
`transformers>=4.25,<5` + `sentencepiece`, while the development stack was
transformers 5.16. Every attribute read from the transformers decoder modules
and caches was audited and the tree runs unchanged on both majors:

* `mokuro/beam.py`: `BertSelfAttention.scaling` exists only on transformers
  >= 5 (`BertSdpaSelfAttention` on 4.x has no such attribute). `_attn_scale()`
  uses `scaling` where present and `attention_head_size ** -0.5` otherwise,
  which is bit-for-bit SDPA's default scale (`tests/test_beam_compat.py`).
  All other attributes read from the decoder (`query` / `key` / `value`,
  `num_attention_heads`, `attention_head_size`, `attention.output`,
  `crossattention.self` / `.output`, `intermediate`, `output`,
  `bert.embeddings(past_key_values_length=)`, `cls`, `generation_config`,
  `enc_to_dec_proj`) are the same on 4.57 and 5.16.
* `mokuro/hf_patches.py` (`generate()` fallback only):
  `EncoderDecoderCache.self_attention_cache` / `cross_attention_cache` and the
  `_reorder_cache` dispatch are the same on 4.57 and 5.16; the cross-cache
  batch probe also reads the pre-4.56 `DynamicCache.key_cache` list. 4.x
  `VisionEncoderDecoderModel` defines no `_reorder_cache`, so the instance
  patch installs as on 5.x. On very old 4.x without `EncoderDecoderCache`
  nothing is patched (stock path).
* `mokuro/page_ops.ocr_preprocess`: the stock `ViTImageProcessor` is the
  torchvision fast processor on 5.x but the PIL "slow" processor on 4.x
  (manga-ocr asks for `ViTImageProcessor` explicitly). PIL's and torchvision's
  antialiased bilinear resize differ by one 8-bit level on ~20% of the pixels
  and the two normalisations round differently, so the single-plane path
  selects its backend from the installed transformers version
  (`OCR_RESIZE_BACKEND`, decided without importing transformers so the
  workers stay light): PIL resize + `float64 * (1/255) -> float32 -> (x - 0.5) / 0.5`
  on 4.x, the torchvision path on 5.x. `tests/test_ocr_preprocess.py` asserts
  `torch.equal` against whichever stock processor is installed.
  Consequence worth knowing: *upstream* mokuro's own `pixel_values` differ
  between the two stacks (different resize), so its OCR text can differ on
  near-tie lines; this fork is identical to upstream on the same stack (on
  the 24-page subset both stacks produced identical text, GPU and CPU).
* Tokenizer loading is manga-ocr's (`AutoTokenizer(..., tokenizer_type="bert-japanese")`
  + sentencepiece / fugashi) and works on both stacks unchanged.

## Precision policy

fp32 OCR is the default on every device and is byte-identical to upstream
mokuro 0.2.5 on all measured volumes and resolution tiers (1080x1530 to
~1790x2800). `--fp16` (config `USE_FP16 = True`) is opt-in and runs the OCR
transformer in half precision. fp16 is **not exact**: measured against the
fp32 output on 140 volumes (26,365 pages, 2.52 M characters), `--fp16`
changes **0.19% of the characters** (4,753 of 2,520,342) on **26 pages per
1000** (695 of 26,365; 137 of the 140 volumes have at least one changed
character). Text boxes are **never** affected by the precision (the detector
always runs in fp32). The changed characters are concentrated on
low-confidence lines (hallucinated text over non-text regions, ellipsis
lengths, colophon / date strings), but real dialogue lines are affected too,
so treat `--fp16` as a speed/accuracy trade-off. fp32 and `--fp16` are the
only two OCR precision modes.

### fp16 vs fp32 character error rate, 140 volumes

Reference = fp32 output of the same pipeline on the RTX 4090 (ROCm fp32
matches it to within 0.002% of the characters).

| GPU | set | volumes | pages | chars | volumes identical | pages with a change | chars changed | CER | fp16 s/page | fp32 s/page |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX 4090 | 100 library volumes | 100 | 18487 | 1,713,612 | 2 | 461 (2.5%) | 2,922 | 0.171% | 0.023 | 0.038 |
| RTX 4090 | 20 Dr Stone HD | 20 | 3952 | 428,584 | 0 | 132 (3.3%) | 1,053 | 0.246% | 0.029 | 0.048 |
| RTX 4090 | 20 '(HD Scan)' series | 20 | 3926 | 378,146 | 1 | 102 (2.6%) | 778 | 0.206% | 0.027 | 0.042 |
| RX 9070 XT | 100 library volumes | 100 | 18487 | 1,713,612 | 4 | 434 (2.3%) | 2,892 | 0.169% | 0.041 | 0.247 |
| RX 9070 XT | 20 Dr Stone HD | 20 | 3952 | 428,584 | 0 | 121 (3.1%) | 1,031 | 0.241% | 0.054 | 0.292 |
| RX 9070 XT | 20 '(HD Scan)' series | 20 | 3926 | 378,146 | 0 | 120 (3.1%) | 839 | 0.222% | 0.047 | 0.232 |
| RTX 4090 | **all 140** | 140 | 26365 | 2,520,342 | 3 | 695 (2.6%) | 4,753 | **0.189%** | 0.024 | 0.039 |

### Speed of the fp32 default vs `--fp16` (full 177-page volume, s/page)

| GPU | fp32 (default) | `--fp16` | fp16 gain |
|---|---|---|---|
| RTX 4090 | 0.039 | 0.024 | 1.6x |
| RX 9070 XT | 0.241 | 0.049 | 4.9x |
| RX 6900 XT | 0.118 | 0.091 | 1.3x |
| M2 Pro (MPS) | 0.339 | 0.318 | 1.07x |

On the 140-volume set the gain is 1.55-1.65x on the RTX 4090 and 5-6x on the
RX 9070 XT. On RDNA4 (gfx1201) fp32 GEMMs are slow enough that the fp32
default is slower than the previous version's fp16 path; use `--fp16` there
if speed matters more than exact OCR text.
