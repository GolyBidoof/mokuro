# Changes in this PR

This PR is the mokuro part of a three-PR performance work. The detector changes
live in a companion PR against kha-white/comic-text-detector and the OCR changes
(batched recognition, single-plane preprocessing, custom beam search, fp16) live
in a companion PR against kha-white/manga-ocr; this PR keeps
`comic_text_detector` as a submodule and calls manga-ocr's public API. The three
PRs together reproduce the fork's measured output; this PR depends on both.

The goal throughout is upstream mokuro v0.2.5's exact CLI and output, with the
same OCR text, while removing the CPU work that used to sit between the two
model forwards.

**Parity contract of the defaults:** text-block boxes, `lines_coords` and
`font_size` are identical to upstream mokuro 0.2.5 on the same device, and the
page text is byte-identical with the fp32 default (GPU and CPU). The opt-in
`--fp16` OCR precision is faster but not exact (see manga-ocr's documentation
and the "Precision policy" section below). Every knob whose default would trade
accuracy for speed is **off** (`NUM_BEAMS`, `FUSE_CONV_BN`,
`ALLOW_CUDNN_TF32`, `USE_FP16`).

The CLI and the `.mokuro` / `_ocr/*.json` / legacy HTML output formats are
unchanged.

## Architecture

```
GPU mode (CUDA / ROCm / MPS)                 CPU-only mode (--force-cpu or no GPU)
---------------------------------------      ----------------------------------------
main process: models, single GPU context     parent: no models, hands 1-page jobs to
  detector forward + NMS (CTD public API)      NUM_WORKERS shard processes
  batched OCR (manga-ocr public API)          each shard: own models, own core block
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

Numbers are for the complete stack (this PR plus the two companion PRs), which
is what the fork's benchmarks measured. This PR on its own is the pipeline
share: it removes the CPU work serialised with the GPU and shards CPU-only
runs; the beam-search, preprocessing and mask-refinement gains come from the
companion PRs.

| Machine | Device | Upstream 0.2.5 | Complete stack | Speedup | `--fp16` |
|---|---|---|---|---|---|
| RTX 4090 + Threadripper 9960X | CUDA | 0.416 | **0.039** | 10.7x | 0.024 |
| Threadripper 9960X (CPU only) | CPU | 1.241 | **0.332** | 3.7x | - |
| RX 9070 XT + Ryzen 9 7950X | ROCm | 0.973 | **0.241** | 4.0x | 0.049 |
| Ryzen 9 7950X (CPU only) | CPU | 2.164 | **0.667** | 3.2x | - |
| MacBook Pro M2 Pro 16 GB | MPS | 1.207 | **0.339** | 3.6x | 0.318 |
| M2 Pro (CPU only) | CPU | 2.351 | **0.609** | 3.9x | - |
| MacBook Pro M4 Pro 24 GB | MPS | 1.868 | **0.432** | 4.3x | 0.424 |
| RX 6900 XT + Ryzen 7 5800X | ROCm | 0.718 | **0.118** | 6.1x | 0.091 |
| Ryzen 7 5800X (CPU only) | CPU | 2.625 | **1.267** | 2.1x | - |

fp32 output is byte-identical to upstream on every row. The RTX 4090 upstream
figure uses PyTorch's default cuDNN TF32 setting; with TF32 off (the parity
setting this stack uses) upstream runs at 0.393 s/page.

The M4 Pro row was measured on a different volume (209 pages instead of 177),
so read it as a separate datapoint rather than a direct comparison with the
other rows.

## What this PR contains

| # | change | measured in isolation | files |
|---|---|---|---|
| 1 | **Worker-process page pipeline**: decode / detector post-processing / crop extraction / OCR preprocessing run in `NUM_WORKERS` spawned processes, the models stay in the main process; tensors cross the queue through shared memory; page-ordered fixed-size OCR batches; per-page failure handling; the pool starts with `init_models()` | RTX 4090 0.111 -> 0.034 (3.3x), RX 9070 XT 0.184 -> 0.077 (2.4x); exact | `mokuro/pipeline.py`, `mokuro/page_ops.py`, `mokuro/mokuro_generator.py`, `mokuro/manga_page_ocr.py` |
| 2 | **Page-level lazy mask refinement**: `refine_mask` / `refine_undetected_mask` run only for pages with a line whose crop ratio exceeds `max_ratio` (their only consumer is `split_into_chunks`) | RTX 4090 2.1x, RX 9070 XT 1.8-2.0x, 7950X 1.14x; exact on all 177 pages | `mokuro/page_ops.py` (knob `LAZY_MASK_REFINE`) |
| 3 | **CPU-only multi-process sharding**: one model process per L3 domain (CCD), pinned, 1-page jobs; the parent loads no models; the config snapshot is re-applied in the children | 9960X 0.729 -> 0.459 (1.6x), 7950X 1.405 -> 0.912 (1.5x); byte-identical | `mokuro/cpu_shards.py`, `mokuro/config.py` (topology helpers), `mokuro/mokuro_generator.py` |
| 4 | **OCR batch-size defaults**: 128 on CUDA/ROCm, 64 on MPS, 32 on CPU; `force_cpu` honoured by the defaults. Batches are continuous across the volume (no ragged chunk-end batches) | RTX 4090 1.07-1.16x, ROCm 1.05x, CPU neutral; boxes identical | `mokuro/config.py` |
| 5 | **cv2 fast path** for plain RGB/grayscale JPEGs in `imread` (`IMREAD_IGNORE_ORIENTATION`; PIL for everything else) | ROCm 1.10x, RTX 4090 ~1.07x; pixel-identical to the PIL path | `mokuro/utils.py` (knob `IMAGE_DECODER`) |
| 6 | **Text detector in `channels_last`** memory format when it runs on the CPU (the memory format lives here; the forward split that makes it useful is in the detector PR) | 7950X 1.17x, 9960X 1.09x; identical | `mokuro/manga_page_ocr.py` (knob `DETECTOR_CPU_CHANNELS_LAST`) |

## Delivered by the companion PRs

- kha-white/comic-text-detector (`inference.py`, `utils/textmask.py`):
  the split of `TextDetector.forward` from post-processing, uint8 host
  transfers with a correctly rounded on-device normalisation, the uint8 mask
  scale before the device-to-host copy, and the vectorised mask-refinement
  helpers (`np.bincount` component merge, closed-form colour selection).
- kha-white/manga-ocr: batched recognition, single-plane crop preprocessing,
  the hand-rolled beam search with its `generate()` fallback, and the fp16
  mode, all behind a public API (`MangaOcr.recognize`,
  `MangaOcr.recognize_preprocessed`, `MangaOcr.preprocess_config`,
  `manga_ocr.preprocessing.preprocess_images`). mokuro no longer reaches into
  `MangaOcr.model` / `.processor` / `.tokenizer`.

## Correctness fixes

* cuDNN TF32 off by default on CUDA (upstream's CUDA boxes differ from its
  CPU boxes by a few pixels because of TF32 convolutions); knob
  `ALLOW_CUDNN_TF32` to re-enable.
* `config.get_device(force_cpu)` / `get_default_num_workers(force_cpu)` /
  `get_default_ocr_batch_size(force_cpu)` honour `--force_cpu` (the previous
  earlier builds of this work picked CUDA defaults under `--force_cpu`).
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
  (PEP 562) so the pipeline workers, which only need `mokuro.page_ops` and
  manga-ocr's preprocessing module, do not import transformers / the models.
* `mokuro/config.py` gained `snapshot()` / `apply_snapshot()` so knobs patched
  at runtime by library callers reach the spawned worker / shard processes.
* `comic_text_detector` stays a git submodule (the branch bumps it to the
  detector PR's commit); nothing in it is vendored.
* `tests/test_textmask.py`, `tests/test_ocr_preprocess.py` and
  `tests/test_beam_compat.py` moved to the repositories that own the code
  (detector and manga-ocr).

## Knobs (`mokuro/config.py`)

| knob | default | meaning |
|---|---|---|
| `NUM_WORKERS` / `--num_workers N` | auto: GPU 4 (<= cores-1); CPU one per L3 domain (7950X 2, 9960X 4), ~2 physical cores per shard on single-L3 CPUs, 1 below 4 cores | GPU: pipeline workers (0 = in-process). CPU: shard processes (0/1 = single process) |
| `PIPELINE_MAX_INFLIGHT` / `max_inflight=` | `2*workers+2` | GPU mode: pages in flight (~40 MB each) |
| `CPU_THREADS_PER_PROCESS` / `cpu_threads=` | physical cores of the shard's block | CPU mode |
| `CPU_CHUNK_PAGES` / `cpu_chunk_pages=` | 1 | CPU mode: pages per shard job |
| `CPU_PIN_CORES` / `cpu_pin=` | True | CPU mode: pin shards to core blocks (Linux) |
| `OCR_BATCH_SIZE` / `--ocr_batch_size N` | auto: 128 CUDA/ROCm, 64 MPS, 32 CPU | crops per OCR call |
| `IMAGE_DECODER` | `"auto"` | `"pil"` forces PIL for every file |
| `NUM_BEAMS` / `--num_beams N` | None (model default 4) | forwarded to manga-ocr; **trade-off** when != 4 |
| `USE_FP16` / `--fp16` | False | forwarded to manga-ocr: faster, **not exact** |
| `FUSE_CONV_BN` | False | **trade-off** on CUDA when True |
| `ALLOW_CUDNN_TF32` | False | **trade-off** on CUDA when True |
| `LAZY_MASK_REFINE` | True | False = eager refinement (same output) |
| `DETECTOR_CPU_CHANNELS_LAST` | True | CPU detector memory format |

Removed knobs of the previous version: `OCR_CHUNK_SIZE`, `IMAGE_LOAD_THREADS`
(there is no chunking and no decode thread pool any more); the OCR-side knobs
(`USE_CUSTOM_BEAM`, `BEAM_SYNC_LAG`, `SKIP_CROSS_ATTN_CACHE_REORDER`,
`OCR_PREPROCESS_SINGLE_PLANE`, `USE_TORCH_COMPILE`) are now manga-ocr's.

## Tried and not adopted

| experiment | reason |
|---|---|
| greedy / 2-beam decoding | trade-off: 4.9% / 3.3% of the characters change. `NUM_BEAMS` still exists, documented as a trade-off |
| detector fp16 autocast | 4-7% of the boxes drift on CUDA/ROCm |
| next-chunk decode prefetch | superseded by the worker pipeline (only the cv2 decoder is kept) |
| 32-page OCR chunking | superseded by continuous page-ordered batching |
| worker pipeline in CPU-only mode | the shards are faster there (1.5x vs 1.16x); with `NUM_WORKERS <= 1` on CPU the pipeline runs in-process |
| `torch.compile` on GPUs, ROCm runtime knobs (TunableOp, hipBLASLt, MIOpen find modes), the manga-ocr-2025 model | slower / within noise / net accuracy loss |

## Tests

`pytest -q tests` runs the upstream `tests/test_mokuro.py` and
`tests/test_cache.py` unchanged. `test_mokuro.py` runs the real detector and
OCR on the repository's test volumes and compares boxes, `lines_coords`,
`font_size` and text against upstream-produced expected results, so it covers
the full split stack (submodule + manga-ocr) end to end.

## transformers 4.x and 5.x

The OCR stack supports both majors; manga-ocr owns that compatibility (its CI
matrix runs transformers 4.51.3 and latest) and the detector is torch-only.
`mokuro` itself imports neither version-specific API.

## Precision policy

fp32 OCR is the default on every device and is byte-identical to upstream
mokuro 0.2.5 on all measured volumes and resolution tiers (1080x1530 to
~1790x2800). `--fp16` (config `USE_FP16 = True`) runs the OCR transformer in
half precision through manga-ocr. fp16 is **not exact**: measured against the
fp32 output on 140 volumes (26,365 pages, 2.52 M characters), `--fp16`
changes **0.19% of the characters** (4,753 of 2,520,342) on **26 pages per
1000** (695 of 26,365; 137 of the 140 volumes have at least one changed
character). Text boxes are **never** affected by the precision (the detector
always runs in fp32). The changed characters are concentrated on
low-confidence lines (hallucinated text over non-text regions, ellipsis
lengths, colophon / date strings), but real dialogue lines are affected too,
so treat `--fp16` as a speed/accuracy trade-off. fp32 and `--fp16` are the
only two OCR precision modes.
