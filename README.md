# mokuro (performance-optimized fork)

Read Japanese manga with selectable text in a browser. This fork of
[kha-white/mokuro](https://github.com/kha-white/mokuro) (rebased on upstream
**v0.2.5**) keeps the exact same CLI, output format and workflow, but runs
several times faster on NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS) and
CPU-only machines.

The defaults produce **identical output to upstream** (same text boxes, same
text; see [Parity](#parity)). The speed comes from restructuring how the work
is scheduled, not from lowering precision or beam width.

**Version: 0.4.0b**

**Demo: https://kha-white.github.io/manga-demo**

mokuro is aimed at Japanese learners who want to read manga in Japanese with a
pop-up dictionary like [Yomitan](https://github.com/themoeway/yomitan). It
works like this:

1. Detect and OCR the text on each page.
2. Generate a `.mokuro` file per volume, containing the OCR results and
   metadata. All processing is done offline, before reading.
3. Open the `.mokuro` file with the manga images in the
   [web reader](https://reader.mokuro.app/).

The older HTML output (mokuro 0.1.x style) is still generated for backward
compatibility, but the `.mokuro` format plus the web reader is the recommended
setup. Text detection uses [comic-text-detector](https://github.com/dmMaze/comic-text-detector),
OCR uses [manga-ocr](https://github.com/kha-white/manga-ocr).

## What's improved

| Stage | Upstream 0.2.5 | This fork |
|---|---|---|
| OCR inference | one `generate()` call per text line | batched beam search, fixed page-ordered batches (128 crops on CUDA/ROCm, 64 on MPS, 32 on CPU) |
| Beam search | transformers' generic `generate()` | hand-rolled beam search with a static KV cache and shared cross-attention; identical results |
| Page decode, detector post-processing, crop extraction | main thread, serialised with the models | worker processes (GPU mode), overlapped with GPU work |
| Text-mask refinement | every page (~half of GPU wall time) | lazy, only for pages with an over-long line; bitwise-identical |
| CPU-only mode | one process, all cores as torch threads | one model process per L3 domain/CCD, pinned cores |
| OCR preprocessing | 3 colour planes resized on the host | single grayscale plane, expanded on the device (bit-identical) |
| JPEG decode | PIL | cv2 fast path (pixel-identical) |
| `--ignore_errors` | a broken image aborted the volume | skips just that page, retried next run |
| Cached volumes | models loaded anyway | models loaded only when there is something to OCR |

The `.mokuro`, `.html` and `_ocr/` cache formats are unchanged.

All knobs are tuned in one file: [`mokuro/config.py`](mokuro/config.py) has a
clearly marked "EDIT ME" block with per-knob comments. Hardware is
auto-detected; command-line flags override the config for a single run.

## Performance

Measured with a 177-page volume (1440x2048 JPEG pages, cold OCR cache, default
beam width, best of two runs after a warm-up volume), unless noted. Baseline
is upstream mokuro 0.2.5 on the same machine with the same dependencies. fp32
output is byte-identical to upstream on every row; `--fp16` is opt-in (see
[Parity](#parity)).

| Machine | Device | Upstream 0.2.5 (s/page) | This fork, fp32 (s/page) | Speedup | This fork, `--fp16` (s/page) |
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

The M4 Pro row was measured on a 209-page volume instead of the 177-page one,
so treat it as a separate datapoint, not a direct comparison with the other
rows.

The RTX 4090 upstream figure uses PyTorch's default cuDNN TF32 setting; with
TF32 off (the parity setting this fork uses) upstream runs at 0.393 s/page.

How the individual optimisations were measured and verified is documented in
[docs/CHANGES.md](docs/CHANGES.md) and [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md).

## Installation

Requires Python 3.10+. Check the [PyTorch website](https://pytorch.org/get-started/locally/)
for supported Python versions, and install PyTorch for your GPU if you have
one.

```commandline
pip3 install git+https://github.com/GolyBidoof/mokuro.git
```

or from a local checkout:

```commandline
pip3 install -e .
```

If you already have the PyPI mokuro installed and want to swap this fork in
(venv, machine-wide, or a zero-install pointer file), see
[docs/REPLACING.md](docs/REPLACING.md).

## Usage

```bash
mokuro /path/to/manga/vol1                    # one volume -> vol1.html
mokuro /path/to/manga/vol1 /path/to/manga/vol2
mokuro --parent_dir /path/to/manga            # every volume under a directory
```

Quote paths containing spaces. Useful flags:

```
--force_cpu             Use CPU even if CUDA/MPS is available
--disable_confirmation  Skip the volume list confirmation prompt
--disable_ocr           Generate mokuro/HTML files without OCR
--ignore_errors         Skip failing pages/volumes instead of aborting
--no_cache              Ignore cached OCR results
--num_workers N         GPU: pipeline workers; CPU: model shards
--ocr_batch_size N      Text-line crops per batched OCR call
--num_beams N           Beam width (default 4 = identical to upstream)
--fp16                  Faster OCR on GPUs, not exact (see Parity)
--version               Print the version and exit
```

Note: the CLI uses python-fire, so a boolean flag directly before a path is
consumed by it (`--fp16 /path/vol` reads as `fp16="/path/vol"`). Write
`--fp16=True` or put flags after the paths.

## Parity

"Identical output" means: on the same machine, the text-block boxes, line
coordinates and font sizes are identical to upstream mokuro 0.2.5, and the OCR
text is identical. This holds for the fp32 default on every device (GPU and
CPU-only output are byte-identical to upstream).

Upstream itself is not identical across devices: on NVIDIA GPUs, cuDNN's TF32
convolutions move some detector boxes by a few pixels relative to the CPU
result. This fork turns TF32 off (`ALLOW_CUDNN_TF32 = False`), so its CUDA
output matches the CPU/ROCm output.

The opt-in `--fp16` mode (config `USE_FP16`) is faster but not exact: measured
on 140 volumes (26,365 pages, 2.52M characters), it changes 0.19% of the
characters on 2.6% of the pages; text boxes are never affected (the detector
always runs in fp32). The full precision policy and fp16-vs-fp32 character
error tables live in [docs/CHANGES.md](docs/CHANGES.md).

## Documentation

The long-form material lives in `docs/`:

- [docs/CHANGES.md](docs/CHANGES.md): the improvement history of this fork,
  with the parity contract, architecture, per-optimisation measurements,
  tried-and-rejected ideas, knobs table and precision policy.
- [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md): a narrative
  overview of where the time went and what was changed.
- [docs/REPLACING.md](docs/REPLACING.md): how to swap this fork in for a
  pip-installed mokuro (venv, machine-wide, or zero-install pointer).

## Development

```bash
pip3 install -e ".[dev]"
python3 -m pytest tests/          # run the test suite (CPU)
python3 -m ruff check . && python3 -m ruff format --check .
```

The fork runs on both transformers major versions (5.x, the current manga-ocr
stack, and 4.x with `transformers>=4.25,<5` + `sentencepiece`). See
[docs/CHANGES.md](docs/CHANGES.md) for the transformers compatibility notes.

## Keeping in sync with upstream

```bash
git remote add upstream https://github.com/kha-white/mokuro.git   # once
git fetch upstream
git merge upstream/master
```

`comic_text_detector/` is vendored in this repository (upstream tracks it as a
git submodule); when a merge touches it, keep this repository's copy.

## License and credits

- **GPL-3.0**, see [LICENSE](LICENSE). This fork inherits upstream mokuro's
  license unmodified.
- Upstream: [kha-white/mokuro](https://github.com/kha-white/mokuro) by
  [Maciej Budyś](https://github.com/kha-white).
- This repository builds on [GolyBidoof/mokuro](https://github.com/GolyBidoof/mokuro)
  (v0.3.0b), the fork this performance work started from.
- Text detection: [comic-text-detector](https://github.com/dmMaze/comic-text-detector);
  OCR: [manga-ocr](https://github.com/kha-white/manga-ocr);
  text segmentation: [Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation).
