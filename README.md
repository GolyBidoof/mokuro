# mokuro — performance-optimized fork

Read Japanese manga with selectable text inside a browser — **optimized for
speed** on NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS) and CPU-only machines.

This is a fork of [kha-white/mokuro](https://github.com/kha-white/mokuro)
(rebased on upstream **v0.2.5**) that keeps the exact same CLI, output format
and workflow while making OCR several times faster. Every default keeps the
OCR output identical to upstream (same text boxes, same text; see
[Parity](#parity)); the speed comes from restructuring *how* the work is
scheduled, not from lowering the model's precision or beam width.

**Version: 0.3.0b** — the `b` marks this fork's *bridge* lineage (it grew out
of the mokuro-bridge project) and distinguishes it from upstream releases.

**See demo: https://kha-white.github.io/manga-demo**

mokuro is aimed towards Japanese learners, who want to read manga in Japanese with a pop-up dictionary like [Yomitan](https://github.com/themoeway/yomitan).
It works like this:
1. Perform text detection and OCR for each page.
2. After processing a whole volume, generate a .mokuro file, which contains OCR results and metadata. All processing is done offline (before reading).
3. Load the .mokuro file together with manga images in [web reader](https://reader.mokuro.app/), which serves both as a manga reader and a catalog for processed series and volumes.

Alternatively, you can still use the old method from mokuro 0.1.*:
Instead of a .mokuro file, generate an HTML file, which you can open in a browser.
You can transfer the resulting HTML file together with manga images to another device (e.g. your mobile phone) and read there.
This method is still supported for backward compatibility, but it is recommended to use the new .mokuro format and the web reader.
For details, see [Legacy HTML vs. new .mokuro format](#legacy-html-vs-new-mokuro-format).

mokuro uses [comic-text-detector](https://github.com/dmMaze/comic-text-detector) for text detection
and [manga-ocr](https://github.com/kha-white/manga-ocr) for OCR.

---

## What's improved in this fork

| Stage | Upstream 0.2.5 | This fork |
|---|---|---|
| OCR inference | one `generate()` call **per text line** | **batched** beam search, page-ordered fixed-size batches (128 crops on CUDA/ROCm, 64 on MPS, 32 on CPU) |
| Beam search | transformers' generic `generate()` | **hand-rolled beam search** (`mokuro/beam.py`) with a static in-place KV cache and per-crop shared cross-attention; identical results, no per-step Python glue |
| Page decode / detector post-processing / crop extraction | on the main thread, serialised with the models | in **worker processes** (GPU mode), overlapped with the GPU work |
| Text-mask refinement | always, for every page (~half of GPU wall time) | **lazy**: only for pages with an over-long line that has to be split; and a ~2x faster bitwise-identical implementation |
| CPU-only mode | one process, all cores as intra-op threads | **one model process per L3 domain / CCD**, pinned; `channels_last` detector |
| OCR preprocessing | 3 identical colour planes resized on the host | **single grayscale plane**, expanded on the device (bit-identical) |
| Precision | fp32 | fp32 by default (identical output); **`--fp16`** opt-in for speed, not exact: 0.19% of characters change, boxes never (see Precision policy) |
| JPEG decode | PIL | cv2 fast path (pixel-identical), PIL for everything else |
| `--ignore_errors` | a broken image aborted the volume; an OCR error wrote empty results for a whole chunk | **per-page**: the page is skipped (no cache file written) and retried next run |
| Cached volumes | models loaded anyway | models loaded only when there is something to OCR |

The `.mokuro`, `.html` and `_ocr/` cache formats are unchanged.

### How it works

- **GPU (CUDA / ROCm / MPS)**: the main process keeps the single GPU context
  (text-detector forward, OCR beam search). `NUM_WORKERS` worker processes
  decode pages, post-process the detector output, extract the text-line
  crops and prepare the OCR inputs, so the GPU never waits for CPU work.
  OCR crops are batched in page order, so the output does not depend on the
  number of workers.
- **CPU only**: the pages are sharded over `NUM_WORKERS` model processes,
  each pinned to its own block of cores (one L3 domain / CCD each on AMD).
  This scales far better than one process with all cores as torch threads.
- **`mokuro/config.py`** auto-detects your hardware and picks the defaults;
  override any of them on the command line or in that file.

## Easy-to-edit parameters

**Everything is tuned in one file: [`mokuro/config.py`](mokuro/config.py).**
Open it and you'll find a clearly marked *"EDIT ME"* block at the top with a
comment on every knob telling you what it does and what values suit which
hardware. Edit it, save, and the new defaults apply everywhere — CLI and
library callers. No code changes needed.

> Command-line flags always override the config file: `--num_workers`,
> `--ocr_batch_size` and `--num_beams` win for that one run.

### The knobs

| Config constant | What it controls | Default |
|---|---|---|
| `NUM_WORKERS` | Worker processes: GPU mode = CPU-side pipeline workers; CPU-only mode = model shard processes. `0` = single process | auto: GPU **4**, CPU **one per L3 domain** |
| `OCR_BATCH_SIZE` | Text-line crops per batched beam-search call | auto: CUDA/ROCm **128** · MPS **64** · CPU **32** |
| `PIPELINE_MAX_INFLIGHT` | GPU mode: pages in flight (~40 MB RAM each) | `2 * workers + 2` |
| `CPU_THREADS_PER_PROCESS`, `CPU_CHUNK_PAGES`, `CPU_PIN_CORES` | CPU-only sharding details | cores of the shard's block · 1 · `True` |
| `IMAGE_DECODER` | `"auto"` (cv2 for plain JPEGs) or `"pil"` | `"auto"` |
| `NUM_BEAMS` | OCR beam width. `None` = model default (4) = upstream output. `1` (greedy) is faster but changes ~5% of the characters | `None` |
| `USE_FP16` | Half-precision OCR on GPUs (`--fp16`); 1.07x-4.9x faster depending on the GPU, not exact: changes 0.19% of the characters on 2.6% of the pages of a 140-volume set, boxes never (see Precision policy) | `False` |
| `FUSE_CONV_BN`, `ALLOW_CUDNN_TF32`, `USE_TORCH_COMPILE` | Off: measured no faster, and the first two change the detector output on CUDA | `False` |
| `LAZY_MASK_REFINE`, `DETECTOR_CPU_CHANNELS_LAST`, `OCR_PREPROCESS_SINGLE_PLANE`, `USE_CUSTOM_BEAM`, `BEAM_SYNC_LAG`, `SKIP_CROSS_ATTN_CACHE_REORDER` | The exact-parity optimisations; each can be switched off to get the reference code path back | on |

### How to choose values for your machine

- **NVIDIA / AMD GPU** — leave the defaults (4 workers, batch 128, fp32). Add `--fp16` for 1.6x (RTX 4090) to 4.9x (RX 9070 XT) more speed at a small accuracy cost (0.19% of characters; see Precision policy).
  If you hit an out-of-memory error, drop `OCR_BATCH_SIZE` to 64 or 32
  (batch 128 peaks at ~1.6 GB of VRAM on a typical volume). Each worker
  process costs ~1 GB of RAM; `--num_workers 2` is nearly as fast on a fast GPU.
- **Apple Silicon (M1–M4)** — defaults: 4 workers, batch 64, fp32 on MPS (`--fp16` opt-in).
  On a 16 GB machine lower `NUM_WORKERS` if memory pressure builds.
- **CPU only** — defaults: one model process per L3 domain (2 on a 7950X,
  4 on a 9960X), each with that domain's cores, batch 32. Each shard needs
  2-3 GB of RAM; use `--num_workers 1` on small machines.
- **Accuracy vs. speed** — the default (`NUM_BEAMS = None`) matches upstream's
  beam search (4) for identical output. `NUM_BEAMS = 1` (greedy) is 25-28%
  faster on CPU and 3-9% on GPU but changes about 5% of the characters on
  our test volume; `2` is not a mild middle ground (3.3% of characters
  change for a 12-14% CPU gain). Both are opt-in only.

### Examples

Greedy OCR, just for one run (faster, but not upstream-identical):

```bash
mokuro --num_beams 1 /path/to/manga/vol1
```

Smaller OCR batch on a small GPU, just for one run:

```bash
mokuro --ocr_batch_size 32 /path/to/manga/vol1
```

Make it permanent for every run — edit `mokuro/config.py`:

```python
OCR_BATCH_SIZE = 32  # 4 GB GPU
NUM_WORKERS = 2  # save RAM
```

## Parity

"Identical output" here means: on the same machine, the text-block boxes,
line coordinates and font sizes are identical to upstream mokuro 0.2.5, and
the OCR text is identical. This holds for the fp32 default on every device
(GPU and CPU-only output are byte-identical to upstream). The opt-in `--fp16`
mode is faster but not exact: see [Precision policy](#precision-policy-2026-09-07).
The per-machine numbers live in `CHANGES.md`; they were measured with an
external benchmark harness (not part of this repository).

Note that upstream itself is not identical across devices: on NVIDIA GPUs
cuDNN's TF32 convolutions move some detector boxes by a few pixels relative
to the CPU result. This fork turns TF32 off (`ALLOW_CUDNN_TF32 = False`), so
its CUDA output matches the CPU/ROCm output.

## Performance

Measured with a 177-page volume (1440x2048 JPEG pages, cold OCR cache,
`--num_beams` default), best of two runs after a warm-up volume, on the
machines listed below. Baseline = upstream mokuro 0.2.5 with the same
dependencies on the same machine, measured with an external benchmark
harness (not part of this repository). fp32 output is byte-identical to
upstream on every row; `--fp16` is opt-in (see Precision policy).

| Machine | Device | Upstream 0.2.5 (s/page) | This fork, fp32 default (s/page) | Speedup | This fork, `--fp16` (s/page) |
|---|---|---|---|---|---|
| RTX 4090 + Threadripper 9960X | CUDA | 0.416 | **0.039** | 10.7x | 0.024 |
| Threadripper 9960X (CPU only) | CPU | 1.241 | **0.332** | 3.7x | – |
| RX 9070 XT + Ryzen 9 7950X | ROCm | 0.973 | **0.241** | 4.0x | 0.049 |
| Ryzen 9 7950X (CPU only) | CPU | 2.164 | **0.667** | 3.2x | – |
| MacBook Pro M2 Pro 16 GB | MPS | 1.207 | **0.339** | 3.6x | 0.318 |
| M2 Pro (CPU only) | CPU | 2.351 | **0.609** | 3.9x | – |
| RX 6900 XT + Ryzen 7 5800X | ROCm | 0.718 | **0.118** | 6.1x | 0.091 |
| Ryzen 7 5800X (CPU only) | CPU | 2.625 | **1.267** | 2.1x | – |

The RTX 4090 upstream figure uses PyTorch's default cuDNN TF32 setting; with
TF32 off (the parity setting this fork uses) upstream runs at 0.393 s/page.

The individual optimisations and their measured, independently verified
contributions are listed in `CHANGES.md`.

## Installation

You need Python 3.10 or newer. Please note, that the newest Python release might not be supported due to a PyTorch dependency,
which often breaks with new Python releases and needs some time to catch up.
Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for a list of supported Python versions.

Some users have reported problems with Python installed from Microsoft Store. If you see an error:
`ImportError: DLL load failed while importing fugashi: The specified module could not be found.`,
try installing Python from the [official site](https://www.python.org/downloads).

If you want to run with GPU, install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally),
otherwise this step can be skipped.

Run in command line:

```commandline
pip3 install git+https://github.com/Gnathonic/mokuro.git
```

or from a local checkout:

```commandline
pip3 install -e .
```

## Replacing the pip-installed mokuro with this fork (no downloads)

The mokuro OCR engine is a **Python package installed via pip** (it has no
npm counterpart — see the note at the end). "Replacing mokuro" therefore
always means making `import mokuro` (and the `mokuro` CLI) resolve to this
local checkout instead of the PyPI wheel. All options below point Python at
the **local checkout** — the fork itself is never downloaded again.

**One-time clone** (skip if you already have a checkout):

```bash
git clone https://github.com/Gnathonic/mokuro.git mokuro-fork
cd mokuro-fork
```

`comic_text_detector/` is vendored in this repository (it is not a git
submodule), so no submodule step is needed.

The fork's runtime dependencies (torch, manga-ocr, transformers, …) must
already be installed in the target environment — install them normally once.
The steps below only swap *which* mokuro code gets used.

### Option A — replace inside one virtualenv (recommended)

```bash
source /path/to/your-venv/bin/activate
pip uninstall -y mokuro
pip install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

`--no-deps` stops pip from fetching anything from the network, and
`--no-build-isolation` reuses the already-installed setuptools (fully offline
install). Because the fork keeps the same distribution name (`mokuro`), pip
cleanly supersedes the previous install — no leftover copies.

Verify from inside that venv:

```bash
mokuro --version          # → 0.3.0b
python -c "import mokuro; print(mokuro.__file__)"   # → .../mokuro-fork/mokuro/__init__.py
```

### Option B — replace machine-wide (your default `python3`)

Same idea, but for the interpreter your scripts use by default:

```bash
pip3 uninstall -y mokuro
pip3 install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

Every `python3` process on the machine now imports the fork, and the global
`mokuro` command runs it too. (If pip refuses with an
`externally-managed-environment` error, use Option A in a venv instead.)
Repeat Option A inside any other virtualenv that should use it — venvs do not
inherit global installs unless created with `--system-site-packages`.

### Option C — zero-install pointer file (library imports only)

If you only use mokuro as a **library** and never call the `mokuro` CLI, a
one-line `.pth` file in the interpreter's site-packages is enough — pip is
never involved:

```bash
python3 - <<'EOF'
import site
site_pkgs = site.getsitepackages()[0]
with open(f"{site_pkgs}/mokuro-fork.pth", "w") as f:
    f.write("/path/to/mokuro-fork\n")
print("wrote", f"{site_pkgs}/mokuro-fork.pth")
EOF
```

Any `import mokuro` under that interpreter now resolves to the fork. Caveats:

- It affects **imports only** — a previously installed `mokuro` console
  command still runs the old version (use Option B to replace the CLI too).
- Remove the `.pth` file before `pip install`-ing mokuro again; the old
  distribution is not uninstalled, so `pip list` shows both.

### Notes

- **mokuro-bridge**: the bridge already auto-uses a sibling `mokuro/`
  checkout and otherwise honours `MOKURO_REPO=/path/to/mokuro-fork` — no pip
  step is needed there at all.
- **npm**: there is no npm package for the mokuro OCR engine, so nothing to
  replace on that side. The web reader (reader.mokuro.app) is a hosted app
  and never runs OCR locally.

## Usage
> Note: the CLI uses python-fire, so a boolean flag placed directly before a path consumes it (`--fp16 /path/vol` reads as `fp16="/path/vol"`). Write `--fp16=True`, or put flags after the paths; `--fp16` also tolerates the bare form by treating the value as the first path.


## Run on one volume

```bash
mokuro /path/to/manga/vol1
```

This will generate `/path/to/manga/vol1.html` file, which you can open in a browser.

If your path contains spaces, enclose it in double quotes, like this:

```bash
mokuro "/path/to/manga/volume 1"
```

## Run on multiple volumes

```bash
mokuro /path/to/manga/vol1 /path/to/manga/vol2 /path/to/manga/vol3
```

For each volume, a separate HTML file will be generated.

## Run on a directory containing multiple volumes

If your directory structure looks somewhat like this:
```
manga_title/
├─vol1/
├─vol2/
├─vol3/
└─vol4/
```

You can process all volumes by running:

```bash
mokuro --parent_dir manga_title/
```

## Other options

```
--pretrained_model_name_or_path: Name or path of the manga-ocr model.
--force_cpu: Force the use of CPU even if CUDA/MPS is available.
--disable_confirmation: Disable confirmation prompt. If False, the user will be prompted to confirm the list of volumes to be processed.
--disable_ocr: Disable OCR processing. Generate mokuro/HTML files without OCR results.
--ignore_errors: Continue processing volumes even if an error occurs.
--no_cache: Do not use cached OCR results from previous runs (_ocr directories).
--unzip: Extract volumes in zip/cbz format in their original location.
--disable_html: Disable legacy HTML output. If True, acts as if --unzip is True.
--as_one_file: Applies only to legacy HTML. If False, generate separate CSS and JS files instead of embedding them in the HTML file.
--num_workers: Worker processes (GPU: pipeline workers; CPU only: model shards). 0 = single process (default: auto-detected).
--ocr_batch_size: Text-line crops per batched OCR call (default: auto-detected).
--num_beams: Beam width for OCR decoding (default: model default 4 = identical to upstream; 1 = greedy, faster but less accurate).
--version: Print the version of mokuro and exit.
```

## Legacy HTML vs. new .mokuro format

Before version 0.2.0, mokuro generated a separate HTML file for each processed volume, which caused some usability issues:
- HTML files contained both the OCR results and the whole web reader GUI, so in order to update the GUI, all volumes needed to be updated with a new mokuro version
- images were stored separately and linked in HTML files, so any change in the directory structure could break the links
- transferring the manga to another device required transferring both the HTML files and the images
- there was no unified GUI for a whole catalog containing multiple volumes
- on some mobile devices, some workarounds were needed to open HTML files

Starting from version 0.2.0, a new .mokuro format is introduced, which is generated for each volume and contains only the OCR results and metadata necessary for the web reader GUI.
Web reader is now a separate web app, which can open manga volumes with their associated .mokuro files.

The old HTML format is still generated for backward compatibility, but it will not be developed further, and it is recommended to use the new .mokuro format and the web reader.

## Development

```bash
pip3 install -e ".[dev]"
python3 -m pytest tests/          # run the test suite (CPU; test_mokuro runs the models)
python3 -m ruff check . && python3 -m ruff format --check .   # lint, as in CI
```

`tests/test_ocr_preprocess.py`, `tests/test_beam_compat.py` and
`tests/test_textmask.py` check the bit-exactness of the rewritten
preprocessing / beam-search / mask-refinement code without running the models.

The fork runs on both transformers major versions: 5.x (the current
manga-ocr stack) and 4.x (`transformers>=4.25,<5` + `sentencepiece`, the
stack tools such as mokuro-bunko install). The stock `ViTImageProcessor`
differs between them (torchvision resize on 5.x, PIL resize on 4.x), so the
OCR text of *upstream* mokuro itself can differ between the two stacks on
near-tie lines; this fork reproduces whichever processor is installed
bit-for-bit and stays identical to upstream on the same stack (see `CHANGES.md`, "transformers 4.x
compatibility").

## Keeping in sync with upstream

This fork tracks [kha-white/mokuro](https://github.com/kha-white/mokuro).
To pull the latest upstream changes into your clone:

```bash
git remote add upstream https://github.com/kha-white/mokuro.git   # once
git fetch upstream
git merge upstream/master          # resolve conflicts, then commit
```

`comic_text_detector/` is vendored here, whereas upstream tracks it as a git
submodule; when a merge touches it, keep this repository's copy.

The fork's changes are confined to `mokuro/` (`config.py`, `manga_page_ocr.py`,
`mokuro_generator.py`, `page_ops.py`, `pipeline.py`, `cpu_shards.py`, `beam.py`,
`hf_patches.py`, `utils.py`, `run.py`, `volume.py`), two files of the vendored
text detector (`comic_text_detector/inference.py`,
`comic_text_detector/utils/textmask.py`) and the docs; see `CHANGES.md`.

## License & credits

- **GPL-3.0** — see [LICENSE](LICENSE). This fork inherits upstream mokuro's
  license unmodified; any use must comply with GPL-3.0.
- Upstream: [kha-white/mokuro](https://github.com/kha-white/mokuro) by
  [Maciej Budyś](https://github.com/kha-white).
- This repository builds on [GolyBidoof/mokuro](https://github.com/GolyBidoof/mokuro)
  (v0.3.0b), the fork this performance work started from. The optimisations
  were measured with an external benchmark harness (not part of this
  repository); see `CHANGES.md`.
- Text detection: [comic-text-detector](https://github.com/dmMaze/comic-text-detector);
  OCR: [manga-ocr](https://github.com/kha-white/manga-ocr);
  text segmentation: [Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation).


## Precision policy (2026-09-07)

fp32 OCR is the default on every device and is byte-identical to upstream mokuro 0.2.5 on all measured volumes and resolution tiers (1080x1530 to ~1790x2800). `--fp16` (config `USE_FP16 = True`) is opt-in and runs the OCR transformer in half precision. fp16 is **not exact**: measured against the fp32 output on 140 volumes (26,365 pages, 2.52 M characters), `--fp16` changes **0.19% of the characters** (4,753 of 2,520,342) on **26 pages per 1000** (695 of 26,365; 137 of the 140 volumes have at least one changed character). Text boxes are **never** affected by the precision (the detector always runs in fp32). The changed characters are concentrated on low-confidence lines (hallucinated text over non-text regions, ellipsis lengths, colophon/date strings), but real dialogue lines are affected too, so treat `--fp16` as a speed/accuracy trade-off.

### fp16 vs fp32 character error rate, 140 volumes

Reference = fp32 output of the same pipeline on the RTX 4090 (ROCm fp32 matches it to within 0.002% of characters).

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

On the 140-volume set the gain is 1.55-1.65x on the RTX 4090 and 5-6x on the RX 9070 XT. On RDNA4 (gfx1201) fp32 GEMMs are slow enough that the fp32 default is slower than the previous fork's fp16 path — use `--fp16` there if speed matters more than exact OCR text. fp32 and `--fp16` are the only two OCR precision modes.

