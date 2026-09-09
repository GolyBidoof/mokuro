"""Central configuration for mokuro's performance-related defaults.

**This is the only file you normally need to edit** to tune how mokuro uses
your machine. Every value below is a *default*: command-line flags
(``--num_workers``, ``--ocr_batch_size``, ``--num_beams``) and library
arguments always take precedence over it, but if you never pass flags, the
values chosen here (or auto-detected from your hardware) apply everywhere —
CLI and library callers alike.

Every default in this file keeps the OCR output identical to upstream mokuro
(same boxes, same text; see CHANGES.md for the exact parity statement). The
only knobs that trade accuracy for speed are ``NUM_BEAMS`` (when set to a
value other than the model's own 4) and ``FUSE_CONV_BN`` / ``ALLOW_CUDNN_TF32``
(which change the detector's output slightly on CUDA); all of them are off by
default.

How the work is split:

* **GPU (CUDA / ROCm / MPS)** — the main process keeps the single GPU context
  (text-detector forward, OCR beam search); ``NUM_WORKERS`` CPU worker
  processes decode pages, post-process the detector output and prepare the
  OCR crops in parallel.
* **CPU only** — the volume's pages are sharded over ``NUM_WORKERS`` model
  processes (each with its own copy of the models and a share of the cores),
  which is much faster than one process using all cores.
* Running out of memory? Lower ``NUM_WORKERS`` (each GPU-mode worker costs
  ~1 GB of RAM, mostly the torch runtime; each CPU-mode shard 2-3 GB) or
  ``OCR_BATCH_SIZE``.
"""

import os
import platform
import subprocess

import torch

# ===========================================================================
# EDIT ME — per-machine tuning knobs
# ===========================================================================
# Set a knob to a concrete value to force it everywhere; leave it ``None`` to
# keep the automatic, hardware-aware default (functions at the bottom of this
# file). CLI flags still override whichever choice you make here.

# -- concurrency ------------------------------------------------------------
# Number of worker *processes*. Meaning depends on the compute device:
#   GPU (CUDA/ROCm/MPS): CPU-side pipeline workers (page decode, detector
#       post-processing, mask refinement, crop extraction, OCR preprocessing);
#       the models stay in the main process.  None -> 4 (capped at cores - 1).
#   CPU only: model shard processes, each running the whole per-page pipeline
#       on its own block of cores.  None -> one per L3 cache domain (CCD on
#       AMD; e.g. 2 on a 7950X, 4 on a 9960X), or physical_cores // 4 when the
#       topology is unknown; 1 when there are fewer than 4 physical cores.
#   0 (or 1 on CPU) -> no extra processes, everything in one process.
# The CLI flag --num_workers overrides this.
NUM_WORKERS = None

# GPU mode: maximum pages in flight between "submitted for decode" and "OCR
# crops received" (bounds RAM: ~40 MB per page). None -> 2 * workers + 2.
PIPELINE_MAX_INFLIGHT = None

# CPU-only mode: torch/OpenCV threads per shard process.
# None -> the physical cores of the shard's core block (= physical cores / shards).
CPU_THREADS_PER_PROCESS = None

# CPU-only mode: pages per work item handed to a shard. 1 gives the best load
# balance (measured 1 > 2 on both test CPUs; CPU OCR cost is batch-insensitive).
CPU_CHUNK_PAGES = 1

# CPU-only mode: pin each shard to its own block of physical cores (+ SMT
# siblings), aligned to L3 domains when the shard count is a multiple of the
# domain count. Linux only; ignored elsewhere.
CPU_PIN_CORES = True

# Text-line crops sent to the OCR model per batched beam-search call. Bigger
# batches use the GPU better but need more memory (bs128 peaks at ~1.6 GB of
# VRAM on the test volume; lower it on small GPUs).
#   CUDA/ROCm: 128 · Apple Silicon (MPS): 64 · CPU: 32       (None = auto)
OCR_BATCH_SIZE = None

# Page image decoder: "auto" decodes plain RGB/grayscale JPEGs with
# cv2.imread (pixel-identical to the PIL path, ~2x faster) and everything else
# with PIL; "pil" forces the PIL path for every file.
IMAGE_DECODER = "auto"

# -- OCR decoding quality vs. speed -----------------------------------------
# Beam width for the OCR transformer:
#   None -> use the model's own generation config (num_beams=4 — identical
#           output to upstream mokuro / manga-ocr, best accuracy)
#   1    -> greedy decoding: measured 25-28% faster on CPU, 3-9% on GPU, but
#           ~5% of characters change (4.9% page CER) on the test volume
#   2    -> ~3.3% CER for 12-14% on CPU (not a mild middle ground)
# Anything other than the model default also disables USE_CUSTOM_BEAM's fast
# path (transformers' generate() is used instead).
NUM_BEAMS = None

# -- GPU feature toggles ----------------------------------------------------
# OCR transformer precision on CUDA/ROCm/MPS (ignored on CPU). Default fp32:
# byte-identical to upstream on every tested volume/tier. fp16 (``--fp16`` on
# the CLI, or ``USE_FP16 = True`` here) is 1.6x faster on an RTX 4090 and 4.9x
# on an RX 9070 XT in this pipeline, but is NOT exact: measured on 140 volumes
# (2.52 M characters) it changes 0.19% of the characters on 26 pages per 1000
# (mostly hallucination-prone lines; occasionally real text, e.g. a dropped
# bracket). Boxes are never affected. Details: README, "Precision policy".
USE_FP16 = False

# Fold batch-norm layers into the preceding conv layers of the text detector
# at load time. Measured: no speed gain on any tested GPU/CPU and the detector
# output changes on CUDA (boxes move by up to 10 px on ~2% of blocks), so it
# is OFF; opt in if you want to experiment.
FUSE_CONV_BN = False

# torch.compile the text detector + OCR encoder (CUDA/ROCm, inductor default
# mode). Measured SLOWER than eager in every mode on an RTX 4090 and an
# RX 9070 XT (torch 2.13), and the fork's original "reduce-overhead" mode
# crashed. Kept only as an off-by-default experiment knob.
USE_TORCH_COMPILE = False

# Let cuDNN use TF32 for the detector's fp32 convolutions on CUDA. TF32 makes
# the CUDA detector output drift from CPU/ROCm output (box edges by 1..10 px
# on some blocks, ~50 character edits per volume) for no measurable speed
# gain, so it is off. Only affects NVIDIA GPUs.
ALLOW_CUDNN_TF32 = False

# -- exact-parity optimisations (all on; each can be switched off to get the
#    reference code path back, e.g. when bisecting a problem) ---------------
# Compute the refined text mask lazily, at page level: only pages that contain
# a text line whose warped crop exceeds max_ratio (and therefore has to be
# split into chunks) ever read it. Identical output; ~2x fewer CPU seconds
# per page on the test volume.
LAZY_MASK_REFINE = True

# Run the text detector in channels_last (NHWC) memory format when it executes
# on the CPU (oneDNN keeps its blocked layout between conv layers). ~1.8-2x on
# the detector forward; no effect on GPU.
DETECTOR_CPU_CHANNELS_LAST = True

# Prepare OCR crops as a single grayscale plane (resized with the very same
# torchvision call transformers' ViTImageProcessor uses) and expand it to the
# model's 3 identical channels on the device. Bit-identical pixel values, a
# third of the preprocessing work and of the host->device bytes.
OCR_PREPROCESS_SINGLE_PLANE = True

# Use mokuro/beam.py (a hand-rolled beam search with an in-place static KV
# cache and per-crop shared cross-attention K/V) instead of transformers'
# generic generate(). Same beam-search semantics and the same kernels, so the
# output is identical; it removes the per-step Python glue that dominates the
# decoder on CPU. Automatically falls back to generate() when the requested
# decoding differs from the model's default (e.g. NUM_BEAMS=1).
USE_CUSTOM_BEAM = True

# Number of decoder steps the host may run ahead of the GPU before checking
# the "all sequences finished" flag (CUDA/ROCm only). 0 = block on the flag
# every step; 1 = read the previous step's flag (overlaps CPU dispatch with GPU
# execution; at most one wasted step per batch, output unchanged).
BEAM_SYNC_LAG = 1

# When transformers' generate() is used (custom beam off or not applicable):
# skip the per-step re-gather of the cross-attention KV cache, which is a
# semantic no-op in beam search (beam indices never leave an item's group and
# every beam holds the same encoder K/V). Identical output.
SKIP_CROSS_ATTN_CACHE_REORDER = True

# ===========================================================================
# Automatic hardware detection — usually nothing to edit below this line.
# ===========================================================================


def get_device(force_cpu: bool = False) -> str:
    """Return the compute device: ``"cuda"`` (also ROCm), ``"mps"`` or ``"cpu"``."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_apple_silicon() -> bool:
    """True when running on an Apple Silicon (arm64) machine."""
    return platform.machine() in ("arm64", "aarch64")


def _read_cpu_list(path):
    out = set()
    with open(path) as f:
        for part in f.read().strip().split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    return out


def get_core_topology():
    """``(l3_domains, sibling_groups)``: lists of sorted logical-CPU lists read
    from Linux sysfs; ``([], [])`` when unavailable (non-Linux)."""
    cpu_count = os.cpu_count() or 1
    base = "/sys/devices/system/cpu"
    try:
        sib, l3 = {}, {}
        for n in range(cpu_count):
            s = frozenset(_read_cpu_list(f"{base}/cpu{n}/topology/thread_siblings_list"))
            sib[s] = None
            try:
                l3[frozenset(_read_cpu_list(f"{base}/cpu{n}/cache/index3/shared_cpu_list"))] = None
            except OSError:
                pass
        sibs = sorted((sorted(s) for s in sib), key=lambda x: x[0])
        l3s = sorted((sorted(s) for s in l3), key=lambda x: x[0])
        return l3s, sibs
    except OSError:
        return [], []


def get_physical_cores() -> int:
    """Physical core count (SMT siblings collapsed)."""
    cpu_count = os.cpu_count() or 4
    _, sibs = get_core_topology()
    if sibs:
        return len(sibs)
    if platform.system() == "Darwin":
        try:
            return max(1, int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return cpu_count
    # Unknown topology: assume SMT.
    return max(1, cpu_count // 2)


def get_default_cpu_processes() -> int:
    """CPU-only mode: number of model shard processes (see NUM_WORKERS)."""
    cores = get_physical_cores()
    if cores < 4:
        return 1
    l3s, _ = get_core_topology()
    if len(l3s) >= 2:
        return len(l3s)
    # Single L3 domain (e.g. Ryzen 7 5800X, Apple M-series): measured best at
    # ~2 physical cores per shard (5800X: 4 shards 1.19x vs 2 shards 1.05x),
    # bounded by memory (each shard holds its own models, ~2.5 GB).
    n = max(2, cores // 2)
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
            total_gb = kb / 2**20
        else:
            total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 2**30
        n = max(1, min(n, int((total_gb - 4) // 2.5)))
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        pass  # total RAM unknown: keep the core-based shard count
    return n


def get_default_num_workers(force_cpu: bool = False) -> int:
    """Number of worker processes (see ``NUM_WORKERS`` for the two meanings)."""
    if NUM_WORKERS is not None:
        return max(0, int(NUM_WORKERS))
    if get_device(force_cpu) != "cpu":
        cpu_count = os.cpu_count() or 4
        return max(1, min(4, cpu_count - 1))
    return get_default_cpu_processes()


def get_default_ocr_batch_size(force_cpu: bool = False) -> int:
    """
    Number of text-line crops fed to the OCR model per beam-search call.

    Dedicated GPUs (CUDA/ROCm) get cheaper per crop up to ~128 crops per call;
    unified memory (Apple Silicon) tolerates 64; CPUs are nearly
    batch-insensitive (32 measured within a few % of 16 either way). Override
    by setting ``OCR_BATCH_SIZE`` above.
    """
    if OCR_BATCH_SIZE is not None:
        return OCR_BATCH_SIZE

    device = get_device(force_cpu)

    if device == "mps":
        return 64

    if device == "cuda":
        return 128

    return 32


# ---------------------------------------------------------------------------
# Propagating runtime overrides to worker processes
# ---------------------------------------------------------------------------
# Worker processes are *spawned* (fresh interpreters), so they re-import this
# module and see the file's defaults. Library callers that change knobs at
# runtime (``mokuro.config.X = ...``) would otherwise not be honoured in the
# workers; the parent therefore snapshots the uppercase knobs and the workers
# re-apply them, also into the modules that imported the names directly.


def snapshot():
    """All uppercase knobs of this module as a plain dict (pickleable)."""
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}


def apply_snapshot(snap):
    import sys

    g = globals()
    for k, v in snap.items():
        g[k] = v
    for name, mod in list(sys.modules.items()):
        if mod is None or name == __name__:
            continue
        if name.startswith(("mokuro", "comic_text_detector")):
            for k, v in snap.items():
                if hasattr(mod, k):
                    setattr(mod, k, v)
