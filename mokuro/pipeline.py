"""Worker-process pool for the page pipeline (GPU mode).

Workers run the pure-CPU per-page stages from :mod:`mokuro.page_ops`; the main
process keeps the single GPU context (detector forward + NMS, OCR beam search).
Page images / detector maps / OCR pixel tensors cross the process boundary as
``torch`` CPU tensors through ``torch.multiprocessing`` queues, i.e. via shared
memory (one memcpy into shm, zero-copy on receive).

Task tuples (main -> worker):
    ("load", idx, path)
    ("post", idx, img_t, det_np, mask_u8_t, lines_t, dw, dh)
Result tuples (worker -> main):
    ("ready", pid)                       once per worker at start-up
    ("load", idx, img_t, img_u8_t, dw, dh)
    ("post", idx, result_dict, crop_metadata, pixel_values_t | None)
    ("err",  idx, stage, message)

``InlinePool`` offers the same interface but executes tasks synchronously in
the caller's process (``num_workers=0`` and the CPU shard processes); the
driver loop in :func:`mokuro.mokuro_generator.process_pages` is identical for
both, so the output is the same by construction.

``cfg`` (shared by both pools): ``input_size`` (detector), ``pp_kwargs``
(kwargs of ``page_ops.postprocess_page``), ``single_plane`` and ``processor``
(OCR preprocessing; the stock ``ViTImageProcessor`` is only shipped when the
single-plane path is off), ``config`` (``mokuro.config.snapshot()`` so that
runtime overrides reach the spawned interpreters).
"""

import atexit
import os
import queue
import traceback
from collections import deque

import torch
from loguru import logger


# --- transport of tensors between processes ---------------------------------
# torch.multiprocessing moves CPU tensors through /dev/shm. Containers often
# ship a 64 MB /dev/shm (Docker default), which makes every page fail with
# "unable to allocate shared memory ... No space left on device". When shm is
# small (or MOKURO_FORCE_PIPE_TRANSFER=1) tensors are sent as numpy arrays
# through the queue's pipe instead: slower, but identical output.
def shm_is_usable(min_bytes=1 << 30):
    if os.environ.get("MOKURO_FORCE_PIPE_TRANSFER"):
        return False
    try:
        import shutil

        if not os.path.isdir("/dev/shm"):
            return True  # macOS & co. do not use /dev/shm for torch sharing
        return shutil.disk_usage("/dev/shm").total >= min_bytes
    except OSError:
        return True


def _to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return ("__np__", obj.detach().cpu().numpy())
    if isinstance(obj, tuple):
        return tuple(_to_numpy(x) for x in obj)
    if isinstance(obj, list):
        return [_to_numpy(x) for x in obj]
    return obj


def _from_numpy(obj):
    if isinstance(obj, tuple):
        if len(obj) == 2 and obj[0] == "__np__":
            return torch.from_numpy(obj[1])
        return tuple(_from_numpy(x) for x in obj)
    if isinstance(obj, list):
        return [_from_numpy(x) for x in obj]
    return obj


def _run_task(task, cfg, ops):
    """Execute one task tuple; returns the result tuple (never raises)."""
    load_page, postprocess_page, ocr_preprocess, torch = ops
    kind, idx = task[0], task[1]
    try:
        if kind == "load":
            img, img_u8, dw, dh = load_page(task[2], cfg["input_size"])
            return ("load", idx, torch.from_numpy(img), torch.from_numpy(img_u8), dw, dh)
        if kind == "post":
            _, _, img_t, det, mask_t, lines_t, dw, dh = task
            result, crops, meta = postprocess_page(
                img_t.numpy(), det, mask_t, lines_t, dw, dh, cfg["input_size"], **cfg["pp_kwargs"]
            )
            pv = ocr_preprocess(crops, cfg.get("processor"), cfg.get("single_plane", True))
            return ("post", idx, result, meta, pv)
        return ("err", idx, kind, f"unknown task {kind}")
    except Exception as e:  # noqa: BLE001 - returned to the driver as an "err" result; the worker keeps running
        return ("err", idx, kind, f"{e!r}\n{traceback.format_exc()}")


def _worker_main(tasks, results, cfg):
    # One BLAS/OpenCV thread per worker: the parallelism comes from the pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    from mokuro import config as _config

    if cfg.get("config"):
        _config.apply_snapshot(cfg["config"])
    from mokuro.page_ops import load_page, ocr_preprocess, postprocess_page

    ops = (load_page, postprocess_page, ocr_preprocess, torch)
    results.put(("ready", os.getpid()))

    while True:
        try:
            task = tasks.get()
        except KeyboardInterrupt:
            break
        if task is None:
            break
        if not cfg.get("shm_ok", True):
            task = _from_numpy(task)
        res = _run_task(task, cfg, ops)
        results.put(res if cfg.get("shm_ok", True) else _to_numpy(res))


class WorkerPool:
    """Spawned worker processes (safe with an initialised CUDA/HIP context in the parent)."""

    def __init__(self, num_workers, cfg):
        import torch.multiprocessing as mp

        self.shm_ok = shm_is_usable()
        cfg = dict(cfg, shm_ok=self.shm_ok)
        if not self.shm_ok:
            logger.warning(
                "/dev/shm is small (Docker default?) - passing page tensors through pipes instead; add --shm-size=8g to the container for full speed"
            )
        ctx = mp.get_context("spawn")
        self.num_workers = num_workers
        self.tasks = ctx.Queue()
        self.results = ctx.Queue()
        self.procs = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_main, args=(self.tasks, self.results, cfg), name=f"mokuro-worker-{i}", daemon=True
            )
            p.start()
            self.procs.append(p)
        self._closed = False
        atexit.register(self.close)
        # Wait until every worker has imported its modules, so the first page
        # is not delayed by spawn/import time (charged to init instead).
        ready = 0
        while ready < len(self.procs):
            try:
                kind, _pid = self.results.get(timeout=2)
            except queue.Empty:
                dead = [p.name for p in self.procs if not p.is_alive()]
                if dead:
                    self.close()
                    raise RuntimeError(f"pipeline worker(s) died during start-up: {dead}")
                continue
            assert kind == "ready", kind
            ready += 1

    def submit(self, task):
        self.tasks.put(task if self.shm_ok else _to_numpy(task))

    def get(self, timeout=None):
        """Next result; raises queue.Empty on timeout, RuntimeError if a worker died."""
        try:
            res = self.results.get(timeout=timeout)
            return res if self.shm_ok else _from_numpy(res)
        except queue.Empty:
            dead = [p.name for p in self.procs if not p.is_alive()]
            if dead:
                raise RuntimeError(f"pipeline worker(s) died: {dead}")
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for _ in self.procs:
                self.tasks.put(None)
            for p in self.procs:
                p.join(timeout=5)
            for p in self.procs:
                if p.is_alive():
                    p.terminate()
        except Exception as e:  # noqa: BLE001 - best-effort shutdown
            logger.debug(f"pipeline pool shutdown: {e}")
        for q in (self.tasks, self.results):
            try:
                q.close()
            except Exception as e:  # noqa: BLE001 - best-effort shutdown
                logger.debug(f"pipeline pool shutdown: {e}")


class InlinePool:
    """Same interface as WorkerPool, but runs the tasks synchronously in-process."""

    num_workers = 0

    def __init__(self, cfg):
        import torch

        from mokuro.page_ops import load_page, ocr_preprocess, postprocess_page

        self._ops = (load_page, postprocess_page, ocr_preprocess, torch)
        self.cfg = cfg
        self._out = deque()

    def submit(self, task):
        self._out.append(_run_task(task, self.cfg, self._ops))

    def get(self, timeout=None):
        if not self._out:
            raise queue.Empty
        return self._out.popleft()

    def close(self):
        pass
