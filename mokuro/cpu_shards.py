"""CPU-only multi-process page sharding.

On CPU a single process running the detector + OCR transformer with all cores
as intra-op threads scales poorly (small GEMMs, op dispatch). This module runs
P worker processes, each owning its own copy of the models and a block of the
cores (one L3 domain / CCD per shard where possible), pulling fixed-size page
chunks from a shared queue. Inside a shard the pages go through exactly the
same driver as the single-process path (``mokuro.mokuro_generator.process_pages``
with an ``InlinePool``), and every page's output depends only on that page
(batched fp32 beam search is exact on CPU), so the result is deterministic and
identical to the single-process path.

The pool is created lazily by ``MokuroGenerator.init_models()`` and reused
across volumes; workers are daemon processes and exit with the parent.
"""

import atexit
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path

from loguru import logger

from mokuro import config as _config


def core_sets(n_procs):
    """One logical-CPU set per shard (each includes its SMT siblings).
    If n_procs is a multiple of the L3-domain count, every set stays inside
    one domain (a CCD on AMD), which measured best; otherwise physical cores
    are split into contiguous blocks. Returns [] if the topology is unknown."""
    l3s, sibs = _config.get_core_topology()
    if not sibs:
        return []
    if l3s and n_procs % len(l3s) == 0:
        per = n_procs // len(l3s)
        groups = []
        for dom in l3s:
            dom_set = set(dom)
            cores = [s for s in sibs if s[0] in dom_set]  # physical cores of this domain
            for k in range(per):
                lo, hi = k * len(cores) // per, (k + 1) * len(cores) // per
                groups.append(sorted(c for core in cores[lo:hi] for c in core))
    else:
        groups = []
        for k in range(n_procs):
            lo, hi = k * len(sibs) // n_procs, (k + 1) * len(sibs) // n_procs
            groups.append(sorted(c for core in sibs[lo:hi] for c in core))
    if any(len(g) == 0 for g in groups):
        return []
    return groups


def physical_cores_in(cpu_set):
    _, sibs = _config.get_core_topology()
    s = set(cpu_set)
    return max(1, sum(1 for core in sibs if core[0] in s))


def _worker_main(rank, n_threads, cores, cfg_snapshot, model_args, worker_opts, task_q, result_q):
    import torch

    _config.apply_snapshot(cfg_snapshot)
    if rank > 0:
        # Only rank 0 echoes the model-loading INFO lines; keep warnings/errors.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if cores:
        try:
            os.sched_setaffinity(0, cores)
        except (AttributeError, OSError):
            pass
    torch.set_num_threads(n_threads)
    try:
        import cv2

        cv2.setNumThreads(n_threads)
    except ImportError:
        logger.debug("cv2 not importable in the shard process; its thread count is left unchanged")

    from mokuro.manga_page_ocr import MangaPageOcr
    from mokuro.mokuro_generator import process_pages
    from mokuro.pipeline import InlinePool

    try:
        mpocr = MangaPageOcr(**model_args)
        pool = InlinePool(mpocr.worker_cfg())
        gen_args = mpocr.generation_args(num_beams=worker_opts.get("num_beams"))
    except Exception as e:  # noqa: BLE001 - any model-load failure is reported to the parent, which raises
        result_q.put(("error", -1, f"{type(e).__name__}: {e}"))
        return
    result_q.put(("ready", rank, None))

    while True:
        task = task_q.get()
        if task is None:
            return
        job_id, path_in, path_ocr_cache, pages, ignore_errors = task
        try:
            n = process_pages(
                mpocr,
                pool,
                Path(path_in),
                Path(path_ocr_cache),
                pages,
                worker_opts["ocr_batch_size"],
                gen_args,
                ignore_errors=ignore_errors,
            )
            result_q.put(("done", job_id, n))
        except Exception as e:  # noqa: BLE001 - reported to the parent; the shard stays alive for the next job
            result_q.put(("error", job_id, f"{type(e).__name__}: {e}"))


def _ranges(cpus):
    out, start, prev = [], None, None
    for c in cpus:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


class ShardPool:
    def __init__(self, n_procs, n_threads, pin, model_args, worker_opts):
        """``n_threads=None`` -> physical cores of each shard's pinned set
        (or physical_cores // n_procs when not pinning)."""
        self.n_procs = n_procs
        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.procs = []
        sets = core_sets(n_procs) if pin else []
        if len(sets) != n_procs:
            sets = [None] * n_procs
        if n_threads:
            threads = [int(n_threads)] * n_procs
        else:
            fallback = max(1, _config.get_physical_cores() // n_procs)
            threads = [physical_cores_in(s) if s else fallback for s in sets]
        self.n_threads = threads[0]
        # Inherited by the spawned interpreters before they import torch.
        os.environ["OMP_NUM_THREADS"] = str(threads[0])
        os.environ["MKL_NUM_THREADS"] = str(threads[0])
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        cfg = _config.snapshot()
        for r in range(n_procs):
            p = self.ctx.Process(
                target=_worker_main,
                args=(r, threads[r], sets[r], cfg, model_args, worker_opts, self.task_q, self.result_q),
                daemon=True,
                name=f"mokuro-shard-{r}",
            )
            p.start()
            self.procs.append(p)
        self._closed = False
        atexit.register(self.close)
        logger.info(
            f"CPU sharding: {n_procs} worker processes x {threads} threads"
            + (f", pinned to logical cpus {[_ranges(s) for s in sets]}" if sets[0] else "")
        )
        ready = 0
        while ready < n_procs:
            kind, _rank, payload = self._get()
            if kind == "ready":
                ready += 1
            elif kind == "error":
                self.close()
                raise RuntimeError(f"shard worker failed to initialise: {payload}")

    def _get(self, timeout=5.0):
        while True:
            try:
                return self.result_q.get(timeout=timeout)
            except queue.Empty:
                dead = [p.name for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"shard worker(s) died: {dead}")

    def run(self, path_in, path_ocr_cache, pages, chunk_pages, ignore_errors=False, on_done=None):
        """Process ``pages`` (list of (key, img_path_rel)) in chunks of
        ``chunk_pages`` across the shards. Raises on the first worker error
        unless ``ignore_errors`` (errors are logged then). Returns the number
        of pages written."""
        jobs = [pages[i : i + chunk_pages] for i in range(0, len(pages), chunk_pages)]
        for j, job in enumerate(jobs):
            self.task_q.put((j, str(path_in), str(path_ocr_cache), job, ignore_errors))
        pending = len(jobs)
        n_done = 0
        first_error = None
        while pending:
            kind, job_id, payload = self._get()
            pending -= 1
            if kind == "done":
                n_done += payload
            else:
                msg = f"Error in shard job {job_id} ({[str(p[1]) for p in jobs[job_id]]}): {payload}"
                if ignore_errors:
                    logger.error(msg)
                elif first_error is None:
                    first_error = msg
            if on_done:
                on_done(len(jobs[job_id]))
        if first_error is not None:
            raise RuntimeError(first_error)
        return n_done

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _ in self.procs:
            try:
                self.task_q.put(None)
            except Exception as e:  # noqa: BLE001 - best-effort shutdown
                logger.debug(f"shard pool shutdown: {e}")
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.procs = []
        for q in (self.task_q, self.result_q):
            try:
                q.close()
                q.join_thread()
            except Exception as e:  # noqa: BLE001 - best-effort shutdown
                logger.debug(f"shard pool shutdown: {e}")
