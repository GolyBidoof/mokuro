import queue
from json import JSONDecodeError

import torch
from loguru import logger
from tqdm import tqdm

from mokuro import __version__
from mokuro.config import (
    CPU_CHUNK_PAGES,
    CPU_PIN_CORES,
    CPU_THREADS_PER_PROCESS,
    PIPELINE_MAX_INFLIGHT,
    get_default_num_workers,
    get_default_ocr_batch_size,
    get_device,
)
from mokuro.manga_page_ocr import MangaPageOcr
from mokuro.pipeline import InlinePool, WorkerPool
from mokuro.utils import dump_json, load_json
from mokuro.volume import Volume


def write_page_json(path_ocr_cache, img_path_rel, result):
    """Write one page's OCR result to ``_ocr/<volume>/<page>.json`` (upstream layout)."""
    json_path = (path_ocr_cache / img_path_rel).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(result, json_path)


def process_pages(
    mpocr,
    pool,
    path_in,
    path_ocr_cache,
    pages,
    ocr_batch_size,
    gen_args,
    ignore_errors=False,
    max_inflight=None,
    on_page_done=None,
):
    """Detect + OCR ``pages`` (list of ``(key, img_path_rel)``) and write one JSON per page.

    Pipelined per-page processing, identical for the GPU driver (``WorkerPool``)
    and for a single process (``InlinePool``):

    pool: decode + letterbox -> [mpocr: detector forward + NMS] -> pool: DB
    polygons / group_output / lazy refine / crops / OCR preprocess -> mpocr:
    fixed-size OCR batches formed strictly in page order (batch k = crops
    ``[k*bs, (k+1)*bs)`` of the page-ordered crop sequence, so the batch
    composition - and hence the output - is independent of scheduling and of
    the worker count; only the last batch is partial) -> per-page JSON as soon
    as all of a page's crops have text.

    Failure semantics: every failure is per page. With ``ignore_errors`` a page
    whose decode / detection / post-processing fails is logged and skipped
    (no JSON is written, so it is retried on the next run); a failing OCR batch
    is retried page by page and only the failing pages are skipped. Without
    ``ignore_errors`` the first failure raises.

    Returns the number of pages written; ``on_page_done(1)`` is called for
    every finished (written or skipped) page.
    """
    bs = int(ocr_batch_size)
    n = len(pages)
    if n == 0:
        return 0
    if max_inflight is None:
        max_inflight = 2 * max(getattr(pool, "num_workers", 0), 1) + 2

    next_submit = 0  # next page index to hand to the pool for decoding
    inflight = 0  # pages between "load submitted" and "post received / failed"
    state = {}  # idx -> {"result", "meta", "pv", "remaining"} once post-processed
    failed = set()
    next_enqueue = 0  # next page index whose crops go to the OCR queue (page order)
    crop_q = []  # [(idx, local_idx)] in page order, not yet OCR'd
    pv_q = []  # matching pixel-value tensors (one per crop)
    finished = 0  # pages written or failed
    written = 0

    def page_done():
        nonlocal finished
        finished += 1
        if on_page_done is not None:
            on_page_done(1)

    def fail_page(idx, stage, msg):
        _key, img_path_rel = pages[idx]
        if not ignore_errors:
            raise RuntimeError(f"Error in {stage} for {img_path_rel}: {msg}")
        logger.error(f"Error in {stage} for {img_path_rel}: {msg}")
        failed.add(idx)
        state.pop(idx, None)
        page_done()

    def maybe_write(idx):
        nonlocal written
        st = state.get(idx)
        if st is not None and st["remaining"] == 0:
            write_page_json(path_ocr_cache, pages[idx][1], st["result"])
            del state[idx]
            written += 1
            page_done()

    def run_batch(k):
        """OCR the first k queued crops; on failure fall back per page."""
        nonlocal crop_q, pv_q
        items, pvs = crop_q[:k], pv_q[:k]
        crop_q, pv_q = crop_q[k:], pv_q[k:]
        try:
            texts = mpocr.generate_texts(torch.cat(pvs), gen_args)
        except Exception as e:
            if not ignore_errors:
                raise
            logger.error(f"Error in OCR batch: {e}; retrying page by page")
            texts = [None] * k
            start = 0
            while start < k:
                idx = items[start][0]
                end = start
                while end < k and items[end][0] == idx:
                    end += 1
                try:
                    texts[start:end] = mpocr.generate_texts(torch.cat(pvs[start:end]), gen_args)
                except Exception as e2:  # noqa: BLE001 - per-page failure, recorded by fail_page
                    if idx not in failed:
                        fail_page(idx, "OCR", str(e2))
                start = end
        for (idx, local_idx), text in zip(items, texts):
            st = state.get(idx)
            if st is None:
                continue
            if text is not None:
                blk_idx, line_idx = st["meta"][local_idx]
                st["result"]["blocks"][blk_idx]["lines"][line_idx] += text
            st["remaining"] -= 1
            maybe_write(idx)

    while finished < n:
        # 1. keep the pool fed (bounded in-flight pages)
        while next_submit < n and inflight < max_inflight:
            pool.submit(("load", next_submit, str(path_in / pages[next_submit][1])))
            next_submit += 1
            inflight += 1

        # 2. advance the page-ordered OCR queue
        while next_enqueue < n and (next_enqueue in state or next_enqueue in failed):
            if next_enqueue in state:
                st = state[next_enqueue]
                if st["pv"] is not None:
                    crop_q.extend((next_enqueue, j) for j in range(len(st["meta"])))
                    pv_q.extend(st["pv"][j : j + 1] for j in range(len(st["meta"])))
                    st["pv"] = None
                else:
                    maybe_write(next_enqueue)
            next_enqueue += 1

        # 3. a full batch (or the final partial one) is ready -> OCR it
        all_enqueued = next_enqueue >= n
        batch_ready = len(crop_q) >= bs or (all_enqueued and crop_q)
        if batch_ready:
            run_batch(min(bs, len(crop_q)))

        # 4. collect pool results (block only if there is nothing else to do)
        block = (not batch_ready) and inflight > 0
        try:
            res = pool.get(timeout=5.0 if block else 0)
        except queue.Empty:
            if not block and inflight == 0 and not crop_q and finished < n:
                raise RuntimeError("pipeline stalled: no work in flight but pages unfinished")
            continue
        while True:
            kind, idx = res[0], res[1]
            if kind == "load":
                _, _, img_t, img_u8, dw, dh = res
                try:
                    det, mask_u8, lines_map = mpocr.detect_forward(img_u8)
                    pool.submit(("post", idx, img_t, det, mask_u8, lines_map, dw, dh))
                except Exception as e:  # noqa: BLE001 - per-page failure, recorded by fail_page
                    inflight -= 1
                    fail_page(idx, "detection", str(e))
            elif kind == "post":
                _, _, result, meta, pv = res
                inflight -= 1
                state[idx] = {"result": result, "meta": meta, "pv": pv, "remaining": len(meta)}
            else:  # "err"
                inflight -= 1
                fail_page(idx, res[2], res[3])
            try:
                res = pool.get(timeout=0)
            except queue.Empty:
                break
    return written


class MokuroGenerator:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        disable_ocr=False,
        num_workers=None,
        ocr_batch_size=None,
        max_inflight=None,
        **kwargs,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.force_cpu = force_cpu
        self.disable_ocr = disable_ocr
        self.device = get_device(force_cpu)

        # num_workers = number of worker processes: CPU-side pipeline workers
        # on a GPU, model shard processes on CPU (0/1 = single process).
        # None -> auto (see mokuro/config.py).
        self.num_workers = num_workers if num_workers is not None else get_default_num_workers(force_cpu)
        self.ocr_batch_size = ocr_batch_size if ocr_batch_size is not None else get_default_ocr_batch_size(force_cpu)
        self.max_inflight = max_inflight if max_inflight is not None else PIPELINE_MAX_INFLIGHT

        # CPU-only sharding knobs (see mokuro/config.py); library callers may
        # override them per instance via kwargs.
        cpu_threads = kwargs.pop("cpu_threads", None)
        self.cpu_threads = int(cpu_threads) if cpu_threads is not None else CPU_THREADS_PER_PROCESS
        self.cpu_chunk_pages = int(kwargs.pop("cpu_chunk_pages", None) or CPU_CHUNK_PAGES)
        cpu_pin = kwargs.pop("cpu_pin", None)
        self.cpu_pin = bool(CPU_PIN_CORES if cpu_pin is None else cpu_pin)

        self.kwargs = kwargs  # num_beams + MangaPageOcr kwargs
        self.mpocr = None
        self.pool = None
        self.shard_pool = None

    def _use_shards(self):
        return self.device == "cpu" and not self.disable_ocr and self.num_workers > 1

    def _mpocr_kwargs(self):
        # num_beams is consumed by the beam search, not by MangaPageOcr
        return {k: v for k, v in self.kwargs.items() if k != "num_beams"}

    def init_models(self):
        """Load the models (and start the worker processes). Called lazily by
        ``process_volume`` for the first volume that has uncached pages."""
        if self.disable_ocr:
            return
        if self._use_shards():
            if self.shard_pool is None:
                from mokuro.cpu_shards import ShardPool

                self.shard_pool = ShardPool(
                    self.num_workers,
                    self.cpu_threads,
                    self.cpu_pin,
                    {
                        "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                        "force_cpu": True,
                        "disable_ocr": False,
                        **self._mpocr_kwargs(),
                    },
                    {"ocr_batch_size": self.ocr_batch_size, "num_beams": self.kwargs.get("num_beams")},
                )
            return
        if self.mpocr is None:
            self.mpocr = MangaPageOcr(
                self.pretrained_model_name_or_path,
                force_cpu=self.force_cpu,
                disable_ocr=False,
                **self._mpocr_kwargs(),
            )
        if self.pool is None:
            cfg = self.mpocr.worker_cfg()
            if self.num_workers > 0:
                logger.info(f"Starting {self.num_workers} pipeline worker process(es)")
                self.pool = WorkerPool(self.num_workers, cfg)
                self.mpocr._share = True
            else:
                self.pool = InlinePool(cfg)
                self.mpocr._share = False

    def close(self):
        """Shut down worker processes (also done at interpreter exit)."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        if self.shard_pool is not None:
            self.shard_pool.close()
            self.shard_pool = None

    def __del__(self):
        try:  # __del__ must never raise; at interpreter shutdown module globals may already be None
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass

    def process_volume(self, volume: Volume, ignore_errors=False, no_cache=False):
        volume.path_ocr_cache.mkdir(parents=True, exist_ok=True)

        if volume.mokuro_data is not None:
            for page in volume.mokuro_data["pages"]:
                json_path = volume.get_ocr_path(page["img_path"])
                if json_path.is_file():
                    continue
                json_path.parent.mkdir(parents=True, exist_ok=True)
                page = page.copy()
                page.pop("img_path")
                dump_json(page, json_path)

        img_paths = volume.get_img_paths()
        img_paths_list = list(img_paths.items())

        # Pages whose OCR cache entry is already valid are skipped up front.
        to_process = []
        n_cached = 0
        for key, img_path_rel in img_paths_list:
            json_path = volume.get_ocr_path(img_path_rel)
            if not no_cache and json_path.is_file():
                try:
                    load_json(json_path)
                    n_cached += 1
                    continue
                except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Error loading cached OCR for {img_path_rel}: {e}. Re-processing.")
            to_process.append((key, img_path_rel))

        with tqdm(total=len(img_paths_list), desc="Processing pages...") as pbar:
            pbar.update(n_cached)
            if to_process and self.disable_ocr:
                for key, img_path_rel in to_process:
                    H, W = _image_size(volume.path_in / img_path_rel)
                    result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}
                    write_page_json(volume.path_ocr_cache, img_path_rel, result)
                    pbar.update(1)
            elif to_process:
                # Models (and worker processes) are only loaded when there is
                # something to do: a fully cached volume costs no model init.
                self.init_models()
                try:
                    if self.shard_pool is not None:
                        self.shard_pool.run(
                            volume.path_in,
                            volume.path_ocr_cache,
                            to_process,
                            self.cpu_chunk_pages,
                            ignore_errors,
                            on_done=pbar.update,
                        )
                    else:
                        process_pages(
                            self.mpocr,
                            self.pool,
                            volume.path_in,
                            volume.path_ocr_cache,
                            to_process,
                            self.ocr_batch_size,
                            self.mpocr.generation_args(num_beams=self.kwargs.get("num_beams")),
                            ignore_errors=ignore_errors,
                            max_inflight=self.max_inflight,
                            on_page_done=pbar.update,
                        )
                except BaseException:
                    # In-flight worker results would leak into the next
                    # volume: drop the pools, init_models() rebuilds them.
                    self.close()
                    raise

        self.generate_mokuro_file(volume, ignore_errors=ignore_errors)

    @staticmethod
    def generate_mokuro_file(volume: Volume, ignore_errors=False):
        json_paths = volume.get_json_paths()
        img_paths = volume.get_img_paths()

        out = {
            "version": __version__,
            "title": volume.title.name,
            "title_uuid": volume.title.uuid,
            "volume": volume.name,
            "volume_uuid": volume.uuid,
            "pages": [],
        }

        for key, json_path_rel in json_paths.items():
            try:
                if key not in img_paths:
                    logger.warning(f"No matching image found for cached OCR: {key}")
                    continue
                img_path_rel = img_paths[key]
                page_json = load_json(volume.path_ocr_cache / json_path_rel)
                page_json["img_path"] = str(img_path_rel).replace("\\", "/")
                out["pages"].append(page_json)
            except Exception as e:
                if ignore_errors:
                    logger.error(e)
                else:
                    raise

        dump_json(out, volume.path_mokuro)


def _image_size(path):
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return h, w
