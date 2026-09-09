import os

import torch
from loguru import logger
from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process as ocr_post_process

from comic_text_detector.inference import TextDetector, letterbox_input, normalize_input
from mokuro import __version__
from mokuro import config as _config
from mokuro.beam import BeamSearchOCR
from mokuro.cache import cache
from mokuro.config import (
    ALLOW_CUDNN_TF32,
    BEAM_SYNC_LAG,
    DETECTOR_CPU_CHANNELS_LAST,
    FUSE_CONV_BN,
    LAZY_MASK_REFINE,
    NUM_BEAMS,
    OCR_PREPROCESS_SINGLE_PLANE,
    SKIP_CROSS_ATTN_CACHE_REORDER,
    USE_CUSTOM_BEAM,
    USE_TORCH_COMPILE,
    get_default_ocr_batch_size,
    get_device,
)
from mokuro.hf_patches import install_skip_cross_attn_cache_reorder
from mokuro.page_ops import extract_crops, ocr_preprocess, postprocess_page
from mokuro.page_ops import split_into_chunks as _split_into_chunks
from mokuro.utils import imread

_log_once_seen: set = set()


def _log_once(msg: str) -> None:
    """Log a warning once per unique message (avoids flooding on batch errors)."""
    if msg in _log_once_seen:
        return
    _log_once_seen.add(msg)
    logger.warning(f"[mokuro] {msg}")


# Suppress noisy transformers warnings (e.g. "Some weights not used")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class MangaPageOcr:
    """Text detector + OCR model for one process.

    The per-page work is split into pieces so that a driver can run the
    CPU-side parts elsewhere (see ``mokuro.mokuro_generator.process_pages``):
    ``page_ops.load_page`` -> :meth:`detect_forward` -> ``page_ops.postprocess_page``
    -> ``page_ops.ocr_preprocess`` -> :meth:`generate_texts`. The single-page
    API (:meth:`__call__`, :meth:`detect_and_extract`, :meth:`recognize_text`)
    runs the same pieces in-process.
    """

    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        detector_input_size=1024,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr

        if not self.disable_ocr:
            device = get_device(force_cpu)
            if device == "cuda":
                # cuDNN TF32 convolutions make the CUDA detector output drift
                # from CPU/ROCm output at no speed benefit (ALLOW_CUDNN_TF32).
                torch.backends.cudnn.allow_tf32 = bool(ALLOW_CUDNN_TF32)
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector, input_size=detector_input_size, device=device, act="leaky"
            )

            # Opt-in: fold batch-norm into the preceding conv layers. Changes the
            # detector output slightly on CUDA and measured no faster (config.py).
            if FUSE_CONV_BN and hasattr(self.text_detector.net, "fuse"):
                try:
                    self.text_detector.net.fuse()
                    logger.warning(
                        "Fused conv+bn layers in text detector (FUSE_CONV_BN): output may differ from upstream"
                    )
                except Exception as e:  # noqa: BLE001 - opt-in fast path; fall back to the unfused net
                    logger.warning(f"FUSE_CONV_BN: fuse() failed ({e}); using the unfused detector")

            # CPU only: channels_last memory format keeps oneDNN's blocked
            # layout across conv layers (~2x on the detector forward).
            if device == "cpu" and DETECTOR_CPU_CHANNELS_LAST:
                try:
                    self.text_detector.net = self.text_detector.net.to(memory_format=torch.channels_last)
                    self.text_detector.channels_last = True
                except Exception as e:  # noqa: BLE001 - optional layout; keep the default memory format
                    logger.warning(f"channels_last for text detector skipped: {e}")

            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

            # OCR precision on GPUs: fp32 (default, identical to upstream) or
            # fp16 (--fp16 / USE_FP16; faster, not exact — see config.py).
            if device != "cpu" and _config.USE_FP16:
                try:
                    self.mocr.model.to(device)
                    self.mocr.model.half()
                    logger.info(f"Moved MangaOcr model to {device} (half precision)")
                except Exception as e:  # noqa: BLE001 - run on the default device/precision instead
                    logger.warning(f"Could not move model to {device}: {e}. Falling back to default.")

            # Beam search: mokuro/beam.py by default; transformers' generate()
            # (with the cache-reorder patch) as the fallback for non-default
            # decoding settings.
            self._beam = BeamSearchOCR(self.mocr.model, sync_lag=BEAM_SYNC_LAG) if USE_CUSTOM_BEAM else None
            if SKIP_CROSS_ATTN_CACHE_REORDER:
                install_skip_cross_attn_cache_reorder(self.mocr.model)

            # Experimental (off by default; measured slower than eager on the
            # GPUs tested): torch.compile in the default inductor mode.
            if device == "cuda" and USE_TORCH_COMPILE and hasattr(torch, "compile"):
                try:
                    self.text_detector.net = torch.compile(self.text_detector.net)
                    self.mocr.model.encoder = torch.compile(self.mocr.model.encoder, dynamic=True)
                    logger.info("Compiled models with torch.compile (USE_TORCH_COMPILE)")
                except Exception as e:  # noqa: BLE001 - experimental; eager mode is the fallback
                    logger.debug(f"torch.compile skipped: {e}")

            self._device = device
            self.single_plane = bool(OCR_PREPROCESS_SINGLE_PLANE)
            # set by the generator when worker processes are in use: the
            # detector's host copies then go straight into shared memory.
            self._share = False

    def __call__(self, img_path):
        """Process a single page image path and return the OCR result dict.

        Equivalent to upstream behaviour; internally it detects text blocks,
        extracts the text-line crops and runs batched OCR over them.
        """
        img = imread(img_path)
        result, all_crops, crop_metadata = self.detect_and_extract(img)

        if not self.disable_ocr and all_crops:
            all_texts = self.recognize_text(all_crops)
            for (blk_idx, line_idx), text in zip(crop_metadata, all_texts):
                result["blocks"][blk_idx]["lines"][line_idx] += text

        return result

    # ---- per-page pipeline pieces --------------------------------------
    @property
    def detector_input_size(self):
        return self.text_detector.input_size

    def page_ops_kwargs(self):
        """kwargs for page_ops.postprocess_page (also shipped to worker processes)."""
        return {
            "text_height": self.text_height,
            "max_ratio_vert": self.max_ratio_vert,
            "max_ratio_hor": self.max_ratio_hor,
            "anchor_window": self.anchor_window,
            "refine_mode": 1,
            "lazy_refine": bool(LAZY_MASK_REFINE),
        }

    def worker_cfg(self):
        """Configuration dict for ``mokuro.pipeline`` pools (picklable)."""
        return {
            "input_size": self.detector_input_size,
            "pp_kwargs": self.page_ops_kwargs(),
            "single_plane": self.single_plane,
            "processor": None if self.single_plane else self.mocr.processor,
            "config": _config.snapshot(),
        }

    @torch.no_grad()
    def detect_forward(self, img_u8):
        """Detector network forward + NMS on the model device.

        ``img_u8``: letterboxed uint8 (1,3,H,W) array or CPU tensor from
        ``page_ops.load_page``. Returns ``(det, mask_u8, lines_map)`` on the
        host: ``det`` (n,6) float32 numpy after NMS (letterbox coordinates),
        ``mask_u8`` uint8 HxW tensor, ``lines_map`` (1,1,H,W) float32 tensor.
        With ``self._share`` the host copies land directly in shared memory so
        they can be handed to a worker process without another copy.
        """
        td = self.text_detector
        x = normalize_input(img_u8, device=td.device, half=td.half)
        det, mask_u8, lines_map = td.forward(x)
        det = det.detach().cpu().numpy()
        if not self._share:
            return det, mask_u8.cpu(), lines_map.detach().cpu()
        mask_cpu = torch.empty(mask_u8.shape, dtype=torch.uint8).share_memory_()
        lines_cpu = torch.empty(lines_map.shape, dtype=lines_map.dtype).share_memory_()
        mask_cpu.copy_(mask_u8)
        lines_cpu.copy_(lines_map)
        return det, mask_cpu, lines_cpu

    def detect_and_extract(self, img):
        """Run text detection on a decoded page image (in-process).

        Returns ``(result_dict, crops, crop_metadata)`` where ``crops`` is a
        list of PIL images (one per text line) and ``crop_metadata`` maps each
        crop back to its ``(block_idx, line_idx)`` in ``result_dict``.
        """
        H, W, *_ = img.shape
        if self.disable_ocr:
            return {"version": __version__, "img_width": W, "img_height": H, "blocks": []}, [], []

        img_u8, _ratio, dw, dh = letterbox_input(img, self.detector_input_size)
        det, mask_u8, lines_map = self.detect_forward(img_u8)
        return postprocess_page(
            img, det, mask_u8, lines_map, dw, dh, self.detector_input_size, **self.page_ops_kwargs()
        )

    def _extract_crops(self, img, blk_list, mask_refined):
        """Split detected blocks into text-line crops (kept for API compatibility)."""
        return extract_crops(
            img, blk_list, mask_refined, self.text_height, self.max_ratio_vert, self.max_ratio_hor, self.anchor_window
        )

    def recognize_text(self, crops, batch_size=None, **generation_kwargs):
        """Run batched OCR over a list of PIL crops, returning one text per crop.

        Decoding defaults match manga-ocr's behaviour exactly: the model's own
        generation config (beam search, ``num_beams=4``) is used unless
        overridden (e.g. ``num_beams=1`` for greedy decoding). The machine-wide
        default for ``batch_size`` and ``num_beams`` lives in ``mokuro/config.py``.
        """
        all_texts = []

        if batch_size is None:
            batch_size = get_default_ocr_batch_size(self._device == "cpu")

        gen_args = self.generation_args(**generation_kwargs)

        for i in range(0, len(crops), batch_size):
            pixel_values = ocr_preprocess(crops[i : i + batch_size], self.mocr.processor, self.single_plane)
            all_texts.extend(self.generate_texts(pixel_values, gen_args))

        return all_texts

    def generation_args(self, **generation_kwargs):
        """Resolved generate() kwargs (model generation config + overrides).

        Explicit ``None`` values fall back to the model's generation config
        (manga-ocr's ``__call__`` passes no decoding overrides at all); a
        machine-wide ``NUM_BEAMS`` in mokuro/config.py wins over the model
        config unless the caller passed ``num_beams`` explicitly.
        """
        gen_config = self.mocr.model.generation_config
        gen_args = {
            "max_length": getattr(gen_config, "max_length", 300),
            "num_beams": getattr(gen_config, "num_beams", 1),
            "do_sample": getattr(gen_config, "do_sample", False),
            "use_cache": True,
        }
        overrides = {k: v for k, v in generation_kwargs.items() if v is not None}
        if NUM_BEAMS is not None and "num_beams" not in overrides:
            overrides["num_beams"] = NUM_BEAMS
        gen_args.update(overrides)
        return gen_args

    def generate_texts(self, pixel_values, gen_args):
        """OCR one preprocessed batch (float32 CPU tensor from ``page_ops.ocr_preprocess``,
        ``(N,1,224,224)`` or ``(N,3,224,224)``) -> list of texts."""
        device = self.mocr.model.device
        model_dtype = next(self.mocr.model.parameters()).dtype

        pixel_values = pixel_values.to(device, non_blocking=True)
        if model_dtype == torch.float16:
            pixel_values = pixel_values.half()
        if pixel_values.shape[1] == 1:
            # single grayscale plane -> the model's 3 identical channels
            pixel_values = pixel_values.expand(-1, 3, -1, -1).contiguous()

        with torch.inference_mode():
            if self._beam is not None and self._beam.matches(gen_args):
                generated_ids = self._beam.generate(pixel_values)
            else:
                generated_ids = self.mocr.model.generate(pixel_values, **gen_args)

        texts = self.mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        out = []
        for text in texts:
            # batch_decode can return bare token ids (int) for some inputs;
            # post_process (jaconv/tokenizer) calls str methods on the item
            # and would raise "'int' object has no attribute 'lower'".
            if not isinstance(text, str):
                _log_once(
                    "non-str decode item in recognize_text batch "
                    f"(type={type(text).__name__}, value={text!r}); coercing to str"
                )
                text = str(text) if text is not None else ""
            out.append(ocr_post_process(text))
        return out

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        return _split_into_chunks(
            img, mask_refined, blk, line_idx, textheight, max_ratio=max_ratio, anchor_window=anchor_window
        )
