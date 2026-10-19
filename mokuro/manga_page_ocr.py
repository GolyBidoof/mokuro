"""Text detector + OCR model for one process.

The per-page work is split into pieces so that a driver can run the CPU-side
parts elsewhere (see ``mokuro.mokuro_generator.process_pages``):
``page_ops.load_page`` -> :meth:`detect_forward` -> ``page_ops.postprocess_page``
-> ``manga_ocr.preprocessing.preprocess_images`` -> :meth:`recognize_pixels`.
The single-page API (:meth:`__call__`, :meth:`detect_and_extract`,
:meth:`recognize_text`) runs the same pieces in-process.

The OCR model itself, its batching, beam search, fp16 mode and crop
preprocessing are manga-ocr's public API (``MangaOcr.recognize`` /
``MangaOcr.recognize_preprocessed`` / ``MangaOcr.preprocess_config``); this
class only drives the detector around it.
"""

import torch
from comic_text_detector.inference import TextDetector, letterbox_input, normalize_input
from loguru import logger
from manga_ocr import MangaOcr

from mokuro import __version__
from mokuro import config as _config
from mokuro.cache import cache
from mokuro.config import (
    ALLOW_CUDNN_TF32,
    DETECTOR_CPU_CHANNELS_LAST,
    FUSE_CONV_BN,
    LAZY_MASK_REFINE,
    NUM_BEAMS,
    USE_FP16,
    get_default_ocr_batch_size,
    get_device,
)
from mokuro.page_ops import extract_crops, postprocess_page
from mokuro.page_ops import split_into_chunks as _split_into_chunks
from mokuro.utils import imread


class MangaPageOcr:
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
        fp16=None,
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

            # The OCR model, its precision (fp32 default / fp16 opt-in), the
            # batched beam search and the crop preprocessing all live in
            # manga-ocr; mokuro only decides when to call them.
            self.mocr = MangaOcr(
                pretrained_model_name_or_path,
                force_cpu=force_cpu,
                fp16=bool(USE_FP16 if fp16 is None else fp16),
            )
            self._device = device
            # Picklable description of manga-ocr's fast crop preprocessing,
            # shipped to the pipeline worker processes (see ``worker_cfg``).
            self.preprocess_config = self.mocr.preprocess_config
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
            "preprocess_config": self.preprocess_config,
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

        ``batch_size`` and the decoding kwargs (e.g. ``num_beams``) are passed
        straight to manga-ocr, whose defaults match the model's own generation
        config (beam search, ``num_beams=4``).
        """
        if batch_size is None:
            batch_size = get_default_ocr_batch_size(self._device == "cpu")
        return self.mocr.recognize(crops, batch_size=batch_size, **self.generation_kwargs(**generation_kwargs))

    def generation_kwargs(self, num_beams=None):
        """Decoding overrides for manga-ocr (empty = the model's own defaults).

        A machine-wide ``NUM_BEAMS`` in mokuro/config.py applies unless the
        caller passed ``num_beams`` explicitly.
        """
        if num_beams is None:
            num_beams = NUM_BEAMS
        kwargs = {}
        if num_beams is not None:
            kwargs["num_beams"] = int(num_beams)
        return kwargs

    def recognize_pixels(self, pixel_values, generation_kwargs=None):
        """OCR one preprocessed CPU batch through manga-ocr's public API.

        ``pixel_values`` is the tensor produced by
        ``manga_ocr.preprocessing.preprocess_images`` (the pipeline
        preprocesses crops in worker processes); ``generation_kwargs`` comes
        from :meth:`generation_kwargs`.
        """
        return self.mocr.recognize_preprocessed(pixel_values, **(generation_kwargs or {}))

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        return _split_into_chunks(
            img, mask_refined, blk, line_idx, textheight, max_ratio=max_ratio, anchor_window=anchor_window
        )
