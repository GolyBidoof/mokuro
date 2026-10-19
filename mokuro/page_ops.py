"""Per-page CPU-side operations of the mokuro pipeline.

Everything here is a pure function of its inputs (no model state) so it can run
either in the main process or in a worker process. The code is a verbatim
split of ``comic_text_detector.inference.TextDetector.__call__`` and of the
crop extraction / OCR preprocessing that used to live in ``MangaPageOcr``:

* :func:`load_page`        - decode the image + detector letterbox (uint8)
* (model process)         - detector forward + NMS (needs the model/GPU)
* :func:`postprocess_page` - YOLO box scaling, DB polygons, group_output,
                             *lazy* mask refinement, text-line crop extraction

Mask refinement is lazy at page level: ``refine_mask``/``refine_undetected_mask``
run (unchanged, same order, same arrays) only if some line of the page needs
chunking (crop ratio > max_ratio). This is bit-exact because blk_list is built
before refinement and nothing but ``split_into_chunks`` reads ``mask_refined``.
(Block-level laziness would NOT be exact: refine windows overlap and
``refine_undetected_mask`` reads the whole refined mask.)
"""

import cv2
import numpy as np
import torch
from comic_text_detector.inference import letterbox_input, postprocess_detections
from comic_text_detector.utils.db_utils import SegDetectorRepresenter
from comic_text_detector.utils.textmask import refine_mask, refine_undetected_mask
from PIL import Image
from scipy.signal.windows import gaussian

from mokuro import __version__
from mokuro.utils import imread

# Cache gaussian windows keyed by (size, std) - they only depend on text_height.
_gaussian_cache = {}
_seg_rep = None


def _get_seg_rep():
    global _seg_rep
    if _seg_rep is None:
        _seg_rep = SegDetectorRepresenter(thresh=0.3)
    return _seg_rep


def load_page(img_path, input_size):
    """Decode a page and letterbox it for the detector (CPU, uint8).

    Returns ``(img, img_u8, dw, dh)``: ``img`` is the BGR uint8 page,
    ``img_u8`` the letterboxed (1,3,H,W) uint8 array the model process
    normalises on its device (``comic_text_detector.inference.normalize_input``).
    """
    img = imread(img_path)
    img_u8, _ratio, dw, dh = letterbox_input(img, input_size)
    return img, img_u8, dw, dh


def postprocess_page(
    img,
    det,
    mask_u8,
    lines_map,
    dw,
    dh,
    input_size,
    text_height,
    max_ratio_vert,
    max_ratio_hor,
    anchor_window,
    refine_mode=1,
    lazy_refine=True,
):
    """Detector post-processing + (lazy) refinement + crop extraction for one page.

    ``img``: BGR uint8 HxWx3; ``det``: NMS output (n,6) float32 numpy in
    letterbox coordinates; ``mask_u8``: uint8 HxW mask (numpy or CPU tensor);
    ``lines_map``: (1,1,H,W) float32 CPU tensor (channel 0 of the DB head).
    Returns ``(result_dict, crops (PIL RGB list), crop_metadata)``.
    """
    im_h, im_w = img.shape[:2]
    if isinstance(mask_u8, torch.Tensor):
        mask_u8 = mask_u8.numpy()
    mask, blk_list = postprocess_detections(det, mask_u8, lines_map, im_w, im_h, input_size, dw, dh, _get_seg_rep())

    # Memoised refinement: identical code and order as the eager detector;
    # `mask` is owned by this closure (refine_undetected_mask mutates it in
    # place) and nothing else reads it after group_output.
    cache = []

    def get_refined():
        if not cache:
            r = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
            r = refine_undetected_mask(img, mask, r, blk_list, refine_mode=refine_mode)
            cache.append(r)
        return cache[0]

    if not lazy_refine:
        get_refined()

    return extract_crops(img, blk_list, get_refined, text_height, max_ratio_vert, max_ratio_hor, anchor_window)


def extract_crops(img, blk_list, get_refined, text_height, max_ratio_vert, max_ratio_hor, anchor_window):
    """Split detected blocks into text-line crops (PIL RGB images).

    ``get_refined`` is the refined mask array or a zero-argument callable
    returning it (only called for lines that have to be split).
    """
    H, W, *_ = img.shape
    result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}
    all_crops = []
    crop_metadata = []

    for blk_idx, blk in enumerate(blk_list):
        result_blk = {
            "box": list(blk.xyxy),
            "vertical": blk.vertical,
            "font_size": blk.font_size,
            "lines_coords": [],
            "lines": [],
        }
        result["blocks"].append(result_blk)

        for line_idx, line in enumerate(blk.lines_array()):
            max_ratio = max_ratio_vert if blk.vertical else max_ratio_hor

            line_crops, _ = split_into_chunks(
                img,
                get_refined,
                blk,
                line_idx,
                textheight=text_height,
                max_ratio=max_ratio,
                anchor_window=anchor_window,
            )

            result_blk["lines_coords"].append(line.tolist())
            result_blk["lines"].append("")

            for line_crop in line_crops:
                if blk.vertical:
                    line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
                all_crops.append(Image.fromarray(line_crop).convert("RGB"))
                crop_metadata.append((blk_idx, line_idx))

    return result, all_crops, crop_metadata


def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
    """Cut an over-long text line into chunks at low-ink columns.

    ``mask_refined`` may be the refined mask array or a zero-argument callable
    returning it; it is only evaluated after the ``ratio <= max_ratio`` early
    return, i.e. right before its only read.
    """
    try:
        line_crop = blk.get_transformed_region(img, line_idx, textheight)
    except (OverflowError, ValueError, ZeroDivisionError):
        # Degenerate line geometry - skip it rather than crash the volume.
        return [], []

    h, w, *_ = line_crop.shape
    if h == 0 or w == 0:
        return [], []

    ratio = w / h

    if ratio <= max_ratio:
        return [line_crop], []

    cache_key = (textheight * 2, textheight / 8)
    if cache_key not in _gaussian_cache:
        _gaussian_cache[cache_key] = gaussian(cache_key[0], cache_key[1])
    k = _gaussian_cache[cache_key]

    if callable(mask_refined):
        mask_refined = mask_refined()
    line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
    num_chunks = int(np.ceil(ratio / max_ratio))

    anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

    line_density = line_mask.sum(axis=0)
    line_density = np.convolve(line_density, k, "same")
    line_density /= line_density.max()

    anchor_window *= textheight

    cut_points = []
    for anchor in anchors:
        anchor = int(anchor)

        n0 = np.clip(anchor - anchor_window // 2, 0, w)
        n1 = np.clip(anchor + anchor_window // 2, 0, w)

        p = line_density[n0:n1].argmin()
        p += n0

        cut_points.append(p)

    return np.split(line_crop, cut_points, axis=1), cut_points
