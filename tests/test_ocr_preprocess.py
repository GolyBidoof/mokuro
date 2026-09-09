"""The single-plane OCR preprocessing must be bit-identical to the stock ViTImageProcessor path."""

import numpy as np
import pytest
import torch
from PIL import Image

from mokuro.page_ops import OCR_RESIZE_BACKEND, ocr_preprocess


@pytest.fixture(scope="module")
def processor():
    try:
        from transformers import ViTImageProcessor

        return ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")
    except Exception as e:  # noqa: BLE001 - any load failure (no model cache / no network) means skip
        pytest.skip(f"manga-ocr processor unavailable: {e}")


def _random_crops(rng, n):
    crops = []
    for i in range(n):
        h = int(rng.integers(20, 120))
        w = int(rng.integers(20, 900))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        if i % 3 == 0:  # some crops with duplicate shapes (stock path batches equal shapes)
            arr = rng.integers(0, 256, size=(64, 300, 3), dtype=np.uint8)
        crops.append(Image.fromarray(arr).convert("RGB"))
    return crops


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_plane_equals_stock(processor, seed):
    rng = np.random.default_rng(seed)
    crops = _random_crops(rng, 24)
    stock = ocr_preprocess(crops, processor, single_plane=False)
    single = ocr_preprocess(crops, None, single_plane=True)
    assert single.shape == (len(crops), 1, 224, 224)
    assert stock.shape == (len(crops), 3, 224, 224)
    expanded = single.expand(-1, 3, -1, -1).contiguous()
    assert torch.equal(stock, expanded)
    assert torch.equal(stock.half(), expanded.half())


def test_backend_follows_transformers_version():
    """The stock ViTImageProcessor is PIL-based on transformers 4.x and torchvision-based on 5.x."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    assert OCR_RESIZE_BACKEND == ("pil" if major < 5 else "torchvision")


def test_backends_differ(processor):
    """Sanity check that the two backends are really distinct arithmetic (else the version switch is moot)."""
    rng = np.random.default_rng(3)
    crops = _random_crops(rng, 8)
    a = ocr_preprocess(crops, None, single_plane=True, backend="pil")
    b = ocr_preprocess(crops, None, single_plane=True, backend="torchvision")
    assert a.shape == b.shape == (len(crops), 1, 224, 224)
    assert not torch.equal(a, b)
    stock = ocr_preprocess(crops, processor, single_plane=False)
    assert torch.equal(stock, ocr_preprocess(crops, None, single_plane=True).expand(-1, 3, -1, -1).contiguous())


def test_empty():
    assert ocr_preprocess([], None, single_plane=True) is None
