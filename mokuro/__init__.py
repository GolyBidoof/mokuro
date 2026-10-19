from ._version import __version__ as __version__

__all__ = ["MangaPageOcr", "MokuroGenerator", "__version__"]


def __getattr__(name):
    # The model classes are imported lazily (PEP 562) so that the light-weight
    # worker processes of the page pipeline, which only need mokuro.page_ops,
    # do not pay for importing transformers / manga-ocr.
    if name == "MangaPageOcr":
        from mokuro.manga_page_ocr import MangaPageOcr

        return MangaPageOcr
    if name == "MokuroGenerator":
        from mokuro.mokuro_generator import MokuroGenerator

        return MokuroGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
