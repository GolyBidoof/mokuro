from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
from loguru import logger

from mokuro import MokuroGenerator, __version__
from mokuro.legacy.overlay_generator import generate_legacy_html
from mokuro.volume import VolumeCollection


def run(
    *paths: Sequence[str | Path] | None,
    parent_dir: str | Path | None = None,
    pretrained_model_name_or_path: str = "kha-white/manga-ocr-base",
    force_cpu: bool = False,
    disable_confirmation: bool = False,
    disable_ocr: bool = False,
    ignore_errors: bool = False,
    no_cache: bool = False,
    unzip: bool = False,
    legacy_html: bool = True,
    as_one_file: bool = True,
    num_workers: int | None = None,
    ocr_batch_size: int | None = None,
    num_beams: int | None = None,
    fp16: bool = False,
    version: bool = False,
):
    """
    Process manga volumes with mokuro.

    Args:
        paths: Paths to manga volumes. Volume can be a directory, a zip file or a cbz file.
        parent_dir: Parent directory to scan for volumes. If provided, all volumes inside this directory will be processed.
        pretrained_model_name_or_path: Name or path of the manga-ocr model.
        force_cpu: Force the use of CPU even if CUDA/MPS is available.
        disable_confirmation: Disable confirmation prompt. If False, the user will be prompted to confirm the list of volumes to be processed.
        disable_ocr: Disable OCR processing. Generate mokuro/HTML files without OCR results.
        ignore_errors: Continue processing volumes even if an error occurs.
        no_cache: Do not use cached OCR results from previous runs (_ocr directories).
        unzip: Extract volumes in zip/cbz format in their original location.
        legacy_html: Enable legacy HTML output. If True, acts as if --unzip is True.
        as_one_file: Applies only to legacy HTML. If False, generate separate CSS and JS files instead of embedding them in the HTML file.
        num_workers: Number of worker processes. On a GPU: CPU-side pipeline workers (decode, post-processing, crops);
            on CPU only: model shard processes. 0 = single process. Default: auto-detected (see mokuro/config.py).
        ocr_batch_size: Text-line crops per batched OCR call. Default: auto-detected from hardware.
        fp16: Run the OCR model in half precision on CUDA/ROCm/MPS (1.07x-4.9x faster depending on the GPU; not exact: changes 0.19% of characters on ~2.6% of pages of a 140-volume set, see README "Precision policy"; boxes unaffected). Default: fp32, identical to upstream.
        num_beams: Beam width for OCR decoding. Default: model default (4, identical output to upstream).
            Other values (e.g. 1 = greedy) are faster but change the OCR text; see mokuro/config.py.
        version: Print the version of mokuro and exit.
    """

    if version:
        print(f"{__version__}")
        return

    if disable_ocr:
        logger.info("Running with OCR disabled")

    if legacy_html:
        logger.warning(
            "Legacy HTML output is deprecated and will not be further developed. "
            "It's recommended to use .mokuro format and web reader instead. "
            "Legacy HTML will be disabled by default in the future. To explicitly enable it, run with option --legacy-html."
        )
        # legacy HTML works only with unzipped output
        unzip = True

    logger.info("Scanning paths...")

    if isinstance(fp16, str):
        # python-fire parses "--fp16 <path>" as fp16="<path>"; keep the path and treat the flag as set.
        paths = (fp16, *paths)
        fp16 = True

    paths_ = []
    for path in paths:
        path_normalized = Path(str(path)).expanduser().absolute()

        try:
            path_valid = path_normalized.exists()
        except OSError:
            path_valid = False

        if path_valid:
            paths_.append(path_normalized)
        else:
            logger.error(f"Invalid path: {path_normalized}")
            return

    paths = paths_

    if parent_dir is not None:
        for p in Path(parent_dir).expanduser().absolute().iterdir():
            if (
                p not in paths
                and (p.is_dir() and p.stem != "_ocr")
                or (p.is_file() and p.suffix.lower() in {".zip", ".cbz"})
            ):
                paths.append(p)

    vc = VolumeCollection()

    for path_in in paths:
        vc.add_path_in(path_in)

    if len(vc) == 0:
        logger.error("Found no paths to process. Did you set the paths correctly?")
        return

    for title in vc.titles.values():
        title.set_uuid()

    status_counter = Counter()

    print(f"\nFound {len(vc)} volumes:\n")

    for volume in vc:
        print(volume)
        status_counter[volume.status] += 1

    msg = "\nEach of the paths above will be treated as one volume.\n"
    print(msg)

    if not disable_confirmation:
        inp = input("\nContinue? [yes/no]")
        if inp.lower() not in ("y", "yes"):
            return

    if fp16:
        from mokuro import config as _cfg

        _cfg.USE_FP16 = True
        logger.warning(
            "fp16 OCR enabled (--fp16): faster, but not exact — a small fraction of characters may differ from fp32"
        )

    mg = MokuroGenerator(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        force_cpu=force_cpu,
        disable_ocr=disable_ocr,
        num_workers=num_workers,
        ocr_batch_size=ocr_batch_size,
        num_beams=num_beams,
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # unzip == True means that zipped volumes will be unzipped in their original location
        # in that case, we don't use a temporary directory
        if unzip:
            tmp_dir = None

        num_sucessful = 0
        try:
            for i, volume in enumerate(vc):
                logger.info(f"Processing {i + 1}/{len(vc)}: {volume.path_in}")

                try:
                    volume.unzip(tmp_dir)
                    mg.process_volume(volume, ignore_errors=ignore_errors, no_cache=no_cache)
                    if legacy_html:
                        generate_legacy_html(volume, as_one_file=as_one_file, ignore_errors=ignore_errors)

                except Exception:  # noqa: BLE001 - logged with traceback; continue with the next volume
                    logger.exception(f"Error while processing {volume.path_in}")
                else:
                    num_sucessful += 1
        finally:
            mg.close()

        logger.info(f"Processed successfully: {num_sucessful}/{len(vc)}")


if __name__ == "__main__":
    fire.Fire(run)
