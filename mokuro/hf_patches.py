"""Instance-scoped patch of the transformers ``generate()`` glue used by the OCR model.

``GenerationMixin._beam_search`` re-orders the KV cache after every decoder
step via ``EncoderDecoderCache.reorder_cache``, which ``index_select``s BOTH
the self-attention cache and the cross-attention cache. The cross-attention
K/V come from ``encoder_outputs`` expanded with ``repeat_interleave`` (all
``num_beams`` rows of an item are identical) and the beam indices always stay
inside the item's own beam group (``batch_offset = arange(batch) *
num_beams``), so the cross gather returns exactly its input. For manga-ocr
(2 decoder layers, 197 encoder tokens, 32 crops x 4 beams) that no-op moves
~310 MB (fp32) / 155 MB (fp16) per step.

``_beam_search`` prefers ``model._reorder_cache(cache, beam_idx)`` when the
model defines it, so we attach one to the *instance* (no global monkeypatching
of transformers classes) that only reorders ``cache.self_attention_cache``.
Safety: if the cache is not an ``EncoderDecoderCache`` or the beam index count
does not match the cached batch size (which would mean the batch was shrunk /
re-indexed), the full stock reorder runs instead. If the transformers
internals are not present, nothing is patched.

Only used on the ``generate()`` fallback path (see ``mokuro/beam.py`` for the
default beam search, which needs no cache reordering at all).
"""

import types

from loguru import logger


def install_skip_cross_attn_cache_reorder(model) -> bool:
    """Attach an instance-level ``_reorder_cache`` that skips the cross-attention gather."""
    try:
        from transformers.cache_utils import EncoderDecoderCache
    except ImportError as e:  # pragma: no cover - very old/new transformers
        logger.debug(f"skip-cross-attn-cache-reorder: EncoderDecoderCache unavailable ({e}); not patching")
        return False

    if hasattr(model, "_reorder_cache"):
        logger.debug("skip-cross-attn-cache-reorder: model already defines _reorder_cache; not patching")
        return False

    def _cross_batch(cache):
        """Batch size held by the cross-attention cache, or None if unknown/empty.

        transformers >= 4.56 / 5.x: ``DynamicCache.layers[i].keys``; 4.42-4.55: the
        ``DynamicCache.key_cache`` list of tensors. Anything else -> None (the
        cross-attention rows are then assumed unchanged, which beam search guarantees).
        """
        try:
            cross = cache.cross_attention_cache
            layers = getattr(cross, "layers", None)
            if layers is not None:
                for layer in layers:
                    keys = getattr(layer, "keys", None)
                    if keys is not None and keys.dim() >= 1 and layer.get_seq_length() > 0:
                        return keys.shape[0]
                return None
            for keys in getattr(cross, "key_cache", []) or []:
                if keys is not None and keys.dim() == 4 and keys.shape[-2] > 0:
                    return keys.shape[0]
        except (AttributeError, IndexError, TypeError):  # unknown cache layout -> rows assumed unchanged
            return None
        return None

    def _reorder_cache(self, past_key_values, beam_idx):
        if (
            isinstance(past_key_values, EncoderDecoderCache)
            and hasattr(past_key_values, "self_attention_cache")
            and hasattr(past_key_values, "cross_attention_cache")
        ):
            n = _cross_batch(past_key_values)
            if n is None or n == beam_idx.numel():
                # Beam search keeps batch*num_beams rows for the whole run and only
                # permutes inside each item's group -> cross rows are unchanged.
                past_key_values.self_attention_cache.reorder_cache(beam_idx)
                return past_key_values
        # Anything unexpected: stock behaviour.
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    model._reorder_cache = types.MethodType(_reorder_cache, model)
    logger.debug("OCR generate(): cross-attention KV cache reorder skipped (identical output)")
    return True
