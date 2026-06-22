"""Semantic (vector) search channel over the global KV cache.

Embeds each message and ranks by cosine similarity to the query, catching
meaning and synonyms that BM25 misses. Embeddings come from a local
sentence-transformers model (all-MiniLM-L6-v2) and are persisted next to the
cache (one index-aligned row per line) so each message is embedded only once.

Brute-force search: the whole embedding matrix is loaded and compared row by
row -- fine at this scale; an on-disk ANN index would be the next step.
"""

import os

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Lazily load the embedding model (importing torch is slow)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embeddings_path(cache_path):
    # Store embeddings alongside the cache, e.g. global_kv_cache.txt.emb.npy.
    return cache_path + ".emb.npy"


def load_messages(cache_path):
    """Read the JSONL cache into a list of message dicts.

    Uses the same tolerant per-line parser as the BM25 channel so a missing
    newline (two records run together on one line) can't crash recall.
    """
    from search import iter_json_objects

    messages = []
    with open(cache_path) as f:
        for line in f:
            messages.extend(iter_json_objects(line))
    return messages


def message_text(message):
    """Text to embed for a message; missing content embeds as ""."""
    return message.get("content") or ""


def ensure_embeddings(messages, cache_path):
    """Return an (len(messages), dim) embedding matrix aligned to `messages`.

    Embeddings are cached on disk and only the newly-appended messages are
    embedded on each call. If the on-disk cache is longer than `messages` (the
    cache file was edited/shrunk), we recompute from scratch to stay aligned.
    """
    emb_path = _embeddings_path(cache_path)

    cached = None
    if os.path.exists(emb_path):
        cached = np.load(emb_path)
        if len(cached) > len(messages):
            cached = None  # desynced -> rebuild

    start = len(cached) if cached is not None else 0
    if start < len(messages):
        new_texts = [message_text(m) for m in messages[start:]]
        new_emb = get_model().encode(new_texts, normalize_embeddings=True)
        new_emb = np.asarray(new_emb, dtype=np.float32)
        embeddings = np.vstack([cached, new_emb]) if cached is not None else new_emb
        np.save(emb_path, embeddings)
    else:
        embeddings = cached

    return embeddings


def vector_rank(query, messages, cache_path, top_k=None, min_similarity=0.0):
    """Rank messages by cosine similarity to the query.

    Returns (index, similarity) best-first, skipping the system message.
    Embeddings are L2-normalized, so a dot product is cosine similarity.
    `min_similarity` is a relevance floor: matches below it are dropped.
    """
    if not messages:
        return []

    embeddings = ensure_embeddings(messages, cache_path)
    query_vec = get_model().encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ np.asarray(query_vec, dtype=np.float32)

    ranked = []
    for i in np.argsort(-sims):  # descending similarity
        if sims[i] < min_similarity:
            break  # the rest are lower; stop (relevance floor)
        if messages[i]["role"] == "system":
            continue
        ranked.append((int(i), float(sims[i])))
        if top_k is not None and len(ranked) >= top_k:
            break
    return ranked
