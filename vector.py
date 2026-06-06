"""Semantic (vector) search channel over the global KV cache.

This is the second of Letta's two recall channels: instead of matching keywords,
it embeds each message into a vector and ranks by cosine similarity to the query
embedding, so it catches meaning and synonyms that BM25 misses.

Embeddings come from a local sentence-transformers model (all-MiniLM-L6-v2) --
offline, free, no API key. They are persisted next to the cache (one row per
cache line, index-aligned) so we only ever embed each message once.

NOTE: This is brute-force vector search -- it loads the whole embedding matrix
into memory and compares against every row. That's fine at our scale; a true
on-disk ANN index (e.g. sqlite-vec / FAISS) is the next step. See comments.md.
"""

import json
import os

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Lazily load the embedding model (importing torch is slow, so defer it)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embeddings_path(cache_path):
    # Store embeddings alongside the cache, e.g. global_kv_cache.txt.emb.npy.
    return cache_path + ".emb.npy"


def load_messages(cache_path):
    """Read the JSONL cache into a list of message dicts."""
    messages = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def message_text(message):
    """The text we embed for a message, across every message kind.

    Plain user/assistant turns embed their content. An assistant message that
    makes tool calls has content None, so we synthesize text from the calls it
    makes (each function name + its arguments) so the *action* is semantically
    findable, not invisible. Tool results embed their content string. Anything
    without usable text embeds as "" (a valid, near-neutral vector).
    """
    content = message.get("content")
    if content:
        return content
    if message.get("tool_calls"):
        return " ".join(
            f'{call["function"]["name"]} {call["function"]["arguments"]}'
            for call in message["tool_calls"])
    return ""


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

    Returns a list of (index, similarity) best-first, skipping the system
    message. Embeddings are L2-normalized, so a dot product is cosine similarity.

    `min_similarity` is a relevance floor: matches below it are dropped. Unlike
    Letta (whose search is agent-invoked, so the LLM filters relevance), we
    auto-inject results, so we need this floor to avoid injecting noise.
    """
    if not messages:
        return []

    embeddings = ensure_embeddings(messages, cache_path)
    query_vec = get_model().encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ np.asarray(query_vec, dtype=np.float32)

    ranked = []
    for i in np.argsort(-sims):  # descending similarity
        if sims[i] < min_similarity:
            break  # everything after this is lower; stop (relevance floor)
        if messages[i]["role"] == "system":
            continue  # the system prompt is always present; skip it
        ranked.append((int(i), float(sims[i])))
        if top_k is not None and len(ranked) >= top_k:
            break
    return ranked
