"""BM25 keyword search over the global KV cache.

The flow this supports: take the user's input, reduce it to keywords, then rank
the persisted conversation cache with BM25 (the same full-text ranking Letta's
recall memory uses). The best-matching chunk(s) (a message plus its reply) are
returned so they can be prepended to the live conversation before the new user
prompt. See comments.md for how this maps to Letta and what comes next.
"""

import json
import math
import re
from collections import Counter

# BM25 tuning constants (Lucene/Elasticsearch defaults).
BM25_K1 = 1.5   # term-frequency saturation: how fast repeated terms stop helping
BM25_B = 0.75   # length normalization: how much long messages are penalized

# Common words that carry little meaning and would pollute keyword matching.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "for", "with", "as", "at",
    "by", "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "me", "my", "your", "do", "does", "did", "can",
    "could", "would", "should", "what", "which", "who", "how", "when", "where",
}


def tokenize(text):
    """Split text into lowercase alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_keywords(text):
    """Reduce text to a set of meaningful keywords, dropping stopwords/noise."""
    # Keep tokens that aren't stopwords and are longer than two characters.
    return {w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2}


def bm25_score(tf, doc_len, avg_doc_len, doc_freq, num_docs):
    """BM25 relevance score for one message given corpus statistics.

    This is the ranking function Letta's recall full-text channel uses. Unlike a
    raw keyword-overlap count it rewards rare terms (IDF), saturates repeated
    terms (k1), and normalizes for message length (b).

    Args:
        tf: {term: count} of query terms found in this message.
        doc_len: number of tokens in this message.
        avg_doc_len: average message length across the corpus.
        doc_freq: {term: number of messages containing it}.
        num_docs: total number of (non-system) messages searched.
    """
    score = 0.0
    for term, term_freq in tf.items():
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        # IDF: rarer terms (low df) contribute more; the +1 keeps it positive.
        idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        # TF component with saturation (k1) and length normalization (b).
        denom = term_freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avg_doc_len)
        score += idf * (term_freq * (BM25_K1 + 1)) / denom
    return score


def search_cache(user_input, cache_path="global_kv_cache.txt",
                 min_coverage=0.3, context=1, top_k=1):
    """Find the best-matching chunk(s) of the cache for the given user input.

    Ranks messages with BM25 (Letta's recall full-text channel) and returns the
    top `top_k` matches, each followed by `context` trailing entries (e.g. its
    assistant reply). Returns an empty list if nothing matches well.

    `min_coverage` is the accept/reject gate: a match must contain at least this
    fraction of the query keywords to be injected. BM25 scores aren't normalized
    to 0..1, so coverage (not the raw score) is what we threshold on; BM25 only
    decides the ranking order.

    Streams the file, but corpus stats (IDF) require holding the *matching*
    messages until the pass completes -- memory scales with the number of
    matches, not the whole cache. See comments.md for the next steps (an on-disk
    index, plus a semantic channel fused with this one via RRF).
    """
    keywords = extract_keywords(user_input)
    if not keywords:
        return []

    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    candidates = []  # each: {"chunk", "remaining", "tf", "doc_len"}

    try:
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)

                # Grow the trailing context of any candidate still collecting it
                # (so a matched question also brings back its answer).
                for cand in candidates:
                    if cand["remaining"] > 0:
                        cand["chunk"].append(message)
                        cand["remaining"] -= 1

                if message["role"] == "system":
                    continue  # the system prompt is always present; skip it

                tokens = tokenize(message["content"])
                num_docs += 1
                total_len += len(tokens)

                # Count how often each query keyword appears in this message.
                counts = Counter(tokens)
                tf = {kw: counts[kw] for kw in keywords if counts[kw] > 0}
                if not tf:
                    continue  # no query term here, not a candidate

                for term in tf:
                    doc_freq[term] += 1
                candidates.append({
                    "chunk": [message],
                    "remaining": context,
                    "tf": tf,
                    "doc_len": len(tokens),
                })
    except FileNotFoundError:
        return []

    if not candidates:
        return []

    # Score every candidate with BM25 now that corpus stats are known.
    avg_doc_len = total_len / num_docs
    for cand in candidates:
        cand["score"] = bm25_score(
            cand["tf"], cand["doc_len"], avg_doc_len, doc_freq, num_docs)
        cand["coverage"] = len(cand["tf"]) / len(keywords)

    # Keep only matches that clear the coverage gate, rank by BM25, take top_k.
    ranked = sorted(
        (c for c in candidates if c["coverage"] >= min_coverage),
        key=lambda c: c["score"],
        reverse=True,
    )[:top_k]

    # Flatten the selected chunks (best first) into one message list.
    result = []
    for cand in ranked:
        result.extend(cand["chunk"])
    return result
