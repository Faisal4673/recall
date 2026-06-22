"""BM25 keyword search over the global KV cache.

Reduce the user's input to keywords, rank the persisted conversation cache with
BM25, and return the best-matching turn(s) for recall.
"""

import json
import math
import re
from collections import Counter

# BM25 tuning constants (Lucene/Elasticsearch defaults).
BM25_K1 = 1.5   # term-frequency saturation
BM25_B = 0.75   # length normalization

# raw_decode recovers every object on a line, so a missing newline (two records
# run together) can't crash recall. An undecodable remainder is skipped.
_DECODER = json.JSONDecoder()


def iter_json_objects(line):
    """Yield each JSON object found in `line`, tolerating several run together."""
    idx, length = 0, len(line)
    while idx < length:
        while idx < length and line[idx].isspace():
            idx += 1  # skip whitespace between (and around) objects
        if idx >= length:
            break
        try:
            obj, idx = _DECODER.raw_decode(line, idx)
        except json.JSONDecodeError:
            break  # undecodable remainder: skip the rest of this line
        yield obj


# Common words dropped from keyword matching.
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
    """Reduce text to keywords, dropping stopwords and tokens of <=2 chars."""
    return {w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2}


def bm25_score(tf, doc_len, avg_doc_len, doc_freq, num_docs):
    """BM25 relevance score for one message given corpus statistics.

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


def bm25_rank(keywords, messages, min_coverage):
    """Rank an in-memory list of messages by BM25; returns [(index, score)].

    Used by the hybrid path, which already has every message loaded. The
    streaming branch in `search_cache` is the memory-light BM25-only equivalent.
    """
    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    docs = []  # (index, tf, doc_len) for messages containing >=1 query term

    for idx, message in enumerate(messages):
        if message["role"] == "system":
            continue
        tokens = tokenize(message.get("content") or "")
        num_docs += 1
        total_len += len(tokens)
        counts = Counter(tokens)
        tf = {kw: counts[kw] for kw in keywords if counts[kw] > 0}
        if not tf:
            continue
        for term in tf:
            doc_freq[term] += 1
        docs.append((idx, tf, len(tokens)))

    if not docs:
        return []

    avg_doc_len = total_len / num_docs
    ranked = []
    for idx, tf, doc_len in docs:
        if len(tf) / len(keywords) < min_coverage:
            continue  # coverage gate
        ranked.append((idx, bm25_score(tf, doc_len, avg_doc_len, doc_freq, num_docs)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def reciprocal_rank_fusion(rankings, k=60, weights=None, top_k=1):
    """Fuse several best-first id-lists into one, returning the top_k ids.

    An id's score is sum(weight / (k + rank)) over the lists it appears in
    (1-based rank). k=60 follows Cormack et al. (2009). Fusing by rank avoids
    reconciling BM25's unbounded scores with cosine similarities.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]


def _turn_cluster(messages, idx):
    """Expand a match into its whole turn (opening user message + its answer).

    Returning the whole turn pairs a match on either side back to its
    counterpart -- the vector channel often matches an assistant reply, which we
    want back with the question that prompted it.
    """
    start = idx
    while start > 0 and messages[start]["role"] != "user":
        start -= 1
    if messages[start]["role"] == "system":
        start += 1  # never start a turn on the leading system prompt
    end = idx + 1
    while end < len(messages) and messages[end]["role"] != "user":
        end += 1
    return messages[start:end]


def _hybrid_search(query, keywords, cache_path, min_coverage, top_k,
                   channel_depth, bm25_weight, vector_weight, min_similarity):
    """Hybrid recall: BM25 + semantic vector channels fused with RRF."""
    import vector  # deferred: importing torch/sentence-transformers is slow

    try:
        messages = vector.load_messages(cache_path)
    except FileNotFoundError:
        return []
    if not messages:
        return []

    # Two independent rankings, each relevance-gated (BM25 by coverage, vector
    # by similarity), then fused by rank.
    bm25_ids = [idx for idx, _ in bm25_rank(keywords, messages, min_coverage)[:channel_depth]]
    vec_ids = [idx for idx, _ in vector.vector_rank(
        query, messages, cache_path, top_k=channel_depth, min_similarity=min_similarity)]
    fused = reciprocal_rank_fusion(
        [bm25_ids, vec_ids],
        weights=[bm25_weight, vector_weight],
        top_k=top_k,
    )

    # Expand each match into its whole turn, deduping turns two matches share.
    clusters = []
    seen_starts = set()
    for idx in fused:
        cluster = _turn_cluster(messages, idx)
        if not cluster:
            continue
        key = id(cluster[0])  # identity of the turn's first message
        if key not in seen_starts:
            seen_starts.add(key)
            clusters.append(cluster)
    return clusters


def search_cache(user_input, cache_path="global_kv_cache.txt",
                 min_coverage=0.3, top_k=3,
                 search_mode="bm25", channel_depth=50,
                 bm25_weight=0.5, vector_weight=0.5, min_similarity=0.3):
    """Find the best-matching turn(s) of the cache for the given user input.

    Returns up to `top_k` turn clusters best first (each a whole turn: its user
    message and the assistant answer), or [] if nothing matches well.

    `search_mode`:
      - "bm25"  (default): keyword ranking only; streams the file, no heavy deps.
      - "hybrid": BM25 + a sentence-transformers vector channel, fused with RRF.
        Loads the model and embeddings, so it's heavier; opt in.

    Each channel has a relevance floor so an off-topic query injects nothing:
      - `min_coverage`: BM25 gate -- fraction of query keywords a match must
        contain (BM25 scores aren't 0..1, so we gate on coverage).
      - `min_similarity`: vector gate -- minimum cosine similarity (0.3 suits
        all-MiniLM-L6-v2).
    """
    keywords = extract_keywords(user_input)
    if not keywords:
        return []

    if search_mode == "hybrid":
        return _hybrid_search(user_input, keywords, cache_path, min_coverage,
                              top_k, channel_depth,
                              bm25_weight, vector_weight, min_similarity)

    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    turns = []       # finalized turns: each {"messages": [...], "matches": [...]}
    current = None   # the turn being accumulated; matches is [(tf, doc_len), ...]

    # Stream the cache, grouping messages into turns (one opens on each user
    # message) and recording per turn the BM25 stats of every matching message.
    try:
        with open(cache_path) as f:
            for line in f:
                for message in iter_json_objects(line):
                    if message["role"] == "system":
                        continue  # the system prompt never belongs to a turn

                    # A user message opens a new turn; so does the first message.
                    if message["role"] == "user" or current is None:
                        if current is not None:
                            turns.append(current)
                        current = {"messages": [], "matches": []}
                    current["messages"].append(message)

                    tokens = tokenize(message.get("content") or "")
                    num_docs += 1
                    total_len += len(tokens)

                    counts = Counter(tokens)
                    tf = {kw: counts[kw] for kw in keywords if counts[kw] > 0}
                    if tf:
                        for term in tf:
                            doc_freq[term] += 1
                        current["matches"].append((tf, len(tokens)))

            if current is not None:
                turns.append(current)
    except FileNotFoundError:
        return []

    if num_docs == 0:
        return []

    # Score each turn by its best message, gate on coverage, return top_k turns.
    avg_doc_len = total_len / num_docs
    ranked = []
    for turn in turns:
        if not turn["matches"]:
            continue
        best_score = 0.0
        best_coverage = 0.0
        for tf, doc_len in turn["matches"]:
            best_score = max(best_score, bm25_score(
                tf, doc_len, avg_doc_len, doc_freq, num_docs))
            best_coverage = max(best_coverage, len(tf) / len(keywords))
        if best_coverage < min_coverage:
            continue  # coverage gate
        ranked.append((best_score, turn["messages"]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [messages for _, messages in ranked[:top_k]]
