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


def bm25_rank(keywords, messages, min_coverage):
    """Rank an in-memory list of messages by BM25; returns [(index, score)].

    The hybrid path uses this because it already has every message loaded (the
    vector channel needs them). The streaming branch in `search_cache` is the
    memory-light equivalent for BM25-only mode.
    """
    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    docs = []  # (index, tf, doc_len) for messages containing >=1 query term

    for idx, message in enumerate(messages):
        if message["role"] == "system":
            continue
        tokens = tokenize(message["content"])
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
    """Fuse several ranked id-lists into one, the way Letta's recall search does.

    Each entry in `rankings` is a list of ids ordered best-first. An id's fused
    score is sum(weight / (k + rank)) over the lists it appears in (rank is
    1-based); an id missing from a list contributes nothing from it. k=60 follows
    Cormack et al. (2009). Fusing by *rank* avoids having to reconcile BM25's
    unbounded scores with cosine similarities. Returns the top_k fused ids.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]


def _conversational_pair(messages, idx):
    """Expand a match into the (user, assistant) exchange it belongs to.

    A match can land on either side of an exchange -- the vector channel often
    matches an assistant reply. So a user match brings back its following reply,
    and an assistant match brings back its preceding question, rather than always
    pulling the (possibly unrelated) *next* turn.
    """
    message = messages[idx]
    if message["role"] == "assistant" and idx > 0 and messages[idx - 1]["role"] == "user":
        return [messages[idx - 1], message]
    if message["role"] == "user" and idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant":
        return [message, messages[idx + 1]]
    return [message]


def _hybrid_search(query, keywords, cache_path, min_coverage, top_k,
                   channel_depth, bm25_weight, vector_weight):
    """Hybrid recall: BM25 + semantic vector channels fused with RRF."""
    import vector  # deferred: importing torch/sentence-transformers is slow

    try:
        messages = vector.load_messages(cache_path)
    except FileNotFoundError:
        return []
    if not messages:
        return []

    # Two independent rankings over the same messages, then fuse by rank.
    bm25_ids = [idx for idx, _ in bm25_rank(keywords, messages, min_coverage)[:channel_depth]]
    vec_ids = [idx for idx, _ in vector.vector_rank(query, messages, cache_path, top_k=channel_depth)]
    fused = reciprocal_rank_fusion(
        [bm25_ids, vec_ids],
        weights=[bm25_weight, vector_weight],
        top_k=top_k,
    )

    # Expand each match into its conversational pair, skipping repeats.
    result = []
    seen = set()
    for idx in fused:
        for message in _conversational_pair(messages, idx):
            key = id(message)
            if key not in seen:
                seen.add(key)
                result.append(message)
    return result


def search_cache(user_input, cache_path="global_kv_cache.txt",
                 min_coverage=0.3, top_k=1,
                 search_mode="bm25", channel_depth=50,
                 bm25_weight=0.5, vector_weight=0.5):
    """Find the best-matching chunk(s) of the cache for the given user input.

    Returns the top `top_k` matches, each expanded to its (user, assistant)
    conversational pair, or an empty list if nothing matches well.

    `search_mode`:
      - "bm25"  (default): keyword ranking only. Streams the file and needs no
        heavy dependencies -- the lightweight default for the chatbot.
      - "hybrid": BM25 + a semantic vector channel (sentence-transformers),
        fused with Reciprocal Rank Fusion, matching Letta's recall search. Loads
        the model and embeddings, so it's heavier; opt in when you want it.

    `min_coverage` is the BM25 accept/reject gate: a keyword match must contain
    at least this fraction of the query keywords. BM25 scores aren't normalized
    to 0..1, so coverage (not the raw score) is what we threshold on. The vector
    channel isn't gated this way -- it surfaces semantic matches BM25 misses.

    The BM25 path streams, but corpus stats (IDF) require holding the *matching*
    messages until the pass completes -- memory scales with the number of
    matches, not the whole cache. The hybrid path loads all messages (brute-force
    vector search). See comments.md for the on-disk-index next step.
    """
    keywords = extract_keywords(user_input)
    if not keywords:
        return []

    if search_mode == "hybrid":
        return _hybrid_search(user_input, keywords, cache_path, min_coverage,
                              top_k, channel_depth,
                              bm25_weight, vector_weight)

    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    candidates = []      # each: {"chunk", "awaiting_reply", "tf", "doc_len"}
    prev_message = None  # one-message lookback, so an assistant match can pair
                         # back to its preceding user question

    try:
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)

                # If a prior user match is waiting for its reply, attach this
                # message when it's the assistant turn, then stop waiting.
                for cand in candidates:
                    if cand["awaiting_reply"]:
                        if message["role"] == "assistant":
                            cand["chunk"].append(message)
                        cand["awaiting_reply"] = False

                if message["role"] != "system":
                    tokens = tokenize(message["content"])
                    num_docs += 1
                    total_len += len(tokens)

                    # Count how often each query keyword appears in this message.
                    counts = Counter(tokens)
                    tf = {kw: counts[kw] for kw in keywords if counts[kw] > 0}
                    if tf:
                        for term in tf:
                            doc_freq[term] += 1

                        # Expand the match into its conversational pair: an
                        # assistant match pairs back to the preceding user turn;
                        # a user match awaits the following assistant reply.
                        if message["role"] == "assistant" and prev_message and prev_message["role"] == "user":
                            chunk, awaiting = [prev_message, message], False
                        elif message["role"] == "user":
                            chunk, awaiting = [message], True
                        else:
                            chunk, awaiting = [message], False
                        candidates.append({
                            "chunk": chunk,
                            "awaiting_reply": awaiting,
                            "tf": tf,
                            "doc_len": len(tokens),
                        })

                prev_message = message
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
