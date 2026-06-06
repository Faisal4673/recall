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
        # content is None on an assistant(tool_calls) message.
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


def _turn_cluster(messages, idx):
    """Expand a match into the whole turn it belongs to.

    A turn is a self-contained unit: the user message that starts it, through
    everything up to (but not including) the next user message -- which now
    includes any assistant(tool_calls) -> tool-result rounds and the final
    answer. Returning the *whole* turn (rather than just a user/assistant pair)
    is what lets tool exchanges be injected back into the conversation as a
    valid, API-legal sequence: a tool result never arrives without its call, and
    a tool call never arrives without its results.

    A match can land anywhere inside the turn (the vector channel often matches
    an assistant reply or a tool result); we walk back to the turn's opening user
    message and forward to the next one to recover the full span.
    """
    start = idx
    # Walk back to the user message that opens this turn.
    while start > 0 and messages[start]["role"] != "user":
        start -= 1
    # Never let a turn start on the leading system prompt.
    if messages[start]["role"] == "system":
        start += 1
    # Walk forward to the message just before the next turn's user message.
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

    # Two independent rankings over the same messages, then fuse by rank. Each
    # channel is gated for relevance (BM25 by coverage, vector by similarity) so
    # an off-topic query injects nothing rather than the least-irrelevant turn.
    bm25_ids = [idx for idx, _ in bm25_rank(keywords, messages, min_coverage)[:channel_depth]]
    vec_ids = [idx for idx, _ in vector.vector_rank(
        query, messages, cache_path, top_k=channel_depth, min_similarity=min_similarity)]
    fused = reciprocal_rank_fusion(
        [bm25_ids, vec_ids],
        weights=[bm25_weight, vector_weight],
        top_k=top_k,
    )

    # Expand each match into its whole turn, deduping turns that two matches
    # share (identified by their opening message). Returns a list of clusters,
    # each a self-contained, injectable turn.
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

    Returns up to `top_k` turn clusters, best first -- each a list of messages
    spanning a whole turn (its user message through any assistant(tool_calls)/
    tool-result rounds to the final answer) -- or an empty list if nothing
    matches well. Returning whole turns keeps tool exchanges injectable as
    valid, self-contained sequences.

    `search_mode`:
      - "bm25"  (default): keyword ranking only. Streams the file and needs no
        heavy dependencies -- the lightweight default for the chatbot.
      - "hybrid": BM25 + a semantic vector channel (sentence-transformers),
        fused with Reciprocal Rank Fusion, matching Letta's recall search. Loads
        the model and embeddings, so it's heavier; opt in when you want it.

    Each channel has its own relevance floor so an off-topic query injects
    nothing rather than the least-irrelevant turn:
      - `min_coverage`: BM25 gate -- a match must contain at least this fraction
        of the query keywords (BM25 scores aren't 0..1, so we gate on coverage).
      - `min_similarity`: vector gate -- a match must reach this cosine
        similarity. The 0.3 default suits all-MiniLM-L6-v2 and is tunable.
    Letta needs neither (its search is agent-invoked, so the LLM filters
    results); we auto-inject, so we gate here instead.

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
                              bm25_weight, vector_weight, min_similarity)

    num_docs = 0
    total_len = 0
    doc_freq = {kw: 0 for kw in keywords}
    turns = []       # finalized turns: each {"messages": [...], "matches": [...]}
    current = None   # the turn being accumulated; matches is [(tf, doc_len), ...]

    # Stream the cache, grouping messages into turns (a turn opens on each user
    # message) and recording, per turn, the BM25 stats of every message that
    # matched. We keep whole matching turns rather than one-off messages so a
    # match anywhere in a tool exchange brings the entire valid sequence with it.
    try:
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                if message["role"] == "system":
                    continue  # the system prompt never belongs to a turn

                # A user message opens a new turn; so does the very first message.
                if message["role"] == "user" or current is None:
                    if current is not None:
                        turns.append(current)
                    current = {"messages": [], "matches": []}
                current["messages"].append(message)

                # content is None on an assistant(tool_calls) message.
                tokens = tokenize(message.get("content") or "")
                num_docs += 1
                total_len += len(tokens)

                # Count how often each query keyword appears in this message.
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

    # Score each matching turn by its best-scoring message, gate on that
    # message's coverage, rank, and return the top_k whole turns.
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
