"""Keyword-based search over the global KV cache.

The flow this supports: take the user's input, reduce it to keywords, then scan
the persisted conversation cache for the message that best matches those
keywords. The best-matching chunk (the message plus its reply) is returned so it
can be prepended to the live conversation before the new user prompt.
"""

import json
import re

# Common words that carry little meaning and would pollute keyword matching.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "for", "with", "as", "at",
    "by", "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "me", "my", "your", "do", "does", "did", "can",
    "could", "would", "should", "what", "which", "who", "how", "when", "where",
}


def extract_keywords(text):
    """Break text into a set of lowercase keywords, dropping stopwords/noise."""
    # Pull out alphanumeric word tokens and lowercase them.
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Keep meaningful words: not a stopword and longer than two characters.
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def load_cache(path):
    """Read the JSONL cache file into a list of {"role", "content"} dicts."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def score(keywords, content):
    """Fraction of the query keywords that appear in a cache entry (0..1)."""
    if not keywords:
        return 0.0
    content_words = set(re.findall(r"[a-z0-9]+", content.lower()))
    matches = sum(1 for k in keywords if k in content_words)
    return matches / len(keywords)


def search_cache(user_input, cache_path="global_kv_cache.txt",
                 threshold=0.3, context=1):
    """Find the best-matching chunk of the cache for the given user input.

    Returns a list of message dicts (the matched entry plus `context` following
    entries, e.g. its assistant reply), or an empty list if nothing matches well.
    """
    keywords = extract_keywords(user_input)
    cache = load_cache(cache_path)

    # Score every non-system entry and remember the best one.
    best_score = 0.0
    best_index = -1
    for i, message in enumerate(cache):
        if message["role"] == "system":
            continue  # the system prompt is always present; skip it
        s = score(keywords, message["content"])
        if s > best_score:
            best_score = s
            best_index = i

    # Nothing matched, or the best match was too weak to be useful.
    if best_index == -1 or best_score < threshold:
        return []

    # Return the matched entry plus a few following entries for context
    # (so a matched question also brings back its answer).
    end = min(len(cache), best_index + context + 1)
    return cache[best_index:end]
