"""Tool: recall -- agent-invoked search over long-term memory (the KV cache).

Nothing is auto-injected; the agent calls this when it needs to remember
something. Keywords run through `search_cache` -- hybrid (BM25 + vector) by
default, or BM25-only when `semantic` is off.
"""

from search import search_cache

# Mirrors KV_CACHE_PATH in main.py.
KV_CACHE_PATH = "global_kv_cache.txt"

SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search your long-term memory -- the persistent record of past "
            "conversations with Mr. Jones -- for relevant earlier turns. You "
            "do NOT see this memory automatically; call this tool whenever you "
            "need to remember something Mr. Jones mentioned before. Pass "
            "space-separated keywords describing what you're trying to recall "
            "(e.g. 'favorite composers music'). Returns the best-matching past "
            "turns, or a note that nothing matched. If you get nothing useful, "
            "call it again with different or synonymous keywords and/or a larger "
            "max_results before concluding the memory isn't there."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "Space-separated keywords to search memory for. Use the "
                        "most distinctive words; on a retry, swap in synonyms or "
                        "related terms."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "How many past turns to return at most (default 3). "
                        "Raise this on a retry to widen the search."
                    ),
                },
                "semantic": {
                    "type": "boolean",
                    "description": (
                        "Default true: hybrid search (keywords plus meaning, so "
                        "it catches paraphrases and synonyms that exact keywords "
                        "miss). Set false for a faster keyword-only search."
                    ),
                },
            },
            "required": ["keywords"],
        },
    },
}


def _render(cluster):
    """Render one retrieved turn (a list of messages) as readable text."""
    lines = []
    for message in cluster:
        text = (message.get("content") or "").strip()
        if not text:
            continue
        who = "Mr. Jones" if message.get("role") == "user" else "You"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def run(keywords, max_results=3, semantic=True):
    miss = ("No matching memories found. Try again with different or synonymous "
            "keywords or a larger max_results.")
    clusters = search_cache(
        keywords, cache_path=KV_CACHE_PATH, top_k=max_results,
        search_mode="hybrid" if semantic else "bm25",
    )
    if not clusters:
        return miss
    blocks = [block for block in (_render(c) for c in clusters) if block]
    return "\n\n---\n\n".join(blocks) if blocks else miss
