# Design comments & notes

Running notes on how the memory/search layer works and where it's headed.

## Current search: streaming keyword scan (search.py)

`search_cache` reads `global_kv_cache.txt` **one line at a time** and keeps only
the best-scoring match (plus a little trailing context) in memory. This means
memory use is **constant** no matter how large the cache grows — we never build
a full in-memory list of the whole history.

What streaming does and does NOT fix:

- ✅ **Memory:** O(1) RAM instead of O(n) — the whole cache is never resident.
- ❌ **I/O / scan cost:** we still *read* every line on every query, so search is
  still O(n) in the number of stored messages. As history grows, each query
  does more disk reads and more scoring work.

## Why we might need SQLite (the next step)

The streaming fix solves *how much memory* we use, but not *how much we read*.
To stop re-scanning the entire cache on every query, we need an **on-disk index**
— a structure that lets us jump straight to the relevant messages without
touching the rest of the file.

The cleanest zero-infrastructure option is **SQLite with the FTS5 full-text
extension**, both built into the Python standard library (`import sqlite3`):

- **Stays on disk, no server.** SQLite is a single file — the same "no
  infrastructure" footprint as our `.txt`, but with real indexing. Queries read
  only the relevant disk *pages* (via B-tree indexes + the OS page cache), not
  the whole file.
- **Inverted index via FTS5.** Full-text search maps `keyword -> matching rows`
  so lookups become **sub-linear** instead of a full scan. This is exactly the
  problem our hand-rolled keyword scan has.
- **Same data model.** We keep exact role-tagged rows (`role`, `content`,
  `timestamp`), so nothing about how we store/inject messages changes.
- **Future-proofing.** It's a natural base for semantic search later (e.g.
  `sqlite-vec` for on-disk vector search), without re-architecting.

This is essentially what Letta's *recall memory* is under the hood — a
relational DB with optional full-text/vector search. SQLite is just the
embedded, serverless version of the same idea.

**When to pull the trigger:** once the cache is large enough that the O(n)
per-query scan becomes a noticeable cost. Until then, the streaming scan is
simpler and perfectly adequate.

## Search ranking: matching Letta's recall memory

Letta's full ("hybrid") recall search has three parts. We've implemented the
first; the other two are the roadmap.

1. **Lexical channel — BM25.** ✅ Done (`bm25_score` in search.py). BM25 is the
   ranking function Letta's full-text channel uses. Vs. our old keyword-overlap
   fraction it rewards rare terms (IDF), saturates repeated terms (k1), and
   normalizes for message length (b). No dependencies — pure stdlib.

   Tradeoff we accepted: BM25 needs corpus stats (IDF), so `search_cache` now
   holds the *matching* messages until the streaming pass finishes. Memory
   scales with the number of matches (a posting list), not the whole cache.

2. **Semantic channel — vector ANN.** ⬜ Next. Letta ranks messages a second way
   with `("vector", "ANN", query_embedding)` over embeddings, catching meaning
   and synonyms that keywords miss. This needs an embedding model + a vector
   store. The blocker is a dependency/provider decision (DeepSeek has no
   embeddings endpoint; options are OpenAI embeddings, a local
   sentence-transformers model, or `sqlite-vec` for an on-disk index).

3. **Fusion — Reciprocal Rank Fusion (RRF).** ⬜ After the vector channel exists.
   Letta merges the two ranked lists by *rank*, not raw score:

       score(doc) = vector_weight / (k + vector_rank)
                  + fts_weight    / (k + fts_rank)

   with `k = 60` (Cormack et al. 2009), ranks 1-based, and a doc missing from a
   list simply contributes nothing from that channel. Sort by combined score,
   take top_k. Default weights are 0.5 / 0.5. RRF is robust because it never has
   to reconcile BM25's unbounded scores with cosine similarities — it only
   compares positions.

So our `search_cache` already returns a BM25-ranked top_k (the FTS half of
Letta's hybrid). Adding the vector channel + RRF is what would make it a true
hybrid recall search.
