# Index cache

Generated retrieval indices. **Nothing here is committed** — the FAISS index
alone is about 740 MB, and all of it is reproducible from the dataset.

## What gets written here

| File | Contents |
|---|---|
| `faiss.index` | `IndexFlatIP` over passage embeddings (~1.1 GB for the 66,635 dev-distractor passages × 4,096 dims) |
| `bm25_tokens.pkl` | Tokenized corpus for `rank_bm25` |
| `passages.json` | Passage titles and sentences, aligned with the index rows |
| `dense_meta.json` | Embedding model name and dimension |
| `config.json` | Retriever settings the index was built with |

## Building it

The index is built automatically on the first run and reused afterwards.
`pipeline/eval.py` checks for all five files and rebuilds if any are missing:

```bash
python -m pipeline.eval --limit 10
```

Expect this to take a while on the first run: every passage has to be embedded
through Ollama. Subsequent runs load the cache.

## Rebuild it when any of these change

- `retriever.embed_model` — embeddings from a different model are not
  comparable, and the dimension may differ
- The dataset or the subset of passages indexed
- `retriever.index_cache_dir` — points at a different cache directory

Delete the directory contents to force a rebuild. The pipeline also rebuilds
automatically when the number of dense vectors doesn't match the number of
passages, which is what an interrupted build leaves behind. The fusion weights
(`alpha*`, `rrf_k`) always come from the current config, not from the cache.

## A note on the pickle

`bm25_tokens.pkl` is written with `pickle`, which executes arbitrary code on
load. That is safe for a file this pipeline generated locally, but it is why
the file is not distributed: never load a `.pkl` from a source you do not
trust. Rebuilding locally is the supported path.

## Why `IndexFlatIP`

Inner product equals cosine similarity on unit vectors, and
`pipeline/embedder.py` normalizes every embedding before indexing. `Flat` means
exhaustive search — exact results, no approximation — which is affordable at
this corpus size and keeps retrieval quality out of the list of things that
could explain a bad score.
