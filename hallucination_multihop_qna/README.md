# Hallucination-Resistant Multi-Hop QA

A retrieval-augmented QA pipeline for [HotpotQA](https://hotpotqa.github.io/)
Every answer carries the specific sentences
it was derived from, those citations are scored against the gold annotations,
and a verifier checks that the answer is supported by them.

Evaluated on the **full 7,405-question distractor dev set**, zero-shot, running
entirely on local models via [Ollama](https://ollama.com/). No fine-tuning and
no cloud API in the final run.

---

## Results

| Metric | Score |
|---|---:|
| Answer Exact Match | 46.8% |
| Answer Token F1 | **59.5%** |
| Supporting-fact EM | 26.9% |
| Supporting-fact F1 | 59.1% |
| Joint EM | 18.8% |
| Joint F1 | 43.2% |
| BERTScore F1 (RoBERTa-large) | 94.3% |

Full 7,405-example distractor dev set (v39, `gemma4:31b` generator), scored
with the official HotpotQA script. Reproduce from the committed predictions:

```bash
python scripts/hotpot_evaluate_v1.py \
  results/predictions/full_dev_v39.json data/hotpot_dev_distractor_v1.json
```

### Compared with the supervised baseline

The HotpotQA paper's BiDAF baseline is a **supervised extractive model trained
on 90,447 labeled examples**. This pipeline uses none of them:

| | Answer F1 | SP F1 | Training examples |
|---|---:|---:|---:|
| BiDAF baseline (supervised) | 58.3% | 66.7% | 90,447 |
| This pipeline (zero-shot) | **59.5%** | 59.1% | **0** |

Answer F1 edges out the supervised baseline with no task-specific training.
Supporting-fact F1 is 7.6 points behind; part of that gap is an index-alignment
artifact discussed [below](#supporting-facts-the-text-is-right-more-often-than-the-index).

### By question type

HotpotQA dev contains two question types. Bridge questions need the pipeline
to find an entity in one passage and follow it to a second; comparison
questions name both entities up front.

| Question type | n | Share | EM | F1 | SP F1 |
|---|---:|---:|---:|---:|---:|
| Bridge | 5,918 | 79.9% | 41.8 | 55.5 | 55.9 |
| Comparison | 1,487 | 20.1% | 66.6 | 75.3 | 71.7 |
| **All** | **7,405** | | **46.8** | **59.5** | **59.1** |

Bridge questions are both the large majority and the hard case, by 25 points
of EM. That is where retrieval work pays off, and where most remaining errors
are. (Every question in the dev split carries the `hard` difficulty label, so a
per-difficulty breakdown adds nothing.)

*Reproduce with `python -m scripts.analyze_v39`.*

---

## How the numbers developed

### Version history

Each column is a real run; all 52 runs are in
[`results/summary.csv`](results/summary.csv).

| | v1 | v7 | v18 | v22 | Earlier full run | **v39** |
|---|---:|---:|---:|---:|---:|---:|
| n | 100 | 100 | 10 | 250 | 7,405 | **7,405** |
| Generator | Mistral 7B | Qwen 2.5 7B | Qwen 2.5 7B | Claude Sonnet | — | **gemma4:31b** |
| Answer EM | 29.0 | 41.0 | 70.0 | 62.0 | 41.6 | 46.8 |
| Answer F1 | 38.2 | 49.3 | 77.5 | 79.1 | 52.8 | 59.5 |
| SP EM | 5.0 | 26.0 | 50.0 | 52.0 | 18.4 | 26.9 |
| SP F1 | 37.3 | 64.5 | 81.0 | 82.7 | 54.2 | 59.1 |
| Joint EM | 1.0 | 18.0 | 30.0 | 37.6 | 12.8 | 18.8 |
| Joint F1 | 19.6 | 37.6 | 63.5 | 68.5 | 35.9 | **43.2** |

- **v1:** basic BM25 + dense hybrid retrieval; the model generates
  `(title, sentence_index)` citations itself.
- **v7:** Qwen 2.5 7B and the `BAAI/bge-reranker-v2-m3` cross-encoder.
- **v18:** title-based hop-2 queries and LLM query decomposition.
- **v22:** Claude Sonnet via API, with prompt rules that forbid abstaining.
- **v39:** final configuration, `gemma4:31b` locally, full dev set.

The two full-dev runs are the only directly comparable pair at scale: v39
improves on the earlier full run by **+7.3 Joint F1** and **+5.2 EM**.

### Small samples overstate, and the generator matters

The mid-range versions look better than v39, but they were run on 10–250
questions. All the sample runs used the *first N* questions of the dev set,
which makes a clean comparison possible: score v39 on exactly the same
questions.

| Same 250 questions | EM | F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|---:|
| v22, Claude Sonnet (API) | **62.0** | **79.1** | **82.7** | **68.5** |
| v39, gemma4:31b (local) | 50.0 | 64.9 | 66.6 | 51.3 |
| v26, Qwen 2.5 7B (local) | 53.6 | 67.5 | 65.9 | 48.1 |
| v25, specialist mode | 44.0 | 53.3 | 63.2 | 45.3 |

Two things follow:

- **Sample size explains only part of the drop.** v39 scores 51.3 Joint F1 on
  the first 250 questions and 42.9 on the remaining 7,155, so the first slice
  is easier. That accounts for about 8 points.
- **The generator explains more.** On identical questions, the Claude Sonnet
  run is 17 points of Joint F1 ahead of the final local run. (The two runs also
  differ in pipeline version, so this isn't a pure model ablation.) The final
  run traded that accuracy for running locally with no rate limits across
  7,405 questions. Among the local options, `gemma4:31b` beats Qwen 2.5 7B on
  the same slice by 3.2 Joint F1.

### Retrieval changes moved the early numbers the most

Holding the generator fixed at Qwen 2.5 7B, retrieval work between v7 and v18
(fuzzy title matching, LLM query decomposition, a wider candidate pool) raised
Joint F1 from 37.6 to 63.5. The caveat is that v7 was measured on 100
questions and v18 on 10, so the size of that jump is directional, not a
precise effect.

### Stage-by-stage recall

The pipeline was instrumented to measure what fraction of gold supporting
facts survive each stage.

| Stage | v18 (n=100) | v22 (n=10) | v22 (n=250) |
|---|---:|---:|---:|
| Retrieve | 88.0% | 100.0% | 97.6% |
| Rerank | 87.0% | 100.0% | 95.2% |
| Sentence selection | 87.3% | 100.0% | 95.1% |
| Prediction | 59.4% | 91.3% | 82.6% |

Once retrieval was fixed, the loss moved downstream: at n=250, retrieval and
reranking keep about 95% of the gold facts, but the model's final selection
keeps 82.6%. The model sees the right facts and doesn't always cite all of
them. This instrumentation wasn't collected for the multi-session v39 run.

*From the project write-up; the logging is in `pipeline/eval.py`.*

---

## Where the answers go wrong

### Exact Match understates correct answers

EM gives zero credit for any character-level difference after normalization,
and a generative model phrases things its own way:

| Gold | Predicted | EM | BERTScore F1 |
|---|---|---:|---:|
| 3,677 seated | 3,677 | 0 | 97% |
| from around 1520 | around 1520 | 0 | 99% |
| NCAA Div. I FBS football | NCAA Div. I FBS | 0 | 98% |
| director | film director | 0 | 96% |
| Canary Islands, Spain | Canary Islands (Tenerife…) | 0 | 91% |

Metrics ordered from strictest to most lenient:

| Metric | Score | What it measures |
|---|---:|---|
| Exact Match | 46.8% | Exact normalized string equality |
| Token F1 | 59.5% | Partial token overlap (official) |
| Containment | 60.8% | Prediction ⊆ gold, or gold ⊆ prediction |
| Fuzzy Match | 74.3% | Mean RapidFuzz `token_set_ratio` (0–100, rescaled) |
| BERTScore P / R / F1 | 94.5 / 94.1 / 94.3% | Contextual embedding similarity |

### How to read the BERTScore number

BERTScore needs calibration before it means anything. Across all 7,405
predictions its **minimum is about 0.74**: RoBERTa embeddings keep residual
overlap between any two English strings, so a completely wrong answer still
scores in the mid-0.70s.

| BERTScore F1 | Gold | Predicted | Pattern |
|---:|---|---|---|
| 1.000 | Animorphs | Animorphs | Exact match |
| 0.952 | Adeline Virginia Woolf | Virginia Woolf | Common alias |
| 0.920 | yes | no | Boolean flip |
| 0.902 | Sonic | Sonic the Hedgehog | Over-specified |
| 0.857 | 230 | more than 230 | Approximation |
| 0.770 | The Big Bang Theory | iCarly | Wrong entity |
| 0.756 | between 7,500 and 40,000 | 0 | Completely wrong |

So the 94.3% mean shouldn't be read on a 0–100 scale. The distribution is more
informative:

| Threshold | Answer BERTScore | Supporting-fact BERTScore |
|---|---:|---:|
| ≥ 0.90 | 71.0% | 72.9% |
| ≥ 0.85 | 89.8% | 88.0% |
| ≥ 0.80 | 99.1% | 99.0% |
| < 0.80 | 0.9% | 1.0% |

At its strictest threshold (≥ 0.999) BERTScore covers 43.9% of predictions,
slightly *below* EM, so it isn't inflating results at the top end. Boolean
flips score 0.92, higher than many wrong entities, because *yes* and *no* are
nearly interchangeable in RoBERTa's embedding space. Excluding the 458 yes/no
questions moves the mean from 0.9426 to 0.9395.

The useful conclusion is narrow. Containment, fuzzy match, and BERTScore are
three independent signals, and all three place answer quality well above what
EM reports. BERTScore on its own shouldn't be read as "94% correct".

### Supporting facts: the text is right more often than the index

SP EM (26.9%) needs an exact `(title, sentence_index)` match, so citing the
right sentence at an index off by one scores zero. Comparing the cited text
instead gives:

| Supporting-fact metric | Score |
|---|---:|
| SP EM (official) | 26.9% |
| SP F1 (official) | 59.1% |
| Title F1 (right source document) | 68.9% |
| Title fuzzy match | 77.7% |
| SP BERTScore F1 | 93.1% |

The pipeline usually finds the right source documents. Sentence-level index
alignment is where it loses most of the official SP score.

### Error patterns

Of the 3,942 EM misses, 1,252 are close paraphrases (`token_set_ratio` ≥ 75)
and 2,690 are genuinely different answers. The most common patterns are:

- **Over-specification:** nationality, occupation, or context added to a bare
  entity ("film director" for "director"). This accounts for much of the
  close-paraphrase group.
- **Units and qualifiers** that the gold answer omits ("3,677 seated" vs.
  "3,677", "more than 230").
- **Multiple entities** given when gold expects one.

Other counts from the committed predictions:

| | Count |
|---|---:|
| Yes/no questions | 458 |
| — answered with the opposite boolean | 57 |
| Predictions with no supporting facts | 140 |
| Abstentions or refusals ("cannot determine…"), 2 of them safety refusals | 20 |

Verbosity also shows up as an F1–EM gap. On the same 250 questions, the Claude
Sonnet run has a 17.1-point gap between F1 and EM; the local Qwen 2.5 7B run
has 13.9. The larger model gives more complete, longer answers, and EM
penalizes them. For EM-scored benchmarks, answer style matters about as much as
reasoning ability.

---

## The verifier: a useful signal of reliability

Every v39 prediction carries a `verification` record with a support score in
[0, 1] and an `is_supported` flag. Broken down by those fields, they separate
right from wrong answers:

| Verifier output | n | Answer EM | Answer F1 |
|---|---:|---:|---:|
| Supported | 6,640 | 49.9% | 62.9% |
| Not supported | 765 | 19.5% | 29.8% |

| Support score | n | Answer EM |
|---|---:|---:|
| 0.8 – 1.0 | 4,843 | 51.2% |
| 0.6 – 0.8 | 1,763 | 47.1% |
| 0.4 – 0.6 | 197 | 17.8% |
| 0.2 – 0.4 | 77 | 13.0% |
| 0.0 – 0.2 | 525 | 20.0% |

Answers the verifier flags are **2.6× less likely to be correct**. That makes
the flag usable for routing: send the roughly 10% of flagged answers to review,
or abstain on them, and the rest are noticeably more reliable.

> **Verifier mode.** All 7,405 v39 records report `"mode": "overlap"`: support
> was scored by lexical claim/evidence overlap, with a separate evidence-quality
> score for yes/no answers. The module also has an `nli` mode
> (`roberta-large-mnli`) and a `qa` mode; the run didn't use them. Retries and
> abstention were both off (`retry_on_failure: false`,
> `abstain_on_unsupported: false`), so the verifier labeled answers but never
> changed one.

*Reproduce with `python -m scripts.analyze_v39`.*

---

## How it works

```
question → hybrid retrieval → rerank → citation selection → generate → verify → decide
            BM25 + FAISS,     cross-    numbered facts,      gemma4:31b  support  accept /
            fuzzy titles,     encoder   evidence-first                   score    regenerate /
            2 hops, RRF                                                           abstain
```

| Stage | Code | Key settings (v39) |
|---|---|---|
| Hybrid retrieval | `pipeline/indexer.py` | BM25 + FAISS, RRF `k=20`, `alpha` 0.6 bridge / 0.5 comparison, 200-candidate pool |
| Two-hop + fuzzy titles | `pipeline/indexer.py` | 20 passages per hop, 5 bridge entities, `token_set_ratio` ≥ 70 |
| Rerank + sentence selection | `pipeline/reranker.py` | `bge-reranker-v2-m3`, top 20 → 5 passages, sentence score ≥ 0.25 |
| Citation-selection prompt | `pipeline/prompt_builder.py`, `configs/prompts.yaml` | Up to 8 numbered facts, separate yes/no prompt |
| Generation + parsing | `pipeline/generator.py` | `gemma4:31b`, temperature 0.1–0.2, 3-stage parse + repair call |
| Verifier | `pipeline/verifier.py` | Overlap mode, support threshold 0.55 |
| Decider | `pipeline/decider.py` | Confidence = 0.7 × support + 0.3 × reranker; retries and abstention off |
| Orchestration | `pipeline/eval.py` | 3 parallel workers, checkpoint every 25 examples |

All settings live in [`configs/default.yaml`](configs/default.yaml).

### Hybrid retrieval with Reciprocal Rank Fusion

BM25 and dense retrieval fail differently. BM25 handles rare exact tokens, like
a film title or a year, but misses paraphrase. Dense retrieval handles
paraphrase but blurs rare tokens. Multi-hop questions need both.

The scores can't be summed directly: BM25 runs 0–25 while cosine similarities
sit in [-1, 1]. RRF discards the scores and fuses **ranks** instead:

```
RRF(d) = Σ  1 / (k + rank_r(d))      for r ∈ {BM25, dense},  k = 20
```

A document's contribution depends only on where each retriever placed it, so
neither scale can dominate. `k=20` rather than the conventional 60 widens the
gap between top ranks. A dense/sparse weight `alpha` shifts by question type:
bridge questions favor semantics, while comparison questions name both entities
explicitly, so keywords carry more weight. Dense embeddings come from
`qwen3-embedding:8b` (4,096 dimensions) in a FAISS index over 66,635 unique
passages.

### Two-hop retrieval with fuzzy title chaining

Bridge questions name an entity that only appears in the *second* passage
("the band that performed a promo for a movie starring…"). One retrieval pass
can't reach it.

Hop 1 retrieves on the question. An LLM then reads the top hop-1 passages and
lists the bridge entities that link them to the answer; passage titles are
always added. If that call fails, a regex extractor takes over (titles,
multi-word capitalized phrases, acronyms, parenthetical context). Hop 2 runs
two strategies on those entities, for every question:
- **Fuzzy title lookup** with RapidFuzz `token_set_ratio` and a threshold of
  70. It's order-invariant and handles "US" vs. "United States", plurals, and
  disambiguators.
- **Query reformulation:** the question plus the entities, sent back through
  hybrid retrieval.

Results from both hops are deduplicated and merged into 20 candidates.

### Reranking and sentence selection

The 20 candidates are rescored by `BAAI/bge-reranker-v2-m3`, a cross-encoder
that reads query and passage together, and the top 5 are kept. The same
cross-encoder then scores each sentence of the top 3 passages against the
question (squashed to [0, 1] with a sigmoid). The top two passages are
guaranteed 2 sentences and the third 1, so a bridge fact isn't dropped for
scoring slightly low; more are added while they score ≥ 0.25, up to 5 per
passage.

### Citation selection instead of citation generation

Asking an LLM to emit `(title, sentence_index)` pairs invites made-up
references. Instead, every selected sentence is pre-numbered as a fact, and the
model **picks numbers**. Numbers that don't correspond to a retrieved fact are
discarded and logged, so every citation points at evidence that was actually
retrieved. The prompt asks for a short reasoning field first, then the fact
numbers and the answer. Up to 3 facts come from each of the top two passages
and 2 from the third. Questions that start like a yes/no question ("Were…",
"Is…") get a dedicated prompt that requires a boolean answer.

### Robust output parsing

LLM JSON is often broken. Parsing runs in three stages: strict JSON, then
targeted regular expressions for the `answer` and `supporting_fact_numbers`
fields in malformed JSON, then a free-form fallback. If both fields still can't
be recovered, a repair call sends the raw output back to the model with a
reformatting instruction, which recovered about 15% of failed parses. Answers
are then normalized: preambles and trailing parentheticals are stripped, yes/no
is lowercased, trailing periods are removed, and "how many" answers without a
digit are replaced by the precise number found in the evidence.

### Verify, then decide

The verifier splits the answer into claims and scores each against the
evidence sentences ([results above](#the-verifier-a-useful-signal-of-reliability)).
The decider combines that score with the top reranker score into a confidence
value. It can then accept the answer, regenerate with feedback
(`retry_on_failure`), or abstain (`abstain_on_unsupported`). For benchmark
scoring both options are off, because an abstention always scores zero; for
uses where a wrong answer costs more than no answer, turn abstention on.

### A worked example

> **Q:** *What year did Guns N' Roses perform a promo for a movie starring
> Arnold Schwarzenegger as a former New York Police detective?*

1. **Hop 1** retrieves *End of Days (film)*: "Jericho Cane is a former NYPD
   detective played by Arnold Schwarzenegger." The bridge entity `End of Days`
   is extracted from the title.
2. **Hop 2** fuzzy-matches that title and retrieves *Oh My God (GN'R song)*:
   "Released November 1999 as a promotional single for the End of Days
   soundtrack."
3. **Reasoning:** "Schwarzenegger played a former NYPD detective in End of Days
   (Facts 0, 1). Guns N' Roses released 'Oh My God' as a promo for End of Days
   in November 1999 (Facts 5, 6)."
4. **Answer:** `1999`, an exact match. **Verifier:** support 79.6%, supported.

### Why no fine-tuning

Fine-tuning on HotpotQA means training on `(question, gold_context, answer)`
triples, where the gold context is the already-identified supporting passages.
That gives the model the very thing the pipeline exists to find. A model
trained that way learns to answer *given* the evidence and never learns to
retrieve it from a pool of distractors. The pipeline is zero-shot by design.

### Model survey

Ten generators were compared on the same 10 questions before choosing one.
With n=10, one question is worth 10 points of EM, so treat this table as a
screening pass, not a ranking.

| Model | Size | EM | SP F1 | Joint F1 | sec/item |
|---|---|---:|---:|---:|---:|
| Claude Sonnet | API | 70 | 94.9 | 83.9 | ~8 |
| Qwen 2.5 7B | 4.7 GB | 70 | 81.0 | 63.5 | ~30 |
| Specialist (Sonnet + Qwen) | hybrid | 57 | 81.7 | 59.2 | ~22 |
| Claude Haiku | API | 50 | 92.0 | 46.0 | ~5 |
| Qwen 3.5 9B | 6.6 GB | 50 | 53.5 | 45.5 | ~130 |
| Gemma 2 9B | 5.5 GB | 50 | 83.7 | 43.0 | ~40 |
| Qwen 3.5 4B | 3.4 GB | 50 | 52.0 | 42.0 | ~90 |
| Llama 3.1 8B | 4.7 GB | 50 | 53.5 | 39.0 | ~30 |
| DeepSeek-R1 8B | 5.2 GB | 40 | 79.3 | 36.0 | ~130 |
| Phi-3 3.8B | 2.2 GB | 30 | 71.8 | 33.8 | ~25 |
| ChatQA 8B | 4.7 GB | 0 | 39.4 | 0.0 | ~33 |

Notable results:
- ChatQA is built specifically for RAG and still produced unusable output.
- The reasoning models (DeepSeek-R1, Qwen 3.5) took 3–4× longer with no
  quality gain.
- Claude Haiku over-abstained, refusing questions the 7B local models answered
  correctly.

`gemma4:31b` (20 GB) was chosen for the full run: it follows instructions and
JSON formats reliably, runs locally, and has no rate limits across 7,405
examples.

**Specialist mode** (a large model selects facts, a small model writes the
answer) was built and then dropped. It ran answer generation locally at about
half the API cost, but joint reasoning beat split reasoning. On the first 100
questions, Sonnet alone scored 67.0 EM and 73.1 Joint F1, compared with 57.0
and 59.2 for specialist mode. When one model does both,
evidence selection and answer extraction share a single chain of thought. When
they're split, the answer model has to work out the reasoning again. The
negative result is kept in `results/summary.csv`.

---

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/). Model weights are
large, so budget about 25 GB of disk.

```bash
git clone https://github.com/Leops21/hallucination_resistant_multihop_qna.git
cd hallucination_resistant_multihop_qna

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen3-embedding:8b     # dense retrieval
ollama pull gemma4:31b             # generator + bridge-entity extraction
ollama pull qwen2.5:7b             # optional: fallback generator for timeouts
```

Download the dataset (see [`data/README.md`](data/README.md)):

```bash
cd data && curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json && cd ..
```

Then run it. The first run builds the retrieval index, which takes a while
because every passage is embedded; later runs load the cache. See
[`index_cache_global/README.md`](index_cache_global/README.md).

```bash
python -m scripts.test_pipeline          # one question, end to end

# Evaluate a slice and score it with the official script
python -m pipeline.eval --limit 100 \
  --output results/predictions/run.json \
  --eval data/hotpot_dev_distractor_v1.json --metrics results/metrics/run.json
```

Everything is configured in [`configs/default.yaml`](configs/default.yaml):
models, alpha weights, thresholds, verifier mode, and timeouts. Prompts live in
[`configs/prompts.yaml`](configs/prompts.yaml). The config comments include
tuning notes learned from real runs. For example, going from 5 to 7 evidence
passages dropped answer EM by about a third because the extra context crowded
out the relevant facts.

### Reproducing the reported numbers

None of these need Ollama; they work from the committed predictions.

```bash
# Official EM / F1 / SP / Joint
python scripts/hotpot_evaluate_v1.py \
  results/predictions/full_dev_v39.json data/hotpot_dev_distractor_v1.json

# Per-type results, same-question comparisons, verifier breakdown, error counts
python -m scripts.analyze_v39

# Containment, fuzzy match, title metrics, BERTScore
# (BERTScore downloads roberta-large; add --no-bert to skip it)
python -m scripts.evaluate_custom \
  --predictions results/predictions/full_dev_v39.json --gold data/hotpot_dev_distractor_v1.json
```

### Scale of the full run

The 7,405-example evaluation ran on Google Colab Pro+ with an A100 GPU and took
about **7 days of cumulative compute** (997 compute units), at 40–100 seconds
per question. [`scripts/colab_run.py`](scripts/colab_run.py) handles the setup:
it installs Ollama, caches model weights on Google Drive so they survive the
24-hour session limit, and runs the evaluation. Predictions are checkpointed
every 25 examples, and `--resume` continues from the last checkpoint after a
disconnect.

Of the 7,405 questions, 91 timed out under `gemma4:31b`. Their IDs are in
[`results/full_dev_v39_fallback_ids.json`](results/full_dev_v39_fallback_ids.json);
they were re-run with `qwen2.5:7b` as a faster fallback and merged back in, so
all 7,405 completed:

```bash
python -m pipeline.eval --ids results/full_dev_v39_fallback_ids.json \
  --generator-model qwen2.5:7b --output results/predictions/fallback.json
```

---

## Repository structure

```text
.
├── configs/
│   ├── default.yaml            # All pipeline parameters
│   └── prompts.yaml            # Prompt templates for every stage
├── data/                       # HotpotQA JSON (downloaded, not committed)
├── index_cache_global/         # FAISS + BM25 indices (generated, not committed)
├── notebooks/
│   └── baseline_training.ipynb # Exploratory baseline work
├── pipeline/
│   ├── data_loader.py          # HotpotQA → Passage / Context / HotpotQAExample
│   ├── embedder.py             # Ollama embeddings, L2-normalized
│   ├── indexer.py              # Hybrid BM25 + FAISS retriever, RRF, 2-hop
│   ├── reranker.py             # Cross-encoder rerank + sentence selection
│   ├── prompt_builder.py       # Evidence-first prompts, citation selection
│   ├── generator.py            # Ollama generation, 3-stage response parsing
│   ├── verifier.py             # Claim-level support scoring (overlap / NLI / QA)
│   ├── decider.py              # Confidence, optional retry, optional abstention
│   └── eval.py                 # End-to-end run, checkpointing, scoring
├── results/
│   ├── summary.csv             # All 52 runs
│   ├── full_dev_v39_fallback_ids.json  # The 91 questions re-run with qwen2.5:7b
│   ├── metrics/                # Curated metric JSONs (+ v39 semantic metrics)
│   └── predictions/            # Matching prediction files
├── scripts/
│   ├── config.py               # YAML → typed dataclasses, with validation
│   ├── colab_run.py            # One-command Colab setup + run (used for v39)
│   ├── evaluate_custom.py      # Containment, fuzzy, title, BERTScore metrics
│   ├── analyze_v39.py          # Reproduces the README breakdowns
│   ├── analyze_predictions.py  # Error analysis over predictions
│   ├── hotpot_evaluate_v1.py   # Official HotpotQA scorer (Apache-2.0, adapted)
│   ├── test_pipeline.py        # Single-example smoke run
│   └── logger.py               # Colored terminal logger
└── tests/                      # Unit tests for the pure logic
```

### What can be reproduced from this repository

| Result | Source |
|---|---|
| EM, F1, SP, and Joint metrics for every run | Committed predictions + `scripts/hotpot_evaluate_v1.py` |
| Per-question-type results, same-question comparisons, verifier breakdown, error counts | `python -m scripts.analyze_v39` |
| Containment, fuzzy match, title metrics, close/wrong error split | `python -m scripts.evaluate_custom --no-bert` (matches the committed file exactly) |
| Answer BERTScore | `python -m scripts.evaluate_custom` (needs roberta-large) |
| SP BERTScore, stage-by-stage recall | Project write-up only: the SP BERTScore code was never committed, and recall was logged for v18/v22 runs whose logs aren't kept |


## Built with

**Retrieval:** FAISS (`IndexFlatIP`), BM25, RapidFuzz, `BAAI/bge-reranker-v2-m3`
**Generation:** Ollama: `gemma4:31b`, `qwen3-embedding:8b`, `qwen2.5:7b`
**Verification:** claim-level support scoring (overlap; optional `roberta-large-mnli` NLI)
**Evaluation:** official HotpotQA scorer, BERTScore (RoBERTa-large), RapidFuzz
**Core:** Python 3.10, NumPy, PyTorch
**Tooling:** pytest, ruff, GitHub Actions




