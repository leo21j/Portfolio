# Results

## `summary.csv`

Every evaluation run, 52 in total, as one table: `run`, `n_examples`, and the
full HotpotQA metric set (`em`, `f1`, `sp_em`, `sp_f1`, `joint_em`, `joint_f1`
plus the precision/recall components). This is the file to read for the
development history.

Caveats worth knowing before quoting any row:

- **Sample sizes vary a lot.** Most early runs are 5, 10, or 25 examples, which
  is far too small to separate configurations. Only four runs reach 250 or
  more, and only one covers the full 7,405-example dev set.
- **The headline number is `v39_final` (n=7,405).** Small-sample runs score
  considerably higher; treating them as the result would be misleading.
- Run labels encode the model and the variant tried, not a version of the
  codebase.

## `metrics/` and `predictions/`

The curated runs kept as raw artifacts:

| File | n | What it is |
|---|---:|---|
| `full_dev_v39` | 7,405 | **The headline result.** Final pipeline, `gemma4:31b`, full dev set. EM 46.8 / F1 59.5 / Joint F1 43.2. Predictions carry the verifier's `verification` and `decision` fields. |
| `full_dev_earlier_qwen` | 7,405 | An earlier full-dev run, superseded by v39. Metrics only. |
| `sample250_sonnet` | 250 | Best-scoring configuration (Claude Sonnet as generator) |
| `sample250_qwen_local` | 250 | Best fully-local configuration |
| `sample250_specialist` | 250 | Specialist mode: large model selects facts, small model answers |
| `sample100_specialist` | 100 | Specialist mode on a smaller slice |
| `sample100_early_baseline` | 100 | An early run, for comparison against the later ones |

`metrics/*.json` are plain JSON metric dicts. `predictions/*.json` map question
ID to the predicted answer, supporting facts, and the raw model response. The
v39 predictions also carry the verifier's `verification` record and the
decider's `decision` record.

Two more v39 artifacts:

| File | What it is |
|---|---|
| `metrics/full_dev_v39_semantic.json` | Output of `scripts/evaluate_custom.py` for v39: containment, fuzzy match, title metrics, answer BERTScore, and SP BERTScore. Everything except the SP BERTScore fields can be regenerated with the committed script. |
| `full_dev_v39_fallback_ids.json` | The 91 question IDs that timed out under `gemma4:31b` and were re-run with `qwen2.5:7b` before being merged into the final predictions. |

`python -m scripts.analyze_v39` computes the per-type, verifier, and
same-question breakdowns quoted in the top-level README.

## Regenerating

```bash
# Produce predictions
python -m pipeline.eval --limit 250

# Score them (predictions first, gold second)
python scripts/hotpot_evaluate_v1.py \
  results/predictions/<run>.json data/hotpot_dev_distractor_v1.json \
  > results/metrics/<run>.json
```

`pipeline/eval.py` can do both steps and writes parsed JSON directly, which is
the safer path — the evaluation script prints its progress to stderr and the
metrics to stdout, so a shell redirect captures only the JSON.
