"""Reproduce the breakdowns reported in the README from committed predictions.

Every number here is computed from a predictions file joined with the HotpotQA
gold data, using the same scoring as the official evaluation script:

  - overall answer / supporting-fact / joint metrics
  - per question type (bridge vs. comparison) and per difficulty level
  - verifier calibration: answer accuracy by `is_supported` and by support score
  - error counts: yes/no flips, empty supporting facts, abstentions
  - same-question comparisons against other runs, plus a first-N vs. rest split

Usage:
    python -m scripts.analyze_v39
    python -m scripts.analyze_v39 --compare results/predictions/sample250_sonnet.json
    python -m scripts.analyze_v39 --json results/analysis_v39.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.hotpot_evaluate_v1 import exact_match_score, f1_score

DEFAULT_PREDICTIONS = Path("results/predictions/full_dev_v39.json")
DEFAULT_GOLD = Path("data/hotpot_dev_distractor_v1.json")
DEFAULT_COMPARE = [
    Path("results/predictions/sample250_sonnet.json"),
    Path("results/predictions/sample250_qwen_local.json"),
    Path("results/predictions/sample250_specialist.json"),
    Path("results/predictions/sample100_specialist.json"),
]
SCORE_BINS = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
# Matches "Cannot determine from evidence", "the distance cannot be determined",
# and model refusals ("I cannot fulfill this request ...").
ABSTAIN_MARKERS = ("cannot",)


def sp_prf(pred: list, gold: list) -> tuple[float, float, float, float]:
    """Supporting-fact (f1, precision, recall, em) over (title, index) pairs."""
    pred_set, gold_set = set(map(tuple, pred)), set(map(tuple, gold))
    tp = len(pred_set & gold_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    em = 1.0 if pred_set == gold_set else 0.0
    return f1, prec, rec, em


def score_example(pred: dict, gold: dict) -> dict:
    """Official-style metrics for one prediction."""
    em = float(exact_match_score(pred["answer"], gold["answer"]))
    f1, prec, rec = f1_score(pred["answer"], gold["answer"])
    sp_f1, sp_prec, sp_rec, sp_em = sp_prf(pred.get("sp", []), gold["supporting_facts"])
    j_prec, j_rec = prec * sp_prec, rec * sp_rec
    joint_f1 = 2 * j_prec * j_rec / (j_prec + j_rec) if j_prec + j_rec else 0.0
    return {"em": em, "f1": f1, "sp_em": sp_em, "sp_f1": sp_f1, "joint_em": em * sp_em, "joint_f1": joint_f1}


def aggregate(scores: list[dict]) -> dict:
    """Mean of each metric, as percentages, plus n."""
    if not scores:
        return {"n": 0}
    out = {"n": len(scores)}
    for key in scores[0]:
        out[key] = round(100 * sum(s[key] for s in scores) / len(scores), 1)
    return out


def score_ids(preds: dict, gold: dict, ids: list[str]) -> dict:
    return aggregate([score_example(preds[i], gold[i]) for i in ids])


def breakdowns(preds: dict, gold: dict) -> dict:
    """Overall, per-type, per-level, verifier, and error-count breakdowns."""
    by_type, by_level = defaultdict(list), defaultdict(list)
    by_supported, by_bin = defaultdict(list), defaultdict(list)
    yes_no = flips = empty_sp = abstain = 0
    modes: dict[str, int] = defaultdict(int)
    overall = []

    for qid, pred in preds.items():
        g = gold[qid]
        s = score_example(pred, g)
        overall.append(s)
        by_type[g.get("type", "")].append(s)
        by_level[g.get("level", "")].append(s)

        verification = pred.get("verification")
        if verification:
            by_supported[str(verification.get("is_supported"))].append(s)
            modes[str((verification.get("metadata") or {}).get("mode"))] += 1
            score = verification.get("support_score")
            if score is not None:
                for lo, hi in SCORE_BINS:
                    if lo <= score <= hi:
                        by_bin[f"{lo:.1f}-{hi:.1f}"].append(s)
                        break

        answer = pred["answer"].strip().lower()
        if g["answer"].lower() in ("yes", "no"):
            yes_no += 1
            if answer in ("yes", "no") and answer != g["answer"].lower():
                flips += 1
        if not pred.get("sp"):
            empty_sp += 1
        if not answer or any(m in answer for m in ABSTAIN_MARKERS):
            abstain += 1

    return {
        "overall": aggregate(overall),
        "by_type": {k: aggregate(v) for k, v in sorted(by_type.items())},
        "by_level": {k: aggregate(v) for k, v in sorted(by_level.items())},
        "verifier": {
            "modes": dict(modes),
            "by_is_supported": {k: aggregate(v) for k, v in sorted(by_supported.items())},
            "by_support_score": {
                f"{lo:.1f}-{hi:.1f}": aggregate(by_bin[f"{lo:.1f}-{hi:.1f}"]) for lo, hi in SCORE_BINS
            },
        },
        "errors": {"yes_no_questions": yes_no, "yes_no_flips": flips, "empty_sp": empty_sp, "abstentions": abstain},
    }


def comparisons(preds: dict, gold: dict, gold_order: list[str], compare: list[Path]) -> dict:
    """Score each comparison run and the main run on exactly the same question IDs."""
    out = {}
    for path in compare:
        if not path.exists():
            continue
        other = json.loads(path.read_text(encoding="utf-8"))
        ids = [i for i in other if i in gold and i in preds]
        out[path.stem] = {"this_run": score_ids(other, gold, ids), "main_run_same_ids": score_ids(preds, gold, ids)}

    n_first = 250
    first = [i for i in gold_order[:n_first] if i in preds]
    rest = [i for i in gold_order[n_first:] if i in preds]
    out[f"main_run_first_{n_first}_vs_rest"] = {
        "first": score_ids(preds, gold, first),
        "rest": score_ids(preds, gold, rest),
    }
    return out


def _row(label: str, m: dict) -> str:
    if not m.get("n"):
        return f"  {label:<28} n=0"
    return (
        f"  {label:<28} n={m['n']:<5} EM={m['em']:5.1f}  F1={m['f1']:5.1f}  "
        f"SP_F1={m['sp_f1']:5.1f}  Joint_F1={m['joint_f1']:5.1f}"
    )


def print_report(report: dict) -> None:
    print("Overall")
    print(_row("all", report["overall"]))
    print("\nBy question type")
    for k, m in report["by_type"].items():
        print(_row(k, m))
    print("\nBy difficulty level")
    for k, m in report["by_level"].items():
        print(_row(k, m))
    v = report["verifier"]
    print(f"\nVerifier (modes: {v['modes']})")
    for k, m in v["by_is_supported"].items():
        print(_row(f"is_supported={k}", m))
    for k, m in v["by_support_score"].items():
        print(_row(f"support_score {k}", m))
    print("\nError counts")
    for k, n in report["errors"].items():
        print(f"  {k:<28} {n}")
    print("\nSame-question comparisons")
    for name, pair in report["comparisons"].items():
        print(f" {name}")
        for k, m in pair.items():
            print(_row(k, m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--compare", type=Path, nargs="*", default=DEFAULT_COMPARE)
    parser.add_argument("--json", type=Path, default=None, help="Also write the report to this JSON file")
    args = parser.parse_args()

    if not args.gold.exists():
        raise SystemExit(f"Gold file not found: {args.gold}. See data/README.md to download it.")

    raw_gold = json.loads(args.gold.read_text(encoding="utf-8"))
    gold = {str(d["_id"]): d for d in raw_gold}
    gold_order = [str(d["_id"]) for d in raw_gold]
    preds = {k: v for k, v in json.loads(args.predictions.read_text(encoding="utf-8")).items() if k in gold}

    report = breakdowns(preds, gold)
    report["comparisons"] = comparisons(preds, gold, gold_order, args.compare)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
