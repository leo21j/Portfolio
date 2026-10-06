from scripts.analyze_v39 import aggregate, breakdowns, score_example, sp_prf

GOLD = {
    "a": {"answer": "yes", "supporting_facts": [["X", 0], ["Y", 1]], "type": "comparison", "level": "hard"},
    "b": {
        "answer": "Chief of Protocol",
        "supporting_facts": [["Shirley Temple", 1]],
        "type": "bridge",
        "level": "hard",
    },
}


def _pred(answer, sp, supported=True, score=0.9):
    return {
        "answer": answer,
        "sp": sp,
        "verification": {"is_supported": supported, "support_score": score, "metadata": {"mode": "overlap"}},
    }


def test_sp_prf_partial_overlap():
    f1, prec, rec, em = sp_prf([["X", 0], ["Z", 2]], [["X", 0], ["Y", 1]])
    assert (prec, rec, f1, em) == (0.5, 0.5, 0.5, 0.0)


def test_score_example_exact():
    s = score_example(_pred("yes", [["X", 0], ["Y", 1]]), GOLD["a"])
    assert s["em"] == s["sp_em"] == s["joint_em"] == 1.0
    assert s["joint_f1"] == 1.0


def test_yes_no_mismatch_scores_zero_f1():
    s = score_example(_pred("no", []), GOLD["a"])
    assert s["em"] == 0.0 and s["f1"] == 0.0


def test_aggregate_percentages():
    m = aggregate([{"em": 1.0}, {"em": 0.0}])
    assert m == {"n": 2, "em": 50.0}


def test_breakdowns_counts():
    preds = {
        "a": _pred("no", [["X", 0]], supported=False, score=0.1),
        "b": _pred("Chief of Protocol", [["Shirley Temple", 1]]),
    }
    r = breakdowns(preds, GOLD)
    assert r["overall"]["n"] == 2
    assert r["by_type"]["bridge"]["em"] == 100.0
    assert r["errors"]["yes_no_flips"] == 1
    assert r["verifier"]["modes"] == {"overlap": 2}
    assert r["verifier"]["by_is_supported"]["False"]["em"] == 0.0
    assert r["verifier"]["by_support_score"]["0.0-0.2"]["n"] == 1
