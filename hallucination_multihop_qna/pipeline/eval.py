"""End-to-end evaluation on HotpotQA.

Pipeline: DataLoader → HybridRetriever → Reranker → PromptBuilder → Generator
→ Verifier → Decider. Writes predictions keyed by question ID, logs
stage-by-stage recall of the gold supporting facts, and optionally scores the
run with the official HotpotQA script.

    python -m pipeline.eval --limit 100 --eval data/hotpot_dev_distractor_v1.json
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from pipeline.data_loader import HotpotQALoader
from pipeline.decider import Decider
from pipeline.generator import Generator
from pipeline.indexer import HybridRetriever
from pipeline.prompt_builder import PromptBuilder
from pipeline.reranker import Reranker
from pipeline.verifier import Verifier
from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("eval")


def build_pipeline(cfg):
    log.info("Loading data...")
    loader = HotpotQALoader(cfg.data.dev_distractor)

    # Load ALL examples to build the global index database
    all_examples = loader.load(limit=None)

    # Collect unique passages from the loaded examples
    seen = set()
    all_passages = []
    for ex in all_examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                all_passages.append(ctx)

    # Slice out examples for limit evaluation
    if cfg.eval.limit:
        eval_examples = all_examples[: cfg.eval.limit]
        log.info(f"Sliced out {cfg.eval.limit} examples for LLM evaluation.")
    else:
        eval_examples = all_examples
        log.info("No limit set, using all examples for LLM evaluation.")

    log.info("Initializing retriever...")
    retriever = HybridRetriever.from_config(cfg)

    cache_path = Path(cfg.retriever.index_cache_dir)
    required_files = ["faiss.index", "bm25_tokens.pkl", "dense_meta.json", "passages.json", "config.json"]

    loaded = False
    if all((cache_path / f).exists() for f in required_files):
        log.info(f"Loading cached global index from {cache_path}...")
        try:
            retriever.load(cache_path)
            loaded = True
        except ValueError as e:
            log.warning(f"{e} Rebuilding.")
    if not loaded:
        log.info(f"Building global retriever with index {len(all_passages)} passages...")
        retriever.index(all_passages, show_progress=True)
        log.info(f"Saving global index to {cache_path}...")
        retriever.save(cache_path)

    log.info("Loading reranker...")
    reranker = Reranker.from_config(cfg)

    log.info("Initializing prompt builder...")
    pb = PromptBuilder.from_config(cfg)

    log.info("Initializing generator...")
    gen = Generator.from_config(cfg)

    verifier = None
    if getattr(cfg.verifier, "enabled", False):
        log.info("Initializing verifier...")
        verifier = Verifier.from_config(cfg)

    log.info("Initializing decider...")
    decider = Decider.from_config(cfg)

    return eval_examples, retriever, reranker, pb, gen, verifier, decider, cfg


def run_pipeline(
    examples,
    retriever,
    reranker,
    pb,
    gen,
    verifier,
    decider,
    cfg,
    limit: int | None = None,
    output_path: Path | None = None,
    resume: bool = True,
):
    predictions = {}

    if limit:
        examples = examples[:limit]

    # Resume from checkpoint if available
    checkpoint_path = Path(str(output_path) + ".checkpoint") if output_path else None
    if resume and checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                predictions = json.load(f)
            done_ids = set(predictions.keys())
            before = len(examples)
            examples = [ex for ex in examples if ex.id not in done_ids]
            log.info(f"Resumed from checkpoint: {len(done_ids)} already done, {len(examples)} remaining (of {before})")
        except Exception as e:
            log.warning(f"Failed to load checkpoint, starting fresh: {e}")
            predictions = {}

    checkpoint_interval = cfg.eval.checkpoint_interval

    # Stage-by-stage recall diagnostics
    diag = {"retrieval": [], "rerank": [], "sentence": [], "prediction": []}
    verification_stats = {"scores": [], "supported": 0, "unsupported": 0, "skipped": 0}

    reranker_top_k = cfg.reranker.top_k
    retrieve_top_k = cfg.retriever.top_k
    top_k_per_hop = cfg.retriever.multihop.top_k_per_hop

    def process_example(ex):
        try:
            # Gold supporting facts for this example
            gold_titles = {sf.title for sf in ex.supporting_facts}
            gold_facts = {(sf.title, sf.sentence_index) for sf in ex.supporting_facts}

            retrieved = retriever.retrieve_multihop(
                ex.question, top_k=retrieve_top_k, top_k_per_hop=top_k_per_hop, question_type=ex.question_type
            )

            if not retrieved:
                return (
                    ex.id,
                    {
                        "answer": "Cannot answer based on the provided evidence.",
                        "sp": [],
                    },
                    {"retrieval": 0.0, "rerank": 0.0, "sentence": 0.0, "prediction": 0.0},
                )

            # Diagnostic: retrieval recall
            retrieved_titles = {r.passage.title for r in retrieved}
            retrieval_recall = len(gold_titles & retrieved_titles) / len(gold_titles) if gold_titles else 0.0

            reranked = reranker.rerank(ex.question, retrieved, top_k=reranker_top_k, select_sentences=True)

            # Diagnostic: rerank recall
            reranked_titles = {r.passage.title for r in reranked}
            rerank_recall = len(gold_titles & reranked_titles) / len(gold_titles) if gold_titles else 0.0

            # Diagnostic: sentence recall
            selected_facts = set()
            for r in reranked:
                for sent_str in r.supporting_sentences:
                    for idx, s in enumerate(r.passage.sentences):
                        if " ".join(s.split()) == " ".join(sent_str.split()) or " ".join(s.split()).startswith(
                            " ".join(sent_str.split())[:50]
                        ):
                            selected_facts.add((r.passage.title, idx))
                            break
            sentence_recall = len(gold_facts & selected_facts) / len(gold_facts) if gold_facts else 0.0

            pb_output = pb.build(ex.question, reranked, use_citation_selection=True, question_type=ex.question_type)
            full_response = ""
            answer = ""
            supporting_facts = []

            # Verification retry config
            retry_enabled = verifier is not None and getattr(cfg.verifier, "retry_on_failure", False)
            max_retries = int(getattr(cfg.verifier, "max_verification_retries", 1)) if retry_enabled else 0
            retry_threshold = float(getattr(cfg.verifier, "retry_score_threshold", 0.4))

            for attempt in range(1 + max_retries):
                try:
                    current_prompt = pb_output.prompt
                    # On retry, prepend feedback to nudge the LLM toward evidence-grounded answers
                    if attempt > 0 and answer:
                        feedback = (
                            f'Your previous answer "{answer}" was not well-supported by the evidence. '
                            f"Re-read the facts carefully and provide a better-grounded answer. "
                            f"Extract the answer directly from the facts.\n\n"
                        )
                        current_prompt = feedback + current_prompt
                        log.info(f"Verification retry {attempt} for {ex.id} (prev answer: '{answer}')")

                    gen_output = gen.generate(
                        prompt=current_prompt,
                        target_model=pb_output.target_model,
                        temperature=(
                            cfg.prompt_builder.temperature_large_model
                            if pb_output.target_model == "complex"
                            else cfg.prompt_builder.temperature_small_model
                        ),
                        supporting_fact_indices=pb_output.supporting_fact_indices,
                        fact_mapping=pb_output.fact_mapping,
                    )
                    answer = gen_output.answer
                    supporting_facts = gen_output.supporting_facts
                    full_response = gen_output.full_response

                    # Check verification — break early if supported or last attempt
                    if retry_enabled and verifier is not None and attempt < max_retries:
                        vr = verifier.verify(
                            answer=answer,
                            evidence=reranked,
                            supporting_facts=supporting_facts,
                        )
                        if vr.is_supported or vr.support_score >= retry_threshold:
                            break  # Answer is good enough, no retry needed
                        # Otherwise loop continues with retry
                    else:
                        break

                except Exception as e:
                    log.warning(f"Generation failed for {ex.id} (attempt {attempt + 1}): {e}")
                    answer = "Cannot answer based on the provided evidence."
                    supporting_facts = []
                    break

            # Diagnostic: prediction recall
            pred_facts = {(sf[0], sf[1]) for sf in supporting_facts} if supporting_facts else set()
            pred_recall = len(gold_facts & pred_facts) / len(gold_facts) if gold_facts else 0.0

            # --- Verification ---
            verification_result = None
            vr_obj = None
            if verifier is not None:
                try:
                    vr_obj = verifier.verify(
                        answer=answer,
                        evidence=reranked,
                        supporting_facts=supporting_facts,
                    )
                    verification_result = {
                        "support_score": vr_obj.support_score,
                        "is_supported": vr_obj.is_supported,
                        "unsupported_claims": vr_obj.unsupported_claims,
                        "evidence_count": vr_obj.evidence_count,
                        "metadata": vr_obj.metadata,
                    }
                except Exception as e:
                    log.warning(f"Verification failed for {ex.id}: {e}")

            # --- Decider: final answer decision ---
            # Build a retry function for the decider (re-generates with feedback)
            def make_retry_fn():
                def retry_fn():
                    try:
                        feedback = (
                            "Your previous answer was not well-supported by the evidence. "
                            "Re-read the facts carefully and provide a better-grounded answer. "
                            "Extract the answer directly from the facts.\n\n"
                        )
                        retry_output = gen.generate(
                            prompt=feedback + pb_output.prompt,
                            target_model=pb_output.target_model,
                            temperature=(
                                cfg.prompt_builder.temperature_large_model
                                if pb_output.target_model == "complex"
                                else cfg.prompt_builder.temperature_small_model
                            ),
                            supporting_fact_indices=pb_output.supporting_fact_indices,
                            fact_mapping=pb_output.fact_mapping,
                        )
                        return {
                            "answer": retry_output.answer,
                            "supporting_facts": retry_output.supporting_facts,
                            "full_response": retry_output.full_response,
                            "reranked_results": reranked,
                        }
                    except Exception as e:
                        log.warning(f"Decider retry failed for {ex.id}: {e}")
                        return None

                return retry_fn

            decision, attempt_data = decider.decide(
                answer=answer,
                verification=vr_obj,
                reranked_results=reranked,
                supporting_facts=supporting_facts,
                verifier=verifier if retry_enabled else None,
                retry_fn=make_retry_fn() if retry_enabled else None,
                attempt_metadata={
                    "full_response": full_response,
                },
            )

            # Use decision output as final answer
            final_answer = decision.answer
            final_sp = attempt_data.get("supporting_facts", supporting_facts) or supporting_facts

            # Update verification result if decider triggered a retry
            if decision.support_score > 0 and verification_result:
                verification_result["support_score"] = decision.support_score
                verification_result["is_supported"] = decision.verifier_supported

            return (
                ex.id,
                {
                    "question": ex.question,
                    "gold_answer": ex.answer,
                    "answer": final_answer,
                    "sp": final_sp,
                    "full_response": attempt_data.get("full_response", full_response),
                    "verification": verification_result,
                    "decision": {
                        "confidence": decision.confidence,
                        "verifier_supported": decision.verifier_supported,
                        "supporting_passage_ids": decision.supporting_passage_ids,
                    },
                },
                {
                    "retrieval": retrieval_recall,
                    "rerank": rerank_recall,
                    "sentence": sentence_recall,
                    "prediction": pred_recall,
                },
            )
        except Exception as e:
            log.error(f"Failed processing example {ex.id}: {e}")
            return (
                ex.id,
                {
                    "answer": "Error occurred during processing.",
                    "sp": [],
                },
                {"retrieval": 0.0, "rerank": 0.0, "sentence": 0.0, "prediction": 0.0},
            )

    max_workers = cfg.eval.parallel_workers
    log.info(f"Running pipeline with {max_workers} parallel workers (cfg.eval.parallel_workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ex = {executor.submit(process_example, ex): ex for ex in examples}

        # Gather results with progress bar
        completed = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_ex), total=len(examples), desc="Running pipeline"):
            ex_id, pred, ex_diag = future.result()
            predictions[ex_id] = pred
            for stage in diag:
                diag[stage].append(ex_diag[stage])
            # Track verification stats
            vr = pred.get("verification")
            if vr is not None:
                verification_stats["scores"].append(vr["support_score"])
                if vr["is_supported"]:
                    verification_stats["supported"] += 1
                else:
                    verification_stats["unsupported"] += 1
            else:
                verification_stats["skipped"] += 1

            # Incremental checkpoint save
            completed += 1
            if checkpoint_path and completed % checkpoint_interval == 0:
                try:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(checkpoint_path, "w") as f:
                        json.dump(predictions, f)
                    log.info(f"Checkpoint saved: {len(predictions)} predictions")
                except Exception as e:
                    log.warning(f"Failed to save checkpoint: {e}")

        # Final checkpoint (ensures last batch is saved)
        if checkpoint_path:
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as f:
                    json.dump(predictions, f)
            except Exception:
                pass

    # Summary stats
    empty_sp = sum(1 for p in predictions.values() if not p["sp"])
    empty_ans = sum(
        1
        for p in predictions.values()
        if not p["answer"] or p["answer"] == "Cannot answer based on the provided evidence."
    )
    log.info(
        f"Pipeline summary: {len(predictions)} predictions, {empty_sp} with empty SP, "
        f"{empty_ans} with no/abstain answer"
    )

    # Stage-by-stage recall diagnostics
    n = len(diag["retrieval"])
    if n > 0:
        log.info(f"\n{'=' * 60}")
        log.info(f"STAGE-BY-STAGE RECALL DIAGNOSTICS (n={n})")
        log.info(f"{'=' * 60}")
        for stage, values in diag.items():
            mean_val = sum(values) / len(values) if values else 0.0
            perfect = sum(1 for v in values if v >= 1.0)
            zero = sum(1 for v in values if v <= 0.0)
            log.info(f"  {stage:>12s}: mean={mean_val:.1%}  perfect={perfect}/{n}  zero={zero}/{n}")
        log.info(f"{'=' * 60}")

    # Verification diagnostics
    v_scores = verification_stats["scores"]
    if v_scores:
        mean_vs = sum(v_scores) / len(v_scores)
        log.info(f"\n{'=' * 60}")
        log.info(f"VERIFICATION DIAGNOSTICS (n={len(v_scores)})")
        log.info(f"{'=' * 60}")
        log.info(f"  Mean support score: {mean_vs:.4f}")
        for label in ("supported", "unsupported"):
            count = verification_stats[label]
            log.info(f"  {label.capitalize() + ':':<13}{count}/{len(v_scores)} ({count / len(v_scores):.1%})")
        if verification_stats["skipped"]:
            log.info(f"  Skipped:     {verification_stats['skipped']}")
        log.info(f"{'=' * 60}")

    return predictions


def save_predictions(predictions: dict, output_path: Path):
    """Save predictions to JSON file (dict indexed by example ID)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"Saved {len(predictions)} predictions to {output_path}")


def evaluate_predictions(pred_path: Path, gold_path: Path, output_path: Path | None = None) -> dict | None:
    """Score predictions with the official HotpotQA script.

    Note the argument order: the reference script takes the predictions file
    first and the gold file second.

    The script's stdout is parsed into a dict before being written, so the
    saved metrics file is always valid JSON. Passing stdout through verbatim
    used to capture the script's progress line alongside the JSON body, which
    left the saved metrics unparseable.
    """
    eval_script = Path("scripts/hotpot_evaluate_v1.py")

    if not eval_script.exists():
        log.error(f"Evaluation script not found at {eval_script}")
        return None

    log.info("Running official HotpotQA evaluation...")
    try:
        result = subprocess.run(
            # sys.executable keeps this on the active interpreter rather than
            # whatever "python" happens to resolve to on PATH.
            [sys.executable, str(eval_script), str(pred_path), str(gold_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as e:
        log.error(f"Failed to run evaluation: {e}")
        return None

    if result.returncode != 0:
        log.error(f"Evaluation failed: {result.stderr.strip()}")
        return None

    if result.stderr.strip():
        log.info(result.stderr.strip())

    try:
        metrics = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        log.error(f"Could not parse evaluation output as JSON: {e}")
        log.error(f"Raw output began: {result.stdout[:300]}")
        return None

    for key in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1"):
        if key in metrics:
            log.info(f"  {key:10} {metrics[key]:.4f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Saved evaluation results to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the full pipeline on HotpotQA")
    parser.add_argument(
        "--split",
        choices=["dev_distractor", "dev_fullwiki", "test_fullwiki"],
        default="dev_distractor",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions/predictions.json"),
        help="Output path for predictions",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Path to gold answers JSON for evaluation",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics/metrics.json"),
        help="Output path for evaluation metrics",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="JSON list of example IDs to run; all others are skipped (used to re-run timeouts)",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Override both generator models, e.g. qwen2.5:7b for the timeout fallback re-run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <output>.checkpoint, skipping examples already predicted",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.split == "dev_fullwiki":
        cfg.data.dev_distractor = cfg.data.dev_fullwiki
    elif args.split == "test_fullwiki":
        cfg.data.dev_distractor = cfg.data.test_fullwiki

    if args.limit:
        cfg.eval.limit = args.limit

    if args.generator_model:
        cfg.generator.model_small = cfg.generator.model_large = args.generator_model

    # Load ID filter if provided
    id_filter = None
    if args.ids:
        with open(args.ids) as f:
            id_filter = set(json.load(f))
        log.info(f"ID filter loaded: {len(id_filter)} examples to re-run")

    log.info(f"{'=' * 80}")
    log.info(f"Evaluating on: {args.split}")
    if args.limit:
        log.info(f"Limit: {args.limit} examples")
    if id_filter:
        log.info(f"Re-running {len(id_filter)} failed IDs only")
    log.info(f"Output: {args.output}")
    log.info(f"{'=' * 80}")

    examples, retriever, reranker, pb, gen, verifier, decider, cfg = build_pipeline(cfg)

    # Filter examples to only the requested IDs
    if id_filter:
        examples = [ex for ex in examples if ex.id in id_filter]
        log.info(f"Filtered to {len(examples)} examples matching ID list")

    predictions = run_pipeline(
        examples,
        retriever,
        reranker,
        pb,
        gen,
        verifier,
        decider,
        cfg,
        limit=args.limit,
        output_path=args.output,
        resume=args.resume,
    )

    save_predictions(predictions, args.output)

    if args.eval:
        if not args.eval.exists():
            log.error(f"Gold file not found: {args.eval}")
        else:
            evaluate_predictions(args.output, args.eval, args.metrics)

    log.info(f"{'=' * 80}")
    log.info("Done!")


if __name__ == "__main__":
    main()
