"""Smoke run: one HotpotQA question through every pipeline stage.

Indexes only the passages of the first 10 dev examples, so it finishes in
minutes rather than hours. Needs a running Ollama server with the configured
models pulled.

    python -m scripts.test_pipeline
"""

from pipeline.data_loader import HotpotQALoader
from pipeline.decider import Decider
from pipeline.generator import Generator
from pipeline.indexer import HybridRetriever
from pipeline.prompt_builder import PromptBuilder
from pipeline.reranker import Reranker
from pipeline.verifier import Verifier
from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("test_pipeline")


def test_single_example():
    cfg = load_config()

    # Load data
    loader = HotpotQALoader(cfg.data.dev_distractor)
    examples = loader.load(limit=10)

    seen = set()
    passages = []
    for ex in examples:
        for ctx in ex.contexts:
            key = (ctx.title, ctx.text)
            if key not in seen:
                seen.add(key)
                passages.append(ctx)

    # Initialize pipeline
    log.info(f"Building indexer with {len(passages)} passages...")
    retriever = HybridRetriever.from_config(cfg)
    retriever.index(passages, show_progress=False)

    reranker = Reranker.from_config(cfg)
    pb = PromptBuilder.from_config(cfg)
    gen = Generator.from_config(cfg)
    verifier = Verifier.from_config(cfg)
    decider = Decider.from_config(cfg)

    ex = examples[0]
    log.info(f"\n{'=' * 80}")
    log.info(f"Example: {ex.id}")
    log.info(f"Question: {ex.question}")
    log.info(f"Gold answer: {ex.answer}")

    # 1. Retrieve (two hops, with fuzzy title matching on hop 2)
    retrieved = retriever.retrieve_multihop(
        ex.question,
        top_k=cfg.retriever.top_k,
        top_k_per_hop=cfg.retriever.multihop.top_k_per_hop,
        question_type=ex.question_type,
    )
    log.info(f"Retrieved {len(retrieved)} candidates")

    # 2. Rerank and select supporting sentences
    reranked = reranker.rerank(ex.question, retrieved, top_k=cfg.reranker.top_k)
    log.info(f"Re-ranked to {len(reranked)} passages")

    # 3. Build the citation-selection prompt
    pb_output = pb.build(ex.question, reranked, question_type=ex.question_type)
    log.info(f"Complexity score: {pb_output.complexity_score:.3f}")
    log.info(f"Target model: {pb_output.target_model}")

    # 4. Generate
    temperature = (
        cfg.prompt_builder.temperature_large_model
        if pb_output.target_model == "complex"
        else cfg.prompt_builder.temperature_small_model
    )
    gen_output = gen.generate(
        prompt=pb_output.prompt,
        target_model=pb_output.target_model,
        temperature=temperature,
        supporting_fact_indices=pb_output.supporting_fact_indices,
        fact_mapping=pb_output.fact_mapping,
    )

    # 5. Verify and decide
    verification = verifier.verify(gen_output.answer, evidence=reranked, supporting_facts=gen_output.supporting_facts)
    decision, _ = decider.decide(
        answer=gen_output.answer,
        verification=verification,
        reranked_results=reranked,
        supporting_facts=gen_output.supporting_facts,
    )

    log.info(f"\n{'=' * 80}")
    log.info(f"Generated answer: {gen_output.answer}")
    log.info(f"Supporting facts: {gen_output.supporting_facts}")
    log.info(f"Model: {gen_output.model_used}")
    log.info(f"Generation time: {gen_output.generation_time:.2f}s")
    log.info(f"Verifier: support={verification.support_score:.2f} supported={verification.is_supported}")
    log.info(f"Decision: answer={decision.answer!r} confidence={decision.confidence:.2f}")
    log.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_single_example()
