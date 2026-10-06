"""Tests for YAML config loading and validation."""

import pytest
import yaml

from scripts.config import Config, load_config


def test_loads_the_shipped_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.retriever.alpha == 0.7
    assert cfg.retriever.multihop.fuzzy_title_threshold == 70
    assert cfg.reranker.top_k == 5
    assert cfg.generator.model_large == "gemma4:31b"
    assert cfg.verifier.mode == "overlap"
    assert cfg.verifier.retry_on_failure is False


def test_dataclass_defaults_mirror_the_yaml():
    """A missing or partial config must not silently change the pipeline.

    Every scalar in default.yaml should equal the dataclass default, so a
    partially-specified file behaves predictably. `device` is excluded because
    it is resolved from the host hardware at load time.
    """
    raw = yaml.safe_load(open("configs/default.yaml"))
    defaults = Config()
    mismatches = []

    def compare(obj, section, path):
        for key, value in (section or {}).items():
            if not hasattr(obj, key) or key == "device":
                continue
            current = getattr(obj, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                compare(current, value, f"{path}{key}.")
            elif not isinstance(value, dict) and current != value:
                mismatches.append(f"{path}{key}: default={current!r} yaml={value!r}")

    for name in ("data", "retriever", "reranker", "prompt_builder", "generator", "verifier", "eval"):
        compare(getattr(defaults, name), raw.get(name), f"{name}.")

    assert not mismatches, "config.py defaults diverge from default.yaml:\n" + "\n".join(mismatches)


def test_missing_config_raises_rather_than_silently_using_defaults():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_unknown_keys_are_reported(tmp_path, monkeypatch):
    """A typo'd key must not pass unnoticed.

    scripts.logger uses its own Logger class rather than stdlib logging, so
    pytest's caplog cannot see it; patch get_logger to record instead.
    """
    warnings = []

    class _Spy:
        def warning(self, msg):
            warnings.append(msg)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr("scripts.logger.get_logger", lambda *a, **k: _Spy())

    cfg_file = tmp_path / "typo.yaml"
    cfg_file.write_text("retriever:\n  aplha: 0.9\n  alpha: 0.6\n", encoding="utf-8")
    cfg = load_config(cfg_file)

    assert any("aplha" in w for w in warnings), f"expected a warning naming 'aplha', got {warnings}"
    # The correctly-spelled key must still be applied.
    assert cfg.retriever.alpha == 0.6


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("retriever", "alpha", 1.5),
        ("retriever", "alpha", -0.1),
        ("retriever", "candidate_pool_size", 0),
        ("reranker", "top_k", 0),
        ("reranker", "sentence_score_threshold", 2.0),
        ("prompt_builder", "complexity_routing_threshold", 1.1),
        ("prompt_builder", "max_evidence_chars", 0),
        ("eval", "parallel_workers", 0),
    ],
)
def test_validate_rejects_out_of_range_values(section, key, bad):
    cfg = Config()
    setattr(getattr(cfg, section), key, bad)
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_accepts_the_shipped_config():
    load_config("configs/default.yaml").validate()
