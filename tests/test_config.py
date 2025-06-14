import pytest

from cabinsense.config import apply_overrides, deep_merge, load_config


def test_env_expansion_with_default(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("root: ${CS_TEST_ROOT:-fallback}\nother: ${CS_TEST_SET}\n")
    monkeypatch.setenv("CS_TEST_SET", "yes")
    monkeypatch.delenv("CS_TEST_ROOT", raising=False)
    assert load_config(cfg) == {"root": "fallback", "other": "yes"}


def test_base_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  epochs: 10\n  lr: 0.1\nname: base\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nname: child\ntrain:\n  lr: 0.01\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.epochs=2", "train.amp=false"])
    assert cfg == {"name": "child", "train": {"epochs": 2, "lr": 0.01, "amp": False}}


def test_bad_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}
    assert merged == {"a": {"b": 1, "c": 2}}
