from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from embodied_data_infra.cleaning.config import ConfigError, load_config


def _minimal_config(tmp_path: Path, **cleaning_overrides) -> Path:
    cleaning = {
        "min_frames": 5,
        "max_frames": 150,
        "action": {"bounds": [[0, 0], [0, 0], [-1, 1], [0, 2]], "allowed_values": {3: [1, 2]}},
        "idle": {"min_trim_frames": 10, "max_trim_ratio": 0.5},
        "success": {"require": True},
    }
    cleaning.update(cleaning_overrides)
    payload = {
        "source": {"repo_id": "test/dataset"},
        "cleaning": cleaning,
        "output": {"repo_id": "test/clean"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_defaults_and_stable_hash(tmp_path):
    path = _minimal_config(tmp_path)
    first = load_config(path)
    second = load_config(path)
    assert first.source.revision == "main"
    assert first.cleaning.min_frames == 5
    assert first.conversion.validate_after is True
    assert first.fingerprint() == second.fingerprint()


def test_hash_changes_with_threshold(tmp_path):
    one = load_config(_minimal_config(tmp_path, min_frames=5))
    two = load_config(_minimal_config(tmp_path, min_frames=6))
    assert one.fingerprint() != two.fingerprint()


def test_unknown_key_rejected(tmp_path):
    path = _minimal_config(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["cleaning"]["not_a_rule"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(path)


def test_invalid_trim_ratio_rejected(tmp_path):
    path = _minimal_config(tmp_path, idle={"min_trim_frames": 10, "max_trim_ratio": 1.5})
    with pytest.raises(ConfigError, match="max_trim_ratio"):
        load_config(path)


def test_missing_repo_id_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"source": {}, "cleaning": {}, "output": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="repo_id"):
        load_config(path)
