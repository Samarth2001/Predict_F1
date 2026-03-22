from __future__ import annotations

import pandas as pd
import pytest

import f1_predictor.utils as utils


def test_ascii_slug_normalizes_text() -> None:
    assert utils._ascii_slug("São Paulo") == "sao-paulo"
    assert utils._ascii_slug("  Max Verstappen ") == "max-verstappen"


def test_add_entity_ids_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "Driver": ["VER", "HAM"],
            "Team": ["Red Bull", "Mercedes"],
            "Circuit": ["Monaco Grand Prix", "Monaco Grand Prix"],
        }
    )
    out = utils.add_entity_ids(df)
    assert {"Driver_ID", "Team_ID", "Circuit_ID"}.issubset(out.columns)
    assert out.loc[0, "Driver_ID"].startswith("drv:")
    assert out.loc[0, "Team_ID"].startswith("team:")
    assert out.loc[0, "Circuit_ID"].startswith("circuit:")


def test_pick_group_col_prefers_id_column() -> None:
    df = pd.DataFrame({"Driver_ID": ["drv:ver"], "Driver": ["VER"]})
    assert utils.pick_group_col(df, "Driver") == "Driver_ID"


def test_safe_merge_raises_when_guard_configured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_get = utils.config.get

    def fake_get(key: str, default=None):
        if key == "feature_engineering.join_guard.enabled":
            return True
        if key == "feature_engineering.join_guard.max_unmatched_ratio":
            return 0.0
        if key == "feature_engineering.join_guard.on_violation":
            return "error"
        return original_get(key, default)

    monkeypatch.setattr(utils.config, "get", fake_get)

    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"id": [1], "value": [10]})

    with pytest.raises(ValueError):
        utils.safe_merge(left, right, on=["id"], how="left", join_name="guard_test")

