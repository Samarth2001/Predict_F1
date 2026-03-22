"""Utility functions for the F1 prediction project.

This module also contains helpers for canonicalizing entity names (drivers/teams/circuits),
deriving stable IDs, and performing safe DataFrame merges with guardrails to prevent
silent mismatches that can cause leakage or dropped rows.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Dict, Any, List
import logging
import re
import unicodedata

from .config import config

import numpy as np
import pandas as pd


def time_to_seconds(time_str: str) -> float:
    """Convert a time string (e.g., '1:23.456') to seconds as float; NaN on invalid."""
    if pd.isna(time_str) or time_str == "":
        return np.nan
    try:
        time_str = str(time_str).strip()
        if ":" in time_str:
            minutes_str, seconds_str = time_str.split(":", 1)
            minutes = float(minutes_str)
            seconds = float(seconds_str)
            return minutes * 60.0 + seconds
        return float(time_str)
    except (ValueError, TypeError):
        return np.nan


def set_global_seeds(seed: int) -> None:
    """Set global random seeds for deterministic behavior where possible.

    - Sets PYTHONHASHSEED for string hashing determinism
    - Seeds Python's random, NumPy
    - Tries to seed PyTorch if available (no hard dependency)
    """
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    except Exception:
        pass
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
                                              
    try:                                          
        import torch                

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():                                     
            torch.cuda.manual_seed_all(int(seed))
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def downcast_dataframe(
    df: pd.DataFrame,
    *,
    downcast_floats: bool = True,
    downcast_ints: bool = True,
    categorize_objects: bool = True,
    max_categories: int = 50,
    object_to_category_ratio: float = 0.5,
) -> pd.DataFrame:
    """Downcast numeric dtypes and optionally convert low-cardinality objects to category.

    Parameters
    - downcast_floats: convert float64 -> float32
    - downcast_ints: convert int64 -> smallest integer subtype when no NaNs
    - categorize_objects: convert object columns to category when cardinality is low
    - max_categories: absolute cap for category conversion
    - object_to_category_ratio: maximum unique/rows ratio to convert to category
    """
    result = df.copy()
            
    if downcast_floats:
        float_cols = result.select_dtypes(include=["float64", "float32"]).columns
        for col in float_cols:
            try:
                result[col] = pd.to_numeric(result[col], errors="coerce", downcast="float")
            except Exception:
                                                            
                try:
                    result[col] = result[col].astype("float32")
                except Exception:
                    pass
                       
    if downcast_ints:
        int_cols = result.select_dtypes(include=["int64", "int32", "int16"]).columns
        for col in int_cols:
            s = result[col]
            if pd.isna(s).any():
                                                             
                try:
                    result[col] = s.astype("float32")
                except Exception:
                    pass
            else:
                try:
                    result[col] = pd.to_numeric(s, downcast="integer")
                except Exception:
                    pass
                         
    if categorize_objects and len(result) > 0:
        obj_cols = result.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            s = result[col].astype("string")
            nunique = int(s.nunique(dropna=True))
            if nunique == 0:
                continue
            if nunique <= max_categories and (nunique / max(1, len(s))) <= object_to_category_ratio:
                try:
                    result[col] = result[col].astype("category")
                except Exception:
                    pass
    return result


# --------------------------------------------------------------------------------------
# Entity canonicalization and ID generation
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _ascii_slug(text: str) -> str:
    """Return a lowercase ASCII slug for a given text for use in stable IDs."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    try:
        txt = str(text).strip()
        # Normalize unicode to ASCII where possible
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
        txt = txt.lower()
        # Remove non-alphanumeric except spaces and hyphens
        txt = re.sub(r"[^a-z0-9\s-]", "", txt)
        txt = re.sub(r"\s+", "-", txt).strip("-")
        txt = re.sub(r"-+", "-", txt)
        return txt
    except Exception:
        return ""


def _normalize_circuit_name(name: str) -> str:
    """Normalize circuit names by ensuring the ' Grand Prix' suffix and applying aliases.

    Aliases are pulled from feature_engineering.canonicalization.circuits in config.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return name
    s = str(name).strip()
    circuits_map: Dict[str, str] = config.get("feature_engineering.canonicalization.circuits", {}) or {}
    # First apply explicit mapping if any
    if s in circuits_map:
        s = circuits_map[s]
    # Ensure standard suffix
    if not s.endswith(" Grand Prix"):
        s = f"{s} Grand Prix"
    return s


def canonicalize_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Apply driver/team/circuit canonicalization configured in YAML.

    - Drivers and Teams use explicit maps if provided; otherwise, values pass through
    - Circuits also get normalized to include the ' Grand Prix' suffix as needed
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        drivers_map: Dict[str, str] = config.get("feature_engineering.canonicalization.drivers", {}) or {}
        teams_map: Dict[str, str] = config.get("feature_engineering.canonicalization.teams", {}) or {}

        if "Driver" in out.columns and drivers_map:
            out["Driver"] = out["Driver"].astype(str).map(lambda x: drivers_map.get(x, x))
        if "Team" in out.columns and teams_map:
            out["Team"] = out["Team"].astype(str).map(lambda x: teams_map.get(x, x))
        if "Circuit" in out.columns:
            out["Circuit"] = out["Circuit"].map(_normalize_circuit_name)
    except Exception as e:
        logger.warning(f"Canonicalization failed; proceeding without changes: {e}")
    return out


def add_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable ID columns for Driver, Team, and Circuit: Driver_ID, Team_ID, Circuit_ID.

    IDs are generated from canonicalized names using ASCII slugs with prefixes:
      - drv:<slug>
      - team:<slug>
      - circuit:<slug>
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        if "Driver" in out.columns and "Driver_ID" not in out.columns:
            out["Driver_ID"] = out["Driver"].map(lambda x: f"drv:{_ascii_slug(x)}")
        if "Team" in out.columns and "Team_ID" not in out.columns:
            out["Team_ID"] = out["Team"].map(lambda x: f"team:{_ascii_slug(x)}")
        if "Circuit" in out.columns and "Circuit_ID" not in out.columns:
            out["Circuit_ID"] = out["Circuit"].map(lambda x: f"circuit:{_ascii_slug(x)}")
    except Exception as e:
        logger.warning(f"Failed to add entity IDs: {e}")
    return out


def pick_group_col(df: pd.DataFrame, base_name: str) -> str:
    """Return ID column name if present (e.g., Driver_ID) else fall back to base (e.g., Driver)."""
    cand = f"{base_name}_ID"
    if isinstance(df, pd.DataFrame) and cand in df.columns:
        return cand
    return base_name


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: List[str],
    how: str = "left",
    join_name: str = "merge",
) -> pd.DataFrame:
    """Merge with guardrails to prevent silent mismatches.

    Behavior controlled by feature_engineering.join_guard in config:
      - enabled (bool)
      - max_unmatched_ratio (float)
      - on_violation: "warn" | "error"
    """
    if left is None or right is None:
        return pd.DataFrame()
    guard_enabled = bool(config.get("feature_engineering.join_guard.enabled", True))
    max_unmatched = float(config.get("feature_engineering.join_guard.max_unmatched_ratio", 0.02))
    on_violation = str(config.get("feature_engineering.join_guard.on_violation", "warn")).lower()

    if not guard_enabled:
        return left.merge(right, on=on, how=how)

    try:
        merged = left.merge(right, on=on, how=how, indicator=True)
        if how in ("left", "outer"):
            total_left = max(1, int(left.shape[0]))
            left_only = int((merged["_merge"] == "left_only").sum())
            ratio = left_only / total_left
            if ratio > max_unmatched:
                msg = (
                    f"Join '{join_name}' resulted in {left_only}/{total_left} unmatched left rows "
                    f"({ratio:.2%}) on keys {on}."
                )
                if on_violation == "error":
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
        return merged.drop(columns=["_merge"], errors="ignore")
    except Exception as e:
        logger.error(f"Safe merge '{join_name}' failed: {e}")
        if on_violation == "error":
            raise
        # Best-effort fallback without indicator
        return left.merge(right, on=on, how=how)

