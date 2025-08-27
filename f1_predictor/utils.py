"""Utility functions for the F1 prediction project."""

from __future__ import annotations

import os
import pickle
import random
from typing import Optional

import numpy as np
import pandas as pd


def save_object(obj, filepath: str) -> None:
    """Save a Python object using pickle to the given file path."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"Object saved successfully to {filepath}")
    except Exception as exc:
        print(f"Error saving object to {filepath}: {exc}")


def load_object(filepath: str):
    """Load a Python object from a pickle file if it exists, else return None."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"Object loaded successfully from {filepath}")
        return obj
    except Exception as exc:
        print(f"Error loading object from {filepath}: {exc}")
        return None


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

