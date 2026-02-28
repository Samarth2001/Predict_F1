"""F1Store: single DuckDB file replacing races/quali CSVs and ingestion_state.json."""

import hashlib
import logging
import os
from typing import Optional

import duckdb
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS races (
    Year        INTEGER NOT NULL,
    Race_Num    INTEGER NOT NULL,
    Circuit     TEXT,
    Date        DATE,
    Driver      TEXT NOT NULL,
    Team        TEXT,
    Grid        REAL,
    Position    REAL,
    Status      TEXT,
    Laps        REAL,
    Points      REAL,
    AirTemp     REAL,
    Humidity    REAL,
    Pressure    REAL,
    WindSpeed   REAL,
    Rainfall    REAL,
    Is_Sprint   INTEGER DEFAULT 0,
    PRIMARY KEY (Year, Race_Num, Driver)
);

CREATE TABLE IF NOT EXISTS qualifying (
    Year        INTEGER NOT NULL,
    Race_Num    INTEGER NOT NULL,
    Circuit     TEXT,
    Date        DATE,
    Driver      TEXT NOT NULL,
    Team        TEXT,
    Position    REAL,
    Q1          TEXT,
    Q2          TEXT,
    Q3          TEXT,
    Is_Sprint   INTEGER DEFAULT 0,
    PRIMARY KEY (Year, Race_Num, Driver)
);

CREATE TABLE IF NOT EXISTS weather_cache (
    circuit     TEXT NOT NULL,
    fetched_at  TIMESTAMP NOT NULL,
    expires_at  TIMESTAMP NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (circuit)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    Year        INTEGER NOT NULL,
    Race_Num    INTEGER NOT NULL,
    session     TEXT NOT NULL,
    checksum    TEXT,
    rows        INTEGER,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (Year, Race_Num, session)
);
"""


class F1Store:
    """Thin DuckDB wrapper — one file, four tables, no CSV juggling.

    Usage:
        with F1Store() as store:
            store.upsert_races(df)
            races = store.get_races()
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.get("paths.db_path", "data/f1.duckdb")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> "F1Store":
        self._con = duckdb.connect(self.db_path)
        self._con.execute(_DDL)
        return self

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self) -> "F1Store":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("F1Store not open — use `with F1Store() as store:`")
        return self._con

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def _upsert(self, table: str, df: pd.DataFrame, pk_cols: list[str]) -> None:
        """Register df as a view, delete matching PKs, then insert."""
        if df.empty:
            return
        self.con.register("_tmp", df)
        where = " AND ".join(f"{table}.{c} = _tmp.{c}" for c in pk_cols)
        self.con.execute(f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM _tmp WHERE {where})")
        self.con.execute(f"INSERT INTO {table} SELECT * FROM _tmp")
        self.con.unregister("_tmp")

    def upsert_races(self, df: pd.DataFrame) -> None:
        self._upsert("races", df, ["Year", "Race_Num", "Driver"])

    def upsert_qualifying(self, df: pd.DataFrame) -> None:
        self._upsert("qualifying", df, ["Year", "Race_Num", "Driver"])

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_races(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        where = self._year_filter("Year", start_year, end_year)
        return self.con.execute(f"SELECT * FROM races{where} ORDER BY Year, Race_Num").df()

    def get_qualifying(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        where = self._year_filter("Year", start_year, end_year)
        return self.con.execute(f"SELECT * FROM qualifying{where} ORDER BY Year, Race_Num").df()

    @staticmethod
    def _year_filter(col: str, start: Optional[int], end: Optional[int]) -> str:
        clauses = []
        if start:
            clauses.append(f"{col} >= {int(start)}")
        if end:
            clauses.append(f"{col} <= {int(end)}")
        return (" WHERE " + " AND ".join(clauses)) if clauses else ""

    # ------------------------------------------------------------------
    # Weather cache
    # ------------------------------------------------------------------

    def get_weather_cache(self, circuit: str) -> Optional[str]:
        row = self.con.execute(
            "SELECT payload FROM weather_cache WHERE circuit = ? AND expires_at > current_timestamp",
            [circuit],
        ).fetchone()
        return row[0] if row else None

    def upsert_weather_cache(self, circuit: str, payload: str, ttl_hours: int = 12) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO weather_cache (circuit, fetched_at, expires_at, payload)
            VALUES (?, current_timestamp, current_timestamp + INTERVAL (?) HOUR, ?)
            """,
            [circuit, ttl_hours, payload],
        )

    # ------------------------------------------------------------------
    # Ingestion log (replaces ingestion_state.json)
    # ------------------------------------------------------------------

    def is_processed(self, year: int, race_num: int, session: str = "R") -> bool:
        row = self.con.execute(
            "SELECT 1 FROM ingestion_log WHERE Year=? AND Race_Num=? AND session=?",
            [int(year), int(race_num), session],
        ).fetchone()
        return row is not None

    def mark_processed(self, year: int, race_num: int, session: str, df: pd.DataFrame) -> None:
        checksum = _df_checksum(df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO ingestion_log (Year, Race_Num, session, checksum, rows, ingested_at)
            VALUES (?, ?, ?, ?, ?, current_timestamp)
            """,
            [int(year), int(race_num), session, checksum, int(len(df))],
        )

    def processed_events(self) -> set[tuple[int, int]]:
        """Return set of (year, race_num) pairs that have a completed race session."""
        rows = self.con.execute(
            "SELECT DISTINCT Year, Race_Num FROM ingestion_log WHERE session='R'"
        ).fetchall()
        return {(int(r[0]), int(r[1])) for r in rows}


def _df_checksum(df: pd.DataFrame) -> str:
    key_cols = [c for c in ["Year", "Race_Num", "Driver", "Team", "Position", "Grid"] if c in df.columns]
    payload = df.sort_values(key_cols).astype(str)[key_cols].to_csv(index=False)
    return hashlib.md5(payload.encode()).hexdigest()
