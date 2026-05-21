"""
롤링 윈도우 재학습 스케줄러.

피처 캐시(Parquet)에서 학습 데이터를 읽어 모델을 재학습하고
models/ 디렉터리에 저장한다.

피처 캐시 스키마:
    market_id  : str
    timestamp  : datetime (UTC, 예측 시점)
    outcome    : int  {0=NO, 1=YES, -1=미정}
    + FEATURE_NAMES 각 컬럼

캐시 파일 경로 규칙:
    data/features_cache/YYYY-MM-DD.parquet   (날짜별 파티션)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from .schemas import FEATURE_NAMES
from .baseline import LogisticBaseline
from .tree_model import TreeModel
from .ensemble import reload_models

# ── 설정 기본값 ───────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR  = Path("data/features_cache")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_WINDOW_DAYS = 90
MIN_SAMPLES = 30

# Known contaminated parquet files (bootstrap data collected at settlement time —
# see Section 5 of CLAUDE.md for the full explanation).
# These are automatically deleted once enough clean labeled rows are available.
_CONTAMINATED_FILES = {"2026-02-27.parquet"}
_MIN_CLEAN_LABELED_FOR_DELETE = 50   # delete only when this many clean rows exist

# ── 12-B: 피처 캐시 스키마 정의 ───────────────────────────────────────────────
# Parquet 파일에 반드시 존재해야 하는 컬럼 (순서 무관 검사용)
_BASE_COLUMNS  = ["market_id", "timestamp", "outcome"]
EXPECTED_COLUMNS = _BASE_COLUMNS + list(FEATURE_NAMES)

_log = logging.getLogger("prediction.training")


# ── 피처 캐시 저장 ────────────────────────────────────────────────────────────

def save_feature_to_cache(
    fv,                         # FeatureVector (순환 임포트 방지로 타입 힌트 생략)
    outcome: int = -1,          # 1=YES, 0=NO, -1=미정
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> None:
    """
    FeatureVector와 실제 결과(outcome)를 Parquet 캐시에 저장한다.

    outcome=-1 은 마켓이 아직 정산되지 않은 상태를 의미한다.
    학습 시에는 outcome ∈ {0,1} 인 행만 사용한다.

    Args:
        fv:         FeatureVector 객체
        outcome:    실제 결과 (1=YES, 0=NO, -1=미정)
        cache_dir:  Parquet 파일 저장 디렉터리
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    flat = fv.to_flat_dict()
    flat["seconds_to_close"] = fv.structured.seconds_to_close

    row = {
        "market_id": fv.market_id,
        "timestamp": fv.timestamp,
        "outcome":   outcome,
    }
    for name in FEATURE_NAMES:
        row[name] = flat.get(name, 0.0)

    df_new = pd.DataFrame([row])
    date_str = fv.timestamp.strftime("%Y-%m-%d")
    file_path = cache_dir / f"{date_str}.parquet"

    if file_path.exists():
        df_existing = pd.read_parquet(file_path)
        # 동일 market_id + timestamp 중복 방지
        df_existing = df_existing[
            ~((df_existing["market_id"] == fv.market_id) &
              (df_existing["timestamp"] == fv.timestamp))
        ]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(file_path, index=False)


def update_outcome(
    market_id: str,
    outcome: int,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> int:
    """
    캐시에서 특정 마켓의 모든 행의 outcome을 업데이트한다.

    마켓이 정산되면 이 함수로 레이블을 채운다.

    Returns:
        업데이트된 행 수.
    """
    cache_dir = Path(cache_dir)
    updated = 0

    for parquet_file in sorted(cache_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        mask = df["market_id"] == market_id
        if mask.any():
            df.loc[mask, "outcome"] = outcome
            df.to_parquet(parquet_file, index=False)
            updated += int(mask.sum())

    return updated


# ── 학습 데이터 로드 ──────────────────────────────────────────────────────────

def load_training_data(
    window_days: int = DEFAULT_WINDOW_DAYS,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    최근 window_days 일치의 캐시에서 학습 데이터를 로드한다.

    outcome ∈ {0, 1} 인 행만 포함한다 (미정 행 제외).

    Returns:
        (X, y)
          X: (n_samples, len(FEATURE_NAMES)) float64 배열
          y: (n_samples,) int 배열 {0, 1}
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    frames: list[pd.DataFrame] = []

    for parquet_file in sorted(cache_dir.glob("*.parquet")):
        # 파일명에서 날짜 추출해 윈도우 밖이면 스킵
        try:
            file_date = datetime.strptime(parquet_file.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                continue
        except ValueError:
            pass

        df = pd.read_parquet(parquet_file)

        # 12-B: 컬럼 존재 검증 — 필수 컬럼이 없으면 파일을 건너뜀
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            _log.warning(
                "Schema mismatch in %s — missing columns: %s — skipping file.",
                parquet_file.name, missing,
            )
            continue

        # 12-B: NaN 행 제거 (피처 컬럼 기준)
        before = len(df)
        df = df.dropna(subset=FEATURE_NAMES)
        dropped = before - len(df)
        if dropped > 0:
            _log.debug("Dropped %d NaN rows from %s", dropped, parquet_file.name)

        frames.append(df)

    if not frames:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)

    data = pd.concat(frames, ignore_index=True)

    # 레이블이 있는 행만
    labeled = data[data["outcome"].isin([0, 1])].copy()
    if labeled.empty:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=int)

    X = labeled[FEATURE_NAMES].astype(float).values
    y = labeled["outcome"].astype(int).values

    return X, y


# ── 롤링 재학습 ───────────────────────────────────────────────────────────────

def _maybe_delete_contaminated(cache_dir: Path) -> bool:
    """
    Delete known contaminated parquet files once enough clean data exists.

    Contaminated files contain features computed at settlement time (see
    CLAUDE.md §5).  They are removed when non-contaminated files collectively
    have at least _MIN_CLEAN_LABELED_FOR_DELETE labeled rows, after which
    retraining from only clean data becomes possible.

    Returns True if any file was deleted.
    """
    contaminated = [
        cache_dir / name
        for name in _CONTAMINATED_FILES
        if (cache_dir / name).exists()
    ]
    if not contaminated:
        return False

    clean_labeled = 0
    for pf in sorted(cache_dir.glob("*.parquet")):
        if pf.name in _CONTAMINATED_FILES:
            continue
        try:
            df = pd.read_parquet(pf)
            if "outcome" in df.columns:
                clean_labeled += int(df["outcome"].isin([0, 1]).sum())
        except Exception:
            pass

    if clean_labeled >= _MIN_CLEAN_LABELED_FOR_DELETE:
        for f in contaminated:
            _log.warning(
                "Deleting contaminated bootstrap parquet '%s' — "
                "%d clean labeled rows now available.",
                f.name, clean_labeled,
            )
            f.unlink()
        return True

    _log.info(
        "Contaminated parquet kept — %d/%d clean labeled rows "
        "(need %d before deletion).",
        clean_labeled, _MIN_CLEAN_LABELED_FOR_DELETE,
        _MIN_CLEAN_LABELED_FOR_DELETE,
    )
    return False


class RollingTrainer:
    """
    주기적 롤링 윈도우 재학습 스케줄러.

    주요 책임:
    - 최근 window_days 일치 캐시에서 학습 데이터 로드
    - baseline + tree 모델 동시 재학습
    - 학습된 모델을 models/ 저장
    - ensemble 싱글턴 모델 갱신 (reload_models 호출)

    사용 예시:
        trainer = RollingTrainer()
        result = trainer.run_training_cycle()
        print(result)
    """

    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW_DAYS,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        models_dir: str | Path = DEFAULT_MODELS_DIR,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self.window_days = window_days
        self.cache_dir   = Path(cache_dir)
        self.models_dir  = Path(models_dir)
        self.min_samples = min_samples

        self.baseline_path = self.models_dir / "baseline.pkl"
        self.tree_path     = self.models_dir / "tree_model.pkl"

    def run_training_cycle(self, window_days: Optional[int] = None) -> dict:
        """
        재학습을 실행하고 결과 요약 딕셔너리를 반환한다.

        Returns:
            {
              "status":         "success" | "skipped" | "error",
              "n_samples":      int,
              "baseline_ok":    bool,
              "tree_ok":        bool,
              "trained_at":     str (ISO 시각),
              "message":        str,
            }
        """
        started_at = datetime.now(timezone.utc)
        win = window_days or self.window_days

        # Remove contaminated bootstrap data once enough clean rows exist.
        _maybe_delete_contaminated(self.cache_dir)

        # 1. 학습 데이터 로드
        try:
            X, y = load_training_data(win, self.cache_dir)
        except Exception as exc:
            return _result("error", 0, False, False, started_at, f"데이터 로드 실패: {exc}")

        n = len(y)
        if n < self.min_samples:
            return _result(
                "skipped", n, False, False, started_at,
                f"샘플 수 부족 ({n} < {self.min_samples}). 학습 건너뜀."
            )

        # 2. 클래스 분포 확인
        n_yes = int(y.sum())
        n_no  = n - n_yes
        if n_yes == 0 or n_no == 0:
            return _result(
                "skipped", n, False, False, started_at,
                f"단일 클래스만 존재 (YES={n_yes}, NO={n_no}). 학습 건너뜀."
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 3. baseline 재학습
        b_ok = False
        try:
            baseline = LogisticBaseline()
            baseline.fit(X, y)
            baseline.save(self.baseline_path)
            b_ok = True
        except Exception as exc:
            print(f"[prediction.training] baseline 학습 실패: {exc}")

        # 4. tree 재학습
        t_ok = False
        try:
            tree = TreeModel()
            tree.fit(X, y)
            tree.save(self.tree_path)
            t_ok = True
        except Exception as exc:
            print(f"[prediction.training] tree 학습 실패: {exc}")

        # 5. ensemble 싱글턴 갱신
        if b_ok or t_ok:
            reload_models(self.baseline_path, self.tree_path)

        msg = (
            f"학습 완료: {n}샘플 (YES={n_yes}, NO={n_no}), "
            f"baseline={'OK' if b_ok else 'FAIL'}, "
            f"tree={'OK' if t_ok else 'FAIL'}"
        )
        return _result("success", n, b_ok, t_ok, started_at, msg)


def _result(
    status: str,
    n: int,
    b_ok: bool,
    t_ok: bool,
    started_at: datetime,
    message: str,
) -> dict:
    return {
        "status":      status,
        "n_samples":   n,
        "baseline_ok": b_ok,
        "tree_ok":     t_ok,
        "trained_at":  started_at.isoformat(),
        "message":     message,
    }


# ── Daily outcome recorder ────────────────────────────────────────────────────

def _parse_outcome_from_tokens(tokens: list) -> int:
    """
    Return 1 (YES won), 0 (NO won), or -1 (unknown) from a Polymarket token list.

    Detection priority:
      1. Explicit YES/NO outcome labels ("Yes", "No", "true", "false", "1", "0").
      2. Any binary (2-token) market: first-listed winning token treated as YES.
    """
    valid = [t for t in tokens if isinstance(t, dict)]

    # priority 1: explicit YES/NO
    for tok in valid:
        if tok.get("winner", False):
            s = str(tok.get("outcome", "")).lower().strip()
            if s in {"yes", "true", "1"}:
                return 1
            if s in {"no", "false", "0"}:
                return 0

    # priority 2: binary market with unambiguous winner
    if len(valid) == 2:
        winners = [t for t in valid if t.get("winner", False)]
        losers  = [t for t in valid if not t.get("winner", False)]
        if len(winners) == 1 and len(losers) == 1:
            return 1 if valid[0].get("winner", False) else 0

    return -1


def record_settled_outcomes(
    since: Optional[datetime] = None,
    clob_host: str = "",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict:
    """
    Fetch markets settled since `since` from the CLOB API, determine the
    winning token for each, and write the outcome (YES=1 / NO=0) to the
    feature cache via `update_outcome()`.

    Only markets that already have rows in the cache are updated; markets with
    no cached features are silently skipped (the update is a no-op).

    Args:
        since:      Start of the settlement window (UTC).  Defaults to 26 hours
                    ago so yesterday's late-resolving markets are captured.
        clob_host:  Override the CLOB API base URL (uses the CLOB_HOST env var
                    if not provided).
        cache_dir:  Feature cache directory.

    Returns:
        {
            "recorded":  int,   # number of markets with outcome written
            "skipped":   int,   # no winner data or bad market data
            "api_error": bool,  # True if the API call itself failed
        }
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=26)

    if not clob_host:
        clob_host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

    recorded  = 0
    skipped   = 0
    api_error = False

    try:
        with httpx.Client(timeout=30.0) as http:
            next_cursor: Optional[str] = None
            page = 0

            while True:
                page += 1
                params: dict = {"closed": "true", "limit": 1000}
                if next_cursor:
                    params["next_cursor"] = next_cursor

                r = http.get(f"{clob_host}/markets", params=params)
                r.raise_for_status()
                payload    = r.json()
                batch      = payload.get("data", []) if isinstance(payload, dict) else payload
                next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None

                if not batch:
                    break

                for raw in batch:
                    if not raw.get("closed", False):
                        continue

                    # Check settlement time against our window
                    end_str = raw.get("end_date_iso") or raw.get("resolution_date", "")
                    try:
                        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        skipped += 1
                        continue

                    if end_dt < since:
                        continue  # older than our window

                    # Determine outcome
                    outcome = _parse_outcome_from_tokens(raw.get("tokens", []))
                    if outcome == -1:
                        skipped += 1
                        continue

                    market_id = raw.get("condition_id") or raw.get("market_id", "")
                    if not market_id:
                        skipped += 1
                        continue

                    rows_updated = update_outcome(market_id, outcome, cache_dir)
                    if rows_updated > 0:
                        recorded += 1

                if not next_cursor:
                    break

    except Exception as exc:
        print(f"[prediction.training] record_settled_outcomes API error: {exc}")
        api_error = True

    return {
        "recorded":  recorded,
        "skipped":   skipped,
        "api_error": api_error,
    }
