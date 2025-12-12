# src/preprocess.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    # 혼잡도 기준 (승차+하차)
    th_low: int = 800
    th_high: int = 2000

    # 컬럼 후보들(데이터셋마다 이름이 다를 수 있어서 후보를 여러 개 둠)
    line_candidates: Tuple[str, ...] = ("호선", "호선명", "line", "LINE")
    station_candidates: Tuple[str, ...] = ("역명", "역", "station", "STATION")
    date_candidates: Tuple[str, ...] = ("사용일자", "날짜", "date", "DATE")
    time_candidates: Tuple[str, ...] = ("시간대", "시간", "hour", "HOUR")
    ride_candidates: Tuple[str, ...] = ("승차", "승차인원", "승차 인원", "ride", "RIDE")
    alight_candidates: Tuple[str, ...] = ("하차", "하차인원", "하차 인원", "alight", "ALIGHT")

    # 이상치 상한(99.5% 분위수로 절단: 극단값 완화)
    clip_quantile: float = 0.995


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    # 부분 매칭도 시도
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in str(c).lower():
                return c
    return None


def _parse_hour(x) -> Optional[int]:
    # "07시~08시", "07", 7, "7:00" 등 대응
    if pd.isna(x):
        return None
    s = str(x).strip()
    # "07시~08시" -> 7
    for token in ["시", "~", "-", " "]:
        if token in s:
            # 가장 앞 숫자만 뽑기
            digits = ""
            for ch in s:
                if ch.isdigit():
                    digits += ch
                else:
                    if digits:
                        break
            return int(digits) if digits else None
    # "7:00" -> 7
    if ":" in s:
        try:
            return int(s.split(":")[0])
        except Exception:
            return None
    # 그냥 숫자
    try:
        return int(float(s))
    except Exception:
        return None


def _label_congestion(total: float, th_low: int, th_high: int) -> int:
    if total < th_low:
        return 0  # 여유
    if total < th_high:
        return 1  # 보통
    return 2      # 혼잡


def preprocess(
    input_csv: Path,
    output_csv: Path,
    cfg: PreprocessConfig,
) -> dict:
    # 1) CSV 로드 (인코딩 자동 대응)
    try:
        df = pd.read_csv(input_csv, encoding="euc-kr")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="cp949")

    # 2) 컬럼 공백 제거(너 데이터에 '20 시', '승차인 원' 같은 게 있어서 필수)
    df.columns = [str(c).replace(" ", "") for c in df.columns]

    # 3) 기본 컬럼 매핑
    # 날짜는 '작업일자' 우선, 없으면 '사용월' 사용
    if "작업일자" in df.columns:
        df["date"] = pd.to_datetime(df["작업일자"], errors="coerce")
    elif "사용월" in df.columns:
        df["date"] = pd.to_datetime(df["사용월"].astype(str) + "01", errors="coerce")
    else:
        raise ValueError("날짜 컬럼(작업일자 또는 사용월)을 찾지 못했어.")

    if "호선명" not in df.columns or "지하철역" not in df.columns:
        raise ValueError("호선명/지하철역 컬럼을 찾지 못했어.")

    df = df.rename(columns={"호선명": "line", "지하철역": "station"})

    # 4) 시간대 승/하차 컬럼 찾기 (wide 형태)
    ride_cols = [c for c in df.columns if c.endswith("승차인원")]
    alight_cols = [c for c in df.columns if c.endswith("하차인원")]

    if not ride_cols or not alight_cols:
        raise ValueError("시간대별 승차인원/하차인원 컬럼을 찾지 못했어.")

    # 5) wide -> long (melt 후 merge)
    id_vars = ["date", "line", "station"]

    ride_long = df.melt(id_vars=id_vars, value_vars=ride_cols,
                        var_name="time_range", value_name="ride")
    alight_long = df.melt(id_vars=id_vars, value_vars=alight_cols,
                          var_name="time_range", value_name="alight")

    ride_long["time_range"] = ride_long["time_range"].str.replace("승차인원", "", regex=False)
    alight_long["time_range"] = alight_long["time_range"].str.replace("하차인원", "", regex=False)

    long = pd.merge(
        ride_long, alight_long,
        on=["date", "line", "station", "time_range"],
        how="inner"
    )

    # 6) hour 추출: "04시-05시" -> 4
    long["hour"] = long["time_range"].str.extract(r"(\d{2})시").astype(float).astype("Int64")

    # 7) 요일 파생
    long["dow"] = pd.to_datetime(long["date"], errors="coerce").dt.dayofweek

    # 8) 숫자 변환
    long["ride"] = pd.to_numeric(long["ride"], errors="coerce")
    long["alight"] = pd.to_numeric(long["alight"], errors="coerce")

    # 9) 결측 제거
    before = len(long)
    long = long.dropna(subset=["date", "line", "station", "hour", "dow", "ride", "alight"])
    after_dropna = len(long)

    # 10) 이상치 완화
    ride_cap = long["ride"].quantile(cfg.clip_quantile)
    alight_cap = long["alight"].quantile(cfg.clip_quantile)
    long["ride"] = long["ride"].clip(lower=0, upper=ride_cap)
    long["alight"] = long["alight"].clip(lower=0, upper=alight_cap)

    # 11) 혼잡도 지표 + 라벨
    long["total"] = long["ride"] + long["alight"]
    long["congestion"] = long["total"].apply(lambda t: _label_congestion(t, cfg.th_low, cfg.th_high))

    out = long[["date", "dow", "hour", "line", "station", "ride", "alight", "total", "congestion"]].copy()

    # 12) 저장
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    meta = {
        "rows_before_long": before,
        "rows_after_dropna": after_dropna,
        "rows_final": len(out),
        "thresholds": {"low": cfg.th_low, "high": cfg.th_high},
        "caps": {"ride_cap": float(ride_cap), "alight_cap": float(alight_cap)},
        "label_meaning": {0: "여유", 1: "보통", 2: "혼잡"},
        "note": "wide(시간대별 컬럼) → long(행=시간대) 변환 후 학습 데이터 생성"
    }
    return meta



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/subway.csv")
    p.add_argument("--output", type=str, default="data/processed.csv")
    p.add_argument("--th_low", type=int, default=800)
    p.add_argument("--th_high", type=int, default=2000)
    p.add_argument("--meta_out", type=str, default="models/meta.json")
    args = p.parse_args()

    cfg = PreprocessConfig(th_low=args.th_low, th_high=args.th_high)
    meta = preprocess(Path(args.input), Path(args.output), cfg)

    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] preprocess done")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
