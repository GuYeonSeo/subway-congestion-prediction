import json
from pathlib import Path

def load_station_map():
    meta_path = Path("models/meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # meta.json에 저장된 인코딩 정보 사용
    line_encoder = meta["encoders"]["line"]
    station_encoder = meta["encoders"]["station"]

    station_map = {}
    for station, station_idx in station_encoder.items():
        # station_encoder: {"강남": 0, ...}
        # meta에 line 정보는 따로 저장되어 있으므로
        # 단순히 전체 역 목록을 모든 호선에 표시
        pass

    # 호선별 역 목록을 meta에 저장해두었다면 (추천)
    return meta["station_map"]
