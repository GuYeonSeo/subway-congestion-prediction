import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="지하철 혼잡도 예측",
    page_icon="🚇",
    layout="centered"
)

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/meta.json")

LABEL_NAME = {0: "여유", 1: "보통", 2: "혼잡"}


# ===============================
# 로딩
# ===============================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    if not META_PATH.exists():
        raise FileNotFoundError(f"메타 파일을 찾을 수 없습니다: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()

# meta.json에 station_map이 있어야 드롭다운 구성 가능
station_map = meta.get("station_map")
if station_map is None:
    st.error("models/meta.json에 'station_map'이 없습니다. preprocess.py로 meta.json을 최신으로 다시 생성해주세요.")
    st.stop()

# encoders는 있을 수도/없을 수도 있음 (모델 학습 방식에 따라 다름)
encoders = meta.get("encoders", None)


# ===============================
# UI
# ===============================
st.title("🚇 지하철 혼잡도 예측 시스템")
st.markdown("호선/역/시간/요일을 선택하면 혼잡도를 예측합니다.")
st.divider()

# 호선/역 선택
line = st.selectbox("🚆 호선 선택", options=sorted(station_map.keys()))
station = st.selectbox("📍 역 선택", options=sorted(station_map[line]))

# 시간/분 선택
st.markdown("⏰ 시간 선택")
c1, c2 = st.columns(2)
with c1:
    hour = st.selectbox("시", list(range(0, 24)), index=8)
with c2:
    minute = st.selectbox("분", [0, 10, 20, 30, 40, 50], index=0)

# 요일 선택
dow_labels = ["월", "화", "수", "목", "금", "토", "일"]
dow = st.selectbox("📅 요일", options=list(range(7)), format_func=lambda x: dow_labels[x])

st.divider()


# ===============================
# 예측 함수 (핵심: DataFrame로 넣기)
# ===============================
def predict_with_dataframe(_model, _line, _station, _hour, _dow):
    """
    모델이 어떤 입력을 기대하든 최대한 호환되게 예측.
    1) 문자열 line/station DataFrame으로 먼저 시도
    2) 실패하면 (encoders가 있을 때) 숫자 인코딩 DataFrame으로 재시도
    """
    # 1) 문자열 그대로 (ColumnTransformer+OneHotEncoder 파이프라인에 일반적으로 맞음)
    X_str = pd.DataFrame([{
        "line": _line,
        "station": _station,
        "hour": int(_hour),
        "dow": int(_dow)
    }])
    try:
        probs = _model.predict_proba(X_str)[0]
        return probs, "string"
    except Exception as e1:
        # 2) 인코딩이 있는 경우 숫자로 변환해서 DataFrame으로 재시도
        if encoders is None:
            raise e1

        line_enc_map = encoders.get("line")
        station_enc_map = encoders.get("station")
        if line_enc_map is None or station_enc_map is None:
            raise e1

        if _line not in line_enc_map or _station not in station_enc_map:
            raise ValueError("선택한 호선/역이 meta.json encoders에 없습니다. meta.json을 최신으로 다시 생성해주세요.")

        X_num = pd.DataFrame([{
            "line": int(line_enc_map[_line]),
            "station": int(station_enc_map[_station]),
            "hour": int(_hour),
            "dow": int(_dow)
        }])

        probs = _model.predict_proba(X_num)[0]
        return probs, "encoded"


# ===============================
# 버튼
# ===============================
if st.button("🔍 혼잡도 예측", use_container_width=True):
    st.info(f"입력값: **{line} / {station} / {hour:02d}:{minute:02d} / {dow_labels[dow]}요일**")

    try:
        probs, mode = predict_with_dataframe(model, line, station, hour, dow)
        pred_label = int(np.argmax(probs))
        pred_name = LABEL_NAME[pred_label]

        st.subheader("📊 예측 결과")
        if pred_label == 2:
            st.error(f"🚨 예측 혼잡도: **{pred_name}**")
        elif pred_label == 1:
            st.warning(f"⚠️ 예측 혼잡도: **{pred_name}**")
        else:
            st.success(f"✅ 예측 혼잡도: **{pred_name}**")

        st.caption(f"예측 입력 모드: {mode} (모델 구조에 맞게 자동 선택)")

        # 확률 표시
        prob_df = pd.DataFrame({
            "혼잡도": ["여유", "보통", "혼잡"],
            "확률": probs
        }).set_index("혼잡도")

        st.markdown("### 클래스별 확률")
        st.bar_chart(prob_df)

        st.markdown("### 확률 값")
        st.write({
            "여유": float(probs[0]),
            "보통": float(probs[1]),
            "혼잡": float(probs[2]),
        })

    except Exception as e:
        st.error("예측 중 오류가 발생했습니다.")
        st.exception(e)
