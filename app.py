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


# ===============================
# 데이터 로딩
# ===============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()

line_encoder = meta["encoders"]["line"]
station_encoder = meta["encoders"]["station"]
station_map = meta["station_map"]

inv_label_map = {
    0: "여유",
    1: "보통",
    2: "혼잡"
}


# ===============================
# UI
# ===============================
st.title("🚇 지하철 혼잡도 예측 시스템")
st.markdown(
    "지하철 **호선, 역, 시간, 요일**을 선택하면 "
    "해당 시점의 **혼잡도**를 예측합니다."
)

st.divider()

# ---- 호선 선택
line = st.selectbox(
    "🚆 호선 선택",
    options=sorted(station_map.keys())
)

# ---- 역 선택
station = st.selectbox(
    "📍 역 선택",
    options=sorted(station_map[line])
)

# ---- 시간/분 선택
col1, col2 = st.columns(2)
with col1:
    hour = st.selectbox("⏰ 시간", list(range(0, 24)))
with col2:
    minute = st.selectbox("⏱️ 분", [0, 10, 20, 30, 40, 50])

# ---- 요일 선택
dow_map = {
    "월요일": 0,
    "화요일": 1,
    "수요일": 2,
    "목요일": 3,
    "금요일": 4,
    "토요일": 5,
    "일요일": 6,
}
dow_label = st.selectbox("📅 요일", list(dow_map.keys()))
dow = dow_map[dow_label]

st.divider()


# ===============================
# 예측
# ===============================
if st.button("🔍 혼잡도 예측", use_container_width=True):
    # 인코딩
    x = np.array([[
        line_encoder[line],
        station_encoder[station],
        hour,
        dow
    ]])

    # 예측
    probs = model.predict_proba(x)[0]
    pred_label = int(np.argmax(probs))
    pred_name = inv_label_map[pred_label]

    # ---- 결과 표시
    st.subheader("📊 예측 결과")

    if pred_label == 2:
        st.error(f"🚨 혼잡도: **{pred_name}**")
    elif pred_label == 1:
        st.warning(f"⚠️ 혼잡도: **{pred_name}**")
    else:
        st.success(f"✅ 혼잡도: **{pred_name}**")

    # ---- 확률 시각화
    prob_df = pd.DataFrame({
        "혼잡도": ["여유", "보통", "혼잡"],
        "확률": probs
    })

    st.bar_chart(prob_df.set_index("혼잡도"))

    st.caption("※ 본 결과는 과거 지하철 이용 데이터를 기반으로 한 예측 결과입니다.")
