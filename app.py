import streamlit as st
import joblib
import json
import pandas as pd

@st.cache_data
def load_station_map():
    df = pd.read_csv("data/processed.csv", usecols=["line", "station"])
    df = df.drop_duplicates()
    return {
        line: sorted(group["station"].unique().tolist())
        for line, group in df.groupby("line")
    }

station_map = load_station_map()


# 모델 로드
model = joblib.load("models/model.joblib")

# 라벨 이름
LABEL_NAME = {0: "여유", 1: "보통", 2: "혼잡"}

st.set_page_config(page_title="지하철 혼잡도 예측", layout="centered")

st.title("🚇 지하철 혼잡도 예측 시스템")
st.markdown("시간대와 역 정보를 입력하면 혼잡도를 예측합니다.")

st.divider()

# 입력 UI
line = st.selectbox("🚇 호선", sorted(station_map.keys()))

station = st.selectbox(
    "📍 역 선택",
    station_map[line]
)

st.markdown("⏰ 시간 선택")

col1, col2 = st.columns(2)

with col1:
    hour = st.selectbox("시", list(range(0, 24)), index=8)

with col2:
    minute = st.selectbox("분", [0, 10, 20, 30, 40, 50], index=0)

dow = st.selectbox(
    "📅 요일",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["월", "화", "수", "목", "금", "토", "일"][x]
)

# 예측 버튼
if st.button("혼잡도 예측"):
    X = pd.DataFrame([{
        "line": line,
        "station": station,
        "hour": hour,
        "dow": dow
    }])

    probs = model.predict_proba(X)[0]
    label = probs.argmax()

    st.subheader("📊 예측 결과")
    st.success(f"**예측 혼잡도: {LABEL_NAME[label]}**")

    st.markdown("### 클래스별 확률")
    st.bar_chart({
        "여유": probs[0],
        "보통": probs[1],
        "혼잡": probs[2]
    })
