import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="뉴욕 주택 인공지능 예측", layout="wide")
st.title("🤖 뉴욕 집값 인공지능 예측기")
st.write("당신이 원하는 집의 조건을 입력하면, 인공지능이 뉴욕의 적정 가격을 예측합니다.")

# 2. 데이터 및 인공지능 모델 불러오기
@st.cache_resource
def load_model():
    # 깃허브에 올릴 다이어트 모델 이름
    return joblib.load('nyc_rf_model_small.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('nyc_final_data.csv')

rf_model = load_model()
df = load_data()

# 원본 훈련 데이터의 컬럼 순서 (더미 변수화 된 상태)
# app.py 작성 시점에서는 X.columns 정보를 하드코딩하거나 
# 훈련 데이터 컬럼 구조를 미리 저장해두는 것이 좋으나, 
# 가장 기초적인 작동을 위해 필수 컬럼 구조를 임의로 맞춥니다.
train_columns = [
    'bedrooms', 'bathrooms', 'livingArea', 'is_harlem',
    'dist_to_central_park', 'dist_to_nearest_subway', 
    'homeType_Condo', 'homeType_MultiFamily', 'homeType_Single Family', 'homeType_Townhouse'
]

# 3. 사이드바: 가상의 집 조건 입력받기
st.sidebar.header("원하는 집의 조건을 만들어보세요")
bed = st.sidebar.slider("방 개수", 1, 5, 2)
bath = st.sidebar.slider("욕실 개수", 1.0, 5.0, 1.0, 0.5)
living_area = st.sidebar.slider("집 크기 (sqft)", 500, 5000, 1500)
subway_dist = st.sidebar.slider("지하철역까지 거리 (km)", 0.1, 5.0, 1.0)
park_dist = st.sidebar.slider("센트럴 파크까지 거리 (km)", 0.1, 20.0, 5.0)

# 4. 사용자 입력값을 데이터프레임으로 변환 (훈련 데이터 구조와 동일하게)
# (이 간단한 데모에서는 할렘 여부는 0, 집 유형은 기본 아파트(Condo)로 가정합니다)
input_data = pd.DataFrame([[
    bed, bath, living_area, 0, park_dist, subway_dist, 1, 0, 0, 0
]], columns=train_columns)

# 5. 인공지능 가격 예측 수행
# (스케일링은 생략된 간단한 버전입니다)
predicted_price = rf_model.predict(input_data)[0]

# 6. 결과 출력
st.subheader("💡 인공지능 예측 결과")
st.metric(label="이 조건의 집, 적정 가격은?", value=f"${predicted_price:,.0f}")

st.write("---")
st.write("이 예측은 실제 시장 가격과 다를 수 있으며, 머신러닝 학습 목적으로 제작되었습니다.")
