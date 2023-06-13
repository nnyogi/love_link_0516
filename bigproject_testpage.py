
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from keras.models import load_model
import os, glob
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import plotly.express as px


os.chdir('/Users/User/빅프로젝트/Data')

# def Lone_Person_Dataset_Loader(group_name, region_name, gender_name, age_name):
    
#     df = pd.read_csv('Lone_Person_Data/group_n.csv')

#     # 20, 25=>20대 ~ 70, 75=>70대 연령대 전처리
#     df['연령대'] = df['연령대'].astype(float)
#     conditions = [
#         (df['연령대'] >= 20) & (df['연령대'] < 30),
#         (df['연령대'] >= 30) & (df['연령대'] < 40),
#         (df['연령대'] >= 40) & (df['연령대'] < 50),
#         (df['연령대'] >= 50) & (df['연령대'] < 60),
#         (df['연령대'] >= 60) & (df['연령대'] < 70),
#         (df['연령대'] >= 70) & (df['연령대'] < 80)
#     ]
#     values = ['20대', '30대', '40대', '50대', '60대', '70대']

#     # Assign the age group based on the conditions
#     df['연령대'] = np.select(conditions, values, default='80대')
#     df['성별'] = df['성별'].replace({1: '남성', 2: '여성'})
#     df['month'] = df['month'].apply(lambda x: str('-'.join(str(x).split('.'))))
#     df['month'] = pd.to_datetime(df['month'])
#     df['month'] = df['month'].dt.to_period('M')
#     x_train = df.loc[(df['month'] != '2023-01') & (df['month'] != '2023-02') & (df['month'] != '2023-03')]
#     x_test = df.loc[(df['month'] == '2023-01') | (df['month'] == '2023-02') | (df['month'] == '2023-03')]
    
#     df_ts = x_train.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

#     # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블) 자치구, 성별, 연령대별 커뮤니케이션이 적은 집단의 미래 수치 예측(다른 집단도 각각 돌려야 함)
#     df_pivot = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
#     frame = pd.DataFrame(df_pivot[region_name, gender_name, age_name])
#     df_col = df_pivot[region_name, gender_name, age_name].fillna(0)
#     column_name = frame.columns.values[0]
    
#     # PeriodIndex를 DatetimeIndex으로 바꾸기 ('M'한달 간격)
#     new_index = df_col.index.to_timestamp(freq='M')

#     # Datetimeindex 바꾼 데이터프레임
#     df_col = pd.DataFrame(df_pivot[region_name, gender_name, age_name].values, index=new_index, columns=[column_name])

#     # 모델 학습
#     model = ARIMA(df_col, order=(2, 1, 1))  # p, d, q
#     fitted_model = model.fit()

#     # 2023 예측
#     start_idx = len(df_col) - 1  # 지난 관측값에서 시작
#     end_idx = start_idx + 12  # 12개월치 예측
#     forecast = fitted_model.predict(start=start_idx, end=end_idx, typ='levels')
#     forecast_index = pd.date_range(start=df_col.index[-1], periods=13, freq='M')[1:]
#     forecast = pd.Series(forecast, index=forecast_index)
    
#     df_ts2 = x_test.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

#     # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블) 자치구, 성별, 연령대별 평일 외출이 적은 집단의 미래 수치 예측(다른 집단도 각각 돌려야 함)
#     df_pivot2 = df_ts2.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
    
#         # 한글 설정
#     import matplotlib.font_manager as fm

#     # 폰트 경로 설정
#     plt.rc('font', family='Malgun Gothic')
#     df_col2 = df_pivot2[region_name, gender_name, age_name].fillna(0)
#     from sklearn.metrics import mean_squared_error

#     # 예측값과 실제값 사이의 MSE 계산
#     mse = mean_squared_error(df_pivot2[region_name, gender_name, age_name], forecast[0:3], squared=False)
    
#     # 예측 결과 시각화
#     plt.figure(figsize=(10, 6))
#     plt.plot(df_col.index, df_pivot[region_name, gender_name, age_name], label='실제값', marker='o', color='pink')
#     for i in range(len(df_col.index)):
#         height = df_pivot[region_name, gender_name, age_name][i]
#         plt.text(df_col.index[i], height + 0.25, '%.1f' %height, ha='center', va='bottom', size = 10)

#     plt.plot(forecast.index, forecast, label='예측값', marker='o', color='gray')
#     for i in range(len(forecast.index)):
#         height = forecast[i]
#         plt.text(forecast.index[i], height + 1, '%.1f' %height, ha='center', va='bottom', size = 10, color='gray')


#     plt.plot(df_col2.index, df_pivot2[region_name, gender_name, age_name], label='미래 실제값', marker='o', color='pink')
#     for i in range(len(df_col2.index)):
#         height = df_pivot2[region_name, gender_name, age_name][i]
#         plt.text(df_col2.index[i], height + 0.25, '%.1f' %height, ha='center', va='bottom', size = 10)

#     plt.title(str(column_name) + ' ' + group_name)

#     plt.xlabel('Date')
#     plt.ylabel('1인 가구 수')
#     plt.legend()
#     plt.show()

    
#페이지 코드 작성
st.set_page_config(layout="wide")

st.markdown("## IoT 사회복지사 통계 포털")
tab1, tab2, tab3 = st.tabs(["IoT 통계", "감정분석 통계", "1인가구 집단 시계열 통계"])

with tab1: # IoT 통계
    
    st.markdown('통계')
#     Lone_Person_Dataset_Loader('평일 외출이 적은 집단', '강남구', '여성', '50대')

#     t1_col1_1, t1_col1_2 = st.columns([0.5, 0.5])
#     with t1_col1_1:
#         st.markdown('기가지니 감정 분류 통계')
#     with t1_col1_2:
#         st.markdown('10개 집단 시계열 차트')
    
#     st.markdown('결과값')


with tab2: # 감정분석 통계
    
    os.chdir('/Users/User/빅프로젝트')
    dff = pd.read_excel('Emotion_Stat_Dataset.xlsx')
    dfff = pd.read_excel('week_df.xlsx')

    t2_col1_1, t2_col1_2, t2_col1_3 = st.columns([0.2, 0.2, 0.05])
    
    with t2_col1_1:
        tab2_selectbox_1 = st.selectbox('사용자 선택', dff['User'].unique(), key = 'tab2 사용자 선택 1')
        dff_Result = dff.loc[dff['User'] == tab2_selectbox_1]
        
        chart_1 = px.bar( dff_Result, x = 'Start_Time', y = 'Negative_Count')     
        st.plotly_chart(chart_1, use_container_width=True)
        
    with t2_col1_2:
        tab2_selectbox_2 = st.selectbox('사용자 선택', dfff['User'].unique(), key = 'tab2 사용자 선택 2')
        dfff_Result = dfff.loc[dfff['User'] == tab2_selectbox_2]
        
        chart_2 = px.bar( dfff_Result, x = 'Week_Start_Date', y = 'Total number of negative sentence predictions per week') 
        st.plotly_chart(chart_2, use_container_width=True)
        
with t2_col1_3:
    st.success("정상입니다")
    st.info("주의가 필요합니다")
    st.warning("심리 상담이 필요합니다")
    st.error("즉시 조치가 필요합니다")
        
with tab3: # 1인가구 집단 시계열 통계
    st.markdown('통계')

    t3_col1_1, t3_col1_2, t3_col1_3, t3_col1_4, t3_col1_5 = st.columns([0.2,0.2,0.2,0.2,0.2])
    with t3_col1_1:
        st.markdown('집단 구분')
        st.selectbox('10개 중 1개 선택', ['cats', 'dogs'])
    with t3_col1_2:
        st.markdown('성별')
    with t3_col1_3:
        st.markdown('연령대')
    with t3_col1_4:
        st.markdown('자치구')
    with t3_col1_5:
        st.button('선택')
        
    t3_col2_1, t3_col2_2 = st.columns([0.5, 0.5])
    with t3_col1_1:
        st.markdown('기가지니 감정 분류 통계')
    with t3_col1_2:
        st.markdown('10개 집단 시계열 차트')
    st.markdown('결과값')
