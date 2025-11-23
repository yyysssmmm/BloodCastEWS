#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMAX 모델을 사용한 헌혈 데이터 시계열 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

print("=" * 80)
print("SARIMAX 모델을 사용한 헌혈 데이터 시계열 예측")
print("=" * 80)

# ============================================
# 1. 입력 변수 정의
# ============================================

# print("\n1. 입력 변수 정의...")
# GENDER = "남자"
# BLOOD_TYPE = "O+"
# BLOOD_CENTER = "서울동부"
# AGE_GROUP = "30~39"
# OCCUPATION = "회사원"

print("\n1. 입력 변수 정의...")
GENDER = "남자"
BLOOD_TYPE = "A+"
BLOOD_CENTER = "부산"
AGE_GROUP = "20~29"
OCCUPATION = "대학생"

print(f"   - 성별: {GENDER}")
print(f"   - 혈액형별: {BLOOD_TYPE}")
print(f"   - 혈액원별: {BLOOD_CENTER}")
print(f"   - 연령대: {AGE_GROUP}")
print(f"   - 직업: {OCCUPATION}")

# Segment 변수 생성 (파일명에 사용)
SEGMENT = f"{GENDER}_{BLOOD_TYPE}_{BLOOD_CENTER}_{AGE_GROUP}_{OCCUPATION}"
print(f"   - Segment: {SEGMENT}")

# 결과 저장 폴더 생성
OUTPUT_DIR = "sarimax_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"   - 결과 저장 폴더: {OUTPUT_DIR}/")

# ============================================
# 2. 데이터 로딩 및 필터링
# ============================================
print("\n2. 데이터 로딩 및 필터링...")
df = pd.read_csv("blood_donation_data_monthly.csv", encoding="utf-8")

# 조건에 맞는 행 필터링
filtered_df = df[
    (df["성별"] == GENDER)
    & (df["혈액형별"] == BLOOD_TYPE)
    & (df["혈액원별"] == BLOOD_CENTER)
    & (df["연령대"] == AGE_GROUP)
    & (df["직업"] == OCCUPATION)
]

if len(filtered_df) == 0:
    raise ValueError("조건에 맞는 데이터를 찾을 수 없습니다!")

print(f"   - 필터링된 행 수: {len(filtered_df)}개")

# ============================================
# 3. 시계열 데이터 추출
# ============================================
print("\n3. 시계열 데이터 추출...")

# 월별 컬럼 추출 (2005-1 ~ 2024-12)
monthly_columns = [col for col in df.columns if "-" in col and col[0].isdigit()]
monthly_columns.sort()

# 시계열 데이터 추출
ts_data = filtered_df[monthly_columns].iloc[0].values

# 날짜 인덱스 생성
dates = pd.date_range(start="2005-01", end="2024-12", freq="MS")  # MS = Month Start
ts_series = pd.Series(ts_data, index=dates)

print(f"   - 시계열 길이: {len(ts_series)}개월")
print(
    f"   - 기간: {ts_series.index[0].strftime('%Y-%m')} ~ {ts_series.index[-1].strftime('%Y-%m')}"
)
print(f"   - 최소값: {ts_series.min():.0f}")
print(f"   - 최대값: {ts_series.max():.0f}")
print(f"   - 평균값: {ts_series.mean():.2f}")

# ============================================
# 4. Train/Test 데이터 분할
# ============================================
print("\n4. Train/Test 데이터 분할...")

# 마지막 N개월을 테스트 데이터로 사용 (기본값: 12개월)
TEST_MONTHS = 12  # 필요시 변경 가능

train_data = ts_series[:-TEST_MONTHS]
test_data = ts_series[-TEST_MONTHS:]

print(
    f"   - Train 데이터: {len(train_data)}개월 ({train_data.index[0].strftime('%Y-%m')} ~ {train_data.index[-1].strftime('%Y-%m')})"
)
print(
    f"   - Test 데이터: {len(test_data)}개월 ({test_data.index[0].strftime('%Y-%m')} ~ {test_data.index[-1].strftime('%Y-%m')})"
)

# ============================================
# 5. Auto ARIMA로 최적 파라미터 찾기
# ============================================
print("\n5. Auto ARIMA로 최적 파라미터 찾기...")
print("   - 이 과정은 몇 분이 걸릴 수 있습니다...")

auto_model = auto_arima(
    train_data,
    seasonal=True,
    m=12,  # 월별 데이터이므로 계절성 주기 = 12
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=5,  # 최대 (p+d+q) 값
    max_p=5,
    max_d=2,
    max_q=5,
    max_P=3,
    max_D=2,
    max_Q=3,
    trace=True,
    n_jobs=-1,
)

print(f"\n   - 최적 파라미터:")
print(f"     Order (p, d, q): {auto_model.order}")
print(f"     Seasonal Order (P, D, Q, s): {auto_model.seasonal_order}")
print(f"     AIC: {auto_model.aic():.2f}")

# ============================================
# 6. SARIMAX 모델 학습
# ============================================
print("\n6. SARIMAX 모델 학습 중...")

# 최적 파라미터로 SARIMAX 모델 생성
model = SARIMAX(
    train_data,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

# 모델 피팅
fitted_model = model.fit(disp=False)

print(f"   - 모델 학습 완료!")
print(f"   - AIC: {fitted_model.aic:.2f}")
print(f"   - BIC: {fitted_model.bic:.2f}")

# ============================================
# 7. 예측 수행
# ============================================
print("\n7. 예측 수행...")

# Train 데이터에 대한 fitted values
train_fitted = fitted_model.fittedvalues

# Test 데이터 예측
forecast = fitted_model.get_forecast(steps=len(test_data))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print(f"   - 예측 완료! ({len(forecast_mean)}개월)")

# ============================================
# 8. 모델 평가
# ============================================
print("\n8. 모델 평가...")

# Train 데이터 평가
train_mae = mean_absolute_error(train_data, train_fitted)
train_rmse = np.sqrt(mean_squared_error(train_data, train_fitted))
train_mape = mean_absolute_percentage_error(train_data, train_fitted) * 100

# Test 데이터 평가
test_mae = mean_absolute_error(test_data, forecast_mean)
test_rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
test_mape = mean_absolute_percentage_error(test_data, forecast_mean) * 100

print("\n   [Train 데이터 평가]")
print(f"   - MAE (Mean Absolute Error): {train_mae:.2f}")
print(f"   - RMSE (Root Mean Squared Error): {train_rmse:.2f}")
print(f"   - MAPE (Mean Absolute Percentage Error): {train_mape:.2f}%")

print("\n   [Test 데이터 평가]")
print(f"   - MAE (Mean Absolute Error): {test_mae:.2f}")
print(f"   - RMSE (Root Mean Squared Error): {test_rmse:.2f}")
print(f"   - MAPE (Mean Absolute Percentage Error): {test_mape:.2f}%")

# ============================================
# 9. 미래 예측 (선택사항)
# ============================================
print("\n9. 미래 예측 (2025년 12개월)...")

future_forecast = fitted_model.get_forecast(steps=12)
future_mean = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

future_dates = pd.date_range(start="2025-01", end="2025-12", freq="MS")
print(f"   - 2025년 예측 완료!")

# ============================================
# 10. 시각화
# ============================================
print("\n10. 시각화 생성 중...")

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# 전체 시계열 및 예측
ax1 = axes[0]
ax1.plot(
    train_data.index, train_data.values, label="Train 데이터", color="blue", linewidth=2
)
ax1.plot(
    train_fitted.index,
    train_fitted.values,
    label="Train Fitted",
    color="green",
    linestyle="--",
    linewidth=1.5,
)
ax1.plot(
    test_data.index,
    test_data.values,
    label="Test 데이터 (실제)",
    color="orange",
    linewidth=2,
)
ax1.plot(
    forecast_mean.index,
    forecast_mean.values,
    label="Test 예측",
    color="red",
    linestyle="--",
    linewidth=2,
)
ax1.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.3,
    color="red",
    label="95% 신뢰구간",
)
ax1.plot(
    future_dates,
    future_mean.values,
    label="2025년 예측",
    color="purple",
    linestyle="--",
    linewidth=2,
)
ax1.fill_between(
    future_ci.index,
    future_ci.iloc[:, 0],
    future_ci.iloc[:, 1],
    alpha=0.3,
    color="purple",
    label="95% 신뢰구간 (2025)",
)

ax1.set_title(
    f"SARIMAX 모델 예측 결과\n{GENDER}, {BLOOD_TYPE}, {BLOOD_CENTER}, {AGE_GROUP}, {OCCUPATION}",
    fontsize=14,
    fontweight="bold",
)
ax1.set_xlabel("날짜", fontsize=12)
ax1.set_ylabel("헌혈 건수", fontsize=12)
ax1.legend(loc="best", fontsize=10)
ax1.grid(True, alpha=0.3)

# Test 데이터 상세 비교
ax2 = axes[1]
x_pos = np.arange(len(test_data))
width = 0.35

ax2.bar(
    x_pos - width / 2,
    test_data.values,
    width,
    label="실제값",
    color="orange",
    alpha=0.7,
)
ax2.bar(
    x_pos + width / 2,
    forecast_mean.values,
    width,
    label="예측값",
    color="red",
    alpha=0.7,
)
ax2.set_xlabel("월", fontsize=12)
ax2.set_ylabel("헌혈 건수", fontsize=12)
ax2.set_title("Test 데이터 실제값 vs 예측값 비교", fontsize=12, fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([date.strftime("%Y-%m") for date in test_data.index], rotation=45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
forecast_filename = os.path.join(OUTPUT_DIR, f"sarimax_forecast_result_{SEGMENT}.png")
plt.savefig(forecast_filename, dpi=300, bbox_inches="tight")
print(f"   - 그래프 저장 완료: {forecast_filename}")

# 잔차 분석
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))

# 잔차 시계열
residuals = train_data - train_fitted
axes2[0, 0].plot(residuals.index, residuals.values, color="blue")
axes2[0, 0].axhline(y=0, color="red", linestyle="--")
axes2[0, 0].set_title("잔차 시계열", fontsize=12, fontweight="bold")
axes2[0, 0].set_xlabel("날짜", fontsize=10)
axes2[0, 0].set_ylabel("잔차", fontsize=10)
axes2[0, 0].grid(True, alpha=0.3)

# 잔차 히스토그램
axes2[0, 1].hist(residuals.values, bins=30, color="blue", alpha=0.7, edgecolor="black")
axes2[0, 1].set_title("잔차 분포", fontsize=12, fontweight="bold")
axes2[0, 1].set_xlabel("잔차", fontsize=10)
axes2[0, 1].set_ylabel("빈도", fontsize=10)
axes2[0, 1].grid(True, alpha=0.3, axis="y")

# Q-Q Plot
from scipy import stats

stats.probplot(residuals.values, dist="norm", plot=axes2[1, 0])
axes2[1, 0].set_title("Q-Q Plot (정규성 검정)", fontsize=12, fontweight="bold")
axes2[1, 0].grid(True, alpha=0.3)

# ACF (자기상관함수)
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(residuals, lags=24, ax=axes2[1, 1], title="잔차 ACF")
axes2[1, 1].set_title("잔차 ACF (자기상관함수)", fontsize=12, fontweight="bold")

plt.tight_layout()
residual_filename = os.path.join(OUTPUT_DIR, f"sarimax_residual_analysis_{SEGMENT}.png")
plt.savefig(residual_filename, dpi=300, bbox_inches="tight")
print(f"   - 잔차 분석 그래프 저장 완료: {residual_filename}")

# ============================================
# 11. 평가 결과를 텍스트 파일로 저장
# ============================================
print("\n11. 평가 결과 저장 중...")

evaluation_filename = os.path.join(OUTPUT_DIR, f"evaluation_{SEGMENT}.txt")

with open(evaluation_filename, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SARIMAX 모델 평가 결과\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("=" * 80 + "\n")
    f.write("입력 변수\n")
    f.write("=" * 80 + "\n")
    f.write(f"성별: {GENDER}\n")
    f.write(f"혈액형별: {BLOOD_TYPE}\n")
    f.write(f"혈액원별: {BLOOD_CENTER}\n")
    f.write(f"연령대: {AGE_GROUP}\n")
    f.write(f"직업: {OCCUPATION}\n")
    f.write(f"Segment: {SEGMENT}\n\n")

    f.write("=" * 80 + "\n")
    f.write("데이터 정보\n")
    f.write("=" * 80 + "\n")
    f.write(f"전체 시계열 길이: {len(ts_series)}개월\n")
    f.write(
        f"기간: {ts_series.index[0].strftime('%Y-%m')} ~ {ts_series.index[-1].strftime('%Y-%m')}\n"
    )
    f.write(
        f"Train 데이터: {len(train_data)}개월 ({train_data.index[0].strftime('%Y-%m')} ~ {train_data.index[-1].strftime('%Y-%m')})\n"
    )
    f.write(
        f"Test 데이터: {len(test_data)}개월 ({test_data.index[0].strftime('%Y-%m')} ~ {test_data.index[-1].strftime('%Y-%m')})\n"
    )
    f.write(f"최소값: {ts_series.min():.0f}\n")
    f.write(f"최대값: {ts_series.max():.0f}\n")
    f.write(f"평균값: {ts_series.mean():.2f}\n")
    f.write(f"표준편차: {ts_series.std():.2f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("모델 정보\n")
    f.write("=" * 80 + "\n")
    f.write(f"최적 Order (p, d, q): {auto_model.order}\n")
    f.write(f"최적 Seasonal Order (P, D, Q, s): {auto_model.seasonal_order}\n")
    f.write(f"AIC: {fitted_model.aic:.2f}\n")
    f.write(f"BIC: {fitted_model.bic:.2f}\n")
    f.write(f"Log Likelihood: {fitted_model.llf:.2f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("Train 데이터 평가 결과\n")
    f.write("=" * 80 + "\n")
    f.write(f"MAE (Mean Absolute Error): {train_mae:.2f}\n")
    f.write(f"RMSE (Root Mean Squared Error): {train_rmse:.2f}\n")
    f.write(f"MAPE (Mean Absolute Percentage Error): {train_mape:.2f}%\n\n")

    f.write("=" * 80 + "\n")
    f.write("Test 데이터 평가 결과\n")
    f.write("=" * 80 + "\n")
    f.write(f"MAE (Mean Absolute Error): {test_mae:.2f}\n")
    f.write(f"RMSE (Root Mean Squared Error): {test_rmse:.2f}\n")
    f.write(f"MAPE (Mean Absolute Percentage Error): {test_mape:.2f}%\n\n")

    f.write("=" * 80 + "\n")
    f.write("Test 데이터 상세 비교\n")
    f.write("=" * 80 + "\n")
    f.write(
        f"{'월':<12} {'실제값':<10} {'예측값':<10} {'오차':<10} {'오차율(%)':<10}\n"
    )
    f.write("-" * 60 + "\n")
    for i, date in enumerate(test_data.index):
        actual = test_data.iloc[i]
        predicted = forecast_mean.iloc[i]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0
        f.write(
            f"{date.strftime('%Y-%m'):<12} {actual:<10.2f} {predicted:<10.2f} {error:<10.2f} {error_pct:<10.2f}\n"
        )
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("2025년 예측 결과\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'월':<12} {'예측값':<15} {'하한(95%)':<15} {'상한(95%)':<15}\n")
    f.write("-" * 60 + "\n")
    for i, date in enumerate(future_dates):
        pred = future_mean.iloc[i]
        lower = future_ci.iloc[i, 0]
        upper = future_ci.iloc[i, 1]
        f.write(
            f"{date.strftime('%Y-%m'):<12} {pred:<15.2f} {lower:<15.2f} {upper:<15.2f}\n"
        )
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("생성된 파일\n")
    f.write("=" * 80 + "\n")
    f.write(f"- sarimax_forecast_result_{SEGMENT}.png: 예측 결과 시각화\n")
    f.write(f"- sarimax_residual_analysis_{SEGMENT}.png: 잔차 분석\n")
    f.write(f"- evaluation_{SEGMENT}.txt: 평가 결과 (본 파일)\n")
    f.write("\n" + "=" * 80 + "\n")

print(f"   - 평가 결과 저장 완료: {evaluation_filename}")

# ============================================
# 12. 결과 요약 출력
# ============================================
print("\n" + "=" * 80)
print("모델 학습 및 예측 완료!")
print("=" * 80)
print(f"\n[모델 정보]")
print(f"  - 최적 Order: {auto_model.order}")
print(f"  - 최적 Seasonal Order: {auto_model.seasonal_order}")
print(f"  - AIC: {fitted_model.aic:.2f}")
print(f"  - BIC: {fitted_model.bic:.2f}")

print(f"\n[예측 결과 요약]")
print(f"  - Test 데이터 MAE: {test_mae:.2f}")
print(f"  - Test 데이터 RMSE: {test_rmse:.2f}")
print(f"  - Test 데이터 MAPE: {test_mape:.2f}%")

print(f"\n[생성된 파일]")
print(f"  - {forecast_filename}")
print(f"  - {residual_filename}")
print(f"  - {evaluation_filename}")

print("\n" + "=" * 80)
