#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
집계된 세그먼트별 SARIMAX 모델 생성 및 비교
- 전체 데이터 모델 vs 세그먼트별 모델 합산 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import warnings
import os
from datetime import datetime
from sarimax_utils import (
    create_sarimax_model,
    save_sarimax_evaluation,
    plot_sarimax_forecast,
)

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

print("=" * 80)
print("집계된 세그먼트별 SARIMAX 모델 생성 및 비교")
print("=" * 80)

# ============================================
# 1. 입력 변수 정의
# ============================================
print("\n1. 입력 변수 정의...")

# 지역 및 혈액형 (고정)
BLOOD_CENTER = "부산"
BLOOD_TYPE = "O+"

# 제외할 카테고리
EXCLUDED_AGE_GROUPS = ["60세 이상"]
EXCLUDED_OCCUPATIONS = ["가사", "종교직", "공무원", "자영업"]
# 참고: RH- 혈액형은 이미 EDA에서 제외 결정되었지만,
# 여기서는 특정 지역/혈액형(O+)에 대해 분석하므로 RH-는 자동으로 제외됨

print(f"   - 지역: {BLOOD_CENTER}")
print(f"   - 혈액형: {BLOOD_TYPE}")
print(f"   - 제외 연령대: {EXCLUDED_AGE_GROUPS}")
print(f"   - 제외 직업: {EXCLUDED_OCCUPATIONS}")

# 결과 저장 폴더
OUTPUT_DIR = "region_bloodtype"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "segments"), exist_ok=True)

# ============================================
# 2. 데이터 로딩
# ============================================
print("\n2. 데이터 로딩...")
df = pd.read_csv("data/blood_donation_data_monthly.csv", encoding="utf-8")

# 월별 컬럼 추출 (2005-1 ~ 2024-12)
monthly_columns = [col for col in df.columns if "-" in col and col[0].isdigit()]
monthly_columns.sort()

print(f"   - 전체 데이터 행 수: {len(df)}")
print(f"   - 월별 컬럼 수: {len(monthly_columns)}")

# ============================================
# 3. 전체 데이터 합산 (부산, O+)
# ============================================
print("\n3. 전체 데이터 합산 (부산, O+)...")

# 부산, O+ 필터링
overall_df = df[(df["혈액원별"] == BLOOD_CENTER) & (df["혈액형별"] == BLOOD_TYPE)]

if len(overall_df) == 0:
    raise ValueError(f"{BLOOD_CENTER}, {BLOOD_TYPE} 데이터를 찾을 수 없습니다!")

# 모든 세그먼트 합산
overall_ts_data = overall_df[monthly_columns].sum(axis=0).values
# float 타입으로 명시적 변환
overall_ts_data = pd.to_numeric(overall_ts_data, errors="coerce").astype(float)
overall_ts_data = np.nan_to_num(overall_ts_data, nan=0.0)

# 날짜 인덱스 생성
dates = pd.date_range(start="2005-01", end="2024-12", freq="MS")
overall_ts_series = pd.Series(overall_ts_data, index=dates, dtype=float)

print(f"   - 전체 세그먼트 수: {len(overall_df)}")
print(f"   - 전체 총합: {overall_ts_series.sum():.0f}")

# ============================================
# 4. 제외된 세그먼트 총량 계산
# ============================================
print("\n4. 제외된 세그먼트 총량 계산...")

# 제외 조건: 부산, O+ 내에서 제외 연령대 또는 제외 직업을 가진 세그먼트
# (RH-는 O+에는 없으므로 제외)
excluded_df = overall_df[
    (overall_df["연령대"].isin(EXCLUDED_AGE_GROUPS))
    | (overall_df["직업"].isin(EXCLUDED_OCCUPATIONS))
]

excluded_ts_data = excluded_df[monthly_columns].sum(axis=0).values
# float 타입으로 명시적 변환
excluded_ts_data = pd.to_numeric(excluded_ts_data, errors="coerce").astype(float)
excluded_ts_data = np.nan_to_num(excluded_ts_data, nan=0.0)
excluded_ts_series = pd.Series(excluded_ts_data, index=dates, dtype=float)

excluded_total = excluded_ts_series.sum()
overall_total = overall_ts_series.sum()
scaling_factor = (overall_total - excluded_total) / overall_total

print(f"   - 제외된 세그먼트 수: {len(excluded_df)}")
print(f"   - 제외된 세그먼트 총합: {excluded_total:.0f}")
print(f"   - 전체 총합: {overall_total:.0f}")
print(f"   - 스케일링 계수: {scaling_factor:.4f}")

# ============================================
# 5. 전체 데이터 SARIMAX 모델 생성
# ============================================
print("\n5. 전체 데이터 SARIMAX 모델 생성...")
print("   (이 과정은 몇 분이 걸릴 수 있습니다...)")

overall_model_result = create_sarimax_model(overall_ts_series, verbose=True)

print(f"   - Train MAE: {overall_model_result['train_mae']:.2f}")
print(f"   - Test MAE: {overall_model_result['test_mae']:.2f}")

# 스케일링 적용
scaled_overall_future = overall_model_result["future_mean"] * scaling_factor
scaled_overall_forecast = overall_model_result["forecast_mean"] * scaling_factor

# ============================================
# 6. 세그먼트별 SARIMAX 모델 생성
# ============================================
print("\n6. 세그먼트별 SARIMAX 모델 생성...")

# 포함할 세그먼트 필터링
included_df = df[
    (df["혈액원별"] == BLOOD_CENTER)
    & (df["혈액형별"] == BLOOD_TYPE)
    & (~df["연령대"].isin(EXCLUDED_AGE_GROUPS))
    & (~df["직업"].isin(EXCLUDED_OCCUPATIONS))
]

print(f"   - 포함할 세그먼트 수: {len(included_df)}")

segment_results = []
segment_forecasts = []  # Test 예측값들
segment_futures = []  # 미래 예측값들

for idx, row in included_df.iterrows():
    gender = row["성별"]
    age_group = row["연령대"]
    occupation = row["직업"]

    segment_name = f"{gender}_{BLOOD_TYPE}_{BLOOD_CENTER}_{age_group}_{occupation}"

    # 시계열 데이터 추출 및 타입 변환
    ts_data = row[monthly_columns].values
    # object 타입을 float로 변환 (NaN 처리 포함)
    ts_data = pd.to_numeric(ts_data, errors="coerce").astype(float)
    # NaN을 0으로 채우기
    ts_data = np.nan_to_num(ts_data, nan=0.0)
    ts_series = pd.Series(ts_data, index=dates, dtype=float)

    # 0이 아닌 데이터만 있는 세그먼트만 처리
    if ts_series.sum() == 0:
        print(f"   - 건너뜀 (합계=0): {segment_name}")
        continue

    print(f"   - 처리 중: {segment_name} ({idx+1}/{len(included_df)})")

    try:
        # SARIMAX 모델 생성 (verbose=False로 설정하여 출력 최소화)
        model_result = create_sarimax_model(ts_series, verbose=False)

        # 결과 저장
        segment_results.append(
            {
                "segment_name": segment_name,
                "gender": gender,
                "age_group": age_group,
                "occupation": occupation,
                "model_result": model_result,
            }
        )

        # 예측값 저장
        segment_forecasts.append(model_result["forecast_mean"])
        segment_futures.append(model_result["future_mean"])

        # 개별 세그먼트 평가 저장
        save_sarimax_evaluation(
            model_result,
            segment_name,
            os.path.join(OUTPUT_DIR, "segments"),
            gender=gender,
            blood_type=BLOOD_TYPE,
            blood_center=BLOOD_CENTER,
            age_group=age_group,
            occupation=occupation,
        )

        # 개별 세그먼트 시각화
        plot_sarimax_forecast(
            model_result,
            segment_name,
            os.path.join(OUTPUT_DIR, "segments"),
            title_suffix="",
        )

    except Exception as e:
        print(f"   - 오류 발생: {segment_name} - {str(e)}")
        continue

print(f"\n   - 성공적으로 처리된 세그먼트 수: {len(segment_results)}")

# ============================================
# 7. 세그먼트 예측값 합산
# ============================================
print("\n7. 세그먼트 예측값 합산...")

if len(segment_forecasts) == 0:
    raise ValueError("처리된 세그먼트가 없습니다!")

# Test 예측값 합산
aggregated_forecast = pd.Series(0.0, index=segment_forecasts[0].index)
for forecast in segment_forecasts:
    aggregated_forecast += forecast

# 미래 예측값 합산
aggregated_future = pd.Series(0.0, index=segment_futures[0].index)
for future in segment_futures:
    aggregated_future += future

print(f"   - 합산된 Test 예측값 총합: {aggregated_forecast.sum():.0f}")
print(f"   - 합산된 미래 예측값 총합: {aggregated_future.sum():.0f}")

# ============================================
# 8. 비교 분석 (시각화 + 통계 지표)
# ============================================
print("\n8. 비교 분석...")

# Test 데이터 추출
test_data = overall_ts_series[-12:]

# 통계 지표 계산
mae_aggregated = mean_absolute_error(test_data, aggregated_forecast)
rmse_aggregated = np.sqrt(mean_squared_error(test_data, aggregated_forecast))
mape_aggregated = mean_absolute_percentage_error(test_data, aggregated_forecast) * 100

mae_scaled = mean_absolute_error(test_data, scaled_overall_forecast)
rmse_scaled = np.sqrt(mean_squared_error(test_data, scaled_overall_forecast))
mape_scaled = mean_absolute_percentage_error(test_data, scaled_overall_forecast) * 100

print(f"\n   [세그먼트 합산 모델]")
print(f"   - MAE: {mae_aggregated:.2f}")
print(f"   - RMSE: {rmse_aggregated:.2f}")
print(f"   - MAPE: {mape_aggregated:.2f}%")

print(f"\n   [스케일링된 전체 모델]")
print(f"   - MAE: {mae_scaled:.2f}")
print(f"   - RMSE: {rmse_scaled:.2f}")
print(f"   - MAPE: {mape_scaled:.2f}%")

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(15, 14))

# 1) 전체 시계열 비교
ax1 = axes[0]
train_data = overall_model_result["train_data"]
ax1.plot(
    train_data.index,
    train_data.values,
    label="Train 데이터 (전체)",
    color="blue",
    linewidth=2,
)
ax1.plot(
    test_data.index,
    test_data.values,
    label="Test 데이터 (실제)",
    color="orange",
    linewidth=2,
    marker="o",
)
ax1.plot(
    aggregated_forecast.index,
    aggregated_forecast.values,
    label="세그먼트 합산 예측",
    color="green",
    linestyle="--",
    linewidth=2,
    marker="s",
)
ax1.plot(
    scaled_overall_forecast.index,
    scaled_overall_forecast.values,
    label="스케일링된 전체 모델 예측",
    color="red",
    linestyle="--",
    linewidth=2,
    marker="^",
)

ax1.set_title(
    f"전체 모델 vs 세그먼트 합산 모델 비교\n{BLOOD_CENTER}, {BLOOD_TYPE}",
    fontsize=14,
    fontweight="bold",
)
ax1.set_xlabel("날짜", fontsize=12)
ax1.set_ylabel("헌혈 건수", fontsize=12)
ax1.legend(loc="best", fontsize=10)
ax1.grid(True, alpha=0.3)

# 2) Test 데이터 상세 비교
ax2 = axes[1]
x_pos = np.arange(len(test_data))
width = 0.25

ax2.bar(
    x_pos - width * 1.5,
    test_data.values,
    width,
    label="실제값",
    color="orange",
    alpha=0.7,
)
ax2.bar(
    x_pos - width * 0.5,
    aggregated_forecast.values,
    width,
    label="세그먼트 합산",
    color="green",
    alpha=0.7,
)
ax2.bar(
    x_pos + width * 0.5,
    scaled_overall_forecast.values,
    width,
    label="스케일링된 전체",
    color="red",
    alpha=0.7,
)

ax2.set_xlabel("월", fontsize=12)
ax2.set_ylabel("헌혈 건수", fontsize=12)
ax2.set_title("Test 데이터 상세 비교", fontsize=12, fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([date.strftime("%Y-%m") for date in test_data.index], rotation=45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# 3) 미래 예측 비교
ax3 = axes[2]
future_dates = pd.date_range(start="2025-01", end="2025-12", freq="MS")

ax3.plot(
    future_dates,
    aggregated_future.values,
    label="세그먼트 합산 예측 (2025)",
    color="green",
    linestyle="--",
    linewidth=2,
    marker="s",
)
ax3.plot(
    future_dates,
    scaled_overall_future.values,
    label="스케일링된 전체 모델 예측 (2025)",
    color="red",
    linestyle="--",
    linewidth=2,
    marker="^",
)

ax3.fill_between(
    future_dates,
    aggregated_future.values * 0.9,  # 간단한 신뢰구간 표시
    aggregated_future.values * 1.1,
    alpha=0.2,
    color="green",
    label="세그먼트 합산 ±10%",
)

ax3.set_title("2025년 미래 예측 비교", fontsize=12, fontweight="bold")
ax3.set_xlabel("날짜", fontsize=12)
ax3.set_ylabel("헌혈 건수", fontsize=12)
ax3.legend(loc="best", fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
comparison_filename = os.path.join(
    OUTPUT_DIR, f"{BLOOD_CENTER}_{BLOOD_TYPE}_comparison.png"
)
plt.savefig(comparison_filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"   - 비교 그래프 저장: {comparison_filename}")

# ============================================
# 9. 결과 저장
# ============================================
print("\n9. 결과 저장...")

analysis_filename = os.path.join(
    OUTPUT_DIR, f"{BLOOD_CENTER}_{BLOOD_TYPE}_analysis.txt"
)

with open(analysis_filename, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"{BLOOD_CENTER}, {BLOOD_TYPE} 집계 SARIMAX 모델 분석 결과\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("=" * 80 + "\n")
    f.write("1. 입력 변수\n")
    f.write("=" * 80 + "\n")
    f.write(f"지역: {BLOOD_CENTER}\n")
    f.write(f"혈액형: {BLOOD_TYPE}\n")
    f.write(f"제외 연령대: {', '.join(EXCLUDED_AGE_GROUPS)}\n")
    f.write(f"제외 직업: {', '.join(EXCLUDED_OCCUPATIONS)}\n")
    f.write(f"참고: RH- 혈액형은 이미 EDA에서 제외 결정되었으며, ")
    f.write(f"본 분석은 {BLOOD_TYPE}에 대해서만 수행되므로 자동으로 제외됨\n\n")

    f.write("=" * 80 + "\n")
    f.write("2. 데이터 정보\n")
    f.write("=" * 80 + "\n")
    f.write(f"전체 세그먼트 수: {len(overall_df)}\n")
    f.write(f"제외된 세그먼트 수: {len(excluded_df)}\n")
    f.write(f"포함된 세그먼트 수: {len(included_df)}\n")
    f.write(f"성공적으로 모델링된 세그먼트 수: {len(segment_results)}\n")
    f.write(f"전체 총합 (240개월): {overall_total:.0f}\n")
    f.write(f"제외된 세그먼트 총합: {excluded_total:.0f}\n")
    f.write(f"스케일링 계수: {scaling_factor:.4f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("3. 전체 모델 정보\n")
    f.write("=" * 80 + "\n")
    f.write(f"최적 Order (p, d, q): {overall_model_result['auto_model'].order}\n")
    f.write(
        f"최적 Seasonal Order (P, D, Q, s): {overall_model_result['auto_model'].seasonal_order}\n"
    )
    f.write(f"AIC: {overall_model_result['fitted_model'].aic:.2f}\n")
    f.write(f"BIC: {overall_model_result['fitted_model'].bic:.2f}\n")
    f.write(f"Train MAE: {overall_model_result['train_mae']:.2f}\n")
    f.write(f"Train RMSE: {overall_model_result['train_rmse']:.2f}\n")
    f.write(f"Train MAPE: {overall_model_result['train_mape']:.2f}%\n")
    f.write(f"Test MAE: {overall_model_result['test_mae']:.2f}\n")
    f.write(f"Test RMSE: {overall_model_result['test_rmse']:.2f}\n")
    f.write(f"Test MAPE: {overall_model_result['test_mape']:.2f}%\n\n")

    f.write("=" * 80 + "\n")
    f.write("4. 세그먼트별 모델 요약\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'세그먼트':<50} {'Test MAE':<12} {'Test RMSE':<12} {'Test MAPE':<12}\n")
    f.write("-" * 90 + "\n")
    for seg in segment_results:
        seg_name = seg["segment_name"]
        result = seg["model_result"]
        f.write(
            f"{seg_name:<50} {result['test_mae']:<12.2f} {result['test_rmse']:<12.2f} {result['test_mape']:<12.2f}\n"
        )
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("5. 비교 분석 결과\n")
    f.write("=" * 80 + "\n")
    f.write("\n[세그먼트 합산 모델]\n")
    f.write(f"Test MAE: {mae_aggregated:.2f}\n")
    f.write(f"Test RMSE: {rmse_aggregated:.2f}\n")
    f.write(f"Test MAPE: {mape_aggregated:.2f}%\n\n")

    f.write("[스케일링된 전체 모델]\n")
    f.write(f"Test MAE: {mae_scaled:.2f}\n")
    f.write(f"Test RMSE: {rmse_scaled:.2f}\n")
    f.write(f"Test MAPE: {mape_scaled:.2f}%\n\n")

    f.write("[차이]\n")
    f.write(f"MAE 차이: {abs(mae_aggregated - mae_scaled):.2f}\n")
    f.write(f"RMSE 차이: {abs(rmse_aggregated - rmse_scaled):.2f}\n")
    f.write(f"MAPE 차이: {abs(mape_aggregated - mape_scaled):.2f}%\n\n")

    f.write("=" * 80 + "\n")
    f.write("6. 생성된 파일\n")
    f.write("=" * 80 + "\n")
    f.write(f"- {BLOOD_CENTER}_{BLOOD_TYPE}_analysis.txt: 본 분석 결과 파일\n")
    f.write(f"- {BLOOD_CENTER}_{BLOOD_TYPE}_comparison.png: 비교 시각화 그래프\n")
    f.write(f"- segments/: 세그먼트별 개별 모델 결과 (평가 파일 및 시각화)\n")
    f.write(f"  - 총 {len(segment_results)}개 세그먼트 모델 결과\n")

print(f"   - 분석 결과 저장: {analysis_filename}")

# ============================================
# 10. 최종 요약
# ============================================
print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n[요약]")
print(f"  - 전체 세그먼트 수: {len(overall_df)}")
print(f"  - 모델링된 세그먼트 수: {len(segment_results)}")
print(f"  - 스케일링 계수: {scaling_factor:.4f}")
print(f"\n[비교 결과]")
print(f"  - 세그먼트 합산 Test MAE: {mae_aggregated:.2f}")
print(f"  - 스케일링된 전체 Test MAE: {mae_scaled:.2f}")
print(f"\n[생성된 파일]")
print(f"  - {analysis_filename}")
print(f"  - {comparison_filename}")
print(f"  - segments/ 폴더 내 {len(segment_results)}개 세그먼트 결과")

print("\n" + "=" * 80)
