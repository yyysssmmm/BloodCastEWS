#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMAX 모델 생성 및 평가 유틸리티 함수
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def create_sarimax_model(
    ts_series,
    test_months=12,
    max_order=5,
    max_p=5,
    max_d=2,
    max_q=5,
    max_P=3,
    max_D=2,
    max_Q=3,
    verbose=True,
):
    """
    SARIMAX 모델 생성 및 학습
    
    Parameters:
    -----------
    ts_series : pd.Series
        시계열 데이터 (날짜 인덱스 포함)
    test_months : int
        테스트 데이터로 사용할 마지막 개월 수
    max_order, max_p, max_d, max_q, max_P, max_D, max_Q : int
        Auto ARIMA 파라미터 제한
    verbose : bool
        진행 상황 출력 여부
    
    Returns:
    --------
    dict : 모델 정보 및 예측 결과를 담은 딕셔너리
        - train_data: 훈련 데이터
        - test_data: 테스트 데이터
        - train_fitted: 훈련 데이터 fitted values
        - forecast_mean: 테스트 데이터 예측값
        - forecast_ci: 테스트 데이터 예측 신뢰구간
        - future_mean: 미래 예측값 (12개월)
        - future_ci: 미래 예측 신뢰구간
        - fitted_model: 학습된 모델
        - auto_model: auto_arima 결과
        - train_mae, train_rmse, train_mape: 훈련 평가 지표
        - test_mae, test_rmse, test_mape: 테스트 평가 지표
    """
    # 데이터 타입 확인 및 변환 (object -> float)
    if ts_series.dtype == 'object' or not pd.api.types.is_numeric_dtype(ts_series):
        ts_series = pd.to_numeric(ts_series, errors='coerce').astype(float)
        ts_series = ts_series.fillna(0.0)
    
    # float 타입으로 명시적 변환
    ts_series = ts_series.astype(float)
    
    if verbose:
        print(f"   - 시계열 길이: {len(ts_series)}개월")
        print(f"   - 기간: {ts_series.index[0].strftime('%Y-%m')} ~ {ts_series.index[-1].strftime('%Y-%m')}")
        print(f"   - 평균값: {ts_series.mean():.2f}")

    # Train/Test 분할
    train_data = ts_series[:-test_months].astype(float)
    test_data = ts_series[-test_months:].astype(float)

    if verbose:
        print(f"   - Train: {len(train_data)}개월, Test: {len(test_data)}개월")

    # Auto ARIMA로 최적 파라미터 찾기
    if verbose:
        print("   - Auto ARIMA 실행 중...")
    
    auto_model = auto_arima(
        train_data,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_order=max_order,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        max_P=max_P,
        max_D=max_D,
        max_Q=max_Q,
        trace=False,
        n_jobs=-1,
    )

    if verbose:
        print(f"   - 최적 파라미터: {auto_model.order}, {auto_model.seasonal_order}")

    # SARIMAX 모델 학습
    model = SARIMAX(
        train_data,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)

    # Train fitted values (0이나 NaN 처리)
    train_fitted = fitted_model.fittedvalues.copy()
    for i in range(len(train_fitted)):
        if pd.isna(train_fitted.iloc[i]) or train_fitted.iloc[i] == 0:
            train_fitted.iloc[i] = train_data.iloc[i]

    # Test 예측
    forecast = fitted_model.get_forecast(steps=len(test_data))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # 미래 예측 (12개월)
    future_forecast = fitted_model.get_forecast(steps=12)
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int()

    # 평가 지표 계산
    train_mae = mean_absolute_error(train_data, train_fitted)
    train_rmse = np.sqrt(mean_squared_error(train_data, train_fitted))
    train_mape = mean_absolute_percentage_error(train_data, train_fitted) * 100

    test_mae = mean_absolute_error(test_data, forecast_mean)
    test_rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
    test_mape = mean_absolute_percentage_error(test_data, forecast_mean) * 100

    return {
        "train_data": train_data,
        "test_data": test_data,
        "train_fitted": train_fitted,
        "forecast_mean": forecast_mean,
        "forecast_ci": forecast_ci,
        "future_mean": future_mean,
        "future_ci": future_ci,
        "fitted_model": fitted_model,
        "auto_model": auto_model,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_mape": train_mape,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
    }


def save_sarimax_evaluation(
    model_result,
    segment_name,
    output_dir,
    gender=None,
    blood_type=None,
    blood_center=None,
    age_group=None,
    occupation=None,
):
    """
    SARIMAX 모델 평가 결과를 파일로 저장
    
    Parameters:
    -----------
    model_result : dict
        create_sarimax_model의 반환값
    segment_name : str
        세그먼트 이름 (파일명에 사용)
    output_dir : str
        출력 디렉토리
    gender, blood_type, blood_center, age_group, occupation : str
        세그먼트 정보 (선택사항)
    """
    os.makedirs(output_dir, exist_ok=True)

    evaluation_filename = os.path.join(output_dir, f"evaluation_{segment_name}.txt")

    with open(evaluation_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SARIMAX 모델 평가 결과\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if gender or blood_type or blood_center or age_group or occupation:
            f.write("=" * 80 + "\n")
            f.write("입력 변수\n")
            f.write("=" * 80 + "\n")
            if gender:
                f.write(f"성별: {gender}\n")
            if blood_type:
                f.write(f"혈액형별: {blood_type}\n")
            if blood_center:
                f.write(f"혈액원별: {blood_center}\n")
            if age_group:
                f.write(f"연령대: {age_group}\n")
            if occupation:
                f.write(f"직업: {occupation}\n")
            f.write(f"Segment: {segment_name}\n\n")

        f.write("=" * 80 + "\n")
        f.write("모델 정보\n")
        f.write("=" * 80 + "\n")
        f.write(f"최적 Order (p, d, q): {model_result['auto_model'].order}\n")
        f.write(
            f"최적 Seasonal Order (P, D, Q, s): {model_result['auto_model'].seasonal_order}\n"
        )
        f.write(f"AIC: {model_result['fitted_model'].aic:.2f}\n")
        f.write(f"BIC: {model_result['fitted_model'].bic:.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Train 데이터 평가 결과\n")
        f.write("=" * 80 + "\n")
        f.write(f"MAE: {model_result['train_mae']:.2f}\n")
        f.write(f"RMSE: {model_result['train_rmse']:.2f}\n")
        f.write(f"MAPE: {model_result['train_mape']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("Test 데이터 평가 결과\n")
        f.write("=" * 80 + "\n")
        f.write(f"MAE: {model_result['test_mae']:.2f}\n")
        f.write(f"RMSE: {model_result['test_rmse']:.2f}\n")
        f.write(f"MAPE: {model_result['test_mape']:.2f}%\n\n")

    return evaluation_filename


def plot_sarimax_forecast(model_result, segment_name, output_dir, title_suffix=""):
    """
    SARIMAX 모델 예측 결과 시각화
    
    Parameters:
    -----------
    model_result : dict
        create_sarimax_model의 반환값
    segment_name : str
        세그먼트 이름 (파일명에 사용)
    output_dir : str
        출력 디렉토리
    title_suffix : str
        그래프 제목에 추가할 텍스트
    """
    os.makedirs(output_dir, exist_ok=True)

    train_data = model_result["train_data"]
    test_data = model_result["test_data"]
    train_fitted = model_result["train_fitted"]
    forecast_mean = model_result["forecast_mean"]
    forecast_ci = model_result["forecast_ci"]
    future_mean = model_result["future_mean"]
    future_ci = model_result["future_ci"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # 전체 시계열 및 예측
    ax1 = axes[0]
    ax1.plot(
        train_data.index,
        train_data.values,
        label="Train 데이터",
        color="blue",
        linewidth=2,
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

    # 미래 예측
    future_dates = pd.date_range(start="2025-01", end="2025-12", freq="MS")
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

    title = f"SARIMAX 모델 예측 결과{title_suffix}"
    if segment_name:
        title += f"\n{segment_name}"

    ax1.set_title(title, fontsize=14, fontweight="bold")
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
    forecast_filename = os.path.join(output_dir, f"sarimax_forecast_{segment_name}.png")
    plt.savefig(forecast_filename, dpi=300, bbox_inches="tight")
    plt.close()

    return forecast_filename

