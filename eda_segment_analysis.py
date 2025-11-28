#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세그멘트 중요성 분석 EDA
연령대, 직업, 혈액형별 통계적 중요성 평가
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    ttest_ind,
    ttest_1samp,
)
import os
from datetime import datetime

print("=" * 80)
print("세그멘트 중요성 분석 EDA")
print("=" * 80)

# 결과 저장 폴더 생성
OUTPUT_DIR = "eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n결과 저장 폴더: {OUTPUT_DIR}/")

# ============================================
# 1. 데이터 로딩
# ============================================
print("\n1. 데이터 로딩 중...")
df = pd.read_csv("data/blood_donation_data_monthly.csv", encoding="utf-8")
print(f"   - 전체 데이터 행 수: {len(df):,}개")
print(f"   - 전체 데이터 열 수: {len(df.columns)}개")

# 월별 컬럼 추출
monthly_columns = [col for col in df.columns if "-" in col and col[0].isdigit()]
print(f"   - 월별 컬럼 수: {len(monthly_columns)}개")

# ============================================
# 2. 기본 통계 계산
# ============================================
print("\n2. 기본 통계 계산 중...")

# 각 세그멘트의 총합 계산
df["total_sum"] = df[monthly_columns].sum(axis=1)
df["mean_value"] = df[monthly_columns].mean(axis=1)
df["std_value"] = df[monthly_columns].std(axis=1)

total_all = df["total_sum"].sum()
print(f"   - 전체 총합: {total_all:,.0f}")

# 전체 데이터의 월별 합계 시계열 생성 (240개월)
print("   - 전체 월별 합계 시계열 생성 중...")
overall_monthly_series = df[monthly_columns].sum(axis=0).values
print(f"   - 시계열 길이: {len(overall_monthly_series)}개월")

# ============================================
# 3. 연령대별 분석
# ============================================
print("\n3. 연령대별 분석 중...")

age_analysis = []

for age in df["연령대"].unique():
    age_df = df[df["연령대"] == age]

    # 기본 통계
    segment_count = len(age_df)
    total_sum = age_df["total_sum"].sum()
    percentage = (total_sum / total_all) * 100
    mean_sum = age_df["total_sum"].mean()
    std_sum = age_df["total_sum"].std()

    age_analysis.append(
        {
            "연령대": age,
            "세그멘트_수": segment_count,
            "총합": total_sum,
            "비율(%)": percentage,
            "평균_총합": mean_sum,
            "표준편차": std_sum,
            "최소값": age_df["total_sum"].min(),
            "최대값": age_df["total_sum"].max(),
        }
    )

age_df_result = pd.DataFrame(age_analysis).sort_values("총합", ascending=False)

# ============================================
# 4. 직업별 분석
# ============================================
print("\n4. 직업별 분석 중...")

occupation_analysis = []

for occ in df["직업"].unique():
    occ_df = df[df["직업"] == occ]

    # 기본 통계
    segment_count = len(occ_df)
    total_sum = occ_df["total_sum"].sum()
    percentage = (total_sum / total_all) * 100
    mean_sum = occ_df["total_sum"].mean()
    std_sum = occ_df["total_sum"].std()

    occupation_analysis.append(
        {
            "직업": occ,
            "세그멘트_수": segment_count,
            "총합": total_sum,
            "비율(%)": percentage,
            "평균_총합": mean_sum,
            "표준편차": std_sum,
            "최소값": occ_df["total_sum"].min(),
            "최대값": occ_df["total_sum"].max(),
        }
    )

occupation_df_result = pd.DataFrame(occupation_analysis).sort_values(
    "총합", ascending=False
)

# ============================================
# 5. 혈액형별 분석 (RH+ vs RH-)
# ============================================
print("\n5. 혈액형별 분석 중...")

# RH+와 RH- 구분
df["RH_type"] = df["혈액형별"].apply(lambda x: "RH+" if "+" in str(x) else "RH-")

blood_type_analysis = []

for rh_type in ["RH+", "RH-"]:
    rh_df = df[df["RH_type"] == rh_type]

    segment_count = len(rh_df)
    total_sum = rh_df["total_sum"].sum()
    percentage = (total_sum / total_all) * 100
    mean_sum = rh_df["total_sum"].mean()
    std_sum = rh_df["total_sum"].std()

    blood_type_analysis.append(
        {
            "혈액형_타입": rh_type,
            "세그멘트_수": segment_count,
            "총합": total_sum,
            "비율(%)": percentage,
            "평균_총합": mean_sum,
            "표준편차": std_sum,
        }
    )

blood_type_df_result = pd.DataFrame(blood_type_analysis)

# ============================================
# 6. 통계적 검정
# ============================================
print("\n6. 통계적 검정 수행 중...")

# 6.1 연령대별 Chi-square test
age_contingency = pd.crosstab(
    df["연령대"],
    pd.cut(
        df["total_sum"],
        bins=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    ),
)
chi2_age, p_age, dof_age, expected_age = chi2_contingency(age_contingency)

# 6.2 직업별 Chi-square test
occ_contingency = pd.crosstab(
    df["직업"],
    pd.cut(
        df["total_sum"],
        bins=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    ),
)
chi2_occ, p_occ, dof_occ, expected_occ = chi2_contingency(occ_contingency)


# 6.3 효과 크기 계산 함수 (다른 검정에서 사용)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# Cramér's V 계산
def cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


cramers_v_age = cramers_v(age_contingency)
cramers_v_occ = cramers_v(occ_contingency)

# ============================================
# 7. 통계적 중요성 검정 (전체 vs 제외 비교)
# ============================================
print("\n7. 통계적 중요성 검정 수행 중...")
print("   각 카테고리별로 전체 데이터 vs 해당 카테고리 제외 데이터 비교")


# TOST (동등성 검정) 함수
def tost_test(diff_series, delta):
    """
    Two One-Sided Tests (TOST) for equivalence testing

    Parameters:
    - diff_series: 차이 시계열 (전체 - 제외)
    - delta: 허용 범위

    Returns:
    - equivalent: 동등함 여부 (두 검정 모두 p < 0.05)
    - t1_pvalue: 첫 번째 검정 p-value (diff_mean > -delta)
    - t2_pvalue: 두 번째 검정 p-value (diff_mean < +delta)
    """
    # H1: diff_mean > -delta (차이가 -delta보다 크다)
    t1_stat, t1_pvalue = ttest_1samp(diff_series, -delta, alternative="greater")
    # H2: diff_mean < +delta (차이가 +delta보다 작다)
    t2_stat, t2_pvalue = ttest_1samp(diff_series, +delta, alternative="less")

    # 두 검정 모두 통과하면 동등함
    equivalent = (t1_pvalue < 0.05) and (t2_pvalue < 0.05)

    return equivalent, t1_pvalue, t2_pvalue


# 검정 결과 저장용 딕셔너리
age_test_results = {
    "tost": [],
    "timeseries_metrics": [],
}

occ_test_results = {
    "tost": [],
    "timeseries_metrics": [],
}

# RH- 검정 결과 저장용 딕셔너리
rh_test_results = {
    "tost": [],
    "timeseries_metrics": [],
}

# 7.1 연령대별 검정
print("\n   [연령대별 검정]")
for age in df["연령대"].unique():
    print(f"      - {age} 검정 중...")

    # 전체 데이터 vs 해당 연령대 제외 데이터
    age_excluded_monthly_series = (
        df[df["연령대"] != age][monthly_columns].sum(axis=0).values
    )

    # 비율 계산
    age_total = df[df["연령대"] == age]["total_sum"].sum()
    age_percentage = (age_total / total_all) * 100

    # 차이 시계열 계산 (TOST와 시계열 지표에서 사용)
    diff_series = overall_monthly_series - age_excluded_monthly_series

    # 검정 1: 동등성 검정 (TOST)
    # 허용 범위 설정 (전체 시계열 평균의 5%)
    delta = np.mean(overall_monthly_series) * 0.05
    equivalent, t1_pvalue, t2_pvalue = tost_test(diff_series, delta)
    age_test_results["tost"].append(
        {
            "연령대": age,
            "equivalent": equivalent,
            "t1_pvalue": t1_pvalue,
            "t2_pvalue": t2_pvalue,
            "delta": delta,
            "비율": age_percentage,
            "제거_권장": equivalent,
        }
    )

    # 검정 2: 시계열 지표 (RMSE, MAPE)
    # RMSE
    rmse = np.sqrt(np.mean((overall_monthly_series - age_excluded_monthly_series) ** 2))
    # MAPE (0으로 나누는 경우 방지)
    mape = (
        np.mean(
            np.abs(
                np.where(
                    overall_monthly_series != 0,
                    (overall_monthly_series - age_excluded_monthly_series)
                    / overall_monthly_series,
                    0,
                )
            )
        )
        * 100
    )
    overall_std_ts = np.std(overall_monthly_series)
    rmse_threshold = overall_std_ts * 0.5
    age_test_results["timeseries_metrics"].append(
        {
            "연령대": age,
            "rmse": rmse,
            "mape": mape,
            "rmse_threshold": rmse_threshold,
            "비율": age_percentage,
            "제거_권장": mape < 5.0 or rmse < rmse_threshold,
        }
    )

# 7.2 직업별 검정
print("\n   [직업별 검정]")
for occ in df["직업"].unique():
    print(f"      - {occ} 검정 중...")

    # 전체 데이터 vs 해당 직업 제외 데이터
    occ_excluded_monthly_series = (
        df[df["직업"] != occ][monthly_columns].sum(axis=0).values
    )

    # 비율 계산
    occ_total = df[df["직업"] == occ]["total_sum"].sum()
    occ_percentage = (occ_total / total_all) * 100

    # 차이 시계열 계산 (TOST와 시계열 지표에서 사용)
    diff_series = overall_monthly_series - occ_excluded_monthly_series

    # 검정 1: 동등성 검정 (TOST)
    # 허용 범위 설정 (전체 시계열 평균의 5%)
    delta = np.mean(overall_monthly_series) * 0.05
    equivalent, t1_pvalue, t2_pvalue = tost_test(diff_series, delta)
    occ_test_results["tost"].append(
        {
            "직업": occ,
            "equivalent": equivalent,
            "t1_pvalue": t1_pvalue,
            "t2_pvalue": t2_pvalue,
            "delta": delta,
            "비율": occ_percentage,
            "제거_권장": equivalent,
        }
    )

    # 검정 2: 시계열 지표 (RMSE, MAPE)
    # RMSE
    rmse = np.sqrt(np.mean((overall_monthly_series - occ_excluded_monthly_series) ** 2))
    # MAPE (0으로 나누는 경우 방지)
    mape = (
        np.mean(
            np.abs(
                np.where(
                    overall_monthly_series != 0,
                    (overall_monthly_series - occ_excluded_monthly_series)
                    / overall_monthly_series,
                    0,
                )
            )
        )
        * 100
    )
    overall_std_ts = np.std(overall_monthly_series)
    rmse_threshold = overall_std_ts * 0.5
    occ_test_results["timeseries_metrics"].append(
        {
            "직업": occ,
            "rmse": rmse,
            "mape": mape,
            "rmse_threshold": rmse_threshold,
            "비율": occ_percentage,
            "제거_권장": mape < 5.0 or rmse < rmse_threshold,
        }
    )

# 7.3 RH- 검정
print("\n   [RH- 검정]")
print(f"      - RH- 검정 중...")

# 전체 데이터 vs RH- 제외 데이터
rh_minus_excluded_monthly_series = (
    df[df["RH_type"] != "RH-"][monthly_columns].sum(axis=0).values
)

# 비율 계산
rh_minus_total = df[df["RH_type"] == "RH-"]["total_sum"].sum()
rh_minus_percentage = (rh_minus_total / total_all) * 100

# 차이 시계열 계산 (TOST와 시계열 지표에서 사용)
diff_series_rh = overall_monthly_series - rh_minus_excluded_monthly_series

# 검정 1: 동등성 검정 (TOST)
# 허용 범위 설정 (전체 시계열 평균의 5%)
delta = np.mean(overall_monthly_series) * 0.05
equivalent, t1_pvalue, t2_pvalue = tost_test(diff_series_rh, delta)
rh_test_results["tost"].append(
    {
        "혈액형": "RH-",
        "equivalent": equivalent,
        "t1_pvalue": t1_pvalue,
        "t2_pvalue": t2_pvalue,
        "delta": delta,
        "비율": rh_minus_percentage,
        "제거_권장": equivalent,
    }
)

# 검정 2: 시계열 지표 (RMSE, MAPE)
# RMSE
rmse = np.sqrt(
    np.mean((overall_monthly_series - rh_minus_excluded_monthly_series) ** 2)
)
# MAPE (0으로 나누는 경우 방지)
mape = (
    np.mean(
        np.abs(
            np.where(
                overall_monthly_series != 0,
                (overall_monthly_series - rh_minus_excluded_monthly_series)
                / overall_monthly_series,
                0,
            )
        )
    )
    * 100
)
overall_std_ts = np.std(overall_monthly_series)
rmse_threshold = overall_std_ts * 0.5
rh_test_results["timeseries_metrics"].append(
    {
        "혈액형": "RH-",
        "rmse": rmse,
        "mape": mape,
        "rmse_threshold": rmse_threshold,
        "비율": rh_minus_percentage,
        "제거_권장": mape < 5.0 or rmse < rmse_threshold,
    }
)

# ============================================
# 8. 검정 결과 정리
# ============================================
print("\n8. 검정 결과 정리 중...")

# ============================================
# 9. 결과 파일 저장
# ============================================
print("\n9. 결과 파일 저장 중...")

output_file = os.path.join(OUTPUT_DIR, "eda_segment_analysis.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("세그멘트 중요성 분석 EDA 결과\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("=" * 80 + "\n")
    f.write("1. 기본 정보\n")
    f.write("=" * 80 + "\n")
    f.write(f"전체 데이터 행 수: {len(df):,}개\n")
    f.write(f"전체 총합: {total_all:,.0f}\n")
    f.write(f"월별 컬럼 수: {len(monthly_columns)}개\n\n")

    f.write("=" * 80 + "\n")
    f.write("2. 연령대별 분석\n")
    f.write("=" * 80 + "\n\n")
    f.write(age_df_result.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("3. 직업별 분석\n")
    f.write("=" * 80 + "\n\n")
    f.write(occupation_df_result.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("4. 혈액형별 분석 (RH+ vs RH-)\n")
    f.write("=" * 80 + "\n\n")
    f.write(blood_type_df_result.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("5. 통계적 검정 결과\n")
    f.write("=" * 80 + "\n\n")

    f.write("[연령대별 Chi-square Test]\n")
    f.write(f"  Chi-square 통계량: {chi2_age:.4f}\n")
    f.write(f"  p-value: {p_age:.4e}\n")
    f.write(f"  자유도: {dof_age}\n")
    f.write(f"  Cramér's V: {cramers_v_age:.4f}\n")
    f.write(
        f"  해석: {'유의미한 차이 있음' if p_age < 0.05 else '유의미한 차이 없음'} (p < 0.05)\n\n"
    )

    f.write("[직업별 Chi-square Test]\n")
    f.write(f"  Chi-square 통계량: {chi2_occ:.4f}\n")
    f.write(f"  p-value: {p_occ:.4e}\n")
    f.write(f"  자유도: {dof_occ}\n")
    f.write(f"  Cramér's V: {cramers_v_occ:.4f}\n")
    f.write(
        f"  해석: {'유의미한 차이 있음' if p_occ < 0.05 else '유의미한 차이 없음'} (p < 0.05)\n\n"
    )

    f.write("=" * 80 + "\n")
    f.write("6. 통계적 검정 결과 (전체 vs 제외 비교)\n")
    f.write("=" * 80 + "\n\n")
    f.write(
        "각 검정별로 전체 데이터와 해당 카테고리 제외 데이터를 비교한 결과입니다.\n\n"
    )

    # 검정 1: 동등성 검정 (TOST)
    f.write("=" * 80 + "\n")
    f.write("[검정 1: 동등성 검정 (TOST: Two One-Sided Tests)]\n")
    f.write("=" * 80 + "\n\n")
    f.write("제거 기준: 두 검정 모두 p-value < 0.05 (차이가 허용 범위 내에 있음)\n")
    f.write("허용 범위(δ): 전체 시계열 평균의 5%\n\n")

    f.write("[연령대]\n")
    age_tost_remove = [r for r in age_test_results["tost"] if r["제거_권장"]]
    if age_tost_remove:
        f.write(f"  제거 권장: {len(age_tost_remove)}개\n\n")
        for r in age_tost_remove:
            f.write(f"  - {r['연령대']}\n")
            f.write(f"    동등함: {r['equivalent']}\n")
            f.write(f"    T1 p-value (diff > -δ): {r['t1_pvalue']:.4e}\n")
            f.write(f"    T2 p-value (diff < +δ): {r['t2_pvalue']:.4e}\n")
            f.write(f"    허용 범위(δ): {r['delta']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    f.write("[직업]\n")
    occ_tost_remove = [r for r in occ_test_results["tost"] if r["제거_권장"]]
    if occ_tost_remove:
        f.write(f"  제거 권장: {len(occ_tost_remove)}개\n\n")
        for r in occ_tost_remove:
            f.write(f"  - {r['직업']}\n")
            f.write(f"    동등함: {r['equivalent']}\n")
            f.write(f"    T1 p-value (diff > -δ): {r['t1_pvalue']:.4e}\n")
            f.write(f"    T2 p-value (diff < +δ): {r['t2_pvalue']:.4e}\n")
            f.write(f"    허용 범위(δ): {r['delta']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    f.write("[혈액형 - RH-]\n")
    rh_tost_remove = [r for r in rh_test_results["tost"] if r["제거_권장"]]
    if rh_tost_remove:
        f.write(f"  제거 권장: {len(rh_tost_remove)}개\n\n")
        for r in rh_tost_remove:
            f.write(f"  - {r['혈액형']}\n")
            f.write(f"    동등함: {r['equivalent']}\n")
            f.write(f"    T1 p-value (diff > -δ): {r['t1_pvalue']:.4e}\n")
            f.write(f"    T2 p-value (diff < +δ): {r['t2_pvalue']:.4e}\n")
            f.write(f"    허용 범위(δ): {r['delta']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    # 검정 2: 시계열 지표 (RMSE, MAPE)
    f.write("=" * 80 + "\n")
    f.write("[검정 2: 시계열 지표 (RMSE, MAPE)]\n")
    f.write("=" * 80 + "\n\n")
    f.write("제거 기준: MAPE < 5% 또는 RMSE < 전체 표준편차의 0.5배\n\n")

    f.write("[연령대]\n")
    age_metrics_remove = [
        r for r in age_test_results["timeseries_metrics"] if r["제거_권장"]
    ]
    if age_metrics_remove:
        f.write(f"  제거 권장: {len(age_metrics_remove)}개\n\n")
        for r in age_metrics_remove:
            f.write(f"  - {r['연령대']}\n")
            f.write(f"    RMSE: {r['rmse']:.2f}\n")
            f.write(f"    MAPE: {r['mape']:.4f}%\n")
            f.write(f"    RMSE 임계값: {r['rmse_threshold']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    f.write("[직업]\n")
    occ_metrics_remove = [
        r for r in occ_test_results["timeseries_metrics"] if r["제거_권장"]
    ]
    if occ_metrics_remove:
        f.write(f"  제거 권장: {len(occ_metrics_remove)}개\n\n")
        for r in occ_metrics_remove:
            f.write(f"  - {r['직업']}\n")
            f.write(f"    RMSE: {r['rmse']:.2f}\n")
            f.write(f"    MAPE: {r['mape']:.4f}%\n")
            f.write(f"    RMSE 임계값: {r['rmse_threshold']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    f.write("[혈액형 - RH-]\n")
    rh_metrics_remove = [
        r for r in rh_test_results["timeseries_metrics"] if r["제거_권장"]
    ]
    if rh_metrics_remove:
        f.write(f"  제거 권장: {len(rh_metrics_remove)}개\n\n")
        for r in rh_metrics_remove:
            f.write(f"  - {r['혈액형']}\n")
            f.write(f"    RMSE: {r['rmse']:.2f}\n")
            f.write(f"    MAPE: {r['mape']:.4f}%\n")
            f.write(f"    RMSE 임계값: {r['rmse_threshold']:.2f}\n")
            f.write(f"    비율: {r['비율']:.2f}%\n\n")
    else:
        f.write("  제거 권장 없음\n\n")

    f.write("=" * 80 + "\n")
    f.write("7. 검정 결과 요약\n")
    f.write("=" * 80 + "\n\n")

    f.write("각 검정별 제거 권장 항목 요약:\n\n")

    # 연령대 요약
    f.write("[연령대]\n")
    all_age_removed = set()
    for test_name in ["tost", "timeseries_metrics"]:
        removed = [r["연령대"] for r in age_test_results[test_name] if r["제거_권장"]]
        all_age_removed.update(removed)
        f.write(
            f"  {test_name}: {len(removed)}개 - {', '.join(removed) if removed else '없음'}\n"
        )
    f.write(f"\n  전체 검정에서 제거 권장된 연령대: {len(all_age_removed)}개\n")
    if all_age_removed:
        f.write(f"  - {', '.join(all_age_removed)}\n")
    f.write("\n")

    # 직업 요약
    f.write("[직업]\n")
    all_occ_removed = set()
    for test_name in ["tost", "timeseries_metrics"]:
        removed = [r["직업"] for r in occ_test_results[test_name] if r["제거_권장"]]
        all_occ_removed.update(removed)
        f.write(
            f"  {test_name}: {len(removed)}개 - {', '.join(removed) if removed else '없음'}\n"
        )
    f.write(f"\n  전체 검정에서 제거 권장된 직업: {len(all_occ_removed)}개\n")
    if all_occ_removed:
        f.write(f"  - {', '.join(all_occ_removed)}\n")
    f.write("\n")

    # 혈액형 요약
    f.write("[혈액형 - RH-]\n")
    all_rh_removed = set()
    for test_name in ["tost", "timeseries_metrics"]:
        removed = [r["혈액형"] for r in rh_test_results[test_name] if r["제거_권장"]]
        all_rh_removed.update(removed)
        f.write(
            f"  {test_name}: {len(removed)}개 - {', '.join(removed) if removed else '없음'}\n"
        )
    f.write(f"\n  전체 검정에서 제거 권장된 혈액형: {len(all_rh_removed)}개\n")
    if all_rh_removed:
        f.write(f"  - {', '.join(all_rh_removed)}\n")
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("8. RH- 혈액형 제거 권장\n")
    f.write("=" * 80 + "\n\n")
    rh_minus_info = blood_type_df_result[
        blood_type_df_result["혈액형_타입"] == "RH-"
    ].iloc[0]
    rh_minus_segments = len(df[df["RH_type"] == "RH-"])
    rh_minus_total = df[df["RH_type"] == "RH-"]["total_sum"].sum()
    f.write(f"  RH- 비율: {rh_minus_info['비율(%)']:.2f}%\n")
    f.write(f"  RH- 세그멘트 수: {rh_minus_segments}개\n")
    f.write(
        f"  RH- 총합: {rh_minus_total:,.0f} ({rh_minus_total/total_all*100:.2f}%)\n"
    )
    f.write(f"  검정 결과: TOST 및 시계열 지표 검정 결과 참조\n")
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("분석 완료\n")
    f.write("=" * 80 + "\n")

print(f"   - 결과 파일 저장 완료: {output_file}")

# ============================================
# 10. 요약 출력
# ============================================
print("\n" + "=" * 80)
print("EDA 분석 완료!")
print("=" * 80)
print(f"\n[주요 결과]")
print(f"  - 연령대 수: {len(age_df_result)}개")
print(f"  - 직업 수: {len(occupation_df_result)}개")
print(f"  - 통계적 검정 완료: 5가지 검정 수행")
print(f"  - 각 검정별 결과는 결과 파일 참조")

print(f"\n[생성된 파일]")
print(f"  - {output_file}")

print("\n" + "=" * 80)
