#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연도별 데이터를 월별 데이터로 변환 스크립트
각 연도 컬럼을 12개월로 분할
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("연도별 데이터를 월별 데이터로 변환 시작")
print("=" * 80)

# 1. 데이터 로딩
print("\n1. 데이터 로딩 중...")
df = pd.read_csv("blood_donation_data_with_age_occupation.csv", encoding="utf-8")
print(f"   - 원본 데이터 행 수: {len(df):,}개")
print(f"   - 원본 데이터 열 수: {len(df.columns)}개")

# 2. 월별 비율 정의 (퍼센트)
print("\n2. 월별 비율 정의...")
monthly_ratios = [
    8.2,  # 1월
    7.7,  # 2월
    7.9,  # 3월
    8.0,  # 4월
    8.7,  # 5월
    8.2,  # 6월
    8.6,  # 7월
    8.4,  # 8월
    8.1,  # 9월
    8.4,  # 10월
    8.9,  # 11월
    8.8,  # 12월
]

# 퍼센트를 소수로 변환
monthly_ratios_decimal = [ratio / 100.0 for ratio in monthly_ratios]
total_ratio = sum(monthly_ratios_decimal)

print(f"   - 월별 비율: {monthly_ratios}")
print(f"   - 총 비율 합계: {sum(monthly_ratios):.1f}%")
print(f"   - 소수 변환 합계: {total_ratio:.4f}")

# 3. 연도 컬럼 추출
year_columns = [col for col in df.columns if "년" in col]
print(f"\n3. 연도 컬럼 확인: {len(year_columns)}개 연도")
print(f"   - 연도 범위: {year_columns[0]} ~ {year_columns[-1]}")

# 4. 기본 컬럼 추출 (연도가 아닌 컬럼)
base_columns = [col for col in df.columns if "년" not in col]
print(f"   - 기본 컬럼: {base_columns}")

# 5. 새로운 데이터프레임 생성
print("\n4. 월별 데이터 생성 중...")
print("   - 각 연도를 12개월로 분할...")

# 기본 컬럼 데이터 복사
new_data = df[base_columns].copy()

# 각 연도에 대해 12개월 컬럼 생성
for year_col in year_columns:
    # 연도 추출 (예: "2005 년" -> "2005")
    year = year_col.replace(" 년", "").strip()

    # 12개월 컬럼 생성
    for month in range(1, 13):
        new_col_name = f"{year}-{month}"
        ratio = monthly_ratios_decimal[month - 1]

        # 원본 연도 값에 월별 비율 곱하기 및 반올림
        new_data[new_col_name] = (df[year_col] * ratio).round().astype(int)

print(f"   - 생성된 월별 컬럼 수: {len(new_data.columns) - len(base_columns)}개")
print(f"   - 총 컬럼 수: {len(new_data.columns)}개")

# 6. 컬럼 순서 정리 (기본 컬럼 + 월별 컬럼 순서대로)
print("\n5. 컬럼 순서 정리 중...")
# 월별 컬럼 이름 생성 (2005-1부터 2024-12까지)
monthly_columns = []
for year in range(2005, 2025):
    for month in range(1, 13):
        monthly_columns.append(f"{year}-{month}")

# 최종 컬럼 순서
final_columns = base_columns + monthly_columns
new_data = new_data[final_columns]

print(f"   - 컬럼 순서 정리 완료")
print(f"   - 첫 번째 월별 컬럼: {monthly_columns[0]}")
print(f"   - 마지막 월별 컬럼: {monthly_columns[-1]}")

# 7. 데이터 검증
print("\n6. 데이터 검증 중...")
sample_row = new_data.iloc[0]
sample_year = "2005"
sample_original_value = df.iloc[0]["2005 년"]

print("\n   [검증 샘플]")
print(f"   원본 연도 컬럼: 2005 년 = {sample_original_value}")
print(f"   월별 분할 결과:")
for month in range(1, 13):
    col_name = f"{sample_year}-{month}"
    monthly_value = sample_row[col_name]
    expected_value = round(sample_original_value * monthly_ratios_decimal[month - 1])
    print(
        f"     {col_name}: {monthly_value} (예상: {expected_value}) {'✓' if monthly_value == expected_value else '✗'}"
    )

# 월별 합계 검증
print(f"\n   월별 합계 검증:")
monthly_sum = sum([sample_row[f"{sample_year}-{month}"] for month in range(1, 13)])
print(f"     2005년 월별 합계: {monthly_sum}")
print(f"     2005년 원본 값: {sample_original_value}")
print(f"     차이: {abs(monthly_sum - sample_original_value)}")

# 8. 최종 데이터 저장
output_file = "blood_donation_data_monthly.csv"
new_data.to_csv(output_file, index=False, encoding="utf-8")
print(f"\n7. 데이터 저장 완료!")
print(f"   - 저장된 파일: '{output_file}'")
print(f"   - 최종 행 수: {len(new_data):,}개")
print(f"   - 최종 열 수: {len(new_data.columns)}개")
print(f"   - 기본 컬럼: {len(base_columns)}개")
print(f"   - 월별 컬럼: {len(monthly_columns)}개")

# 9. 통계 요약
print("\n8. 통계 요약")
print(f"   - 원본 연도 컬럼 수: {len(year_columns)}개")
print(f"   - 생성된 월별 컬럼 수: {len(monthly_columns)}개")
print(f"   - 확장 배수: {len(monthly_columns) / len(year_columns)}배")

print("\n" + "=" * 80)
print("연도별 데이터를 월별 데이터로 변환 완료!")
print("=" * 80)
