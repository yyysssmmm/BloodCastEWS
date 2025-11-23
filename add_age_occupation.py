#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연령대와 직업 컬럼 추가 스크립트
각 원본 행을 54개 조합(6개 연령대 × 9개 직업)으로 확장
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("연령대 및 직업 컬럼 추가 시작")
print("=" * 80)

# 1. 전처리된 데이터 로딩
print("\n1. 데이터 로딩 중...")
df = pd.read_csv("blood_donation_data_preprocessed.csv", encoding="utf-8")
print(f"   - 원본 데이터 행 수: {len(df):,}개")
print(f"   - 원본 데이터 열 수: {len(df.columns)}개")

# 2. 연령대 및 직업 비율 정의
print("\n2. 연령대 및 직업 비율 정의...")

# 연령대 비율 (퍼센트)
age_ratios = {
    "16~19": 18.0,
    "20~29": 37.0,
    "30~39": 16.0,
    "40~49": 17.0,
    "50~59": 10.0,
    "60세 이상": 2.0,
}

# 직업 비율 (퍼센트)
occupation_ratios = {
    "가사": 2.0,
    "종교직": 1.0,
    "고등학생": 10.0,
    "대학생": 25.0,
    "군인": 11.0,
    "회사원": 35.0,
    "공무원": 4.0,
    "자영업": 2.0,
    "기타": 10.0,
}

print(f"   - 연령대 수: {len(age_ratios)}개")
print(f"   - 직업 수: {len(occupation_ratios)}개")
print(f"   - 총 조합 수: {len(age_ratios) * len(occupation_ratios)}개")

# 3. 조합 비율 계산 (연령대 비율 × 직업 비율)
print("\n3. 조합 비율 계산 중...")
combinations = []
for age, age_ratio in age_ratios.items():
    for occupation, occ_ratio in occupation_ratios.items():
        # 퍼센트를 소수로 변환하여 곱하기
        combined_ratio = (age_ratio / 100.0) * (occ_ratio / 100.0)
        combinations.append({"연령대": age, "직업": occupation, "비율": combined_ratio})

print(f"   - 총 {len(combinations)}개 조합 생성 완료")
print("\n   [조합 비율 샘플 (처음 5개)]")
for i, combo in enumerate(combinations[:5]):
    print(f"     {i+1}. {combo['연령대']}, {combo['직업']}: {combo['비율']*100:.2f}%")

# 4. 연도 컬럼 추출
year_columns = [col for col in df.columns if "년" in col]
print(f"\n4. 연도 컬럼 확인: {len(year_columns)}개 연도")

# 5. 각 원본 행을 54개 조합으로 확장
print("\n5. 데이터 확장 중...")
print("   - 각 원본 행을 54개 조합으로 확장...")

expanded_rows = []

for idx, row in df.iterrows():
    # 원본 행의 기본 정보 (성별, 혈액형별, 혈액원별)
    base_info = {
        "성별": row["성별"],
        "혈액형별": row["혈액형별"],
        "혈액원별": row["혈액원별"],
    }

    # 각 조합에 대해 새 행 생성
    for combo in combinations:
        new_row = base_info.copy()
        new_row["연령대"] = combo["연령대"]
        new_row["직업"] = combo["직업"]

        # 각 연도별 값에 비율 곱하기 및 반올림
        for year_col in year_columns:
            original_value = row[year_col]
            if pd.notna(original_value):
                new_value = original_value * combo["비율"]
                new_row[year_col] = round(new_value)
            else:
                new_row[year_col] = 0.0

        expanded_rows.append(new_row)

    # 진행 상황 출력 (100행마다)
    if (idx + 1) % 10 == 0:
        print(f"   - 처리 중: {idx + 1}/{len(df)} 행 완료...")

# 6. 확장된 데이터프레임 생성
print("\n6. 확장된 데이터프레임 생성 중...")
df_expanded = pd.DataFrame(expanded_rows)

# 컬럼 순서 정리: 성별, 혈액형별, 혈액원별, 연령대, 직업, 연도들
column_order = ["성별", "혈액형별", "혈액원별", "연령대", "직업"] + year_columns
df_expanded = df_expanded[column_order]

print(f"   - 확장된 데이터 행 수: {len(df_expanded):,}개")
print(f"   - 확장된 데이터 열 수: {len(df_expanded.columns)}개")

# 7. 데이터 검증
print("\n7. 데이터 검증 중...")
# 첫 번째 원본 행의 첫 번째 조합 확인
sample_original = df.iloc[0]
sample_expanded = df_expanded[
    (df_expanded["성별"] == sample_original["성별"])
    & (df_expanded["혈액형별"] == sample_original["혈액형별"])
    & (df_expanded["혈액원별"] == sample_original["혈액원별"])
    & (df_expanded["연령대"] == "30~39")
    & (df_expanded["직업"] == "회사원")
].iloc[0]

print("\n   [검증 샘플]")
print(
    f"   원본: {sample_original['성별']}, {sample_original['혈액형별']}, {sample_original['혈액원별']}"
)
print(f"   2005년 원본 값: {sample_original['2005 년']}")
print(
    f"   확장: {sample_expanded['성별']}, {sample_expanded['혈액형별']}, {sample_expanded['혈액원별']}, {sample_expanded['연령대']}, {sample_expanded['직업']}"
)
print(f"   2005년 확장 값: {sample_expanded['2005 년']}")
print(
    f"   계산: {sample_original['2005 년']} × 0.056 = {round(sample_original['2005 년'] * 0.056)}"
)
print(
    f"   검증: {'✓ 일치' if abs(sample_expanded['2005 년'] - round(sample_original['2005 년'] * 0.056)) < 0.01 else '✗ 불일치'}"
)

# 8. 최종 데이터 저장
output_file = "blood_donation_data_with_age_occupation.csv"
df_expanded.to_csv(output_file, index=False, encoding="utf-8")
print(f"\n8. 데이터 저장 완료!")
print(f"   - 저장된 파일: '{output_file}'")
print(f"   - 최종 행 수: {len(df_expanded):,}개")
print(f"   - 최종 열 수: {len(df_expanded.columns)}개")

# 9. 통계 요약
print("\n9. 통계 요약")
print(f"   - 원본 행 수: {len(df):,}개")
print(f"   - 확장 행 수: {len(df_expanded):,}개")
print(f"   - 확장 배수: {len(df_expanded) / len(df):.0f}배")
print(f"   - 연령대별 분포:")
for age in age_ratios.keys():
    count = len(df_expanded[df_expanded["연령대"] == age])
    print(f"     {age}: {count:,}개 행 ({count/len(df_expanded)*100:.1f}%)")
print(f"   - 직업별 분포:")
for occ in occupation_ratios.keys():
    count = len(df_expanded[df_expanded["직업"] == occ])
    print(f"     {occ}: {count:,}개 행 ({count/len(df_expanded)*100:.1f}%)")

print("\n" + "=" * 80)
print("연령대 및 직업 컬럼 추가 완료!")
print("=" * 80)
