#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
혈액 기부 데이터 전처리 스크립트
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("혈액 기부 데이터 전처리 시작")
print("=" * 80)

# 1. CSV 파일 읽기
print("\n1. 데이터 로딩 중...")
df = pd.read_csv("blood_donation_data.csv", encoding="utf-8")
print(f"   - 원본 데이터 행 수: {len(df):,}개")
print(f"   - 원본 데이터 열 수: {len(df.columns)}개")

# 2. 불필요한 컬럼 제거 (Unnamed: 25)
if "Unnamed: 25" in df.columns:
    df = df.drop(columns=["Unnamed: 25"])
    print("\n2. 불필요한 컬럼 제거 완료 (Unnamed: 25)")

# 3. 항목 컬럼에서 "구성비[%]" 행 삭제
print("\n3. 항목='구성비[%]' 행 삭제 중...")
before_count = len(df)
df = df[df["항목"] != "구성비[%]"].copy()
after_count = len(df)
print(f"   - 삭제 전: {before_count:,}개 행")
print(f"   - 삭제 후: {after_count:,}개 행")
print(f"   - 삭제된 행 수: {before_count - after_count:,}개")

# 4. 성별 컬럼에서 "계" 행 삭제
print("\n4. 성별='계' 행 삭제 중...")
before_count = len(df)
df = df[df["성별"] != "계"].copy()
after_count = len(df)
print(f"   - 삭제 전: {before_count:,}개 행")
print(f"   - 삭제 후: {after_count:,}개 행")
print(f"   - 삭제된 행 수: {before_count - after_count:,}개")

# 5. 혈액형별 컬럼에서 "합계" 행 삭제
print("\n5. 혈액형별='합계' 행 삭제 중...")
before_count = len(df)
df = df[df["혈액형별"] != "합계"].copy()
after_count = len(df)
print(f"   - 삭제 전: {before_count:,}개 행")
print(f"   - 삭제 후: {after_count:,}개 행")
print(f"   - 삭제된 행 수: {before_count - after_count:,}개")

# 6. 혈액원별 컬럼에서 "합계"와 "대한적십자사" 행 삭제
print("\n6. 혈액원별='합계' 및 '대한적십자사' 행 삭제 중...")
before_count = len(df)
df = df[(df["혈액원별"] != "합계") & (df["혈액원별"] != "대한적십자사")].copy()
after_count = len(df)
print(f"   - 삭제 전: {before_count:,}개 행")
print(f"   - 삭제 후: {after_count:,}개 행")
print(f"   - 삭제된 행 수: {before_count - after_count:,}개")

# 7. 혈액형별 컬럼 값 변환 (RH+/- 처리)
print("\n7. 혈액형별 컬럼 값 변환 중...")
print("   - RH(+) 및 RH(-) 행 삭제 및 혈액형 변환 처리...")

# 인덱스 리셋 (삭제 후 인덱스가 꼬이지 않도록)
df = df.reset_index(drop=True)

# 삭제할 행의 인덱스와 변환할 행의 인덱스를 저장
rows_to_delete = []
rows_to_update = {}  # {인덱스: 새로운_값}

current_rh_type = None  # 현재 RH 타입 추적 ('+' 또는 '-')
i = 0

while i < len(df):
    blood_type = df.loc[i, "혈액형별"]

    # RH(+) 행을 만나면
    if blood_type == "RH(+)":
        rows_to_delete.append(i)
        current_rh_type = "+"
        print(f"   - RH(+) 감지 (행 {i+1}): 삭제 예정, 이후 혈액형을 RH+로 변환")
        i += 1
        continue

    # RH(-) 행을 만나면
    elif blood_type == "RH(-)":
        rows_to_delete.append(i)
        current_rh_type = "-"
        print(f"   - RH(-) 감지 (행 {i+1}): 삭제 예정, 이후 혈액형을 RH-로 변환")
        i += 1
        continue

    # O형, A형, B형, AB형을 만나면
    elif blood_type in ["O형", "A형", "B형", "AB형"]:
        if current_rh_type is not None:
            # 현재 RH 타입에 따라 변환
            if current_rh_type == "+":
                new_value = blood_type.replace("형", "+")
                rows_to_update[i] = new_value
            elif current_rh_type == "-":
                new_value = blood_type.replace("형", "-")
                rows_to_update[i] = new_value
        i += 1
        continue

    # 다른 혈액형을 만나면 (RH 타입 리셋)
    else:
        current_rh_type = None
        i += 1
        continue

# 혈액형 값 업데이트
print(f"\n   - 변환할 행 수: {len(rows_to_update)}개")
for idx, new_value in rows_to_update.items():
    df.loc[idx, "혈액형별"] = new_value

# RH(+) 및 RH(-) 행 삭제
print(f"   - 삭제할 행 수: {len(rows_to_delete)}개")
df = df.drop(index=rows_to_delete).reset_index(drop=True)

print(f"\n   - 변환 완료 후 행 수: {len(df):,}개")

# 8. 변환 결과 확인
print("\n8. 변환 결과 확인")
print("\n   [혈액형별 값 분포]")
blood_type_counts = df["혈액형별"].value_counts().sort_index()
print(blood_type_counts)

# 9. 연도별 빈 값 채우기 (각 행의 나머지 연도 평균으로)
print("\n9. 연도별 빈 값 채우기 중...")

# 연도 컬럼 추출 (2005 년 ~ 2024 년)
year_columns = [col for col in df.columns if "년" in col]
print(f"   - 연도 컬럼 수: {len(year_columns)}개")

# 각 행의 연도 데이터를 숫자형으로 변환하고 빈 값 채우기
filled_count = 0
total_missing = 0

for idx in df.index:
    row = df.loc[idx, year_columns].copy()

    # 빈 문자열을 NaN으로 변환
    row = row.replace("", np.nan)

    # 숫자형으로 변환 (변환 실패 시 NaN)
    row_numeric = pd.to_numeric(row, errors="coerce")

    # 빈 값(NaN)이 있는지 확인
    missing_mask = row_numeric.isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        total_missing += missing_count

        # 유효한 값들의 평균 계산
        valid_values = row_numeric[~missing_mask]

        if len(valid_values) > 0:
            mean_value = valid_values.mean()
            # 소수점 첫째 자리에서 반올림
            mean_value = round(mean_value)

            # 빈 값에 평균값 채우기
            for year_col in year_columns:
                if pd.isna(row_numeric[year_col]):
                    df.loc[idx, year_col] = mean_value
                    filled_count += 1
        else:
            # 모든 값이 비어있는 경우 0으로 채우기
            for year_col in year_columns:
                if pd.isna(row_numeric[year_col]):
                    df.loc[idx, year_col] = 0
                    filled_count += 1

print(f"   - 총 빈 값 개수: {total_missing:,}개")
print(f"   - 채워진 값 개수: {filled_count:,}개")
print(f"   - 빈 값 채우기 완료!")

# 10. 모든 연도 컬럼을 float 타입으로 변환
print("\n10. 연도 컬럼 타입 변환 중...")
for year_col in year_columns:
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("float64")
print(f"   - 모든 연도 컬럼을 float 타입으로 변환 완료!")

# 11. "항목" 컬럼 삭제 (구성비[%] 삭제 후 실적[건]만 남아 불필요)
print("\n11. '항목' 컬럼 삭제 중...")
if "항목" in df.columns:
    df = df.drop(columns=["항목"])
    print(f"   - '항목' 컬럼 삭제 완료!")
else:
    print(f"   - '항목' 컬럼이 존재하지 않습니다.")

# 12. "단위" 컬럼 삭제
print("\n12. '단위' 컬럼 삭제 중...")
if "단위" in df.columns:
    df = df.drop(columns=["단위"])
    print(f"   - '단위' 컬럼 삭제 완료!")
else:
    print(f"   - '단위' 컬럼이 존재하지 않습니다.")

# 13. 최종 데이터 저장
output_file = "blood_donation_data_preprocessed.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\n13. 전처리 완료!")
print(f"   - 전처리된 데이터가 '{output_file}'에 저장되었습니다.")
print(f"   - 최종 행 수: {len(df):,}개")
print(f"   - 최종 열 수: {len(df.columns)}개")

print("\n" + "=" * 80)
print("전처리 완료!")
print("=" * 80)
