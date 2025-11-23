#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
혈액 기부 데이터 특성 분석 스크립트
"""

import pandas as pd
import numpy as np

# CSV 파일 읽기
print("=" * 80)
print("혈액 기부 데이터 특성 분석")
print("=" * 80)

df = pd.read_csv('blood_donation_data.csv', encoding='utf-8')

print(f"\n1. 기본 정보")
print(f"   - 총 행 수: {len(df):,}개")
print(f"   - 총 열 수: {len(df.columns)}개")

print(f"\n2. 컬럼 정보")
print(f"   - 컬럼명: {list(df.columns)}")

print(f"\n3. 카테고리 변수 분석")
print(f"\n   [성별]")
print(df['성별'].value_counts())
print(f"\n   [혈액형별]")
print(df['혈액형별'].value_counts())
print(f"\n   [혈액원별]")
print(df['혈액원별'].value_counts())
print(f"\n   [항목]")
print(df['항목'].value_counts())
print(f"\n   [단위]")
print(df['단위'].value_counts())

# 연도 컬럼 추출
year_columns = [col for col in df.columns if '년' in col]
print(f"\n4. 시계열 정보")
print(f"   - 연도 범위: {year_columns[0]} ~ {year_columns[-1]}")
print(f"   - 총 연도 수: {len(year_columns)}개")

# 데이터 타입별 분석
print(f"\n5. 데이터 타입")
print(df.dtypes)

# 결측치 분석
print(f"\n6. 결측치 분석")
missing_data = df[year_columns].isnull().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    print(f"   - 결측치가 있는 연도:")
    for year, count in missing_data.items():
        print(f"     {year}: {count}개 ({count/len(df)*100:.1f}%)")
else:
    print("   - 결측치 없음")

# 빈 문자열 체크
print(f"\n7. 빈 문자열 분석")
empty_strings = {}
for col in year_columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        empty_strings[col] = empty_count

if empty_strings:
    print(f"   - 빈 문자열이 있는 연도:")
    for year, count in empty_strings.items():
        print(f"     {year}: {count}개 ({count/len(df)*100:.1f}%)")
else:
    print("   - 빈 문자열 없음")

# 실적 데이터만 추출 (구성비 제외)
performance_df = df[df['항목'] == '실적[건]'].copy()
print(f"\n8. 실적 데이터 분석 (항목='실적[건]')")
print(f"   - 실적 데이터 행 수: {len(performance_df):,}개")

# 숫자 데이터로 변환
for col in year_columns:
    performance_df[col] = pd.to_numeric(performance_df[col], errors='coerce')

# 통계 요약
print(f"\n9. 연도별 통계 요약 (실적[건])")
stats_summary = performance_df[year_columns].describe()
print(stats_summary)

# 총합 데이터 확인
total_row = performance_df[
    (performance_df['성별'] == '계') & 
    (performance_df['혈액형별'] == '합계') & 
    (performance_df['혈액원별'] == '합계')
]

if len(total_row) > 0:
    print(f"\n10. 전체 합계 데이터 (2005-2024)")
    total_values = total_row[year_columns].iloc[0]
    print(total_values.to_string())
    
    # 시계열 트렌드
    print(f"\n11. 시계열 트렌드 분석")
    print(f"   - 최소값: {total_values.min():,.0f}건 ({total_values.idxmin()})")
    print(f"   - 최대값: {total_values.max():,.0f}건 ({total_values.idxmax()})")
    print(f"   - 평균값: {total_values.mean():,.0f}건")
    print(f"   - 중앙값: {total_values.median():,.0f}건")
    
    # 증감률 계산
    if len(total_values) > 1:
        growth_rates = total_values.pct_change() * 100
        growth_rates = growth_rates.dropna()
        print(f"\n   연도별 증감률:")
        for year, rate in growth_rates.items():
            print(f"     {year}: {rate:+.2f}%")

print(f"\n12. 데이터 샘플 (처음 5행)")
print(performance_df.head().to_string())

print(f"\n" + "=" * 80)
print("분석 완료")
print("=" * 80)

