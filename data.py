"""데이터 추출 모듈 (MDT 파일 기반)"""

import csv
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
import geopandas as gpd

from models import MDTRecord, CellInfo, RUInfo


def _parse_int(value: str) -> int:
    """쉼표 포함된 숫자 파싱"""
    return int(value.strip().replace(',', ''))


def _parse_float(value: str) -> float:
    """쉼표 포함된 숫자 파싱"""
    return float(value.strip().replace(',', ''))


def load_mdt_data(filepath: str | Path) -> List[MDTRecord]:
    """
    MDT 측정 데이터 로드 (GPOT 형식)

    파일 형식: 세미콜론 구분자
    컬럼: 기간_월;RU_ID;GPOT_ID;POT_IN/OUT;UTMKX;UTMKY;PCI;수집건수;RSRP평균;RSRQ평균;...
    """
    # 1. 데이터 로드
    df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')

    # 2. 불필요한 컬럼 삭제
    drop_cols = ['POT_IN/OUT', 'UTMKX', 'UTMKY', 'PCI']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. 기간_월이 여러 종류면 RU_ID, GPOT_ID 기준으로 집계
    if df['기간_월'].nunique() > 1:
        agg_dict = {'수집건수': 'sum'}
        # 나머지 수치 컬럼은 평균
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['기간_월', '수집건수']:
                agg_dict[col] = 'mean'

        df = df.groupby(['RU_ID', 'GPOT_ID'], as_index=False).agg(agg_dict)

    # 4. GPOT_ID에서 위도/경도 계산
    df[['POT_X', 'POT_Y']] = df['GPOT_ID'].str.split('_', expand=True).astype(float) * 25
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['POT_X'], df['POT_Y']),
        crs='EPSG:5179'
    ).to_crs('EPSG:4326')
    df['위도'] = gdf.geometry.y
    df['경도'] = gdf.geometry.x

    # 5. MDTRecord 형식으로 변환
    data = []
    for _, row in df.iterrows():
        data.append({
            'grid_id': row['GPOT_ID'],
            'cell_id': row['RU_ID'],
            'sample_count': int(str(row['수집건수']).replace(',', '')),
            'avg_rsrp': float(str(row['RSRP평균']).replace(',', '')) if pd.notna(row['RSRP평균']) else 0.0,
            'avg_rsrq': float(str(row['RSRQ평균']).replace(',', '')) if pd.notna(row['RSRQ평균']) else 0.0,
            'avg_sinr': float(str(row['SINR평균']).replace(',', '')) if pd.notna(row['SINR평균']) else 0.0,
            'avg_latency': float(str(row['Latency평균']).replace(',', '')) if pd.notna(row['Latency평균']) else 0.0,
            'avg_dl_ri': float(str(row['DL_RI평균']).replace(',', '')) if pd.notna(row['DL_RI평균']) else 0.0,
            'avg_cqi': float(str(row['CQI평균']).replace(',', '')) if pd.notna(row['CQI평균']) else 0.0,
            'avg_distance': float(str(row['평균거리']).replace(',', '')) if pd.notna(row['평균거리']) else 0.0,
            'latitude': float(row['위도']),
            'longitude': float(row['경도']),
        })
    return data


def extract_cell_info(mdt_data: List[MDTRecord]) -> List[CellInfo]:
    """MDT 데이터에서 고유한 기지국 정보 추출"""
    seen: Set[str] = set()
    cells = []

    for record in mdt_data:
        if record['cell_id'] not in seen:
            seen.add(record['cell_id'])
            cells.append({'cell_id': record['cell_id']})
    return cells


def filter_problem_grids(
    mdt_data: List[MDTRecord],
    rsrp_threshold: float = -105,
    rsrq_threshold: float = -15
) -> List[MDTRecord]:
    """
    문제 격자 필터링: RSRP 양호 & RSRQ 불량

    Args:
        rsrp_threshold: RSRP 최소값 (이 값보다 커야 "양호")
        rsrq_threshold: RSRQ 최대값 (이 값보다 작으면 "불량")
    """
    return [
        record for record in mdt_data
        if record['avg_rsrp'] >= rsrp_threshold
        and record['avg_rsrq'] <= rsrq_threshold
    ]


def group_by_cell(mdt_data: List[MDTRecord]) -> Dict[str, List[MDTRecord]]:
    """기지국별로 MDT 레코드 그룹핑"""
    grouped: Dict[str, List[MDTRecord]] = {}
    for record in mdt_data:
        cell_id = record['cell_id']
        if cell_id not in grouped:
            grouped[cell_id] = []
        grouped[cell_id].append(record)
    return grouped


def get_cell_statistics(mdt_data: List[MDTRecord]) -> Dict[str, Dict]:
    """기지국별 통계 계산"""
    grouped = group_by_cell(mdt_data)
    stats = {}

    for cell_id, records in grouped.items():
        total_samples = sum(r['sample_count'] for r in records)
        weighted_rsrp = sum(r['avg_rsrp'] * r['sample_count'] for r in records)
        weighted_rsrq = sum(r['avg_rsrq'] * r['sample_count'] for r in records)

        stats[cell_id] = {
            'cell_id': cell_id,
            'grid_count': len(records),
            'total_samples': total_samples,
            'avg_rsrp': weighted_rsrp / total_samples if total_samples > 0 else 0,
            'avg_rsrq': weighted_rsrq / total_samples if total_samples > 0 else 0,
        }
    return stats


# ============== RU 데이터 로드 ==============

def _dms_to_decimal(dms_str: str) -> float:
    """
    도-분-초 형식을 십진수로 변환
    예: "37-23-08.586" -> 37.385718333...
    """
    if not dms_str or dms_str.strip() == '':
        return 0.0

    parts = dms_str.strip().split('-')
    if len(parts) != 3:
        return 0.0

    try:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return degrees + minutes / 60 + seconds / 3600
    except ValueError:
        return 0.0


def _safe_float(value: str) -> float:
    """안전한 float 변환 (이상한 값 처리)"""
    if not value or value.strip() == '':
        return 0.0
    try:
        return float(value.strip())
    except ValueError:
        return 0.0


def _safe_int(value: str) -> int:
    """안전한 int 변환"""
    if not value or value.strip() == '':
        return 0
    try:
        return int(value.strip())
    except ValueError:
        return 0


def load_ru_data(filepath: str | Path) -> List[RUInfo]:
    """
    RU(기지국) 정보 로드

    파일 형식: 탭 구분자 (TSV)
    주요 컬럼: 장비ID, 장비명, 장비유형, PCI, 방위각, M_Tilt, E_Tilt, 위도, 경도
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            data.append({
                'cell_id': row['장비ID'].strip(),
                'cell_name': row['장비명'].strip(),
                'equipment_type': row['장비유형'].strip(),
                'pci': _safe_int(row.get('PCI', '')),
                'azimuth': _safe_float(row.get('안테나1 방위각', '')),
                'm_tilt': _safe_float(row.get('안테나1 M_Tilt', '')),
                'e_tilt': _safe_float(row.get('안테나1 E_Tilt', '')),
                'latitude': _dms_to_decimal(row.get('위도', '')),
                'longitude': _dms_to_decimal(row.get('경도', '')),
                'address': row.get('도로명주소', '').strip(),
                'dong': row.get('읍/면/동(리)', '').strip(),
            })
    return data


def get_ru_by_cell_id(ru_data: List[RUInfo]) -> Dict[str, RUInfo]:
    """cell_id를 키로 하는 딕셔너리 생성"""
    return {ru['cell_id']: ru for ru in ru_data}


# ============== 추후 DB 확장 시 ==============
#
# def load_mdt_from_db(connection, query_params) -> List[MDTRecord]:
#     """DB에서 MDT 데이터 로드"""
#     pass
