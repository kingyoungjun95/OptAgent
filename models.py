"""데이터 모델 정의 (TypedDict 기반)"""

from typing import TypedDict, List, Dict, Optional


# ============== 입력 데이터 모델 ==============

class MDTRecord(TypedDict):
    """MDT 측정 데이터 (격자 단위)"""
    grid_id: str           # GPOT_ID (예: 38090_76620)
    cell_id: str           # RU_ID
    sample_count: int      # 수집건수
    avg_rsrp: float        # RSRP평균 (dBm)
    avg_rsrq: float        # RSRQ평균 (dB)
    avg_sinr: float        # SINR평균 (dB)
    avg_latency: float     # Latency평균 (ms)
    avg_dl_ri: float       # DL_RI평균
    avg_cqi: float         # CQI평균
    avg_distance: float    # 평균거리 (m)
    latitude: float        # 위도 (EPSG:4326)
    longitude: float       # 경도 (EPSG:4326)


class CellInfo(TypedDict):
    """기지국 정보 (MDT에서 추출)"""
    cell_id: str


class RUInfo(TypedDict):
    """RU(기지국) 상세 정보"""
    cell_id: str           # 장비ID
    cell_name: str         # 장비명
    equipment_type: str    # 장비유형
    pci: int               # PCI
    azimuth: float         # 안테나1 방위각
    m_tilt: float          # 안테나1 M_Tilt (기계적 틸트)
    e_tilt: float          # 안테나1 E_Tilt (전기적 틸트)
    latitude: float        # 위도 (십진수)
    longitude: float       # 경도 (십진수)
    address: str           # 도로명주소
    dong: str              # 읍/면/동(리)


# ============== 분석 결과 모델 ==============

class LLMThresholds(TypedDict):
    """임계값 설정"""
    rsrp_threshold: float
    rsrq_threshold: float
    sinr_threshold: float
    problem_ratio_threshold: float
    reasoning: str


class LLMTargetAnalysis(TypedDict):
    """LLM의 타겟 셀 분석 결과"""
    severity: str  # "low", "medium", "high", "critical"
    primary_issue: str
    analysis: str


class SINRAnalysisResult(TypedDict):
    """SINR Agent 분석 결과"""
    timestamp: str
    problem_grids: List[MDTRecord]
    serving_cells: List[CellInfo]
    grid_cell_mapping: Dict[str, str]  # grid_id -> cell_id
    severity_score: float  # 0~1
    recommendation: str  # "OPTIMIZE" or "MONITOR"
    # LLM 분석 필드
    llm_thresholds: Optional[LLMThresholds]
    llm_target_analysis: Optional[Dict[str, LLMTargetAnalysis]]  # cell_id -> analysis


class LLMNeighborAnalysis(TypedDict):
    """LLM의 이웃셀 분석 결과"""
    root_cause: str  # "NO_INTERFERENCE", "MILD_INTERFERENCE", "MODERATE_INTERFERENCE", "SEVERE_INTERFERENCE"
    risk_level: str  # "low", "medium", "high"
    main_interferer: Optional[str]
    analysis: str
    recommendation: str  # "PROCEED", "CAUTION", "ABORT"


class NeighborAnalysisResult(TypedDict):
    """Neighbor Agent 분석 결과"""
    timestamp: str
    serving_cell_id: str
    interference_sources: List[str]  # 간섭 유발 cell_ids
    root_cause: str
    optimization_risk: float  # 0~1
    recommendation: str  # "PROCEED", "CAUTION", "ABORT"
    # LLM 분석 필드
    llm_analysis: Optional[LLMNeighborAnalysis]


class LLMDecision(TypedDict):
    """LLM의 최종 판단 결과"""
    decision: str  # "OPTIMIZE", "HOLD", "REJECT"
    confidence: float
    reasoning: str
    actions: List[Dict]


class OptimizationDecision(TypedDict):
    """최종 최적화 결정"""
    timestamp: str
    cell_id: str
    decision: str  # "OPTIMIZE", "HOLD", "REJECT"
    confidence: float
    reasoning: str
    actions: List[Dict]
    llm_explanation: Optional[str]  # LLM 생성 설명 (선택적)
    llm_decision: Optional[LLMDecision]  # LLM 직접 판단 결과
