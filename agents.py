"""Agent 클래스 모듈 (LLM 기반)

각 Agent가 자체 LLM 로직을 포함하여 분석/판단 수행
"""

from datetime import datetime
from typing import List, Dict, Optional
import ollama
import json
import re

from models import MDTRecord, CellInfo, RUInfo, SINRAnalysisResult, NeighborAnalysisResult, OptimizationDecision


# ============== 공통 LLM 유틸리티 ==============

def extract_json(text: str) -> Optional[Dict]:
    """LLM 응답에서 JSON 추출"""
    # 1. ```json ... ``` 블록에서 추출 시도
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. { } 블록 직접 추출 시도
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def llm_generate(model: str, prompt: str, system_prompt: str = None) -> str:
    """LLM 응답 생성"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"[LLM 오류: {e}]"


def llm_generate_json(model: str, prompt: str, system_prompt: str = None) -> Optional[Dict]:
    """JSON 응답 생성 및 파싱"""
    response = llm_generate(model, prompt, system_prompt)
    return extract_json(response)


# ============== SINR Agent ==============

class SINRAgent:
    """
    SINR(RSRQ) 분석 Agent (Positive) - Cell 단위 분석

    - Cell별 그룹화 후 평균 SINR로 1차 필터링
    - 문제 격자 비율 또는 지리적 클러스터링으로 2차 필터링
    - LLM이 타겟 셀별 분석값 도출
    """

    SYSTEM_PROMPT = """당신은 RAN(Radio Access Network) 최적화 전문가입니다.
MDT(Minimization of Drive Test) 데이터를 분석하여 SINR/RSRQ 문제를 진단합니다.

[역할]
- RSRP는 양호하나 RSRQ가 낮은 격자 탐지
- 문제 셀별 심각도 평가

[응답 규칙]
- 반드시 JSON 형식으로 응답
- 기술적 근거를 명확히 제시
- 한국어로 작성"""

    def __init__(
        self,
        rsrp_threshold: float = -105,
        rsrq_threshold: float = -15,
        sinr_threshold: float = 10.0,           # 셀 평균 SINR 임계값 (이 값 이하면 문제)
        problem_ratio_threshold: float = 0.3,   # 문제 격자 비율 임계값 (30% 이상이면 문제)
        cluster_distance_threshold: float = 0.005,  # 클러스터링 판단 위도/경도 표준편차 (약 500m)
        verbose: bool = False,
        llm_model: str = "gemma3:27b"
    ):
        self.rsrp_threshold = rsrp_threshold
        self.rsrq_threshold = rsrq_threshold
        self.sinr_threshold = sinr_threshold
        self.problem_ratio_threshold = problem_ratio_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        self.verbose = verbose
        self.llm_model = llm_model

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [SINRAgent] {msg}")

    def _group_by_cell(self, mdt_data: List[MDTRecord]) -> Dict[str, List[MDTRecord]]:
        """cell_id별로 MDT 레코드 그룹화"""
        grouped: Dict[str, List[MDTRecord]] = {}
        for record in mdt_data:
            cell_id = record['cell_id']
            if cell_id not in grouped:
                grouped[cell_id] = []
            grouped[cell_id].append(record)
        return grouped

    def _calculate_cell_statistics(self, cell_grids: List[MDTRecord]) -> Dict:
        """셀 내 격자들의 통계 계산"""
        if not cell_grids:
            return {}

        total_samples = sum(g['sample_count'] for g in cell_grids)

        # 가중 평균 계산 (샘플 수 기준)
        if total_samples > 0:
            avg_sinr = sum(g['avg_sinr'] * g['sample_count'] for g in cell_grids) / total_samples
            avg_rsrp = sum(g['avg_rsrp'] * g['sample_count'] for g in cell_grids) / total_samples
            avg_rsrq = sum(g['avg_rsrq'] * g['sample_count'] for g in cell_grids) / total_samples
        else:
            avg_sinr = sum(g['avg_sinr'] for g in cell_grids) / len(cell_grids)
            avg_rsrp = sum(g['avg_rsrp'] for g in cell_grids) / len(cell_grids)
            avg_rsrq = sum(g['avg_rsrq'] for g in cell_grids) / len(cell_grids)

        return {
            'grid_count': len(cell_grids),
            'total_samples': total_samples,
            'avg_sinr': avg_sinr,
            'avg_rsrp': avg_rsrp,
            'avg_rsrq': avg_rsrq,
        }

    def _is_problem_grid(self, grid: MDTRecord) -> bool:
        """개별 격자가 문제 격자인지 판단 (RSRP 양호 & RSRQ 불량)"""
        return (
            grid['avg_rsrp'] >= self.rsrp_threshold and
            grid['avg_rsrq'] <= self.rsrq_threshold
        )

    def _is_clustered(self, grids: List[MDTRecord]) -> bool:
        """문제 격자들이 지리적으로 클러스터링 되어 있는지 판단"""
        if len(grids) < 2:
            return False

        lats = [g['latitude'] for g in grids]
        lons = [g['longitude'] for g in grids]

        # 위도/경도 표준편차 계산
        lat_mean = sum(lats) / len(lats)
        lon_mean = sum(lons) / len(lons)

        lat_std = (sum((lat - lat_mean) ** 2 for lat in lats) / len(lats)) ** 0.5
        lon_std = (sum((lon - lon_mean) ** 2 for lon in lons) / len(lons)) ** 0.5

        # 표준편차가 임계값보다 작으면 클러스터링으로 판단
        return lat_std < self.cluster_distance_threshold and lon_std < self.cluster_distance_threshold

    def _identify_problem_cells(self, mdt_data: List[MDTRecord]) -> Dict[str, Dict]:
        """
        문제 셀 식별 (2단계 필터링)

        1단계: 셀 평균 SINR이 임계값 이하
        2단계: 문제 격자 비율이 임계값 이상 OR 문제 격자가 클러스터링 됨

        Returns:
            Dict[cell_id, {'grids': List[MDTRecord], 'stats': Dict, 'problem_grids': List, 'reason': str}]
        """
        grouped = self._group_by_cell(mdt_data)
        problem_cells = {}

        for cell_id, grids in grouped.items():
            stats = self._calculate_cell_statistics(grids)

            # 1단계: 평균 SINR 필터
            if stats['avg_sinr'] > self.sinr_threshold:
                continue  # SINR이 양호하면 스킵

            # 문제 격자 식별
            problem_grids = [g for g in grids if self._is_problem_grid(g)]
            problem_ratio = len(problem_grids) / len(grids) if grids else 0

            # 2단계: 비율 또는 클러스터링 필터
            is_ratio_problem = problem_ratio >= self.problem_ratio_threshold
            is_clustered = self._is_clustered(problem_grids) if problem_grids else False

            if is_ratio_problem or is_clustered:
                reason_parts = []
                if is_ratio_problem:
                    reason_parts.append(f"문제격자 비율 {problem_ratio:.1%}")
                if is_clustered:
                    reason_parts.append("문제격자 클러스터링")

                problem_cells[cell_id] = {
                    'grids': grids,
                    'stats': stats,
                    'problem_grids': problem_grids,
                    'problem_ratio': problem_ratio,
                    'is_clustered': is_clustered,
                    'reason': ' & '.join(reason_parts)
                }

        return problem_cells

    def _llm_analyze_targets(self, cell_summaries: List[Dict]) -> Dict[str, Dict]:
        """LLM: 타겟 셀 분석"""
        top_cells = cell_summaries[:10]

        cells_info = "\n".join([
            f"- {c['cell_id']}: 문제격자 {c['grid_count']}개, "
            f"평균RSRQ {c['avg_rsrq']:.1f}dB, 샘플 {c['total_samples']}개"
            for c in top_cells
        ])

        prompt = f"""다음 문제 셀들을 분석하고 각 셀의 상태를 평가하세요.

[문제 셀 목록]
{cells_info}

각 셀에 대해 다음 JSON 형식으로 응답하세요:
```json
{{
    "<cell_id>": {{
        "severity": "<low/medium/high/critical>",
        "primary_issue": "<주요 문제점>",
        "analysis": "<상세 분석>"
    }},
    ...
}}
```"""

        result = llm_generate_json(self.llm_model, prompt, self.SYSTEM_PROMPT)
        if result:
            return result

        return {
            cell['cell_id']: {
                "severity": "medium",
                "primary_issue": "RSRQ 저하",
                "analysis": "LLM 응답 파싱 실패"
            }
            for cell in top_cells
        }

    def analyze(self, mdt_data: List[MDTRecord]) -> SINRAnalysisResult:
        """SINR 분석 실행 (Cell 단위)"""
        self._log(f"분석 시작 - 입력 레코드: {len(mdt_data):,}개")
        self._log(f"설정: SINR<={self.sinr_threshold}, 문제격자비율>={self.problem_ratio_threshold:.0%}")

        # 1. Cell 단위 문제 셀 식별 (2단계 필터링)
        self._log("문제 셀 식별 중...")
        problem_cells = self._identify_problem_cells(mdt_data)
        self._log(f"문제 셀: {len(problem_cells)}개")

        # 2. 문제 격자 및 서빙 셀 추출
        problem_grids = []
        serving_cells = []
        for cell_id, cell_data in problem_cells.items():
            problem_grids.extend(cell_data['problem_grids'])
            serving_cells.append({'cell_id': cell_id})
            if self.verbose:
                stats = cell_data['stats']
                self._log(f"  {cell_id}: SINR={stats['avg_sinr']:.1f}dB, "
                         f"문제격자={len(cell_data['problem_grids'])}/{len(cell_data['grids'])}, "
                         f"사유={cell_data['reason']}")

        self._log(f"총 문제 격자: {len(problem_grids):,}개")

        # 3. LLM 타겟 분석 (문제 셀이 있는 경우)
        llm_target_analysis = None
        if problem_cells:
            self._log("LLM에게 타겟 분석 요청 중...")
            cell_summaries = self._generate_cell_summaries_from_problem_cells(problem_cells)
            llm_target_analysis = self._llm_analyze_targets(cell_summaries)
            self._log(f"타겟 분석 완료: {len(llm_target_analysis)}개 셀")

        # 4. 심각도 계산
        severity = self._calculate_severity_from_cells(problem_cells)

        # threshold 정보 (LLM 없이 설정값 사용)
        thresholds = {
            "rsrp_threshold": self.rsrp_threshold,
            "rsrq_threshold": self.rsrq_threshold,
            "sinr_threshold": self.sinr_threshold,
            "problem_ratio_threshold": self.problem_ratio_threshold,
            "reasoning": "설정값 기반 (규칙 기반)"
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'problem_grids': problem_grids,
            'serving_cells': serving_cells,
            'grid_cell_mapping': {r['grid_id']: r['cell_id'] for r in problem_grids},
            'severity_score': severity,
            'recommendation': "OPTIMIZE" if severity > 0.5 else "MONITOR",
            'llm_thresholds': thresholds,
            'llm_target_analysis': llm_target_analysis,
        }

    def _generate_cell_summaries_from_problem_cells(self, problem_cells: Dict[str, Dict]) -> List[Dict]:
        """문제 셀 정보에서 LLM용 요약 생성"""
        summaries = []
        for cell_id, cell_data in problem_cells.items():
            stats = cell_data['stats']
            summaries.append({
                'cell_id': cell_id,
                'grid_count': len(cell_data['problem_grids']),
                'avg_rsrq': stats['avg_rsrq'],
                'avg_sinr': stats['avg_sinr'],
                'total_samples': stats['total_samples'],
                'problem_ratio': cell_data['problem_ratio'],
                'is_clustered': cell_data['is_clustered'],
                'reason': cell_data['reason']
            })

        return sorted(summaries, key=lambda x: x['grid_count'], reverse=True)

    def _calculate_severity_from_cells(self, problem_cells: Dict[str, Dict]) -> float:
        """문제 셀 기반 심각도 계산"""
        if not problem_cells:
            return 0.0

        # 모든 문제 격자 수집
        all_problem_grids = []
        for cell_data in problem_cells.values():
            all_problem_grids.extend(cell_data['problem_grids'])

        if not all_problem_grids:
            return 0.0

        total_samples = sum(g['sample_count'] for g in all_problem_grids)
        if total_samples == 0:
            return 0.0

        # RSRQ 기반 심각도
        weighted_rsrq = sum(g['avg_rsrq'] * g['sample_count'] for g in all_problem_grids) / total_samples
        rsrq_factor = max(0, (self.rsrq_threshold - weighted_rsrq) / 10)

        # 샘플 수 기반 심각도
        sample_factor = min(1, total_samples / 500)

        # 문제 셀 수 기반 심각도
        cell_factor = min(1, len(problem_cells) / 10)

        return min(1.0, (rsrq_factor + sample_factor + cell_factor) / 3)

    def _calculate_severity(self, problem_grids: List[MDTRecord]) -> float:
        """심각도 계산 (하위 호환용)"""
        if not problem_grids:
            return 0.0

        total_samples = sum(g['sample_count'] for g in problem_grids)
        if total_samples == 0:
            return 0.0

        weighted_rsrq = sum(g['avg_rsrq'] * g['sample_count'] for g in problem_grids) / total_samples
        rsrq_factor = max(0, (self.rsrq_threshold - weighted_rsrq) / 10)
        sample_factor = min(1, total_samples / 500)

        return min(1.0, (rsrq_factor + sample_factor) / 2)


# ============== Neighbor Agent ==============

class NeighborAgent:
    """
    Neighbor 분석 Agent (Negative) - LLM 기반

    - LLM이 타겟 셀의 이웃셀 간섭 상황 분석
    - 근본 원인 및 위험도 판단
    """

    SYSTEM_PROMPT = """당신은 RAN(Radio Access Network) 최적화 전문가입니다.
이웃 셀 간의 간섭 상황을 분석하여 최적화 위험도를 평가합니다.

[역할]
- 동일 격자에서 측정되는 다중 셀 분석
- 간섭 유발 셀 식별 및 영향도 평가
- 최적화 수행 시 위험도 판단

[응답 규칙]
- 반드시 JSON 형식으로 응답
- 기술적 근거를 명확히 제시
- 한국어로 작성"""

    def __init__(
        self,
        interference_threshold: float = 0.3,
        verbose: bool = False,
        llm_model: str = "gemma3:27b"
    ):
        self.interference_threshold = interference_threshold
        self.verbose = verbose
        self.llm_model = llm_model

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [NeighborAgent] {msg}")

    def _collect_interference_data(
        self,
        cell_id: str,
        cell_grids: List[MDTRecord],
        grid_to_cells: Dict[str, List[str]],
        mdt_data: List[MDTRecord]
    ) -> Dict:
        """간섭 데이터 수집"""
        interference_cells = {}
        for grid in cell_grids:
            for other_cell in grid_to_cells.get(grid['grid_id'], []):
                if other_cell != cell_id:
                    if other_cell not in interference_cells:
                        interference_cells[other_cell] = {'overlap_count': 0, 'rsrp_values': []}
                    interference_cells[other_cell]['overlap_count'] += 1

        for record in mdt_data:
            if record['cell_id'] in interference_cells:
                interference_cells[record['cell_id']]['rsrp_values'].append(record['avg_rsrp'])

        interference_list = []
        for other_cell, data in interference_cells.items():
            avg_rsrp = sum(data['rsrp_values']) / len(data['rsrp_values']) if data['rsrp_values'] else None
            interference_list.append({
                'cell_id': other_cell,
                'overlap_count': data['overlap_count'],
                'avg_rsrp': avg_rsrp
            })

        interference_list.sort(key=lambda x: x['overlap_count'], reverse=True)
        avg_rsrq = sum(g['avg_rsrq'] for g in cell_grids) / len(cell_grids) if cell_grids else 0

        return {
            'problem_grid_count': len(cell_grids),
            'avg_rsrq': avg_rsrq,
            'interference_cells': interference_list
        }

    def _llm_analyze_neighbors(
        self,
        cell_id: str,
        interference_data: Dict,
        sinr_analysis: Optional[Dict] = None
    ) -> Dict:
        """LLM: 이웃셀 분석"""
        interference_list = interference_data.get('interference_cells', [])[:5]
        interference_info = "\n".join([
            f"- {c['cell_id']}: 중첩격자 {c['overlap_count']}개, 평균RSRP {c.get('avg_rsrp', 'N/A')}dBm"
            for c in interference_list
        ]) if interference_list else "- 간섭 셀 없음"

        sinr_info = ""
        if sinr_analysis:
            sinr_info = f"""
[SINR 분석 결과]
- 심각도: {sinr_analysis.get('severity', 'N/A')}
- 주요 문제: {sinr_analysis.get('primary_issue', 'N/A')}"""

        prompt = f"""다음 셀의 이웃셀 간섭 상황을 분석하세요.

[대상 셀]
- 셀 ID: {cell_id}
- 문제 격자 수: {interference_data.get('problem_grid_count', 0)}개
- 평균 RSRQ: {interference_data.get('avg_rsrq', 0):.1f} dB
{sinr_info}

[간섭 유발 셀 목록]
{interference_info}

다음 JSON 형식으로 응답하세요:
```json
{{
    "root_cause": "<NO_INTERFERENCE/MILD_INTERFERENCE/MODERATE_INTERFERENCE/SEVERE_INTERFERENCE>",
    "risk_level": "<low/medium/high>",
    "main_interferer": "<가장 영향 큰 간섭 셀 ID 또는 null>",
    "analysis": "<간섭 상황 상세 분석>",
    "recommendation": "<PROCEED/CAUTION/ABORT>"
}}
```"""

        result = llm_generate_json(self.llm_model, prompt, self.SYSTEM_PROMPT)
        if result:
            return result

        return {
            "root_cause": "MODERATE_INTERFERENCE" if interference_list else "NO_INTERFERENCE",
            "risk_level": "medium" if interference_list else "low",
            "main_interferer": interference_list[0]['cell_id'] if interference_list else None,
            "analysis": "LLM 응답 파싱 실패",
            "recommendation": "CAUTION" if interference_list else "PROCEED"
        }

    def analyze(
        self,
        sinr_result: SINRAnalysisResult,
        mdt_data: List[MDTRecord]
    ) -> List[NeighborAnalysisResult]:
        """Neighbor 분석 실행"""
        self._log(f"분석 시작 - 대상 셀: {len(sinr_result['serving_cells'])}개")

        results = []
        grid_to_cells = self._get_grid_cell_overlap(mdt_data)
        llm_target_analysis = sinr_result.get('llm_target_analysis', {})

        for i, cell in enumerate(sinr_result['serving_cells']):
            cell_id = cell['cell_id']
            cell_grids = [g for g in sinr_result['problem_grids'] if g['cell_id'] == cell_id]

            # 간섭 소스 식별
            interference_sources = self._find_interference_sources(cell_grids, grid_to_cells, cell_id)

            # LLM 분석
            interference_data = self._collect_interference_data(cell_id, cell_grids, grid_to_cells, mdt_data)
            sinr_analysis = llm_target_analysis.get(cell_id)

            self._log(f"LLM에게 셀 {cell_id} 분석 요청 중...")
            llm_analysis = self._llm_analyze_neighbors(cell_id, interference_data, sinr_analysis)

            root_cause = llm_analysis.get('root_cause', 'MODERATE_INTERFERENCE')
            recommendation = llm_analysis.get('recommendation', 'CAUTION')
            risk_mapping = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
            risk = risk_mapping.get(llm_analysis.get('risk_level', 'medium'), 0.5)

            if self.verbose and i < 3:
                self._log(f"셀 {cell_id}: {root_cause}, 위험도 {risk:.2f} → {recommendation}")

            results.append({
                'timestamp': datetime.now().isoformat(),
                'serving_cell_id': cell_id,
                'interference_sources': interference_sources,
                'root_cause': root_cause,
                'optimization_risk': risk,
                'recommendation': recommendation,
                'llm_analysis': llm_analysis,
            })

        return results

    def _get_grid_cell_overlap(self, mdt_data: List[MDTRecord]) -> Dict[str, List[str]]:
        """격자별 셀 목록"""
        grid_to_cells: Dict[str, List[str]] = {}
        for record in mdt_data:
            grid_id = record['grid_id']
            if grid_id not in grid_to_cells:
                grid_to_cells[grid_id] = []
            if record['cell_id'] not in grid_to_cells[grid_id]:
                grid_to_cells[grid_id].append(record['cell_id'])
        return grid_to_cells

    def _find_interference_sources(
        self,
        cell_grids: List[MDTRecord],
        grid_to_cells: Dict[str, List[str]],
        serving_cell_id: str
    ) -> List[str]:
        """간섭 유발 셀 식별"""
        interference_cells: Dict[str, int] = {}
        for grid in cell_grids:
            for other_cell in grid_to_cells.get(grid['grid_id'], []):
                if other_cell != serving_cell_id:
                    interference_cells[other_cell] = interference_cells.get(other_cell, 0) + 1

        sorted_cells = sorted(interference_cells.items(), key=lambda x: x[1], reverse=True)
        return [cell_id for cell_id, _ in sorted_cells[:5]]


# ============== Decision Agent ==============

class DecisionAgent:
    """
    최종 결정 Agent - LLM 기반

    - LLM이 SINR/Neighbor 분석 결과 + 유연한 가이드라인을 기반으로 판단
    - 최종 최적화 여부 및 액션 결정
    """

    SYSTEM_PROMPT = """당신은 RAN(Radio Access Network) 최적화 전문가입니다.
SINR 분석 결과와 이웃셀 간섭 분석 결과를 종합하여 최적화 여부를 판단합니다.

[역할]
- SINR 심각도와 간섭 위험도를 종합 평가
- 최적화 실행 여부 결정 (OPTIMIZE/HOLD/REJECT)
- 구체적인 액션 권고

[판단 가이드라인]
- RSRQ 심각도가 높고 간섭 위험이 낮으면 → OPTIMIZE 권고
- 간섭 셀이 많고 위험도가 높으면 → REJECT 또는 신중한 접근 필요
- 심각도와 위험도가 비슷하면 → HOLD로 추가 모니터링 권고
- Tilt 조정 시 인접 셀에 미치는 영향 고려
- 현재 Tilt가 이미 높으면(>10°) 추가 다운틸트 주의

[응답 규칙]
- 반드시 JSON 형식으로 응답
- 기술적 근거를 명확히 제시
- 한국어로 작성"""

    REPORT_SYSTEM_PROMPT = """당신은 RAN(Radio Access Network) 최적화 전문가입니다.
주어진 분석 결과를 바탕으로 명확하고 실용적인 보고서를 작성하세요.

[작성 지침]
- 기술적 용어는 간결하게 설명
- 구체적인 수치와 근거 포함
- 실행 가능한 권고사항 제시
- 한국어로 작성"""

    def __init__(
        self,
        positive_weight: float = 0.6,
        negative_weight: float = 0.4,
        verbose: bool = False,
        llm_model: str = "gemma3:27b"
    ):
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.verbose = verbose
        self.llm_model = llm_model

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DecisionAgent] {msg}")

    def _llm_make_decision(
        self,
        cell_id: str,
        sinr_analysis: Dict,
        neighbor_analysis: Dict,
        ru_info: Optional[Dict] = None
    ) -> Dict:
        """LLM: 최종 판단"""
        ru_info_str = ""
        if ru_info:
            total_tilt = ru_info.get('m_tilt', 0) + ru_info.get('e_tilt', 0)
            ru_info_str = f"""
[현재 RU 설정]
- 방위각: {ru_info.get('azimuth', 'N/A')}°
- 기계적 틸트: {ru_info.get('m_tilt', 'N/A')}°
- 전기적 틸트: {ru_info.get('e_tilt', 'N/A')}°
- 총 틸트: {total_tilt}°"""

        prompt = f"""다음 분석 결과를 종합하여 최적화 여부를 판단하세요.

[대상 셀]
- 셀 ID: {cell_id}
{ru_info_str}

[SINR 분석 결과]
- 심각도: {sinr_analysis.get('severity', 'N/A')}
- 주요 문제: {sinr_analysis.get('primary_issue', 'N/A')}
- 분석: {sinr_analysis.get('analysis', 'N/A')}

[이웃셀 분석 결과]
- 간섭 원인: {neighbor_analysis.get('root_cause', 'N/A')}
- 위험도: {neighbor_analysis.get('risk_level', 'N/A')}
- 주요 간섭 셀: {neighbor_analysis.get('main_interferer', '없음')}
- 분석: {neighbor_analysis.get('analysis', 'N/A')}
- 권고: {neighbor_analysis.get('recommendation', 'N/A')}

다음 JSON 형식으로 응답하세요:
```json
{{
    "decision": "<OPTIMIZE/HOLD/REJECT>",
    "confidence": <0.0~1.0 사이 신뢰도>,
    "reasoning": "<판단 근거 설명>",
    "actions": [
        {{
            "type": "<TILT_ADJUSTMENT/NEIGHBOR_COORDINATION/MONITORING>",
            "description": "<액션 설명>",
            "priority": <우선순위 1,2,3...>
        }}
    ]
}}
```"""

        result = llm_generate_json(self.llm_model, prompt, self.SYSTEM_PROMPT)
        if result:
            return result

        return {
            "decision": "HOLD",
            "confidence": 0.5,
            "reasoning": "LLM 응답 파싱 실패로 보수적 판단",
            "actions": [{"type": "MONITORING", "description": "추가 데이터 수집 후 재분석", "priority": 1}]
        }

    def _llm_generate_explanation(
        self,
        cell_id: str,
        decision: str,
        confidence: float,
        reasoning: str,
        actions: List[Dict],
        severity_score: float,
        interference_sources: List[str],
        ru_info: Dict = None
    ) -> str:
        """LLM: 상세 설명 생성"""
        tilt_info = ""
        if ru_info:
            total_tilt = ru_info.get('m_tilt', 0) + ru_info.get('e_tilt', 0)
            tilt_info = f"현재 Tilt: M={ru_info.get('m_tilt')}° + E={ru_info.get('e_tilt')}° = {total_tilt}°"

        action_details = []
        for action in actions:
            if action['type'] == 'TILT_ADJUSTMENT':
                action_details.append(f"Tilt 조정: {action.get('current_tilt', '?')}° → {action.get('recommended_tilt', '?')}°")
            elif action['type'] == 'NEIGHBOR_COORDINATION':
                action_details.append(f"인접 셀 검토: {action.get('cell_id')}")
            elif action['type'] == 'MONITORING':
                action_details.append(f"모니터링: {action.get('duration_hours', 24)}시간")

        prompt = f"""다음 기지국 최적화 분석 결과를 간결하게 설명해주세요 (3-4문장):

[분석 대상]
- 셀 ID: {cell_id}
- {tilt_info}

[분석 결과]
- 결정: {decision}
- 신뢰도: {confidence:.1%}
- 사유: {reasoning}
- RSRQ 심각도: {severity_score:.2f}
- 간섭 유발 셀: {', '.join(interference_sources) if interference_sources else '없음'}

[권고 액션]
{chr(10).join(f'- {a}' for a in action_details) if action_details else '- 없음'}

왜 이런 결정이 내려졌는지, 어떤 조치가 필요한지 설명하세요."""

        return llm_generate(self.llm_model, prompt, self.REPORT_SYSTEM_PROMPT)

    def _format_llm_actions(
        self,
        llm_actions: List[Dict],
        cell_id: str,
        ru_info: Dict = None,
        neighbor_result: Dict = None
    ) -> List[Dict]:
        """LLM 액션 형식 변환"""
        formatted = []

        for action in llm_actions:
            action_type = action.get('type', 'MONITORING')

            if action_type == 'TILT_ADJUSTMENT':
                current_tilt = ru_info['m_tilt'] + ru_info['e_tilt'] if ru_info else None
                formatted.append({
                    "type": "TILT_ADJUSTMENT",
                    "cell_id": cell_id,
                    "direction": "DOWN",
                    "value": 1,
                    "current_tilt": current_tilt,
                    "recommended_tilt": current_tilt + 1 if current_tilt else None,
                    "azimuth": ru_info['azimuth'] if ru_info else None,
                    "priority": action.get('priority', 1),
                    "description": action.get('description', '')
                })
            elif action_type == 'NEIGHBOR_COORDINATION':
                target = neighbor_result['interference_sources'][0] if neighbor_result and neighbor_result.get('interference_sources') else cell_id
                formatted.append({
                    "type": "NEIGHBOR_COORDINATION",
                    "cell_id": action.get('target_cell', target),
                    "action": "REVIEW_TILT",
                    "priority": action.get('priority', 2),
                    "description": action.get('description', '')
                })
            else:
                formatted.append({
                    "type": "MONITORING",
                    "cell_id": cell_id,
                    "duration_hours": 24,
                    "metrics": ["RSRP", "RSRQ"],
                    "priority": action.get('priority', 1),
                    "description": action.get('description', '')
                })

        if not formatted:
            formatted.append({
                "type": "MONITORING",
                "cell_id": cell_id,
                "duration_hours": 24,
                "metrics": ["RSRP", "RSRQ"],
                "priority": 1
            })

        return formatted

    def decide(
        self,
        sinr_result: SINRAnalysisResult,
        neighbor_results: List[NeighborAnalysisResult],
        ru_dict: Dict[str, RUInfo] = None
    ) -> List[OptimizationDecision]:
        """최종 결정"""
        self._log(f"결정 시작 - 대상 셀: {len(neighbor_results)}개")

        decisions = []
        ru_dict = ru_dict or {}
        llm_target_analysis = sinr_result.get('llm_target_analysis', {})

        for i, neighbor_result in enumerate(neighbor_results):
            cell_id = neighbor_result['serving_cell_id']
            ru_info = ru_dict.get(cell_id)

            # SINR/Neighbor 분석 결과
            sinr_analysis = llm_target_analysis.get(cell_id, {
                'severity': 'medium',
                'primary_issue': 'RSRQ 저하',
                'analysis': f"severity_score: {sinr_result['severity_score']:.2f}"
            })

            neighbor_analysis = neighbor_result.get('llm_analysis', {
                'root_cause': neighbor_result['root_cause'],
                'risk_level': 'medium' if neighbor_result['optimization_risk'] > 0.4 else 'low',
                'main_interferer': neighbor_result['interference_sources'][0] if neighbor_result['interference_sources'] else None,
                'analysis': f"optimization_risk: {neighbor_result['optimization_risk']:.2f}",
                'recommendation': neighbor_result['recommendation']
            })

            # LLM 판단
            self._log(f"LLM에게 셀 {cell_id} 최종 판단 요청 중...")
            llm_decision = self._llm_make_decision(cell_id, sinr_analysis, neighbor_analysis, ru_info)

            decision = llm_decision.get('decision', 'HOLD')
            confidence = llm_decision.get('confidence', 0.5)
            reasoning = llm_decision.get('reasoning', 'LLM 판단')
            actions = llm_decision.get('actions', [])

            formatted_actions = self._format_llm_actions(actions, cell_id, ru_info, neighbor_result)

            if self.verbose and i < 5:
                self._log(f"셀 {cell_id}: {decision} (신뢰도: {confidence:.2f})")

            # 설명 생성 (OPTIMIZE인 경우)
            llm_explanation = None
            if decision == "OPTIMIZE":
                llm_explanation = self._llm_generate_explanation(
                    cell_id, decision, confidence, reasoning, formatted_actions,
                    sinr_result['severity_score'], neighbor_result['interference_sources'], ru_info
                )

            decisions.append({
                'timestamp': datetime.now().isoformat(),
                'cell_id': cell_id,
                'decision': decision,
                'confidence': confidence,
                'reasoning': reasoning,
                'actions': formatted_actions,
                'llm_explanation': llm_explanation,
                'llm_decision': llm_decision,
            })

        if self.verbose:
            opt_cnt = sum(1 for d in decisions if d['decision'] == 'OPTIMIZE')
            hold_cnt = sum(1 for d in decisions if d['decision'] == 'HOLD')
            rej_cnt = sum(1 for d in decisions if d['decision'] == 'REJECT')
            self._log(f"완료 - OPTIMIZE: {opt_cnt}, HOLD: {hold_cnt}, REJECT: {rej_cnt}")

        return decisions

    def generate_summary_report(
        self,
        total_cells: int,
        optimize_count: int,
        hold_count: int,
        reject_count: int,
        top_optimize_cells: List[Dict],
        avg_severity: float,
        avg_interference: float
    ) -> str:
        """요약 보고서 생성"""
        cell_details = []
        for cell in top_optimize_cells[:5]:
            cell_details.append(f"- {cell['cell_id']}: 신뢰도 {cell['confidence']:.1%}, {cell['reasoning']}")

        prompt = f"""다음 기지국 최적화 분석 결과를 요약 보고서로 작성해주세요:

[분석 개요]
- 총 분석 기지국: {total_cells}개
- 최적화 필요 (OPTIMIZE): {optimize_count}개
- 모니터링 필요 (HOLD): {hold_count}개
- 최적화 불필요/위험 (REJECT): {reject_count}개

[주요 지표]
- 평균 RSRQ 심각도: {avg_severity:.2f}
- 평균 간섭 셀 수: {avg_interference:.1f}개

[최적화 우선순위 셀]
{chr(10).join(cell_details) if cell_details else '- 없음'}

위 결과를 바탕으로:
1. 전체적인 네트워크 상태 평가
2. 주요 문제점 요약
3. 권고 조치사항
을 포함한 요약 보고서를 작성하세요."""

        return llm_generate(self.llm_model, prompt, self.REPORT_SYSTEM_PROMPT)
