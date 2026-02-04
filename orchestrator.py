"""파이프라인 오케스트레이터"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path

from models import OptimizationDecision, RUInfo
from data import load_mdt_data, load_ru_data, get_ru_by_cell_id
from agents import SINRAgent, NeighborAgent, DecisionAgent


class OptimizationPipeline:
    """
    기지국 간섭 최적화 파이프라인 (LLM 기반)

    Pipeline: SINR Agent (LLM) → Neighbor Agent (LLM) → Decision Agent (LLM)

    모든 Agent가 LLM을 활용하여 분석/판단 수행
    """

    def __init__(
        self,
        rsrp_threshold: float = -100,
        rsrq_threshold: float = -15,
        sinr_threshold: float = 10.0,           # 셀 평균 SINR 임계값
        problem_ratio_threshold: float = 0.3,   # 문제 격자 비율 임계값
        interference_threshold: float = 0.3,
        positive_weight: float = 0.6,
        negative_weight: float = 0.4,
        verbose: bool = False,
        llm_model: str = "exaone3.5:7.8b"
    ):
        self.verbose = verbose
        self.llm_model = llm_model

        # 모든 Agent에 LLM 설정 전달
        self.sinr_agent = SINRAgent(
            rsrp_threshold, rsrq_threshold,
            sinr_threshold, problem_ratio_threshold,
            verbose=verbose, llm_model=llm_model
        )
        self.neighbor_agent = NeighborAgent(
            interference_threshold,
            verbose, llm_model
        )
        self.decision_agent = DecisionAgent(
            positive_weight, negative_weight,
            verbose, llm_model
        )

        self.ru_dict: Dict[str, RUInfo] = {}

    def load_ru_info(self, ru_filepath: str | Path):
        """RU 정보 로드"""
        ru_data = load_ru_data(ru_filepath)
        self.ru_dict = get_ru_by_cell_id(ru_data)
        if self.verbose:
            print(f"RU 정보 로드: {len(self.ru_dict):,}개")

    def run(
        self,
        mdt_filepath: str | Path,
        ru_filepath: Optional[str | Path] = None,
        verbose: bool = None
    ) -> Tuple[List[OptimizationDecision], str]:
        """파이프라인 실행

        Returns:
            Tuple[List[OptimizationDecision], str]: (결정 목록, LLM 요약 보고서)
        """
        show_progress = verbose if verbose is not None else self.verbose

        if show_progress:
            print("=" * 60)
            print("기지국 간섭 최적화 파이프라인 시작 (LLM 기반)")
            print(f"LLM 모델: {self.llm_model}")
            print("=" * 60)

        # 0. RU 정보 로드 (옵션)
        if ru_filepath:
            if show_progress:
                print(f"\n[0/5] RU 정보 로드: {ru_filepath}")
            self.load_ru_info(ru_filepath)
            if show_progress:
                print(f"      총 {len(self.ru_dict):,}개 RU 정보 로드")

        # 1. MDT 데이터 로드
        if show_progress:
            print(f"\n[1/5] MDT 데이터 로드: {mdt_filepath}")
        mdt_data = load_mdt_data(mdt_filepath)
        if show_progress:
            print(f"      총 {len(mdt_data):,}개 레코드 로드")

        # 2. SINR 분석 (Positive Agent + LLM)
        if show_progress:
            print("\n[2/5] SINR 분석 (Positive Agent + LLM)...")
        sinr_result = self.sinr_agent.analyze(mdt_data)
        if show_progress:
            print(f"      문제 격자: {len(sinr_result['problem_grids']):,}개")
            print(f"      관련 기지국: {len(sinr_result['serving_cells'])}개")
            print(f"      심각도: {sinr_result['severity_score']:.2f}")
            print(f"      권고: {sinr_result['recommendation']}")

        if not sinr_result['problem_grids']:
            if show_progress:
                print("\n문제 격자 없음. 파이프라인 종료.")
            return [], None

        # 3. Neighbor 분석 (Negative Agent + LLM)
        if show_progress:
            print("\n[3/5] Neighbor 분석 (Negative Agent + LLM)...")
        neighbor_results = self.neighbor_agent.analyze(sinr_result, mdt_data)

        # 4. 최종 결정 (Decision Agent + LLM) - RU 정보 전달
        if show_progress:
            print("\n[4/5] 최종 결정 (Decision Agent + LLM)...")
        decisions = self.decision_agent.decide(sinr_result, neighbor_results, self.ru_dict)

        # 5. LLM 요약 보고서 생성
        if show_progress:
            print("\n[5/5] LLM 요약 보고서 생성...")

        optimize_decisions = [d for d in decisions if d['decision'] == 'OPTIMIZE']
        hold_decisions = [d for d in decisions if d['decision'] == 'HOLD']
        reject_decisions = [d for d in decisions if d['decision'] == 'REJECT']

        # 평균 심각도 및 간섭 계산
        avg_severity = sinr_result['severity_score']
        avg_interference = sum(
            len(nr['interference_sources']) for nr in neighbor_results
        ) / len(neighbor_results) if neighbor_results else 0

        llm_summary = self.decision_agent.generate_summary_report(
            total_cells=len(decisions),
            optimize_count=len(optimize_decisions),
            hold_count=len(hold_decisions),
            reject_count=len(reject_decisions),
            top_optimize_cells=sorted(
                optimize_decisions,
                key=lambda x: x['confidence'],
                reverse=True
            )[:5],
            avg_severity=avg_severity,
            avg_interference=avg_interference
        )

        if show_progress:
            self._print_summary(decisions, llm_summary)

        return decisions, llm_summary

    def _print_summary(self, decisions: List[OptimizationDecision], llm_summary: str = None):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("최적화 결과 요약")
        print("=" * 60)

        optimize_count = sum(1 for d in decisions if d['decision'] == "OPTIMIZE")
        hold_count = sum(1 for d in decisions if d['decision'] == "HOLD")
        reject_count = sum(1 for d in decisions if d['decision'] == "REJECT")

        print(f"분석 기지국: {len(decisions)}개")
        print(f"  - OPTIMIZE: {optimize_count}")
        print(f"  - HOLD: {hold_count}")
        print(f"  - REJECT: {reject_count}")

        # OPTIMIZE 셀 상세 출력
        if optimize_count > 0:
            print("\n최적화 대상 셀:")
            for d in decisions:
                if d['decision'] == "OPTIMIZE":
                    print(f"  {d['cell_id']}: {d['reasoning']}")
                    for action in d['actions']:
                        if action['type'] == 'TILT_ADJUSTMENT':
                            current = action.get('current_tilt', '?')
                            recommended = action.get('recommended_tilt', '?')
                            print(f"    → Tilt: {current}° → {recommended}°")

                    # LLM 설명 출력
                    if d.get('llm_explanation'):
                        print(f"    📝 AI 분석: {d['llm_explanation'][:200]}...")

        # LLM 요약 보고서 출력
        if llm_summary:
            print("\n" + "=" * 60)
            print("📋 AI 요약 보고서")
            print("=" * 60)
            print(llm_summary)
