"""
기지국 최적화 파이프라인 (LLM 기반)
# 실행
python main.py
"""

from pathlib import Path
from orchestrator import OptimizationPipeline


# 데이터 경로
DATA_DIR = Path("/Users/youngjun/Documents/AIProject/data/Tilt")
MDT_FILE = DATA_DIR / "MDT_서부엔지/20260202_봉담읍_GPOT.txt"
RU_FILE = DATA_DIR / "서부광본ru_info.txt"

def main():
    """메인 실행 (LLM 기반)"""

    # 파이프라인 생성
    pipeline = OptimizationPipeline(
        rsrp_threshold=-100,          # RSRP 임계값 (dBm) - 이 값 이상이면 신호 강도 양호
        rsrq_threshold=-15,           # RSRQ 임계값 (dB) - 이 값 이하면 신호 품질 불량
        sinr_threshold=10.0,          # 셀 평균 SINR 임계값 (이 값 이하면 문제 셀 후보)
        problem_ratio_threshold=0.3,  # 문제 격자 비율 임계값 (30% 이상이면 문제 셀)
        verbose=True,                 # Agent 상세 로그 활성화
        llm_model="gemma3:27b"        # LLM 모델
    )

    # 실행 (RU 정보 포함)
    decisions, llm_summary = pipeline.run(MDT_FILE, ru_filepath=RU_FILE, verbose=True)

    # 최적화 대상만 출력
    if decisions:
        optimize_targets = [d for d in decisions if d['decision'] == "OPTIMIZE"]

        if optimize_targets:
            print("\n" + "=" * 60)
            print(f"최적화 대상 기지국: {len(optimize_targets)}개")
            print("=" * 60)

            for d in optimize_targets:
                print(f"\n셀: {d['cell_id']}")
                print(f"  신뢰도: {d['confidence']:.2f}")
                print(f"  사유: {d['reasoning']}")
                print("  액션:")
                for action in d['actions']:
                    print(f"    - {action['type']}: {action.get('cell_id', '')} "
                          f"(priority: {action.get('priority', '-')})")

                # LLM 설명 출력
                if d.get('llm_explanation'):
                    print(f"\n  📝 AI 상세 분석:")
                    print(f"  {d['llm_explanation']}")


if __name__ == "__main__":
    main()
