"""CLI: python -m qa_classifier 입력.xlsx [-o 리포트.xlsx] [--sheet 시트명]

UI 없이 시트를 넣으면 분석·분류 후 QA 리포트(Excel)를 생성한다.
"""
import argparse
import sys
from pathlib import Path

from . import report
from .pipeline import detect_columns, read_sheet, run
from .rules import load_rules


def main():
    ap = argparse.ArgumentParser(description="LLM QA 이슈 자동 분류 (오프라인)")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--utterance-col")
    ap.add_argument("--response-col")
    ap.add_argument("--no-model", action="store_true", help="로컬 학습 모델 사용 안 함")
    args = ap.parse_args()

    df = read_sheet(args.input, args.sheet)
    cols = detect_columns(df)
    utt = args.utterance_col or cols.get("utterance")
    resp = args.response_col or cols.get("response")
    if not utt or not resp:
        sys.exit(f"발화/응답 컬럼을 찾지 못했습니다. 컬럼: {list(df.columns)}  → --utterance-col/--response-col 지정")
    result = run(df, utt, resp, load_rules(), cols.get("stt"), cols.get("issue"), use_model=not args.no_model)
    print(f"컬럼 인식: 발화={utt}, 응답={resp}, 문제점={cols.get('issue')}, STT={cols.get('stt')}")
    out = Path(args.output or Path(args.input).with_name(Path(args.input).stem + "_QA리포트.xlsx"))
    out.write_bytes(report.to_excel(result))
    s = report.summary(result)
    print(f"전체 {s['total']}건 | 이슈 {s['issues']}건 ({s['issue_rate']:.0%}) | 검토 필요 {s['review']}건")
    print("등급:", s["by_grade"])
    for t, n in s["by_type"].items():
        print(f"  {t}: {n}")
    print(f"리포트: {out}")


if __name__ == "__main__":
    main()
