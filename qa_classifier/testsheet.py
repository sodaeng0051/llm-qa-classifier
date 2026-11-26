"""테스트 결과 시트(테스트케이스 × 시험자별 pass/fail) 읽기.

예) 한 시트 안에 표가 여러 개 있을 수 있다 (일상대화 / 일반QA …).

    📌 제목
    (빈 줄)
                                         TT261103051                TT261103050      ← 시험자(SAID)
    일상대화                              4                          3                ← 건수 등 (무시)
    순번 | 카테고리 | 사용자 의도 | 발화 | 시험방법 | 기대결과 | 실행결과 | 비고1 | 재현일시 | | 실행결과 | 비고2 | 재현일시
    1    | 일상대화 | Chit Chat  | …    | …       | …       | pass     |       | 15:03    | | fail     | …    | 14:46

시험자별 실행결과 하나하나를 한 행으로 펼친다:
    pass → 정상,  fail → 이슈 (비고 = 문제점, 시험자 ID = SAID)
"""
from __future__ import annotations

import re

import pandas as pd

RESULT_VALUES = {"pass", "fail"}
RESULT_HEADERS = ("실행결과", "결과", "판정", "result", "pass/fail", "테스트결과", "테스트 결과")
NOTE_HEADERS = ("비고", "문제점", "이슈", "현상", "코멘트", "comment", "note", "메모")
TIME_HEADERS = ("재현일시", "일시", "시간", "date", "time")
UTT_HEADERS = ("발화", "사용자 발화", "발화문", "질의", "utterance")
FIELD_HEADERS = {"순번": ("순번", "no", "번호", "tc", "tc id"), "카테고리": ("카테고리", "분류", "category"),
                 "사용자 의도": ("사용자 의도", "의도", "intent"), "시험방법": ("시험방법", "테스트 방법", "방법", "경로"),
                 "기대결과": ("기대결과", "기대 결과", "기대", "expected")}


def _norm(v) -> str:
    return str(v or "").strip().lower()


def _is_result(v) -> bool:
    return _norm(v) in RESULT_VALUES


def _header_rows(raw: pd.DataFrame) -> list[int]:
    """'발화' 컬럼과 '실행결과' 컬럼이 같이 있는 행 = 표 헤더."""
    rows = []
    for i in range(len(raw)):
        cells = [_norm(v) for v in raw.iloc[i]]
        if any(c in UTT_HEADERS for c in cells) and any(c in RESULT_HEADERS for c in cells):
            rows.append(i)
    return rows


def is_test_sheet(raw: pd.DataFrame) -> bool:
    raw = raw.fillna("").astype(str)
    return bool(_header_rows(raw)) and int(raw.map(_is_result).to_numpy().sum()) >= 3


def _find(header: list[str], names, start: int = 0, stop: int | None = None) -> int | None:
    stop = len(header) if stop is None else stop
    for j in range(start, stop):
        if header[j] in names or any(header[j].startswith(n) for n in names):
            return j
    return None


def parse_test_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """테스트 결과 시트 → 실행 1건당 1행. 컬럼: 구분, 순번, 카테고리, 사용자 의도, 발화, 시험방법, 기대결과,
    판정(pass/fail), 문제점, SAID, 재현일시"""
    raw = raw.fillna("").astype(str)
    heads = _header_rows(raw)
    out = []
    for k, h in enumerate(heads):
        header = [_norm(v) for v in raw.iloc[h]]
        end = heads[k + 1] if k + 1 < len(heads) else len(raw)
        body = raw.iloc[h + 1:end]
        utt_j = _find(header, UTT_HEADERS)
        fields = {name: _find(header, alias) for name, alias in FIELD_HEADERS.items()}
        # 시험자 블록: 헤더가 '실행결과'이고 실제 값이 pass/fail인 컬럼
        result_js = [j for j, c in enumerate(header) if c in RESULT_HEADERS and body.iloc[:, j].map(_is_result).any()]
        # 표 위쪽(제목·건수 줄)에서 구분 이름과 시험자 ID 찾기
        top = h
        while top > 0 and h - top < 5 and not raw.iloc[top - 1].map(_is_result).any():
            top -= 1          # 바로 위 제목·시험자·건수 줄까지만 (앞 표의 데이터 행은 제외)
        above = raw.iloc[top:h]
        section = next((r.iloc[0].strip() for _, r in above.iloc[::-1].iterrows()
                        if r.iloc[0].strip() and not r.iloc[0].strip().startswith("📌")), "")
        for n, rj in enumerate(result_js):
            nxt = result_js[n + 1] if n + 1 < len(result_js) else len(header)
            note_j = _find(header, NOTE_HEADERS, rj + 1, nxt)
            time_j = _find(header, TIME_HEADERS, rj + 1, nxt)
            tester = next((v.strip() for v in above.iloc[::-1][rj] if v.strip() and not re.fullmatch(r"\d+", v.strip())),
                          f"시험자{n + 1}")
            for _, r in body.iterrows():
                verdict = _norm(r.iloc[rj])
                if verdict not in RESULT_VALUES or not r.iloc[utt_j].strip():
                    continue   # 건수 줄·빈 줄·미실행(빈칸)은 제외
                row = {"구분": section}
                for name, j in fields.items():
                    row[name] = r.iloc[j].strip() if j is not None else ""
                row.update({"발화": r.iloc[utt_j].strip(), "판정": verdict,
                            "문제점": r.iloc[note_j].strip() if note_j is not None else "",
                            "SAID": tester, "재현일시": r.iloc[time_j].strip() if time_j is not None else ""})
                out.append(row)
    cols = ["구분", "순번", "카테고리", "사용자 의도", "발화", "시험방법", "기대결과", "판정", "문제점", "SAID", "재현일시"]
    return pd.DataFrame(out, columns=cols)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """시험자별 pass / fail 건수."""
    t = pd.crosstab(df["SAID"], df["판정"]).reindex(columns=["pass", "fail"], fill_value=0)
    t["합계"] = t.sum(axis=1)
    t["fail 비율"] = (t["fail"] / t["합계"]).round(3)
    return t.reset_index()
