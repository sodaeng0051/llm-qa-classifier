"""① 입력 → ② 분석 → ③ 분류 전체 파이프라인."""
from __future__ import annotations

import re

import pandas as pd

from . import model as local_model
from .analyzer import INTENT_LABELS, RESPONSE_LABELS, analyze
from .classifier import analysis_note, apply_grade_override, judge_with_desc

# 앞쪽 별칭일수록 우선. 값이 전부 비어 있는 컬럼은 건너뛴다.
COLUMN_ALIASES = {
    "utterance": ["사용자 발화", "사용자발화", "발화", "발화문", "질의", "사용자 입력", "테스트 발화", "방법",
                  "utterance", "query"],
    "response": ["실제 응답", "실제응답", "응답", "답변", "시스템 응답", "결과", "기대", "response", "answer"],
    "issue": ["문제점", "이슈 내용", "이슈내용", "현상", "문제", "오류 내용", "issue", "problem"],
    "stt": ["stt", "stt 결과", "인식 결과", "인식결과", "asr", "음성인식 결과"],
}
HEADER_HINTS = {a.lower() for v in COLUMN_ALIASES.values() for a in v} | {"순번", "no", "번호", "등급", "시험자"}
_UTT_WRAP = re.compile(r"""^\s*["“”'‘’]\s*(.+?)\s*["“”'‘’]\s*(발화|말하기|입력|질문)?\s*$""", re.S)


def clean_utterance(text: str) -> str:
    """'"현재 프로야구 팀 순위 알려줘" 발화' → '현재 프로야구 팀 순위 알려줘'"""
    m = _UTT_WRAP.match(str(text or ""))
    return m.group(1).strip() if m else str(text or "").strip()


def _find_header(raw: pd.DataFrame) -> int:
    """제목줄·빈 줄이 위에 있는 시트에서 실제 헤더 행을 찾는다."""
    for i in range(min(20, len(raw))):
        cells = {str(v).strip().lower() for v in raw.iloc[i] if str(v).strip()}
        if len(cells & HEADER_HINTS) >= 2:
            return i
    return 0


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.fillna("").astype(str)
    h = _find_header(raw)
    header = [str(c).strip() or f"col{j + 1}" for j, c in enumerate(raw.iloc[h])]
    seen = {}
    for j, c in enumerate(header):  # 중복 헤더 이름 정리
        seen[c] = seen.get(c, 0) + 1
        if seen[c] > 1:
            header[j] = f"{c}_{seen[c]}"
    df = raw.iloc[h + 1:].copy()
    df.columns = header
    named = [not re.fullmatch(r"col\d+(_\d+)?", c) for c in header]  # 원본에 헤더 이름이 있던 컬럼
    keep = [(df[c] != "").any() or named[j] for j, c in enumerate(df.columns)]
    df = df.loc[:, keep]                              # 헤더도 값도 없는 컬럼만 제거 (빈 '등급' 칸 등은 유지)
    df = df[(df != "").any(axis=1)]                  # 완전히 빈 행 제거
    return df.reset_index(drop=True)


def read_sheet(file, sheet: str | int | None = 0) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".csv", ".txt")):
        for enc in ("utf-8-sig", "cp949"):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                return _tidy(pd.read_csv(file, encoding=enc, dtype=str, header=None,
                                         keep_default_na=False, skip_blank_lines=False))
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 인코딩을 인식하지 못했습니다 (UTF-8 / CP949 지원).")
    return _tidy(pd.read_excel(file, sheet_name=sheet or 0, dtype=str, header=None))


def detect_columns(df: pd.DataFrame) -> dict:
    found = {}
    filled = {c for c in df.columns if (df[c].astype(str).str.strip() != "").any()}
    norm = {c: str(c).strip().lower().replace("_", " ") for c in df.columns if c in filled}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:  # 정확히 일치 우선, 그다음 포함
            a = alias.lower()
            hit = next((c for c, n in norm.items() if n == a and c not in found.values()), None) or \
                next((c for c, n in norm.items() if len(a) > 1 and a in n and c not in found.values()), None)
            if hit is not None:
                found[key] = hit
                break
    if "utterance" not in found:  # 값이 '"..." 발화' 형식인 컬럼
        for c in filled:
            vals = df[c][df[c].str.strip() != ""]
            if len(vals) and vals.str.contains(r"발화\s*$").mean() > 0.5:
                found["utterance"] = c
                break
    return found


RULE_STRONG = ("음성인식", "음성합성")   # 규칙 정밀도가 높아 모델 결과에 더해 주는 관련시스템


def _merge(rule_row: pd.Series, pr: dict, desc: str, rules: dict) -> dict:
    """규칙 판정(세부 이슈·근거 문구)과 학습 모델 판정(관련시스템·등급·유사 이슈)을 합친다."""
    # 학습 데이터는 이슈리스트(문제 있는 건)만 있어 '정상'을 모른다 → 문제점 기술이 없고 규칙상 정상이면 그대로 둔다
    if not str(desc or "").strip() and rule_row["관련시스템"] == "정상":
        return {"관련시스템": "정상", "등급": "", "세부 이슈": "", "판단 근거": rule_row["판단 근거"],
                "신뢰도": rule_row["신뢰도"], "_review": False}
    systems = local_model.norm_systems(pr["systems"])
    rule_hit = str(rule_row["적용 규칙"]).startswith("D") and rule_row["적용 규칙"] != "D90"
    if rule_hit and rule_row["관련시스템"] in RULE_STRONG and rule_row["관련시스템"] not in systems:
        systems.append(rule_row["관련시스템"])
    grade = pr["grade"] or rule_row["등급"]
    grade, note = apply_grade_override(desc or "", grade, rules)

    parts = []
    if rule_hit:
        parts.append(str(rule_row["판단 근거"]).split(" · 등급 보정")[0])
    nb = pr["neighbors"][0] if pr["neighbors"] else None
    if nb:
        snippet = re.sub(r"\s+", " ", nb["문제점"]).strip()[:45]
        parts.append(f"유사 과거 이슈 [{nb['출처']} #{nb['순번']}] '{snippet}…' → "
                     f"{nb['관련시스템'] or '-'} / {nb['등급'] or '-'} (유사도 {nb['sim']:.0%})")
    if note:
        parts.append(f"등급 보정({note})")

    sys_str = local_model.join_systems(systems) or rule_row["관련시스템"]
    # 아주 비슷한 과거 이슈와 판정이 다르면 검토 (과거 판정 자체가 회차마다 다를 수 있음)
    disagree = bool(nb and nb["sim"] >= 0.6 and (
        (nb["관련시스템"] and nb["관련시스템"] != sys_str) or (nb["등급"] and nb["등급"] != grade)))
    novel = bool(nb is None or nb["sim"] < rules["params"].get("novel_similarity", 0.3))
    if disagree:
        parts.append("유사 과거 이슈와 판정이 달라 확인 필요")
    if novel:
        parts.append("비슷한 과거 이슈가 없어 확인 필요 (새 유형일 수 있음)")
    return {"관련시스템": sys_str, "등급": grade, "세부 이슈": rule_row["세부 이슈"] if rule_hit else "",
            "판단 근거": " · ".join(parts) or rule_row["판단 근거"], "신뢰도": pr["confidence"],
            "_review": disagree or novel}


def run(df: pd.DataFrame, utt_col: str, resp_col: str, rules: dict,
        stt_col: str | None = None, issue_col: str | None = None, use_model: bool = True,
        model=None) -> pd.DataFrame:
    """원본 컬럼은 그대로 두고 분석·분류 결과 컬럼을 덧붙인 DataFrame을 반환."""
    p = rules["params"]
    df = df[(df[utt_col].astype(str).str.strip() != "")
            | ((df[issue_col].astype(str).str.strip() != "") if issue_col else False)]  # 빈 행(순번만 있는 행) 제외
    utts = df[utt_col].map(clean_utterance)
    descs = df[issue_col].astype(str) if issue_col else pd.Series("", index=df.index)
    rows = []
    for idx, r in df.iterrows():
        a = analyze(utts[idx], r[resp_col], rules, r[stt_col] if stt_col else None)
        v = judge_with_desc(a, descs[idx], rules, utts[idx], str(r[resp_col]))
        rows.append({"발화 의도": INTENT_LABELS[a.intent], "응답 유형": RESPONSE_LABELS[a.response_type],
                     "연관도": a.overlap, "분석 내용": analysis_note(a), "관련시스템": v.issue_type,
                     "세부 이슈": v.sub, "등급": v.grade, "보조 이슈": ", ".join(v.secondary),
                     "판단 근거": v.reason, "신뢰도": v.confidence, "적용 규칙": v.rule_id})
    res = pd.DataFrame(rows, index=df.index)

    review = pd.Series(False, index=res.index)
    model = model if model is not None else (local_model.load() if use_model else None)
    if model is not None and len(df):
        preds = local_model.predict(model, utts, df[resp_col].astype(str), descs)
        merged = [_merge(res.loc[idx], pr, descs[idx], rules) for idx, pr in zip(res.index, preds)]
        m = pd.DataFrame(merged, index=res.index)
        res["규칙 판정"] = res["관련시스템"].astype(str) + " / " + res["등급"].astype(str)
        for c in m.columns:
            res[c] = m[c]
        review = m["_review"]
        res = res.drop(columns="_review")

    need = ((res["신뢰도"] < p["review_threshold"])
            | res["관련시스템"].map(lambda v: any(t in str(v).split(", ") for t in p.get("always_review_types", [])))
            | res["등급"].isin(p.get("always_review_grades", []))
            | (res["관련시스템"] == "미분류")
            | review)
    if model is None:  # 규칙만 쓸 때: 서로 다른 관련시스템 이슈가 섞이면 검토
        need |= pd.Series([p.get("review_cross_system", True) and any(
            lbl.split("·")[0] != sys for lbl in str(sec).split(", ") if lbl)
            for sys, sec in zip(res["관련시스템"], res["보조 이슈"])], index=res.index)
    res["검토 필요"] = need.map({True: "Y", False: "N"})

    # 이미 결과 컬럼이 있는 시트(이전 리포트)를 다시 넣어도 덮어쓰도록
    out = df.copy()
    # 시트에 QA가 직접 입력한 등급/관련시스템이 있으면 'XX(QA)'로 보존해 자동 판정과 비교할 수 있게 한다
    for col in ("관련시스템", "등급"):
        if col in out.columns:
            if (out[col].astype(str).str.strip() != "").any():
                out = out.rename(columns={col: f"{col}(QA)"})
            else:
                out = out.drop(columns=col)
    out = out.drop(columns=[c for c in res.columns if c in out.columns]).join(res)
    out[utt_col] = utts
    front = [(utt_col, "사용자 발화"), (resp_col, "실제 응답")] + ([(issue_col, "문제점")] if issue_col else [])
    for pos, (src, name) in enumerate(front):
        col = out.pop(src)
        if name in out.columns:  # 같은 이름의 다른 원본 컬럼이 있으면 보존
            out = out.rename(columns={name: f"{name}(원본)"})
        out.insert(pos, name, col)
    return out
