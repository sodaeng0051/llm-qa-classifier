"""① 입력 → ② 분석 → ③ 분류 전체 파이프라인."""
from __future__ import annotations

import re

import pandas as pd

from . import model as local_model
from .analyzer import INTENT_LABELS, RESPONSE_LABELS, analyze
from .classifier import analysis_note, apply_grade_override, judge_with_desc, match_desc
from .testsheet import is_test_sheet, parse_test_sheet

# 앞쪽 별칭일수록 우선. 값이 전부 비어 있는 컬럼은 건너뛴다.
COLUMN_ALIASES = {
    "utterance": ["사용자 발화", "사용자발화", "발화", "발화문", "질의", "사용자 입력", "테스트 발화", "방법",
                  "utterance", "query"],
    "response": ["실제 응답", "실제응답", "응답", "답변", "시스템 응답", "결과", "기대", "response", "answer"],
    "issue": ["문제점", "이슈설명", "이슈 설명", "이슈 내용", "이슈내용", "이슈 상세", "문제 설명", "오류 설명", "현상 설명",
              "현상", "문제", "오류 내용", "issue description", "issue", "problem"],
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


def _load(raw: pd.DataFrame) -> pd.DataFrame:
    """테스트 결과 시트(시험자별 pass/fail)면 실행 1건당 1행으로 펼치고, 아니면 헤더를 찾아 정리한다."""
    raw = raw.fillna("").astype(str)
    return parse_test_sheet(raw) if is_test_sheet(raw) else _tidy(raw)


def is_issue_list(df: pd.DataFrame, cols: dict) -> bool:
    """이슈리스트(모든 행이 이슈)인지: 문제점 컬럼이나 값이 있는 '등급' 컬럼이 있으면 이슈리스트로 본다.
    테스트 결과 시트(pass/fail)는 pass 행이 정상이므로 제외."""
    if is_test_result(df):
        return False
    has_grade = any(str(c).strip() == "등급" and (df[c].astype(str).str.strip() != "").any() for c in df.columns)
    return bool(cols.get("issue")) or has_grade


def is_test_result(df: pd.DataFrame) -> bool:
    """read_sheet 결과가 테스트 결과 시트(판정 컬럼 = pass/fail)를 펼친 것인지."""
    return "판정" in df.columns and "SAID" in df.columns


def read_sheet(file, sheet: str | int | None = 0) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".csv", ".txt")):
        for enc in ("utf-8-sig", "cp949"):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                return _load(pd.read_csv(file, encoding=enc, dtype=str, header=None,
                                         keep_default_na=False, skip_blank_lines=False))
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 인코딩을 인식하지 못했습니다 (UTF-8 / CP949 지원).")
    return _load(pd.read_excel(file, sheet_name=sheet or 0, dtype=str, header=None))


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
    if is_test_result(df):
        found["result"] = "판정"
    if "utterance" not in found:  # 값이 '"..." 발화' 형식인 컬럼
        for c in filled:
            vals = df[c][df[c].str.strip() != ""]
            if len(vals) and vals.str.contains(r"발화\s*$").mean() > 0.5:
                found["utterance"] = c
                break
    return found


RULE_STRONG = ("음성인식", "음성합성")   # 규칙 정밀도가 높아 모델 결과에 더해 주는 관련시스템


def _merge(rule_row: pd.Series, pr: dict, desc: str, rules: dict, issue_list: bool = False) -> dict:
    """규칙 판정(세부 이슈·근거 문구)과 학습 모델 판정(관련시스템·등급·유사 이슈)을 합친다."""
    # 학습 데이터는 이슈리스트(문제 있는 건)만 있어 '정상'을 모른다 → 문제점 기술이 없고 규칙상 정상이면 그대로 둔다
    if not issue_list and not str(desc or "").strip() and rule_row["관련시스템"] == "정상":
        return {"관련시스템": "정상", "등급": "", "세부 이슈": "", "판단 근거": rule_row["판단 근거"],
                "신뢰도": rule_row["신뢰도"], "_review": False}
    systems = local_model.norm_systems(pr["systems"])
    rule_hit = str(rule_row["적용 규칙"]).startswith("D") and rule_row["적용 규칙"] != "D90"
    if rule_hit and rule_row["관련시스템"] in RULE_STRONG and rule_row["관련시스템"] not in systems:
        systems.append(rule_row["관련시스템"])
    grade = pr["grade"] or rule_row["등급"]
    grade, note = apply_grade_override(desc or "", grade, rules)

    # 세부 이슈·근거 문구는 최종 관련시스템과 맞는 규칙에서만 가져온다 (설명이 판정과 어긋나지 않도록)
    sub, rule_reason, rule_note = "", "", ""
    if rule_hit:
        hits = match_desc(desc or "", rules)
        same = [h for h in hits if h[1]["type"] in systems]
        if same:
            _, r0, ph = same[0]
            sub, rule_reason = r0.get("sub", ""), f"{r0['reason']} [근거: '{ph}']"
        else:
            rule_note = f"키워드 규칙은 '{rule_row['관련시스템']}'로 보았으나 과거 판정 기준으로 다르게 판단"
    parts = [rule_reason] if rule_reason else []
    nb = pr["neighbors"][0] if pr["neighbors"] else None
    if nb:
        snippet = re.sub(r"\s+", " ", nb["문제점"]).strip()[:45]
        parts.append(f"유사 과거 이슈 [{nb['출처']} #{nb['순번']}] '{snippet}…' → "
                     f"{nb['관련시스템'] or '-'} / {nb['등급'] or '-'} (유사도 {nb['sim']:.0%})")
    if note:
        parts.append(f"등급 보정({note})")
    if rule_note:
        parts.append(rule_note)

    sys_str = local_model.join_systems(systems) or rule_row["관련시스템"]
    # 아주 비슷한 과거 이슈와 판정이 다르면 검토 (과거 판정 자체가 회차마다 다를 수 있음)
    disagree = bool(nb and nb["sim"] >= 0.6 and (
        (nb["관련시스템"] and nb["관련시스템"] != sys_str) or (nb["등급"] and nb["등급"] != grade)))
    novel = bool(nb is None or nb["sim"] < rules["params"].get("novel_similarity", 0.3))
    if disagree:
        parts.append("유사 과거 이슈와 판정이 달라 확인 필요")
    if novel:
        parts.append("비슷한 과거 이슈가 없어 확인 필요 (새 유형일 수 있음)")
    return {"관련시스템": sys_str, "등급": grade, "세부 이슈": sub,
            "판단 근거": " · ".join(parts) or rule_row["판단 근거"], "신뢰도": pr["confidence"],
            "_review": disagree or novel}


def run(df: pd.DataFrame, utt_col: str, resp_col: str, rules: dict,
        stt_col: str | None = None, issue_col: str | None = None, use_model: bool = True,
        model=None, result_col: str | None = None, issue_list: bool = False) -> pd.DataFrame:
    """원본 컬럼은 그대로 두고 분석·분류 결과 컬럼을 덧붙인 DataFrame을 반환.

    result_col(pass/fail)이 있으면 pass 행은 '정상'으로 두고 fail 행만 분류한다.
    issue_list=True(이슈리스트)면 모든 행이 이슈이므로 '정상'으로 판정하지 않는다.
    """
    if result_col:
        passed = df[result_col].astype(str).str.strip().str.lower().eq("pass")
        parts = []
        if (~passed).any():
            parts.append(run(df[~passed], utt_col, resp_col, rules, stt_col, issue_col, use_model, model))
        if passed.any():
            parts.append(_pass_rows(df[passed], utt_col, resp_col, issue_col))
        if not parts:
            return _assemble(df.iloc[0:0], pd.DataFrame(index=df.index[0:0]), utt_col, resp_col, issue_col)
        out = pd.concat(parts)
        out = out.loc[[i for i in df.index if i in out.index]]   # 원래 순서
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].fillna("")
        return out
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
        merged = [_merge(res.loc[idx], pr, descs[idx], rules, issue_list) for idx, pr in zip(res.index, preds)]
        m = pd.DataFrame(merged, index=res.index)
        res["규칙 판정"] = res["관련시스템"].astype(str) + " / " + res["등급"].astype(str)
        for c in m.columns:
            res[c] = m[c]
        review = m["_review"]
        res = res.drop(columns="_review")

    crossed = pd.Series(False, index=res.index)
    if model is None:
        # 규칙만 쓸 때: 보조 이슈에 나온 다른 관련시스템도 관련시스템 칸에 함께 붙인다
        # 예) 대표 음성합성 + 보조 GA·기가지니 UI 오류 → "음성합성, GA"
        merged_sys = []
        for sys_, sec in zip(res["관련시스템"], res["보조 이슈"]):
            others = [lbl.split("·")[0] for lbl in str(sec).split(", ") if lbl and lbl.split("·")[0] != sys_]
            merged_sys.append(local_model.join_systems([sys_] + others) if others else sys_)
        crossed = pd.Series([m != o for m, o in zip(merged_sys, res["관련시스템"])], index=res.index)
        res["관련시스템"] = merged_sys

    if issue_list:   # 이슈리스트 행은 정상일 수 없다 (학습 모델을 끈 경우 등)
        ok = res["관련시스템"] == "정상"
        res.loc[ok, "관련시스템"] = "미분류"
        res.loc[ok, "판단 근거"] = "이슈리스트 행이지만 문제점 기술이 없어 관련시스템을 판단할 수 없음"
        res.loc[ok, "신뢰도"] = 0.0

    need = ((res["신뢰도"] < p["review_threshold"])
            | res["관련시스템"].map(lambda v: any(t in str(v).split(", ") for t in p.get("always_review_types", [])))
            | res["등급"].isin(p.get("always_review_grades", []))
            | (res["관련시스템"] == "미분류")
            | review
            | (crossed & bool(p.get("review_cross_system", False))))
    res["검토 필요"] = need.map({True: "Y", False: "N"})

    return _assemble(df, res, utt_col, resp_col, issue_col)


def _pass_rows(df: pd.DataFrame, utt_col: str, resp_col: str, issue_col: str | None) -> pd.DataFrame:
    """테스트 결과 pass 행 = 정상. pass인데 비고가 적힌 경우 판단 근거에 남긴다."""
    notes = df[issue_col].astype(str).str.strip() if issue_col else pd.Series("", index=df.index)
    res = pd.DataFrame({"관련시스템": "정상", "세부 이슈": "", "등급": "", "보조 이슈": "",
                        "판단 근거": ["테스트 결과 pass" + (f" (비고: {n})" if n else "") for n in notes],
                        "신뢰도": 1.0, "적용 규칙": "PASS", "검토 필요": "N"}, index=df.index)
    return _assemble(df, res, utt_col, resp_col, issue_col)


def _assemble(df, res, utt_col, resp_col, issue_col) -> pd.DataFrame:
    # 이미 결과 컬럼이 있는 시트(이전 리포트)를 다시 넣어도 덮어쓰도록
    utts = df[utt_col].map(clean_utterance)
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
