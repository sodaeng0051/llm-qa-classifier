"""③ 이슈 유형 자동 분류: 관련시스템 · 세부 이슈 · 등급 · 판단 근거 · 신뢰도를 만든다.

판정 기준은 운영 지침(관련시스템 분류 기준 / 등급 기준)을 rules.py 에 옮긴 것이다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analyzer import INTENT_LABELS, RESPONSE_LABELS, Analysis
from .rules import GRADES


@dataclass
class Verdict:
    issue_type: str          # 관련시스템
    sub: str                 # 세부 이슈 (대표 이슈)
    grade: str               # Critical / Major / Minor / Request ('' = 이슈 아님)
    reason: str
    confidence: float
    rule_id: str
    secondary: list[str] = field(default_factory=list)


def _rank(grade: str) -> int:
    return GRADES.index(grade) if grade in GRADES else len(GRADES)


def judge(a: Analysis, rules: dict) -> Verdict:
    """문제점 기술이 없을 때: 발화 의도 × 응답 유형 조합으로 판정 (위에서부터 우선)."""
    p = rules["params"]
    low = a.overlap < p["low_overlap"]
    it, rt = a.intent, a.response_type

    if a.stt_similarity is not None and a.stt_similarity < p["stt_mismatch"]:
        return Verdict("음성인식", "오인식", "Major",
                       f"실제 발화와 인식(STT) 결과가 다름 (유사도 {a.stt_similarity:.0%})", 0.85, "R01")
    if rt == "empty":
        return Verdict("GA", "응답미출력", "Major", "응답이 비어 있음 (응답 미표출)", 0.9, "R02")
    if rt == "error":
        return Verdict("GA", "응답미출력", "Major", "시스템 오류·일시적 장애 안내로 응답 흐름이 끊김", 0.8, "R03")
    if rt == "not_understood":
        return Verdict("음성인식", "미인식", "Minor", "발화를 인식하지 못해 재발화를 요청함", 0.8, "R04")
    if a.asr_suspect and it == "unknown":
        return Verdict("음성인식", "오인식", "Minor", "발화 텍스트가 비정상적임 (인식 오류 의심)", 0.6, "R05")
    if it == "question" and rt == "action":
        return Verdict("GA", "의도오해", "Major", "정보성 질문을 기능 실행 의도로 오인", 0.85, "R10")
    if it == "command" and rt == "no_result":
        return Verdict("GA", "동작미연계", "Major", "요청 기능 실행 실패 및 검색 결과 없음", 0.85, "R11")
    if it == "question" and rt == "no_result":
        return Verdict("GA", "검색/뷰서치 실패", "Major", "정보성 질문에 대한 답변/검색 결과 없음", 0.65, "R12")
    if rt == "unsupported":
        target = "기능 요청" if it == "command" else "질문"
        return Verdict("대화 정책", "미지원 안내", "Request",
                       f"{target}에 미지원·불가 안내로 응답 (현재 기준 미지원 기능인지 확인 필요)", 0.6, "R20")
    if it == "command" and rt == "action" and low:
        return Verdict("GA", "의도오해", "Major", "요청과 상관없는 기능을 실행함 (발화-응답 핵심어 불일치)", 0.65, "R13")
    if it == "command" and rt == "info":
        return Verdict("GA", "동작미연계", "Major", "기능 실행 요청에 실행 없이 일반 응답만 제공", 0.6, "R14")
    if it == "question" and rt == "info" and low:
        return Verdict("GA", "의도오해", "Minor", "질문과 관련 없는 응답 (엉뚱한 답변 의심)", 0.55, "R21")
    if it == "unknown":
        return Verdict("미분류", "", "", "발화 의도를 규칙으로 판단할 수 없음", 0.3, "R90")
    conf = 0.7 + min(a.overlap, 1.0) * 0.25
    return Verdict("정상", "", "", f"{INTENT_LABELS[it]} → {RESPONSE_LABELS[rt]} (의도와 응답 유형 일치)",
                   round(conf, 2), "R99")


def analysis_note(a: Analysis) -> str:
    """② 분석 단계 요약 문자열 (리포트·화면 표시용)."""
    parts = [f"발화: {INTENT_LABELS[a.intent]}" + (f"[{', '.join(a.intent_hits[:3])}]" if a.intent_hits else ""),
             f"응답: {RESPONSE_LABELS[a.response_type]}" + (f"[{', '.join(a.response_hits[:3])}]" if a.response_hits else ""),
             f"연관도 {a.overlap:.0%}" + (f"({', '.join(a.shared_terms[:3])})" if a.shared_terms else "")]
    if a.stt_similarity is not None:
        parts.append(f"STT 유사도 {a.stt_similarity:.0%}")
    return " / ".join(parts)


def match_desc(desc: str, rules: dict) -> list[tuple[int, dict, str]]:
    """QA 문제점 텍스트에 매치되는 (규칙 번호, 규칙, 매치 문구) 목록."""
    out = []
    for i, rule in enumerate(rules.get("issue_desc", [])):
        try:
            m = re.search(rule["pattern"], desc, flags=re.IGNORECASE)
        except (re.error, TypeError, KeyError):
            continue
        if m:
            out.append((i, rule, m.group(0)))
    return out


def apply_grade_override(text: str, grade: str, rules: dict) -> tuple[str, str]:
    """Critical 문구는 등급을 올리고, Request 문구는 Request로 바꾼다. (등급, 보정 사유)"""
    for target in ("Critical", "Request"):
        pat = rules.get("grade_override", {}).get(target)
        if pat:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return target, f"'{m.group(0)}' → {target}"
    return grade, ""


def judge_with_desc(a: Analysis, desc: str, rules: dict, utterance: str = "", response: str = "") -> Verdict:
    """문제점(QA 기술)이 있으면 그것을 1순위 근거로, 발화·응답 분석을 보조 근거로 사용."""
    desc = (desc or "").strip()
    if not desc:
        v = judge(a, rules)
        if v.grade:
            v.grade, note = apply_grade_override(f"{utterance} {response}", v.grade, rules)
            if note:
                v.reason += f" · 등급 보정({note})"
        return v

    hits = match_desc(desc, rules)
    if not hits:
        v = judge(a, rules)
        if v.issue_type == "정상":  # 이슈 리스트에 올라온 건이므로 '정상'으로 두지 않는다
            v = Verdict("미분류", "", "Minor", f"문제점 기술에서 관련시스템을 판단할 수 없음: '{desc[:40]}'", 0.3, "D90")
        else:
            v.reason += f" (문제점 기술 규칙 미매치: '{desc[:30]}')"
            v.confidence = min(v.confidence, 0.5)
    else:
        # 대표: 등급이 높은 규칙 → 목록 위쪽 규칙 순
        i, rule, phrase = min(hits, key=lambda h: (_rank(h[1].get("grade", "")), h[0]))
        reason = rule["reason"]
        conf = float(rule.get("confidence") or 0.7)
        if str(rule.get("sub", "")).startswith("의도오해") and "정보성 질문을" in reason:
            if a.intent == "command":
                reason = reason.replace("정보성 질문을", "발화 의도를")
            elif a.intent == "question":
                conf = min(conf + 0.05, 0.99)
        main_label = f"{rule['type']}·{rule.get('sub', '')}"
        secondary, extra = [], []
        for _, r2, ph2 in sorted(hits, key=lambda h: h[0]):
            label = f"{r2['type']}·{r2.get('sub', '')}"
            if label != main_label and label not in secondary:
                secondary.append(label)
                extra.append(f"{label}('{ph2}')")
        text = f"{reason} [근거: '{phrase}']"
        if extra:
            text += " · 추가 이슈: " + ", ".join(extra)
        v = Verdict(rule["type"], rule.get("sub", ""), rule.get("grade") or "Minor", text,
                    round(conf, 2), f"D{i + 1:02d}", secondary)

    v.grade, note = apply_grade_override(desc, v.grade or "Minor", rules)
    if note:
        v.reason += f" · 등급 보정({note})"
    return v
