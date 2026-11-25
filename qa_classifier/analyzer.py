"""② 이슈 분석: 사용자 발화 의도, 실제 응답 유형, 발화-응답 연관도를 계산한다 (외부 API 없음)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

INTENT_LABELS = {"question": "정보성 질문", "command": "기능 실행 요청", "unknown": "판단 불가"}
RESPONSE_LABELS = {
    "empty": "무응답", "error": "오류 안내", "not_understood": "인식 실패/재발화 요청",
    "no_result": "결과 없음", "unsupported": "미지원/불가 안내", "action": "기능 실행/화면 이동",
    "info": "정보 제공/일반 응답",
}

# 핵심어 비교 시 무시할 어미·조사·군더더기
_STRIP = re.compile(
    r"(해\s*줘|해\s*주세요|줘|주세요|줄래|할래|좀|요|은|는|이|가|을|를|에|에서|으로|로|의|도|만|"
    r"합니다|했습니다|습니다|입니다|니다|었|았|겠|해요|돼|되|나요|니|까|야|지)$")


@dataclass
class Analysis:
    intent: str
    intent_hits: list[str] = field(default_factory=list)
    response_type: str = "info"
    response_hits: list[str] = field(default_factory=list)
    overlap: float = 0.0
    shared_terms: list[str] = field(default_factory=list)
    asr_suspect: list[str] = field(default_factory=list)
    stt_similarity: float | None = None


def _hits(patterns: list[str], text: str) -> list[str]:
    out = []
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            out.append(m.group(0).strip() or p)
    return out


def _terms(text: str) -> set[str]:
    """어절 단위로 자르고 조사/어미를 떼어 핵심어 집합을 만든다."""
    terms = set()
    for w in re.findall(r"[가-힣A-Za-z0-9]+", text):
        prev = None
        while w and w != prev:
            prev, w = w, _STRIP.sub("", w)
        if len(w) >= 2:
            terms.add(w.lower())
    return terms


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def keyword_overlap(utterance: str, response: str) -> tuple[float, list[str]]:
    """핵심어 기준 연관도(0~1). 어절이 정확히 같지 않아도 2글자 조각이 겹치면 부분 점수."""
    ut, rt = _terms(utterance), _terms(response)
    if not ut or not rt:
        return 0.0, []
    shared = sorted(t for t in ut if any(t in r or r in t for r in rt))
    exact = len(shared) / len(ut)
    ub, rb = set().union(*map(_bigrams, ut)), set().union(*map(_bigrams, rt))
    fuzzy = len(ub & rb) / len(ub) if ub else 0.0
    return round(max(exact, fuzzy * 0.8), 3), shared


def classify_intent(text: str, rules: dict) -> tuple[str, list[str]]:
    u = rules["utterance"]
    q, c, pc = _hits(u["question"], text), _hits(u["command"], text), _hits(u["polite_command"], text)
    if pc:
        return "command", pc + c
    if q and (not c or text.rstrip().endswith("?") or len(q) >= len(c)):
        return "question", q
    if c:
        return "command", c
    if q:
        return "question", q
    return "unknown", []


def classify_response(text: str, rules: dict) -> tuple[str, list[str]]:
    if not text.strip():
        return "empty", []
    for rtype in ("error", "not_understood", "no_result", "unsupported", "action"):
        hits = _hits(rules["response"][rtype], text)
        if hits:
            return rtype, hits
    return "info", []


def analyze(utterance: str, response: str, rules: dict, stt: str | None = None) -> Analysis:
    utterance, response = str(utterance or "").strip(), str(response or "").strip()
    intent, ih = classify_intent(utterance, rules)
    rtype, rh = classify_response(response, rules)
    overlap, shared = keyword_overlap(utterance, response)
    a = Analysis(intent, ih, rtype, rh, overlap, shared, _hits(rules["asr_suspect"], utterance))
    if stt is not None and str(stt).strip():
        a.stt_similarity = round(SequenceMatcher(None, re.sub(r"\s", "", utterance),
                                                 re.sub(r"\s", "", str(stt))).ratio(), 3)
    return a
