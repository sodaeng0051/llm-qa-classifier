"""로컬 학습 모델: 과거 이슈리스트(QA가 등급·관련시스템을 입력한 시트)로 학습한다. 외부 API 없음.

- 기본 학습 데이터: training/base_labeled.csv  (프로젝트에 포함, 과거 회차 이슈리스트에서 추출)
- 추가 학습 데이터: data/labeled.csv           (앱에서 검수 결과 저장 / 과거 시트 업로드로 누적)
- 모델: 문자·단어 n-gram TF-IDF + 로지스틱 회귀
    · 등급: 4개 중 1개 (Critical / Major / Minor / Request)
    · 관련시스템: 시스템별 예/아니오 (여러 개 동시 가능, 예: "대화 정책, GA")
- 판단 근거로 '가장 비슷한 과거 이슈'와 그때의 QA 판정을 함께 보여준다.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "training" / "base_labeled.csv"
DATA_DIR = ROOT / "data"
LABELED_PATH = DATA_DIR / "labeled.csv"
MODEL_PATH = DATA_DIR / "model.pkl"

SYSTEMS = ["음성인식", "음성합성", "대화 정책", "GA", "홈포털"]   # 표기 순서
GRADES = ["Critical", "Major", "Minor", "Request"]
COLS = ["출처", "순번", "사용자 발화", "실제 응답", "문제점", "등급", "관련시스템"]
MIN_SAMPLES = 20


# ───────────────────────── 라벨 정리 ─────────────────────────
def norm_systems(value) -> list[str]:
    """'대화정책, GA' / '대화 정책,GA' → ['대화 정책', 'GA'] (SYSTEMS 순서)"""
    parts = {re.sub(r"\s+", "", p) for p in re.split(r"[,/\n·]", str(value or "")) if p.strip()}
    return [s for s in SYSTEMS if s.replace(" ", "") in parts]


def join_systems(systems) -> str:
    return ", ".join(s for s in SYSTEMS if s in set(systems))


def norm_grade(value) -> str:
    v = str(value or "").strip().lower()
    return next((g for g in GRADES if g.lower() == v), "")


def source_name(filename: str) -> str:
    """'260727-260729_GG3_QAT_1__MAR4510C.csv' → '260727 GG3'"""
    stem = Path(str(filename)).stem
    m = re.match(r"(\d{6})(?:-\d{6})?_([A-Za-z0-9]+)", stem)
    return f"{m.group(1)} {m.group(2)}" if m else stem[:30]


def extract_labeled(df: pd.DataFrame, utt_col: str, resp_col: str, issue_col: str | None,
                    source: str, sys_col: str | None = "관련시스템", grade_col: str | None = "등급") -> pd.DataFrame:
    """QA가 등급 또는 관련시스템을 입력한 행만 학습용 형식으로 뽑는다."""
    from .pipeline import clean_utterance
    get = lambda c: df[c].astype(str) if c and c in df.columns else pd.Series("", index=df.index)
    out = pd.DataFrame({
        "출처": source, "순번": get("순번"), "사용자 발화": get(utt_col).map(clean_utterance),
        "실제 응답": get(resp_col), "문제점": get(issue_col),
        "등급": get(grade_col).map(norm_grade), "관련시스템": get(sys_col).map(lambda v: join_systems(norm_systems(v))),
    })
    return out[(out["등급"] != "") | (out["관련시스템"] != "")].reset_index(drop=True)


# ───────────────────────── 학습 데이터 관리 ─────────────────────────
def load_training() -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str).reindex(columns=COLS).fillna("")
              for p in (BASE_PATH, LABELED_PATH) if p.exists()]
    if not frames:
        return pd.DataFrame(columns=COLS)
    t = pd.concat(frames, ignore_index=True)
    # 기본 데이터와 같은 회차를 다시 올린 경우 중복 제거 (추가 데이터 쪽 판정을 우선)
    return t.drop_duplicates(["출처", "순번", "사용자 발화", "실제 응답", "문제점"], keep="last").reset_index(drop=True)


def append_labeled(df: pd.DataFrame, source: str = "검수") -> int:
    """df: 사용자 발화, 실제 응답, 문제점, 관련시스템, 등급 (+출처, 순번). 같은 (출처, 발화, 응답, 문제점)은 최신으로 덮어씀."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new = df.reindex(columns=COLS).fillna("").astype(str)
    new.loc[new["출처"] == "", "출처"] = source
    new["관련시스템"] = new["관련시스템"].map(lambda v: join_systems(norm_systems(v)))
    new["등급"] = new["등급"].map(norm_grade)
    new = new[(new["등급"] != "") | (new["관련시스템"] != "")]
    if LABELED_PATH.exists():
        new = pd.concat([pd.read_csv(LABELED_PATH, dtype=str).reindex(columns=COLS).fillna(""), new])
    new = new.drop_duplicates(["출처", "사용자 발화", "실제 응답", "문제점"], keep="last")
    new.to_csv(LABELED_PATH, index=False, encoding="utf-8-sig")
    MODEL_PATH.unlink(missing_ok=True)   # 다음 사용 시 자동 재학습
    return len(new)


def training_summary() -> dict:
    t = load_training()
    return {"total": len(t), "base": int(pd.read_csv(BASE_PATH).shape[0]) if BASE_PATH.exists() else 0,
            "added": int(pd.read_csv(LABELED_PATH).shape[0]) if LABELED_PATH.exists() else 0,
            "sources": t["출처"].value_counts().to_dict(),
            "grade": int((t["등급"] != "").sum()), "system": int((t["관련시스템"] != "").sum())}


def labeled_count() -> int:
    return training_summary()["total"]


def reset_added() -> None:
    """추가 학습 데이터와 모델 파일 삭제 (기본 학습 데이터는 유지)."""
    LABELED_PATH.unlink(missing_ok=True)
    MODEL_PATH.unlink(missing_ok=True)


# ───────────────────────── 모델 ─────────────────────────
def _text(utt, resp, desc) -> list[str]:
    return [f"[D] {d} [U] {u} [R] {r}" for u, r, d in zip(utt, resp, desc)]


def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    return make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                      TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))


def _fit(t: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    vec = _vectorizer().fit(_text(t["사용자 발화"], t["실제 응답"], t["문제점"]))
    X = normalize(vec.transform(_text(t["사용자 발화"], t["실제 응답"], t["문제점"])))
    m = {"vec": vec, "X": X, "meta": t.reset_index(drop=True), "grade": None, "systems": {}}
    g = (t["등급"] != "").to_numpy()
    if g.sum() >= MIN_SAMPLES and t.loc[g, "등급"].nunique() >= 2:
        m["grade"] = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(X[g], t.loc[g, "등급"])
    s = (t["관련시스템"] != "").to_numpy()
    if s.sum() >= MIN_SAMPLES:
        sets = t.loc[s, "관련시스템"].map(norm_systems)
        for name in SYSTEMS:
            y = sets.map(lambda x: name in x).to_numpy()
            if y.sum() >= 3 and (~y).sum() >= 3:
                m["systems"][name] = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(X[s], y)
    return m


def train(evaluate: bool = True) -> dict:
    """전체 학습 데이터로 학습. evaluate=True면 '회차 하나를 빼고 학습 → 그 회차 예측' 방식으로 정확도도 잰다."""
    t = load_training()
    if len(t) < MIN_SAMPLES:
        raise ValueError(f"학습 데이터가 부족합니다 (현재 {len(t)}건, 최소 {MIN_SAMPLES}건).")
    info = {"samples": len(t), "sources": t["출처"].nunique()}
    if evaluate and t["출처"].nunique() >= 2:
        ga, sa = [], []
        for src in t["출처"].unique():
            tr, te = t[t["출처"] != src], t[t["출처"] == src]
            if len(tr) < MIN_SAMPLES or te.empty:
                continue
            p = predict(_fit(tr), te["사용자 발화"], te["실제 응답"], te["문제점"], k=0)
            for row, pr in zip(te.itertuples(index=False), p):
                if row.등급:
                    ga.append(pr["grade"] == row.등급)
                if row.관련시스템:
                    sa.append(pr["systems"] == row.관련시스템)
        info["holdout_grade_acc"] = float(np.mean(ga)) if ga else None
        info["holdout_system_acc"] = float(np.mean(sa)) if sa else None
    model = _fit(t)
    model["info"] = info
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    return info


def load(auto_train: bool = True):
    """저장된 모델을 읽는다. 없거나 학습 데이터보다 오래됐으면 자동으로 다시 학습한다."""
    sources = [p for p in (BASE_PATH, LABELED_PATH) if p.exists()]
    stale = MODEL_PATH.exists() and any(p.stat().st_mtime > MODEL_PATH.stat().st_mtime for p in sources)
    if (not MODEL_PATH.exists() or stale) and auto_train and sources:
        try:
            train(evaluate=False)
        except ValueError:
            return None
    # 이 앱이 직접 만든 로컬 파일만 로드한다 (외부에서 받은 pkl 파일은 넣지 말 것)
    return pickle.loads(MODEL_PATH.read_bytes()) if MODEL_PATH.exists() else None


def predict(model, utterances, responses, descs, k: int = 3) -> list[dict]:
    """각 행: grade, grade_prob, systems('대화 정책, GA'), system_probs, confidence, neighbors."""
    from sklearn.preprocessing import normalize

    X = normalize(model["vec"].transform(_text(list(utterances), list(responses), list(descs))))
    n = X.shape[0]
    out = [{"grade": "", "grade_prob": 0.0, "systems": "", "system_probs": {}, "confidence": 0.0,
            "neighbors": []} for _ in range(n)]
    confs = [[] for _ in range(n)]
    if model["grade"] is not None:
        P = model["grade"].predict_proba(X)
        for i, p in enumerate(P):
            j = int(p.argmax())
            out[i]["grade"], out[i]["grade_prob"] = model["grade"].classes_[j], float(p[j])
            confs[i].append(float(p[j]))
    if model["systems"]:
        probs = {name: clf.predict_proba(X)[:, list(clf.classes_).index(True)] for name, clf in model["systems"].items()}
        for i in range(n):
            ps = {name: float(v[i]) for name, v in probs.items()}
            chosen = [s for s, p in ps.items() if p >= 0.5] or [max(ps, key=ps.get)]
            out[i]["systems"], out[i]["system_probs"] = join_systems(chosen), ps
            confs[i].append(min(max(p, 1 - p) for p in ps.values()))
    for i in range(n):
        out[i]["confidence"] = round(min(confs[i]), 2) if confs[i] else 0.0
    if k:
        sim = (X @ model["X"].T).toarray()
        meta = model["meta"]
        for i in range(n):
            for j in np.argsort(-sim[i])[:k]:
                r = meta.iloc[j]
                out[i]["neighbors"].append({"sim": float(sim[i, j]), "출처": r["출처"], "순번": r["순번"],
                                            "문제점": r["문제점"], "등급": r["등급"], "관련시스템": r["관련시스템"]})
    return out
