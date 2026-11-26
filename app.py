"""LLM QA 이슈 자동 분류 플랫폼 (외부 API 미사용).

① 테스트 결과 업로드 → ② 이슈 분석 → ③ 이슈 유형 자동 분류 → ④ QA 리포트 자동 생성
실행: streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from qa_classifier import model as local_model
from qa_classifier import report
from qa_classifier.pipeline import detect_columns, is_test_result, read_sheet, run
from qa_classifier.testsheet import summary as tester_summary
from qa_classifier.rules import (GRADE_DESC, GRADES, ISSUE_TYPE_DESC, ISSUE_TYPES, load_rules, reset_rules,
                                 save_rules)

st.set_page_config(page_title="QA 이슈 자동 분류", page_icon="🧪", layout="wide")
ss = st.session_state
SAMPLE = Path(__file__).resolve().parent / "sample_qa_results.xlsx"
rules = load_rules()
flash = ss.pop("flash", None)   # 저장 직후 새로고침에서 한 번만 보여줄 메시지

def save_and_refresh(frames: list[pd.DataFrame], what: str):
    """학습 데이터에 추가 → 바로 재학습 → 페이지 새로고침 (사이드바 건수·모델 상태가 즉시 바뀌도록)."""
    before = local_model.training_summary()["total"]
    for f in frames:
        local_model.append_labeled(f)
    after = local_model.training_summary()["total"]
    with st.spinner("학습 데이터 반영 중 (모델 재학습)..."):
        try:
            local_model.train(evaluate=False)
            trained = "모델 재학습 완료, 다음 분류부터 반영됩니다."
        except ValueError as e:
            trained = str(e)
    ss.flash = (f"{what} 저장 완료 — 새로 추가 {after - before}건"
                + (f" (나머지 {sum(len(f) for f in frames) - (after - before)}건은 이미 있던 행이라 갱신)"
                   if sum(len(f) for f in frames) > after - before else "")
                + f" · 학습 데이터 총 {after}건 · {trained}")
    st.rerun()


# ───────────────────────── 사이드바: 규칙 / 학습 모델 ─────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    st.caption("모든 분석은 이 PC/서버 안에서만 실행됩니다. 외부 API 호출 없음.")
    prm = rules["params"]
    prm["review_threshold"] = st.slider("검토 필요 기준 (신뢰도 미만)", 0.0, 1.0,
                                        float(prm["review_threshold"]), 0.05)
    prm["low_overlap"] = st.slider("발화-응답 연관도 '낮음' 기준", 0.0, 0.5, float(prm["low_overlap"]), 0.01)
    prm["always_review_types"] = st.multiselect("항상 검토할 관련시스템", ISSUE_TYPES, prm.get("always_review_types", []))
    prm["always_review_grades"] = st.multiselect("항상 검토할 등급", GRADES, prm.get("always_review_grades", []))
    prm["review_cross_system"] = st.toggle(
        "관련시스템이 여러 개면 검토 (규칙만 쓸 때)", prm.get("review_cross_system", False),
        help="관련시스템은 여러 개(예: 음성합성, GA)가 나올 수 있습니다. 켜면 규칙만으로 분류할 때 "
             "여러 개로 나온 행을 검토 필요로 표시합니다. 학습 모델 사용 시에는 적용되지 않습니다.")

    with st.expander("분석 규칙 편집 (정규식, 한 줄에 하나)"):
        groups = [("발화: 정보성 질문", "utterance", "question"), ("발화: 기능 실행 요청", "utterance", "command"),
                  ("발화: 질문형 명령 (켜줄래?)", "utterance", "polite_command"),
                  ("응답: 오류 안내", "response", "error"), ("응답: 인식 실패", "response", "not_understood"),
                  ("응답: 결과 없음", "response", "no_result"), ("응답: 미지원/불가", "response", "unsupported"),
                  ("응답: 기능 실행/화면 이동", "response", "action")]
        for title, g, k in groups:
            txt = st.text_area(title, "\n".join(rules[g][k]), height=110, key=f"r_{g}_{k}")
            rules[g][k] = [line.strip() for line in txt.splitlines() if line.strip()]
    with st.expander("문제점 분류 규칙 편집 (위에서부터 우선)"):
        st.caption("‘문제점’ 컬럼 문구에 정규식(| 로 구분)이 매치되면 해당 유형으로 분류합니다.")
        rdf = st.data_editor(pd.DataFrame(rules["issue_desc"]), num_rows="dynamic", hide_index=True,
                             key="r_issue_desc", column_config={
                                 "type": st.column_config.SelectboxColumn("관련시스템", options=ISSUE_TYPES, required=True),
                                 "sub": "세부 이슈",
                                 "grade": st.column_config.SelectboxColumn("등급", options=GRADES, required=True),
                                 "reason": "판단 근거 문구", "pattern": "패턴(정규식)",
                                 "confidence": st.column_config.NumberColumn("신뢰도", min_value=0.0, max_value=1.0)})
        rules["issue_desc"] = [r for r in rdf.fillna("").to_dict("records") if r.get("type") and r.get("pattern")]
    c1, c2 = st.columns(2)
    if c1.button("규칙 저장", width="stretch"):
        save_rules(rules)
        st.toast("저장되었습니다. 다시 분석하면 반영됩니다.")
    if c2.button("기본값 복원", width="stretch"):
        reset_rules()
        for key in [k for k in ss if str(k).startswith("r_")]:
            del ss[key]
        st.rerun()

    st.divider()
    st.subheader("🧠 학습 모델")
    if flash:
        st.success(flash)
    tsum = local_model.training_summary()
    st.caption(f"학습 데이터 {tsum['total']}건 (기본 {tsum['base']} + 추가 {tsum['added']}) · "
               f"회차 {len(tsum['sources'])}개\n\n과거 이슈리스트의 QA 판정(등급·관련시스템)을 학습해 "
               "관련시스템·등급을 정하고, 가장 비슷한 과거 이슈를 판단 근거로 보여줍니다.")
    use_model = st.toggle("학습 모델 사용", value=tsum["total"] >= local_model.MIN_SAMPLES,
                          disabled=tsum["total"] < local_model.MIN_SAMPLES,
                          help="끄면 키워드 규칙만으로 분류합니다.")
    with st.expander(f"관련시스템 분류표 ({tsum['guideline']}행{', 수정됨' if tsum['guideline_custom'] else ''})"):
        st.caption("분류 기준표의 판단 기준·대표 이슈 문장도 학습에 씁니다. 대표 이슈는 쉼표로 구분하세요. "
                   "저장하면 바로 다시 학습합니다.")
        gdf = st.data_editor(local_model.load_guideline(), num_rows="dynamic", hide_index=True, key="guide_ed",
                             column_config={
                                 "관련시스템": st.column_config.TextColumn(
                                     width="small", required=True,
                                     help="여러 개를 함께 붙이는 기준은 '대화 정책, GA'처럼 쉼표로 적으세요."),
                                 "판단 기준": st.column_config.TextColumn(width="medium"),
                                 "대표 이슈": st.column_config.TextColumn(width="medium")})
        gup = st.file_uploader("분류표 파일로 바꾸기 (관련시스템 · 판단 기준 · 대표 이슈 컬럼)", type=["csv", "xlsx"],
                               key="guide_up")
        b1, b2 = st.columns(2)
        if b1.button("분류표 저장", width="stretch"):
            new = gdf
            if gup is not None:
                new = pd.read_csv(gup, dtype=str) if gup.name.lower().endswith(".csv") else pd.read_excel(gup, dtype=str)
                new.columns = [str(c).strip() for c in new.columns]
                if not {"관련시스템", "대표 이슈"} <= set(new.columns):
                    st.error("파일에 '관련시스템', '대표 이슈' 컬럼이 필요합니다. ('분류' 컬럼명도 인식)")
                    new = None
            if new is not None:
                new = new.rename(columns={"분류": "관련시스템"})
                local_model.save_guideline(new)
                with st.spinner("분류표 반영 중 (모델 재학습)..."):
                    local_model.train(evaluate=False)
                ss.flash = f"관련시스템 분류표 {len(local_model.load_guideline())}행 저장 · 모델 재학습 완료"
                st.rerun()
        if b2.button("기본 분류표로", width="stretch", disabled=not tsum["guideline_custom"]):
            local_model.reset_guideline()
            ss.flash = "기본 관련시스템 분류표로 되돌렸습니다."
            st.rerun()
    with st.expander("과거 이슈리스트로 추가 학습"):
        st.caption("등급·관련시스템이 입력된 과거 시트를 올리면 학습 데이터에 추가됩니다. 같은 행은 최신으로 덮어씁니다.")
        hist = st.file_uploader("이슈리스트 (여러 개 가능)", type=["csv", "xlsx"], accept_multiple_files=True,
                                key="hist_up", label_visibility="collapsed")
        if hist and st.button(f"{len(hist)}개 파일 학습 데이터에 추가", width="stretch"):
            added, frames = [], []
            for f in hist:
                try:
                    hdf = read_sheet(f)
                    hc = detect_columns(hdf)
                    if "utterance" not in hc or "response" not in hc:
                        st.error(f"{f.name}: 발화/응답 컬럼을 찾지 못했습니다.")
                        continue
                    x = local_model.extract_labeled(hdf, hc["utterance"], hc["response"], hc.get("issue"),
                                                    local_model.source_name(f.name))
                    frames.append(x)
                    added.append(f"{f.name} {len(x)}건")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if frames:
                save_and_refresh(frames, "과거 이슈리스트(" + ", ".join(added) + ")")
    if st.button("모델 다시 학습 + 정확도 측정", width="stretch", disabled=tsum["total"] < local_model.MIN_SAMPLES,
                 help="회차 하나를 빼고 학습해 그 회차를 맞히는 방식으로 정확도를 잽니다 (수십 초 소요)."):
        with st.spinner("학습 중..."):
            info = local_model.train(evaluate=True)
        msg = f"{info['samples']}건 학습 완료"
        if info.get("holdout_grade_acc") is not None:
            msg += f" · 처음 보는 회차 기준 등급 일치 {info['holdout_grade_acc']:.0%}"
        if info.get("holdout_system_acc") is not None:
            msg += f", 관련시스템 완전일치 {info['holdout_system_acc']:.0%}"
        st.success(msg)
    if st.button("추가 학습 데이터 초기화", width="stretch", disabled=tsum["added"] == 0,
                 help="기본 학습 데이터(training/base_labeled.csv)는 유지됩니다."):
        local_model.reset_added()
        ss.flash = "추가 학습 데이터를 초기화했습니다 (기본 학습 데이터는 유지)."
        st.rerun()

# ───────────────────────── 상단: 진행 단계 ─────────────────────────
step = 4 if "result" in ss else 2 if "df" in ss else 1
steps = [("①", "테스트 결과 업로드", "Excel / CSV 파일 입력"),
         ("②", "이슈 분석", "사용자 발화와 실제 응답의 적절성 분석"),
         ("③", "이슈 유형 자동 분류", "관련시스템 · 세부 이슈 · 등급"),
         ("④", "QA 리포트 자동 생성", "이슈 현황 / 유형별 통계 / 검토 필요 항목")]
cards = "".join(
    f'<div class="stp {"done" if i < step else "on" if i == step else ""}"><b>{n_} {t}</b><span>{d}</span></div>'
    + ('<div class="arr">→</div>' if i < 4 else "")
    for i, (n_, t, d) in enumerate(steps, 1))
st.markdown(f"""<style>
.flow{{display:flex;gap:6px;align-items:stretch;flex-wrap:wrap;margin:4px 0 18px}}
.stp{{flex:1;min-width:170px;border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;opacity:.55}}
.stp b{{display:block;font-size:15px}} .stp span{{font-size:12px;opacity:.8}}
.stp.done{{opacity:.9}} .stp.on{{opacity:1;border-color:#22c55e;background:rgba(34,197,94,.08)}}
.arr{{align-self:center;opacity:.5}} @media (max-width:800px){{.arr{{display:none}}}}
</style><div class="flow">{cards}</div>""", unsafe_allow_html=True)
st.title("LLM QA 이슈 자동 분류")

# ───────────────────────── ① 업로드 ─────────────────────────
with st.container(border=True):
    st.subheader("① 테스트 결과 업로드")
    st.caption("세 가지 양식을 받습니다 — ① 테스트 결과 시트(발화·기대결과·시험자별 실행결과 pass/fail·비고), "
               "② 이슈리스트(방법·기대·문제점·등급·관련시스템), ③ 사용자 발화·실제 응답만 있는 시트. "
               "위에 제목줄이 있어도 헤더 행을 자동으로 찾습니다. 자세한 형식은 README 참고.")
    up = st.file_uploader("Excel / CSV", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
    if st.button("샘플 데이터로 해보기"):
        ss.file_key = "sample"
        ss.df = read_sheet(SAMPLE)
        ss.src_name = "sample_qa_results.xlsx"
        ss.pop("result", None)
        st.rerun()

    if up is not None:
        sheet = 0
        if not up.name.lower().endswith(".csv"):
            try:
                names = pd.ExcelFile(up).sheet_names
            except Exception as e:
                st.error(f"파일을 읽지 못했습니다: {e}")
                st.stop()
            if len(names) > 1:
                sheet = st.selectbox("시트 선택", names)
        key = f"{up.name}:{up.size}:{sheet}"
        if ss.get("file_key") != key:
            try:
                ss.df = read_sheet(up, sheet)
                ss.file_key, ss.src_name = key, up.name
                ss.pop("result", None)
            except Exception as e:
                st.error(f"파일을 읽지 못했습니다: {e}")

    if "df" in ss:
        df = ss.df
        auto = detect_columns(df)
        cols = list(df.columns)
        c1, c2, c3, c4 = st.columns(4)
        utt_col = c1.selectbox("사용자 발화 컬럼", cols, index=cols.index(auto["utterance"]) if "utterance" in auto else 0)
        resp_col = c2.selectbox("실제 응답 컬럼", cols,
                                index=cols.index(auto["response"]) if "response" in auto else min(1, len(cols) - 1))
        opt = ["(없음)"] + cols
        issue_col = c3.selectbox("문제점 컬럼 (선택)", opt,
                                 index=opt.index(auto["issue"]) if "issue" in auto else 0)
        stt_col = c4.selectbox("STT 결과 컬럼 (선택)", opt,
                               index=opt.index(auto["stt"]) if "stt" in auto else 0)
        test_mode = is_test_result(df)
        if test_mode:
            tsum = tester_summary(df)
            st.success(f"테스트 결과 시트로 인식했습니다 — 시험자 {len(tsum)}명, 실행 {len(df)}건 "
                       f"(pass {int(tsum['pass'].sum())} / fail {int(tsum['fail'].sum())}). "
                       "pass는 정상으로, fail은 비고 내용을 문제점으로 보고 분류합니다.")
            st.dataframe(tsum, hide_index=True, width="content",
                         column_config={"fail 비율": st.column_config.NumberColumn(format="percent")})
        st.caption(f"`{ss.src_name}` · {len(df)}행" + (" (시험자별 실행 1건 = 1행으로 펼침)" if test_mode else ""))
        st.dataframe(df.head(5), width="stretch", hide_index=True)

        if st.button("② 분석 · ③ 분류 실행", type="primary", disabled=utt_col == resp_col):
            data = df[df[utt_col].astype(str).str.strip() != ""]
            ss.result = run(data, utt_col, resp_col, rules,
                            stt_col=None if stt_col == "(없음)" else stt_col,
                            issue_col=None if issue_col == "(없음)" else issue_col, use_model=use_model,
                            result_col=auto.get("result"))
            ss.run_at = datetime.now()
            st.rerun()

if "result" not in ss:
    st.stop()
res: pd.DataFrame = ss.result

# ───────────────────────── ② 분석 ─────────────────────────
with st.container(border=True):
    st.subheader("② 이슈 분석")
    st.caption("발화 의도(질문/명령)와 응답 유형(실행·결과 없음·미지원 등), 핵심어 연관도를 비교해 적절성을 판단합니다."
               + (" 문제점 컬럼이 있어 QA 기술 내용의 키워드를 1순위 근거로 사용했습니다." if "문제점" in res.columns else ""))
    c1, c2 = st.columns(2)
    cross = pd.crosstab(res["발화 의도"], res["응답 유형"])
    c1.markdown("**발화 의도 × 응답 유형**")
    c1.dataframe(cross, width="stretch")
    with c2.expander("행별 분석 내용", expanded=False):
        st.dataframe(res[["사용자 발화", "실제 응답", "분석 내용"]], width="stretch", hide_index=True)

# ───────────────────────── ③ 분류 + 검수 ─────────────────────────
with st.container(border=True):
    st.subheader("③ 이슈 유형 자동 분류")
    g1, g2, _ = st.columns([1, 1, 4])
    with g1.popover("관련시스템 기준"):
        for r in local_model.load_guideline().itertuples(index=False):
            st.markdown(f"**{r[0]}** — {r[1]}  \n대표 이슈: {r[2]}")
    with g2.popover("등급 기준"):
        for g in GRADES:
            st.markdown(f"**{g}** — {GRADE_DESC[g]}")
    f1, f2 = st.columns([1, 2])
    only = f1.radio("보기", ["전체", "이슈만", "검토 필요만"], horizontal=True)
    types = f2.multiselect("관련시스템 필터", ISSUE_TYPES)
    view = res
    if only == "이슈만":
        view = view[view["관련시스템"] != "정상"]
    elif only == "검토 필요만":
        view = view[view["검토 필요"] == "Y"]
    if types:
        view = view[view["관련시스템"].map(lambda v: any(t in str(v).split(", ") for t in types))]

    show_cols = [c for c in ["사용자 발화", "실제 응답", "문제점", "관련시스템", "세부 이슈", "등급", "판단 근거",
                             "검토 필요", "신뢰도", "보조 이슈", "규칙 판정", "판정", "SAID", "관련시스템(QA)", "등급(QA)"] if c in res.columns]
    if "모델 예측" in res.columns:
        show_cols.append("모델 예측")
    st.caption("관련시스템·세부 이슈·등급·판단 근거·검토 필요 칸은 직접 수정할 수 있습니다. 수정 내용은 리포트에 반영됩니다. "
               "관련시스템은 여러 개일 수 있습니다 (예: 대화 정책, GA).")
    edited = st.data_editor(
        view[show_cols], width="stretch", hide_index=True, key="editor",
        disabled=["사용자 발화", "실제 응답", "문제점", "보조 이슈", "신뢰도", "규칙 판정", "판정", "SAID", "관련시스템(QA)", "등급(QA)"],
        column_config={
            "관련시스템": st.column_config.TextColumn(width=150, help="여러 개는 쉼표로: 대화 정책, GA, 홈포털"),
            "등급": st.column_config.SelectboxColumn(options=GRADES + [""], width=80),
            "검토 필요": st.column_config.SelectboxColumn(options=["Y", "N"], required=True, width="small"),
            "판단 근거": st.column_config.TextColumn(width="large"),
            "신뢰도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", width="small"),
        })
    EDIT_COLS = ["관련시스템", "세부 이슈", "등급", "판단 근거", "검토 필요"]
    changed = edited[EDIT_COLS].ne(view[EDIT_COLS]).any(axis=1)
    if changed.any():
        res.loc[edited.index, EDIT_COLS] = edited[EDIT_COLS]
        st.rerun()

    c1, c2 = st.columns([1, 3])
    reviewed = res[res["검토 필요"] == "N"]
    if c1.button(f"검수 결과 학습 데이터로 저장 ({len(reviewed)}건)", disabled=reviewed.empty):
        save_and_refresh([reviewed.assign(출처=local_model.source_name(ss.src_name))], f"검수 결과 {len(reviewed)}건")
    if flash:
        c2.success(flash)
    c2.caption("‘검토 필요 = N’인 행(확인 완료된 판정)만 저장됩니다. 검토 후 N으로 바꿔 저장하세요.")

# ───────────────────────── ④ 리포트 ─────────────────────────
with st.container(border=True):
    st.subheader("④ QA 리포트 자동 생성")
    s = report.summary(res)
    k = st.columns(4)
    k[0].metric("전체", s["total"])
    k[1].metric("이슈", s["issues"])
    k[2].metric("이슈율", f"{s['issue_rate']:.0%}")
    k[3].metric("검토 필요", s["review"])

    tst = report.tester_stats(res)
    if not tst.empty:
        st.markdown("**시험자별 테스트 결과**")
        st.dataframe(tst, hide_index=True, width="content",
                     column_config={"fail 비율": st.column_config.NumberColumn(format="percent")})
    gcols = st.columns(len(GRADES))
    for col, g in zip(gcols, GRADES):
        col.metric(g, s["by_grade"].get(g, 0), help=GRADE_DESC[g])
    agree = report.qa_agreement(res)
    if agree:
        seen = local_model.source_name(ss.src_name) in local_model.training_summary()["sources"]
        st.info("시트에 입력된 QA 판정과 자동 판정 일치율 — " + " · ".join(
            f"{k} {v['agree']:.0%} ({v['n']}건)" for k, v in agree.items())
            + (" · ⚠️ 이 회차는 이미 학습 데이터에 포함되어 있어 실제보다 높게 나옵니다. "
               "실제 정확도는 사이드바 '모델 다시 학습 + 정확도 측정'의 수치를 보세요." if seen else ""))

    ts = report.type_stats(res)
    c1, c2 = st.columns([3, 2])
    fig = px.bar(ts, x="건수", y="관련시스템", orientation="h", text="건수", color="관련시스템",
                 color_discrete_map={"정상": "#22c55e", "GA": "#3b82f6", "대화 정책": "#a855f7",
                                     "음성인식": "#14b8a6", "음성합성": "#f59e0b", "홈포털": "#ec4899",
                                     "미분류": "#9ca3af"})
    fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=10, t=10, b=0),
                      yaxis={"categoryorder": "array", "categoryarray": list(ts["관련시스템"])[::-1]})
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(ts[["관련시스템", "건수", "비율", "검토 필요"]], hide_index=True, width="stretch",
                 column_config={"비율": st.column_config.NumberColumn(format="percent")})

    gm = report.grade_matrix(res)
    if not gm.empty:
        m1, m2 = st.columns([3, 2])
        m1.markdown("**관련시스템 × 등급** (시스템별 중복 집계)")
        m1.dataframe(gm, hide_index=True, width="stretch")
        m2.markdown("**관련시스템 조합**")
        m2.dataframe(report.combo_stats(res), hide_index=True, width="stretch")
    st.markdown("**세부 이슈별 통계**")
    st.dataframe(report.reason_stats(res), hide_index=True, width="stretch")
    st.markdown(f"**검토 필요 항목 ({s['review']}건)**")
    st.dataframe(res[res["검토 필요"] == "Y"][[c for c in ["사용자 발화", "실제 응답", "문제점", "관련시스템",
                                                         "세부 이슈", "등급", "판단 근거", "보조 이슈"]
                                              if c in res.columns]],
                 hide_index=True, width="stretch")

    stem = ss.src_name.rsplit(".", 1)[0]
    d1, d2, d3 = st.columns([1, 1, 2])
    d1.download_button("📊 Excel 리포트", report.to_excel(res), f"{stem}_QA리포트.xlsx", type="primary",
                       width="stretch")
    d2.download_button("🌐 HTML 리포트", report.to_html(res), f"{stem}_QA리포트.html", "text/html",
                       width="stretch")
    if is_test_result(res):
        d3.download_button(f"📝 이슈리스트 양식으로 받기 (fail {s['issues']}건)", report.to_issue_list(res),
                           f"{stem}_이슈리스트.xlsx", width="stretch",
                           help="fail 건만 모아 순번·등급·경로·방법·결과·기대·문제점·특이사항·관련시스템·시험자·SAID·재현일시 "
                                "양식으로 만듭니다 (세부 이슈·판단 근거·검토 필요는 뒤에 추가).")
    else:
        d3.download_button("📝 원본 양식에 등급·관련시스템 채워 받기", report.fill_original(ss.df, res),
                           f"{stem}_등급분류.xlsx", width="stretch",
                           help="올린 시트의 컬럼 순서 그대로, 등급·관련시스템을 채우고 세부 이슈·판단 근거·검토 필요를 뒤에 붙입니다.")
