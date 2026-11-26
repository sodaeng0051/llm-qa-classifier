"""④ QA 리포트 자동 생성: 이슈 현황 / 관련시스템·등급별 통계 / 검토 필요 항목 → Excel, HTML."""
from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd

from .rules import GRADE_DESC, GRADES, ISSUE_TYPE_DESC, ISSUE_TYPES

MAIN_COLS = ["사용자 발화", "실제 응답", "문제점", "관련시스템", "세부 이슈", "등급", "판단 근거", "검토 필요",
             "신뢰도", "보조 이슈"]


def _order(values, ref):
    vals = list(dict.fromkeys(values))
    return [v for v in ref if v in vals] + sorted(v for v in vals if v not in ref)


def _explode(df: pd.DataFrame) -> pd.DataFrame:
    """'대화 정책, GA' 행을 시스템별 행으로 펼친다 (시스템별 건수는 중복 포함)."""
    return df.assign(관련시스템=df["관련시스템"].astype(str).str.split(r"\s*,\s*")).explode("관련시스템").reset_index(drop=True)


def summary(df: pd.DataFrame) -> dict:
    total = len(df)
    issues = int((df["관련시스템"] != "정상").sum())
    review = int((df["검토 필요"] == "Y").sum())
    ex = _explode(df)["관련시스템"]
    by_type = ex.value_counts().reindex(_order(ex, ISSUE_TYPES)).fillna(0).astype(int)
    g = df.loc[df["등급"].astype(str) != "", "등급"]
    by_grade = g.value_counts().reindex(_order(g, GRADES)).fillna(0).astype(int)
    return {"total": total, "issues": issues, "issue_rate": issues / total if total else 0.0,
            "review": review, "by_type": by_type.to_dict(), "by_grade": by_grade.to_dict(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}


def type_stats(df: pd.DataFrame) -> pd.DataFrame:
    """관련시스템별 통계. 한 이슈에 시스템이 여러 개면 각각 집계한다 (비율은 전체 이슈 대비)."""
    g = _explode(df).groupby("관련시스템")
    st = pd.DataFrame({
        "건수": g.size(),
        "비율": (g.size() / len(df)).round(3),
        "검토 필요": g["검토 필요"].apply(lambda s: int((s == "Y").sum())),
        "평균 신뢰도": g["신뢰도"].mean().round(2),
    })
    st["분류 기준"] = [ISSUE_TYPE_DESC.get(t, "") for t in st.index]
    return st.loc[_order(st.index, ISSUE_TYPES)].reset_index()


def combo_stats(df: pd.DataFrame) -> pd.DataFrame:
    """관련시스템 조합별 건수 (예: '대화 정책, GA')."""
    c = df["관련시스템"].value_counts().reset_index()
    c.columns = ["관련시스템 조합", "건수"]
    return c


def grade_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """관련시스템 × 등급 건수표 (시스템별 중복 포함)."""
    d = _explode(df[df["등급"].astype(str) != ""])
    if d.empty:
        return pd.DataFrame()
    m = pd.crosstab(d["관련시스템"], d["등급"])
    m = m.reindex(index=_order(m.index, ISSUE_TYPES), columns=_order(m.columns, GRADES), fill_value=0)
    m["합계"] = m.sum(axis=1)
    return m.reset_index()


def reason_stats(df: pd.DataFrame) -> pd.DataFrame:
    """관련시스템 · 세부 이슈 · 등급별 건수 (대표 이슈 순위)."""
    issues = df[df["관련시스템"] != "정상"]
    return (issues.groupby(["관련시스템", "세부 이슈", "등급"]).size().reset_index(name="건수")
            .sort_values("건수", ascending=False))


def qa_agreement(df: pd.DataFrame) -> dict:
    """시트에 QA가 입력한 관련시스템/등급이 있으면 자동 판정과의 일치율."""
    out = {}
    for col in ("관련시스템", "등급"):
        qa = f"{col}(QA)"
        if qa in df.columns:
            m = df[qa].astype(str).str.strip() != ""
            if m.any():
                from .model import join_systems, norm_grade, norm_systems
                norm = (lambda v: join_systems(norm_systems(v))) if col == "관련시스템" else norm_grade
                out[col] = {"n": int(m.sum()),
                            "agree": float((df.loc[m, qa].map(norm) == df.loc[m, col].map(norm)).mean())}
    return out


def fill_original(original: pd.DataFrame, res: pd.DataFrame) -> bytes:
    """원본 시트 양식 그대로, 등급·관련시스템 칸을 채우고 세부 이슈·판단 근거·검토 필요를 뒤에 붙인 Excel."""
    out = original.copy()
    for col in ("등급", "관련시스템"):   # QA가 이미 입력한 칸은 그대로 두고 빈 칸만 채운다
        if col not in out.columns:
            out[col] = ""
        empty = out.loc[res.index, col].astype(str).str.strip() == ""
        out.loc[empty[empty].index, col] = res.loc[empty[empty].index, col]
    for col in ("세부 이슈", "판단 근거", "검토 필요"):
        out[col] = ""
        out.loc[res.index, col] = res[col]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        out.to_excel(xw, sheet_name="이슈리스트", index=False)
        w = xw.sheets["이슈리스트"]
        hdr = xw.book.add_format({"bold": True, "bg_color": "#1f2937", "font_color": "white", "border": 1})
        new = xw.book.add_format({"bold": True, "bg_color": "#2563eb", "font_color": "white", "border": 1})
        for j, c in enumerate(out.columns):
            w.write(0, j, c, new if c in ("등급", "관련시스템", "세부 이슈", "판단 근거", "검토 필요") else hdr)
            w.set_column(j, j, 40 if len(out) and out[c].astype(str).str.len().mean() > 20 else 12)
        w.freeze_panes(1, 0)
    return buf.getvalue()


ISSUE_LIST_COLS = ["순번", "등급", "경로", "방법", "결과", "기대", "문제점", "특이사항", "관련시스템", "시험자", "SAID", "재현일시"]


def tester_stats(df: pd.DataFrame) -> pd.DataFrame:
    """테스트 결과 시트일 때 시험자별 pass / fail."""
    if "판정" not in df.columns or "SAID" not in df.columns:
        return pd.DataFrame()
    from .testsheet import summary as _s
    return _s(df)


def to_issue_list(res: pd.DataFrame) -> bytes:
    """테스트 결과(pass/fail) → fail 건만 모아 이슈리스트 양식 Excel로.
    컬럼: 순번 · 등급 · 경로 · 방법 · 결과 · 기대 · 문제점 · 특이사항 · 관련시스템 · 시험자 · SAID · 재현일시
    (+ 세부 이슈 · 판단 근거 · 검토 필요 · TC 순번 · 구분)"""
    f = res[res["관련시스템"] != "정상"].copy()
    if "SAID" in f.columns:   # 시험자별로 묶기 (시트에 나온 순서)
        order = {s: i for i, s in enumerate(dict.fromkeys(f["SAID"]))}
        f = f.assign(_o=f["SAID"].map(order)).sort_values("_o", kind="stable").drop(columns="_o")
    get = lambda c: f[c].astype(str) if c in f.columns else pd.Series("", index=f.index)
    out = pd.DataFrame({
        "순번": range(1, len(f) + 1), "등급": get("등급"), "경로": get("시험방법"),
        "방법": get("사용자 발화").map(lambda u: f'"{u}" 발화' if u else ""), "결과": "",
        "기대": get("실제 응답"), "문제점": get("문제점"), "특이사항": "", "관련시스템": get("관련시스템"),
        "시험자": "", "SAID": get("SAID"), "재현일시": get("재현일시"),
        "세부 이슈": get("세부 이슈"), "판단 근거": get("판단 근거"), "검토 필요": get("검토 필요"),
        "TC 순번": get("순번"), "구분": get("구분"),
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        title = xw.book.add_format({"bold": True, "font_size": 14})
        hdr = xw.book.add_format({"bold": True, "bg_color": "#1f2937", "font_color": "white", "border": 1})
        new = xw.book.add_format({"bold": True, "bg_color": "#2563eb", "font_color": "white", "border": 1})
        wrap = xw.book.add_format({"text_wrap": True, "valign": "top"})
        out.to_excel(xw, sheet_name="이슈리스트", index=False, startrow=2)
        w = xw.sheets["이슈리스트"]
        w.write(0, 0, "📌 이슈리스트", title)
        for j, c in enumerate(out.columns):
            w.write(2, j, c, new if c in ("등급", "관련시스템", "세부 이슈", "판단 근거", "검토 필요") else hdr)
            w.set_column(j, j, 40 if c in ("방법", "기대", "문제점", "판단 근거", "경로") else 12,
                         wrap if c in ("방법", "기대", "문제점", "판단 근거", "경로") else None)
        w.freeze_panes(3, 0)
        if len(out):
            w.autofilter(2, 0, 2 + len(out), len(out.columns) - 1)
        ts = tester_stats(res)
        if not ts.empty:
            ts.to_excel(xw, sheet_name="시험자별 결과", index=False)
    return buf.getvalue()


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    rest = [c for c in df.columns if c not in MAIN_COLS]
    return df[[c for c in MAIN_COLS if c in df.columns] + rest]


def to_excel(df: pd.DataFrame) -> bytes:
    s = summary(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book
        h1 = wb.add_format({"bold": True, "font_size": 16})
        bold = wb.add_format({"bold": True})
        pct = wb.add_format({"num_format": "0.0%"})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        hdr = wb.add_format({"bold": True, "bg_color": "#1f2937", "font_color": "white", "border": 1})
        yfmt = wb.add_format({"bg_color": "#fde68a"})
        gfmt = {"Critical": wb.add_format({"bg_color": "#fca5a5"}), "Major": wb.add_format({"bg_color": "#fdba74"}),
                "Minor": wb.add_format({"bg_color": "#fef08a"}), "Request": wb.add_format({"bg_color": "#bfdbfe"})}

        def table(ws, r0, c0, data: pd.DataFrame, pct_cols=()):
            for j, c in enumerate(data.columns):
                ws.write(r0, c0 + j, c, hdr)
            for i, row in enumerate(data.itertuples(index=False)):
                for j, v in enumerate(row):
                    ws.write(r0 + 1 + i, c0 + j, v, pct if data.columns[j] in pct_cols else None)
            return r0 + 1 + len(data)

        # 요약 시트
        ws = wb.add_worksheet("요약")
        ws.write(0, 0, "LLM QA 이슈 분석 리포트", h1)
        ws.write(1, 0, f"생성: {s['generated_at']}")
        kpis = [("전체 건수", s["total"], None), ("이슈 건수", s["issues"], None),
                ("이슈율", s["issue_rate"], pct), ("검토 필요", s["review"], None)]
        kpis += [(f"등급 {g}", n, None) for g, n in s["by_grade"].items()]
        for i, (k, v, f) in enumerate(kpis):
            ws.write(3 + i, 0, k, bold)
            ws.write(3 + i, 1, v, f)
        r = 4 + len(kpis)
        tst = tester_stats(df)
        if not tst.empty:
            ws.write(r, 0, "시험자별 테스트 결과", bold)
            r = table(ws, r + 1, 0, tst, ("fail 비율",)) + 1
        ws.write(r, 0, "관련시스템별 통계", bold)
        ts = type_stats(df)
        end = table(ws, r + 1, 0, ts, ("비율",))
        type_rows = (r + 2, end - 1)
        ws.write(end + 1, 0, "관련시스템 조합", bold)
        end = table(ws, end + 2, 0, combo_stats(df))
        gm = grade_matrix(df)
        if not gm.empty:
            ws.write(end + 1, 0, "관련시스템 × 등급", bold)
            end = table(ws, end + 2, 0, gm)
        ws.write(end + 1, 0, "등급 기준", bold)
        for i, g in enumerate(GRADES):
            ws.write(end + 2 + i, 0, g, gfmt[g])
            ws.write(end + 2 + i, 1, GRADE_DESC[g])
        ws.set_column(0, 0, 14)
        ws.set_column(1, 4, 11)
        ws.set_column(5, 5, 70)
        if len(ts):
            chart = wb.add_chart({"type": "bar"})
            chart.add_series({"name": "건수", "categories": ["요약", type_rows[0], 0, type_rows[1], 0],
                              "values": ["요약", type_rows[0], 1, type_rows[1], 1], "fill": {"color": "#3b82f6"}})
            chart.set_title({"name": "관련시스템별 건수"})
            chart.set_legend({"none": True})
            ws.insert_chart(1, 7, chart, {"x_scale": 1.2, "y_scale": 1.1})

        # 상세 시트
        sheets = [("검토 필요 항목", _ordered(df[df["검토 필요"] == "Y"])), ("전체 결과", _ordered(df)),
                  ("세부 이슈별 통계", reason_stats(df))]
        for name, data in sheets:
            data.to_excel(xw, sheet_name=name, index=False)
            w = xw.sheets[name]
            for j, c in enumerate(data.columns):
                w.write(0, j, c, hdr)
                width = 40 if c in ("사용자 발화", "실제 응답", "문제점", "판단 근거", "분석 내용") else 13
                w.set_column(j, j, width, wrap if width == 40 else None)
            w.freeze_panes(1, 0)
            if len(data):
                w.autofilter(0, 0, len(data), len(data.columns) - 1)
                cols = list(data.columns)
                if "검토 필요" in cols:
                    c = cols.index("검토 필요")
                    w.conditional_format(1, c, len(data), c, {"type": "cell", "criteria": "==",
                                                              "value": '"Y"', "format": yfmt})
                if "등급" in cols:
                    c = cols.index("등급")
                    for g, f in gfmt.items():
                        w.conditional_format(1, c, len(data), c, {"type": "cell", "criteria": "==",
                                                                  "value": f'"{g}"', "format": f})
    return buf.getvalue()


def to_html(df: pd.DataFrame) -> bytes:
    s = summary(df)
    ts = type_stats(df)
    mx = max(ts["건수"].max(), 1) if len(ts) else 1
    e = html.escape

    bars = "".join(
        f'<div class="bar"><span class="lbl">{e(r["관련시스템"])}</span>'
        f'<span class="track"><span class="fill{" ok" if r["관련시스템"] == "정상" else ""}" '
        f'style="width:{r["건수"] / mx * 100:.0f}%"></span></span><b>{r["건수"]}</b></div>'
        for _, r in ts.iterrows())

    def table(d: pd.DataFrame) -> str:
        if d.empty:
            return "<p class='muted'>해당 항목 없음</p>"
        head = "".join(f"<th>{e(str(c))}</th>" for c in d.columns)
        body = "".join("<tr>" + "".join(
            f"<td class='g-{e(str(v))}'>{e(str(v))}</td>" for v in row) + "</tr>" for row in d.itertuples(index=False))
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    cols = [c for c in MAIN_COLS if c in df.columns]
    review = df[df["검토 필요"] == "Y"][cols]
    grades = "".join(f'<div class="kpi">{g}<b>{n}</b></div>' for g, n in s["by_grade"].items())
    guide = "".join(f"<tr><td class='g-{g}'>{g}</td><td>{e(GRADE_DESC[g])}</td></tr>" for g in GRADES)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QA 이슈 리포트</title>
<style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;--acc:#2563eb;--ok:#16a34a;
--cr:#fecaca;--ma:#fed7aa;--mi:#fef9c3;--rq:#dbeafe}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e5e7eb;--muted:#9ca3af;--line:#2a2f3a;--card:#171a21;
--acc:#60a5fa;--ok:#4ade80;--cr:#7f1d1d;--ma:#7c2d12;--mi:#713f12;--rq:#1e3a8a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:auto}} h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:28px 0 10px}}
.muted{{color:var(--muted)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:16px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .kpi b{{display:block;font-size:24px}}
.bar{{display:flex;align-items:center;gap:10px;margin:6px 0}} .lbl{{width:90px}} .track{{flex:1;background:var(--card);border-radius:4px;height:14px}}
.fill{{display:block;height:100%;background:var(--acc);border-radius:4px}} .fill.ok{{background:var(--ok)}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:var(--card);white-space:nowrap}} .g-Critical{{background:var(--cr)}} .g-Major{{background:var(--ma)}}
.g-Minor{{background:var(--mi)}} .g-Request{{background:var(--rq)}}
</style></head><body><main>
<h1>LLM QA 이슈 분석 리포트</h1><div class="muted">생성 {s['generated_at']} · 외부 API 미사용(규칙/로컬 모델 분석)</div>
<div class="kpis"><div class="kpi">전체<b>{s['total']}</b></div><div class="kpi">이슈<b>{s['issues']}</b></div>
<div class="kpi">검토 필요<b>{s['review']}</b></div>{grades}</div>
{"<h2>시험자별 테스트 결과</h2>" + table(tester_stats(df)) if not tester_stats(df).empty else ""}
<h2>관련시스템별 통계</h2>{bars}{table(ts)}
<h2>관련시스템 조합</h2>{table(combo_stats(df))}
<h2>관련시스템 × 등급</h2>{table(grade_matrix(df))}
<h2>세부 이슈별 통계</h2>{table(reason_stats(df))}
<h2>검토 필요 항목 ({len(review)})</h2>{table(review)}
<h2>전체 결과</h2>{table(df[cols])}
<h2>등급 기준</h2><table>{guide}</table>
</main></body></html>"""
    return page.encode("utf-8")
