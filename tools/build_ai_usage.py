"""Fill the school's AI usage report form (HWPX) from a JSON content file.

    python tools/build_ai_usage.py docs/ai_usage/round1.json [--check]

Reads docs/ai_usage/template.hwpx (the official form) and writes
docs/ai_usage/<output>. --check only lists what is missing.

The form has Part 1 (models, purposes), Part 2 (prompt/answer/verification
cases) and Part 3 (reflection). The challenge notice also asks for algorithm
understanding, references, before/after validation results and analysis, so
those are added as Parts 3-6 in the form's own box style, and the form's
reflection becomes Part 7:

  Part 1  AI 활용 현황          models, purposes                (form)
  Part 2  프롬프트 및 검증 로그  cases                           (form)
  Part 3  알고리즘 이해          algorithm[]
  Part 4  참고 문헌              references[] (ids in docs/references.json), external_code[]
  Part 5  개선 결과              results.rows[] -> table, results.notes[]
  Part 6  분석 및 고찰           analysis[]  (spectrograms, listening, limits, next steps)
  Part 7  종합 고찰              reflection[]                    (form's Part 3)

JSON keys: output, affiliation, team, members, models (subset of
ChatGPT/Claude/Gemini/기타), other_model, purposes (subset of
library/code_draft/concept/debugging), cases [{title, prompt[], answer[],
verify[]}], algorithm[], references[], external_code[], results {rows
[{label, source, method}], notes[]}, analysis[], reflection[].

A paragraph starting with "## " is written as a bold sub-heading.
results.rows[].source: "roundN" (results/submissions/roundN/summary.csv) or a
run id logged in results/experiments.csv.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "ai_usage"
REFERENCES = ROOT / "docs" / "references.json"
MODELS = ["ChatGPT", "Claude", "Gemini", "기타"]
PURPOSES = ["library", "code_draft", "concept", "debugging"]
CHECK = "[ V ]"
PLACEHOLDER = re.compile(r"TODO|TBD|\?\?\?")

# Parts added between the form's Part 2 and Part 3: (key, title, guidance)
EXTRA_PARTS = [
    ("algorithm", "Part 3. 알고리즘 이해",
     "(적용한 방법의 핵심 원리와 해당 방법을 선택한 이유를 적어주세요.)"),
    ("references", "Part 4. 참고 문헌",
     "(직접 확인한 문헌·논문의 제목, 저자, 연도, 링크 또는 DOI와 구현에 참고한 내용. "
     "외부 코드를 활용했다면 출처를 함께 적어주세요.)"),
    ("results", "Part 5. 개선 결과",
     "(알고리즘·파라미터 변경 이유와 변경 전후 Validation의 SNR 및 PESQ 비교)"),
    ("analysis", "Part 6. 분석 및 고찰",
     "(스펙트로그램과 청취에서 확인한 변화, 남은 한계와 개선 방향)"),
]


# --------------------------------------------------------------------------
# content helpers
# --------------------------------------------------------------------------
def load_references() -> dict:
    if not REFERENCES.exists():
        return {}
    return {r["id"]: r for r in json.loads(REFERENCES.read_text(encoding="utf-8"))}


def format_reference(r: dict) -> str:
    s = f"{r['authors']}, \"{r['title']},\" {r['venue']}, {r['year']}."
    if r.get("doi"):
        s += f" DOI: {r['doi']}"
    elif r.get("url"):
        s += f" {r['url']}"
    return s


def result_value(source: str, method: str) -> dict:
    """{'pesq', 'snr_db', 'n'} of `method` in a round folder or a logged run."""
    if re.fullmatch(r"round\d+", source):
        path = ROOT / "results" / "submissions" / source / "summary.csv"
        key = "method"
    else:
        path = ROOT / "results" / "experiments.csv"
        key = None
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["method"] == method and (key or row["run_id"] == source):
                return {"pesq": float(row["pesq"]), "snr_db": float(row["snr_db"]),
                        "n": int(row["n"])}
    raise KeyError(f"{method} not found in {source}")


def results_table(rows: list[dict]) -> list[list[str]]:
    table = [["방법", "PESQ-WB", "SNR (dB)", "ΔPESQ", "ΔSNR (dB)"]]
    base = None
    for r in rows:
        v = result_value(r["source"], r["method"])
        base = base or v
        table.append([r["label"], f"{v['pesq']:.3f}", f"{v['snr_db']:.2f}",
                      f"{v['pesq'] - base['pesq']:+.3f}", f"{v['snr_db'] - base['snr_db']:+.2f}"])
    return table


def validate_spec(spec: dict) -> list[str]:
    """Everything the notice requires that is missing or unverified."""
    problems = []
    for k, what in [("team", "조"), ("members", "조원")]:
        if not spec.get(k):
            problems.append(f"{k} 비어 있음 (양식 머리말의 {what})")
    if not spec.get("models"):
        problems.append("models 비어 있음 (사용한 AI 모델)")
    if not spec.get("purposes"):
        problems.append("purposes 비어 있음 (활용 목적)")
    cases = spec.get("cases") or []
    if not cases:
        problems.append("cases 없음 (핵심 질문·답변 요약·검증)")
    for i, c in enumerate(cases, 1):
        for k in ("title", "prompt", "answer", "verify"):
            if not c.get(k):
                problems.append(f"cases[{i}].{k} 비어 있음")
    for k in ("algorithm", "analysis", "reflection"):
        if not spec.get(k):
            problems.append(f"{k} 비어 있음")
    refs = load_references()
    if not spec.get("references"):
        problems.append("references 없음 (참고 문헌)")
    for rid in spec.get("references") or []:
        r = refs.get(rid)
        if r is None:
            problems.append(f"참고 문헌 {rid}: docs/references.json에 없음")
            continue
        if not r.get("used_for"):
            problems.append(f"참고 문헌 {rid}: used_for(구현에 참고한 내용) 비어 있음")
        if not r.get("checked_by"):
            problems.append(f"참고 문헌 {rid}: checked_by 비어 있음 — 팀원이 원문을 직접 확인한 뒤 이름 기입")
        if r.get("doi") and not r.get("doi_verified"):
            problems.append(f"참고 문헌 {rid}: DOI 미확인 → python tools/check_refs.py")
    res = spec.get("results") or {}
    if len(res.get("rows") or []) < 2:
        problems.append("results.rows 2개 이상 필요 (변경 전후 비교)")
    else:
        try:
            results_table(res["rows"])
        except Exception as e:
            problems.append(f"results.rows 값 읽기 실패: {e}")
    text = json.dumps(spec, ensure_ascii=False)
    for m in sorted(set(PLACEHOLDER.findall(text))):
        problems.append(f"미완성 표시 '{m}'가 남아 있음")
    return problems


# --------------------------------------------------------------------------
# HWPX XML helpers
# --------------------------------------------------------------------------
def t(text: str) -> str:
    return f"<hp:t>{escape(text)}</hp:t>"


def para(text: str, para_pr: str, char_pr: str) -> str:
    return (f'<hp:p id="0" paraPrIDRef="{para_pr}" styleIDRef="24" pageBreak="0" '
            f'columnBreak="0" merged="0"><hp:run charPrIDRef="{char_pr}">{t(text)}'
            f'</hp:run></hp:p>')


def body(paragraphs: list[str], para_pr="28", char_pr="11", bold_pr="16") -> str:
    out = []
    for p in paragraphs:
        if p.startswith("## "):
            out.append(para(p[3:], para_pr, bold_pr))
        else:
            out.append(para(p, para_pr, char_pr))
    return "".join(out)


def table(rows: list[list[str]], width: int, tbl_id: int, first_col: float = 0.36) -> str:
    """A simple bordered table, placed inline (treated as a character)."""
    n_cols = len(rows[0])
    w0 = int(width * first_col)
    widths = [w0] + [(width - w0) // (n_cols - 1)] * (n_cols - 1)
    row_h = 1400
    trs = []
    for r, row in enumerate(rows):
        tcs = []
        for c, cell in enumerate(row):
            tcs.append(
                f'<hp:tc name="" header="{int(r == 0)}" hasMargin="0" protect="0" editable="0" '
                f'dirty="0" borderFillIDRef="4"><hp:subList id="" textDirection="HORIZONTAL" '
                f'lineWrap="BREAK" vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" '
                f'textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
                f'{para(cell, "40", "25" if r == 0 else "10")}</hp:subList>'
                f'<hp:cellAddr colAddr="{c}" rowAddr="{r}"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
                f'<hp:cellSz width="{widths[c]}" height="{row_h}"/>'
                f'<hp:cellMargin left="510" right="510" top="141" bottom="141"/></hp:tc>')
        trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
    return (f'<hp:p id="0" paraPrIDRef="40" styleIDRef="24" pageBreak="0" columnBreak="0" '
            f'merged="0"><hp:run charPrIDRef="10"><hp:tbl id="{tbl_id}" zOrder="{tbl_id % 100}" '
            f'numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
            f'dropcapstyle="None" pageBreak="CELL" repeatHeader="1" rowCnt="{len(rows)}" '
            f'colCnt="{n_cols}" cellSpacing="0" borderFillIDRef="4" noAdjust="0">'
            f'<hp:sz width="{sum(widths)}" widthRelTo="ABSOLUTE" height="{row_h * len(rows)}" '
            f'heightRelTo="ABSOLUTE" protect="0"/><hp:pos treatAsChar="1" affectLSpacing="0" '
            f'flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" '
            f'horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
            f'<hp:outMargin left="0" right="0" top="0" bottom="0"/>'
            f'<hp:inMargin left="510" right="510" top="141" bottom="141"/>'
            f'{"".join(trs)}</hp:tbl><hp:t/></hp:run></hp:p>')


def replace_once(x: str, old: str, new: str, start: int = 0) -> str:
    i = x.find(old, start)
    if i < 0:
        raise ValueError(f"not found in template: {old[:60]}")
    return x[:i] + new + x[i + len(old):]


def replace_nth(x: str, old: str, n: int, new: str) -> str:
    i = -1
    for _ in range(n + 1):
        i = x.find(old, i + 1)
        if i < 0:
            raise ValueError(f"occurrence {n} of {old!r} not found")
    return x[:i] + new + x[i + len(old):]


def fill_cell_after(block: str, label: str, paragraphs: list[str]) -> str:
    """Replace the content of the first table cell that follows `label`."""
    i = block.find(label)
    j = block.find("<hp:subList", i)
    j = block.find(">", j) + 1
    k = block.find("</hp:subList>", j)
    content = "".join(para(p, "40", "10") for p in paragraphs) or para("", "40", "10")
    return block[:j] + content + block[k:]


def renumber_tables(block: str, offset: int) -> str:
    return re.sub(r'<hp:tbl id="(\d+)"', lambda m: f'<hp:tbl id="{int(m.group(1)) + offset}"',
                  block)


def section_part(part_block: str, title: str, guide: str, content: str) -> str:
    """Copy of the form's Part 3 box with a new title, guidance line and content."""
    b = part_block.replace(t("Part 3. 종합 고찰(Self-Reflection)"), t(title))
    g = b.find("(이번 챌린지에서AI를 활용해")
    b = b[:g] + escape(guide) + b[b.find("</hp:t>", g):]
    s = b.find("</hp:p>", b.find(escape(guide))) + len("</hp:p>")
    e = b.find("</hp:subList>", s)
    return b[:s] + content + para("", "28", "11") + b[e:]


# --------------------------------------------------------------------------
def build(spec: dict, out: Path | None = None) -> Path:
    tpl = DOCS / "template.hwpx"
    with zipfile.ZipFile(tpl) as z:
        items = [(info, z.read(info.filename)) for info in z.infolist()]
    x = next(d for i, d in items if i.filename == "Contents/section0.xml").decode("utf-8")
    # cached line layout no longer matches the text; Hangul recomputes it
    x = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", x, flags=re.S)

    # header table
    if spec.get("affiliation"):
        x = replace_once(x, t("연세대학교 미래캠퍼스___________________학과/학부/전공"),
                         t(f"연세대학교 미래캠퍼스 {spec['affiliation']}"))
    if spec.get("team") or spec.get("members"):
        x = replace_once(x, t("______조 / 조원: ___________________________________________"),
                         t(f"{spec.get('team') or '______'}조 / 조원: {spec.get('members') or ''}"))
    if spec.get("team"):  # pledge at the end; signatures stay handwritten
        x = replace_once(x, t("(       ) 조"), t(f"(  {spec['team']}  ) 조"))
    # Part 1 check boxes (models use 4 spaces, purposes 3 spaces in the form)
    # fill from the last box backwards so earlier occurrence indices stay valid
    for name in sorted(spec.get("models", []), key=MODELS.index, reverse=True):
        x = replace_nth(x, t("[    ]"), MODELS.index(name), t(CHECK))
    if spec.get("other_model"):
        x = replace_once(x, t("기타"), t(f"기타({spec['other_model']})"))
    for p in sorted(spec.get("purposes", []), key=PURPOSES.index, reverse=True):
        x = replace_nth(x, t("[   ]"), PURPOSES.index(p), t(CHECK))

    # the form's Part 3 box: heading paragraph + paragraph holding a 1x1 table
    p3 = x.rfind("<hp:p ", 0, x.find("Part 3. 종합 고찰"))
    p3_end = x.find("</hp:p>", x.find("</hp:tbl>", p3)) + len("</hp:p>")
    part_block = x[p3:p3_end]

    # Part 2 cases: rebuild every case from the form's empty "사례2" block
    c1 = x.rfind("<hp:p ", 0, x.find(t("사례1 (예시 포함)")))
    c2 = x.rfind("<hp:p ", 0, x.find(t("사례2")))
    template_block = x[c2:p3]
    blocks = []
    for n, case in enumerate(spec["cases"], 1):
        b = template_block.replace(t("사례2"), t(f"사례{n}: {case['title']}"))
        b = fill_cell_after(b, "나의 질문(Prompt):", case["prompt"])
        b = fill_cell_after(b, "AI의 답변 요약:", case["answer"])
        b = fill_cell_after(b, "나의 검증 및 수정 과정", case["verify"])
        blocks.append(renumber_tables(b, 1000 * n))

    # Parts 3-6 (notice items not in the form)
    refs = load_references()
    for n, (key, title, guide) in enumerate(EXTRA_PARTS, 1):
        if key == "references":
            lines = []
            for i, rid in enumerate(spec.get("references") or [], 1):
                r = refs.get(rid)
                if r is None:
                    lines.append(f"[{i}] (docs/references.json에 없는 항목: {rid})")
                    continue
                lines += [f"[{i}] {format_reference(r)}", f"    → 참고한 내용: {r.get('used_for', '')}"]
            ext = spec.get("external_code") or ["외부 코드: 사용하지 않음 (numpy·scipy·soundfile "
                                                "라이브러리 함수만 사용)"]
            content = body(lines + [""] + ext)
        elif key == "results":
            res = spec.get("results") or {}
            content = ""
            if res.get("rows"):
                content = table(results_table(res["rows"]), 45000, 2000000000 + n)
            content += body(res.get("notes") or [])
        else:
            content = body(spec.get(key) or [])
        blocks.append(renumber_tables(section_part(part_block, title, guide, content), 7000 * n))

    x = x[:c1] + "".join(blocks) + x[p3:]

    # form's Part 3 -> Part 7, filled with the reflection
    x = replace_once(x, t("Part 3. 종합 고찰(Self-Reflection)"), t("Part 7. 종합 고찰(Self-Reflection)"))
    g = x.find("(이번 챌린지에서AI를 활용해")
    s = x.find("</hp:p>", g) + len("</hp:p>")
    e = x.find("</hp:subList>", s)
    x = x[:s] + body(spec.get("reflection", [])) + para("", "28", "11") + x[e:]

    out = out or DOCS / spec["output"]
    with zipfile.ZipFile(out, "w") as z:
        for info, data in items:
            if info.filename == "Contents/section0.xml":
                data = x.encode("utf-8")
            comp = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            z.writestr(zipfile.ZipInfo(info.filename, date_time=info.date_time), data,
                       compress_type=comp)
    return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1:
        sys.exit(__doc__)
    spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    problems = validate_spec(spec)
    for p in problems:
        print("[미완성]", p)
    if not check_only:
        print("written:", build(spec))
    print("내역서 내용 완비" if not problems else f"미완성 항목 {len(problems)}개")
    return 1 if problems and check_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
