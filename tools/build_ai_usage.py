"""Fill the school's AI usage report template (HWPX) from a JSON content file.

    python tools/build_ai_usage.py docs/ai_usage/round1.json

Reads docs/ai_usage/template.hwpx (the official form, unchanged) and writes
docs/ai_usage/<output>. Only text is changed; layout and styles come from the
form. Any number of cases is supported (each copies the form's "사례2" block).
Hangul recomputes line layout on open, so cached line positions are dropped.

JSON keys: output, affiliation, team, members (null = leave the blank line),
models (subset of ChatGPT/Claude/Gemini/기타), other_model, purposes (subset of
library/code_draft/concept/debugging), cases [{title, prompt[], answer[],
verify[]}], reflection[].
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parents[1] / "docs" / "ai_usage"
MODELS = ["ChatGPT", "Claude", "Gemini", "기타"]
PURPOSES = ["library", "code_draft", "concept", "debugging"]
CHECK = "[ V ]"


def t(text: str) -> str:
    return f"<hp:t>{escape(text)}</hp:t>"


def para(text: str, para_pr: str, char_pr: str) -> str:
    return (f'<hp:p id="0" paraPrIDRef="{para_pr}" styleIDRef="24" pageBreak="0" '
            f'columnBreak="0" merged="0"><hp:run charPrIDRef="{char_pr}">{t(text)}'
            f'</hp:run></hp:p>')


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
    body = "".join(para(p, "40", "10") for p in paragraphs) or para("", "40", "10")
    return block[:j] + body + block[k:]


def build(spec: dict) -> Path:
    tpl = DOCS / "template.hwpx"
    with zipfile.ZipFile(tpl) as z:
        items = [(info, z.read(info.filename)) for info in z.infolist()]
    x = next(d for i, d in items if i.filename == "Contents/section0.xml").decode("utf-8")

    # header table
    if spec.get("affiliation"):
        x = replace_once(x, t("연세대학교 미래캠퍼스___________________학과/학부/전공"),
                         t(f"연세대학교 미래캠퍼스 {spec['affiliation']}"))
    if spec.get("team") or spec.get("members"):
        x = replace_once(x, t("______조 / 조원: ___________________________________________"),
                         t(f"{spec.get('team') or '______'}조 / 조원: {spec.get('members') or ''}"))
    # Part 1 check boxes (models use 4 spaces, purposes 3 spaces in the form)
    # fill from the last box backwards so earlier occurrence indices stay valid
    for name in sorted(spec.get("models", []), key=MODELS.index, reverse=True):
        x = replace_nth(x, t("[    ]"), MODELS.index(name), t(CHECK))
    if spec.get("other_model"):
        x = replace_once(x, t("기타"), t(f"기타({spec['other_model']})"))
    for p in sorted(spec.get("purposes", []), key=PURPOSES.index, reverse=True):
        x = replace_nth(x, t("[   ]"), PURPOSES.index(p), t(CHECK))

    # Part 2 cases: rebuild every case from the form's empty "사례2" block
    c1 = x.rfind("<hp:p ", 0, x.find(t("사례1 (예시 포함)")))
    c2 = x.rfind("<hp:p ", 0, x.find(t("사례2")))
    p3 = x.rfind("<hp:p ", 0, x.find("Part 3. 종합 고찰"))
    template_block = x[c2:p3]
    blocks = []
    for n, case in enumerate(spec["cases"], 1):
        b = template_block.replace(t("사례2"), t(f"사례{n}: {case['title']}"))
        b = fill_cell_after(b, "나의 질문(Prompt):", case["prompt"])
        b = fill_cell_after(b, "AI의 답변 요약:", case["answer"])
        b = fill_cell_after(b, "나의 검증 및 수정 과정", case["verify"])
        b = re.sub(r'<hp:tbl id="(\d+)"',
                   lambda m: f'<hp:tbl id="{int(m.group(1)) + 1000 * n}"', b)
        blocks.append(b)
    x = x[:c1] + "".join(blocks) + x[p3:]

    # Part 3: replace the empty lines under the guidance sentence
    g = x.find("(이번 챌린지에서AI를 활용해")
    s = x.find("</hp:p>", g) + len("</hp:p>")
    e = x.find("</hp:subList>", s)
    x = x[:s] + "".join(para(p, "28", "11") for p in spec.get("reflection", [])) \
        + para("", "28", "11") + x[e:]

    # cached line layout no longer matches the text; Hangul recomputes it
    x = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", x, flags=re.S)

    out = DOCS / spec["output"]
    with zipfile.ZipFile(out, "w") as z:
        for info, data in items:
            if info.filename == "Contents/section0.xml":
                data = x.encode("utf-8")
            comp = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            z.writestr(zipfile.ZipInfo(info.filename, date_time=info.date_time), data,
                       compress_type=comp)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print("written:", build(spec))
