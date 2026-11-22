#!/usr/bin/env python3
"""Build the public GitHub Pages copy of the Korea datacenter tracker.

The project copy (korea_datacenter_tracker.html) is the internal working file:
it keeps builder/cooling fields, the data-completeness score and long research
notes. The public page drops those and shows short, reader-friendly notes.

Usage:
    python3 _tools/build_public_tracker.py <internal.html> datacenter-tracker/index.html

Public notes live in _tools/public_notes.json ({site id: one or two sentences}).
The build fails if any site id has no public note, and lists the missing ids,
so new sites always get a reviewed summary before they are published.
Pass --allow-missing to publish missing ones with an empty note instead.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
NOTES_PATH = HERE / "public_notes.json"
JS_STR = r'"(?:[^"\\]|\\.)*"'


def replace_once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[build] expected exactly 1 match for {label}, found {n}")
    return s.replace(old, new)


def sub_once(s, pattern, repl, label, flags=0):
    out, n = re.subn(pattern, repl, s, flags=flags)
    if n != 1:
        raise SystemExit(f"[build] expected exactly 1 match for {label}, found {n}")
    return out


INTERNAL_WORDS = re.compile(
    r"재확인|확인 필요|다만|미확인|미공개|추정|보도마다|상이|혼동|참고|기준으로|보도 기준|별도 항목|별개|좌표|시공|냉각|정정|오기")
CLEAN_FIELDS = ("operator", "region", "power", "scale", "floors", "start", "complete", "pue")


def _strip_internal_parens(text):
    """Drop (...) groups whose content reads like an internal research remark."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "(":
            depth, j = 0, i
            while j < len(text):
                depth += text[j] == "("
                depth -= text[j] == ")"
                if depth == 0:
                    break
                j += 1
            inner = text[i + 1:j]
            if j < len(text) and INTERNAL_WORDS.search(inner):
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _split_top(text, sep):
    """Split on sep only outside parentheses."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def clean_field(key, text):
    if text.startswith("확인 안 됨"):
        return "확인 안 됨"
    t = text
    if key == "operator":
        # remove items naming the builder or cooling vendor, e.g. "·신세계건설(시공)", ", 시공 현대건설"
        t = re.sub(r"[·,]?\s*[^·,()]+\((?:출자 및 )?시공[^)]*\)", "", t)
        t = re.sub(r",\s*시공\s+[^,·()]+", "", t)
        t = re.sub(r",?\s*[^,()]+\s+(?:시공|침지냉각)(?=,|$)", "", t)
    t = _strip_internal_parens(t)
    # top-level " — ..." tails that explain sourcing or uncertainty
    head, *tails = _split_top(t, "—")
    kept = [head] + [x for x in tails if not INTERNAL_WORDS.search(x)]
    t = "—".join(kept) if len(kept) > 1 else head
    if key == "operator":
        # anything still mentioning them: drop that top-level comma part
        t = ",".join(p for p in _split_top(t, ",") if not re.search(r"시공|냉각", p))
    t = re.sub(r"\s{2,}", " ", t).strip(" ,·;—")
    return t or text


def js_string(text):
    return json.dumps(text, ensure_ascii=False).replace("</", "<\\/")


def build(src, notes, allow_missing):
    # ---- site data: drop builder/cooling, swap in public notes ----
    start = src.index("const sites = [")
    end = src.index("\n];", start)
    head, body, tail = src[:start], src[start:end], src[end:]

    missing, count, out_lines = [], 0, []
    for line in body.split("\n"):
        m = re.match(r'\s*\{\s*id:"([^"]+)"', line)
        if m:
            sid = m.group(1)
            count += 1
            line, nb = re.subn(r"\bbuilder:" + JS_STR + r",\s*", "", line)
            line, nc = re.subn(r"\bcooling:" + JS_STR + r",\s*", "", line)
            if nb != 1 or nc != 1:
                raise SystemExit(f"[build] {sid}: builder/cooling field not found exactly once")
            if sid in notes:
                note = notes[sid]
            else:
                missing.append(sid)
                note = ""
            line, nn = re.subn(r"\bnote:" + JS_STR, lambda _m: "note:" + js_string(note), line)
            if nn != 1:
                raise SystemExit(f"[build] {sid}: note field not found exactly once")
            line = re.sub(r"\{name:(" + JS_STR + r"),",
                          lambda m: "{name:" + js_string(_strip_internal_parens(json.loads(m.group(1))).strip()) + ",",
                          line)
            for key in CLEAN_FIELDS:
                line = re.sub(r"\b" + key + r":(" + JS_STR + ")",
                              lambda m, k=key: k + ":" + js_string(clean_field(k, json.loads(m.group(1)))),
                              line, count=1)
        out_lines.append(line)
    if missing and not allow_missing:
        raise SystemExit("[build] public note missing for: " + ", ".join(missing)
                         + "\nAdd them to _tools/public_notes.json and run again.")
    s = head + "\n".join(out_lines) + tail

    # ---- table: remove builder / cooling / completeness columns ----
    s = replace_once(s, '        <th data-key="builder">시공사</th>\n', "", "th builder")
    s = replace_once(s, '        <th data-key="cooling">냉각방식</th>\n', "", "th cooling")
    s = replace_once(s, '        <th data-key="completenessScore">정보완성도</th>\n', "", "th completeness")
    s = replace_once(s, "      <td class=\"muted\">${s.builder||'확인 안 됨'}</td>\n", "", "td builder")
    s = replace_once(s, "      <td class=\"muted\">${s.cooling||'확인 안 됨'}</td>\n", "", "td cooling")
    s = replace_once(s, "      <td>${completenessBadge(s)}</td>\n", "", "td completeness")

    # ---- detail panel ----
    s = replace_once(s, "    ${completenessBadge(s)}\n    <dl>", "    <dl>", "detail badge")
    s = replace_once(s, "      <dt>시공사</dt><dd>${fieldOrUnknown(s.builder)}</dd>\n", "", "detail builder")
    s = replace_once(s, "      <dt>냉각방식</dt><dd>${fieldOrUnknown(s.cooling)}</dd>\n", "", "detail cooling")

    # ---- '정보 부족' tile, filter chip, region-view count, legend ----
    s = sub_once(s, r"\n\s*<div class=\"stat gap\"><b>\$\{gapCount\}</b>[^<]*</div>", "", "gap stat tile")
    s = sub_once(s, r"\n\s*<span class=\"chip\" data-filter=\"gap\" id=\"gapFilterChip\">[^<]*</span>\n"
                    r"\s*<span style=\"width:1px;height:20px;background:var\(--border\);margin:0 2px;\"></span>",
                 "", "gap chip")
    s = sub_once(s, r"document\.getElementById\('gapFilterChip'\)\.addEventListener\('click', function\(\)\{.*?\n\}\);\n",
                 "", "gap chip listener", flags=re.S)
    s = sub_once(s, r"\n\s*\$\{gapN>0\?`<span style=\"color:var\(--gap\)\">정보부족 \$\{gapN\}</span>`:''\}", "",
                 "region gap count")
    s = sub_once(s, r"<span class=\"muted\">정보완성도\([^\n]*?</span>",
                 '<span class="muted">"일정 보기"에서 착공·완공 전망을 확인할 수 있습니다</span>', "legend")

    s = replace_once(s, "'사이트 목록 (헤더 클릭 시 정렬 · 정보완성도 컬럼 클릭 시 미확인 필드 많은 순 정렬)'",
                     "'사이트 목록 (헤더 클릭 시 정렬)'", "table caption")

    # ---- notes: show the public note as one paragraph ----
    s = replace_once(s, "function renderNoteHtml(note){\n",
                     "function renderNoteHtml(note){\n"
                     "  return note ? `<p class=\"note-base\">${escapeHtml(note)}</p>` : '<p class=\"note-base unknown-val\">-</p>';\n",
                     "renderNoteHtml")

    # ---- footer: last update date only, no internal changelog ----
    m = re.search(r"마지막 갱신: (\d{4}-\d{2}-\d{2})", s)
    if not m:
        raise SystemExit("[build] could not find the last update date in genInfo")
    last = m.group(1)
    s = sub_once(s, r"document\.getElementById\('genInfo'\)\.textContent = `.*?`;\n?", "", "genInfo", flags=re.S)
    s = sub_once(s, r"document\.getElementById\('genLatestLine'\)\.textContent = '(?:[^'\\]|\\.)*';",
                 f"document.getElementById('genLatestLine').textContent = '{last}';", "genLatestLine")
    s = sub_once(s, r"\n\s*<details id=\"changelogDetails\">.*?</details>", "", "changelog details", flags=re.S)

    return s, count, missing, last


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    src = pathlib.Path(args[0]).read_text(encoding="utf-8")
    notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    out, count, missing, last = build(src, notes, "--allow-missing" in argv)
    pathlib.Path(args[1]).write_text(out, encoding="utf-8")
    print(f"[build] {count} sites, last update {last}, missing notes: {len(missing)}")


if __name__ == "__main__":
    main(sys.argv)
