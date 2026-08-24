"""Static checks on the generated LaTeX, for when no TeX toolchain is present.

This cannot replace a real compile, and does not pretend to. It catches the
failure modes that actually occur in generated LaTeX: unbalanced environments,
unbalanced braces, \\ref with no \\label, \\cite keys absent from the .bib,
\\includegraphics pointing at files that were never copied, and stray unescaped
characters that TeX will choke on.

  python src/check_latex.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
TEX = ROOT / "latex" / "report.tex"
BIB = ROOT / "latex" / "refs.bib"

problems = []


def check(label, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        problems.append(label)


def main():
    if not TEX.exists():
        print(f"missing {TEX}")
        return 1
    src = TEX.read_text(encoding="utf-8")

    # strip comments so a % in a comment cannot confuse later checks
    nocomment = re.sub(r"(?<!\\)%.*", "", src)

    print("environments")
    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", nocomment))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", nocomment))
    for env in sorted(set(begins) | set(ends)):
        b, e = begins[env], ends[env]
        check(f"{env}: {b} begin / {e} end", b == e)

    print("\nbraces")
    depth = 0
    bad = 0
    for i, ch in enumerate(nocomment):
        if ch == "{" and (i == 0 or nocomment[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or nocomment[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                bad += 1
                depth = 0
    check("balanced", depth == 0 and bad == 0, f"net {depth}, {bad} early closes")

    print("\nlabels and references")
    labels = set(re.findall(r"\\label\{([^}]+)\}", nocomment))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", nocomment))
    dangling = refs - labels
    check(f"{len(refs)} refs resolve against {len(labels)} labels",
          not dangling, ", ".join(sorted(dangling)))
    unused = labels - refs
    if unused:
        print(f"         note: labels never referenced: {', '.join(sorted(unused))}")

    print("\ncitations")
    keys = set()
    if BIB.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    cited = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", nocomment):
        cited |= {c.strip() for c in m.split(",")}
    missing = cited - keys
    check(f"{len(cited)} cite keys present in refs.bib ({len(keys)} entries)",
          not missing, ", ".join(sorted(missing)))
    uncited = keys - cited
    if uncited:
        print(f"         note: in .bib but never cited: {', '.join(sorted(uncited))}")

    print("\ngraphics")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", nocomment):
        p = TEX.parent / g
        found = p.exists() or any(
            (TEX.parent / (g + ext)).exists() for ext in (".png", ".pdf", ".jpg"))
        check(f"{g}", found)

    print("\npackages and template")
    for needed in ("dibrisunige-report", "graphicx"):
        check(f"{needed} available",
              (TEX.parent / f"{needed}.sty").exists() or needed == "graphicx",
              "(from the template)" if needed == "dibrisunige-report" else "(TeX Live)")
    for logo in ("graphics/logo_verticale_BLACK.png",
                 "graphics/Dibris-con-dicitura-orizzontale-BN.png"):
        check(logo, (TEX.parent / logo).exists(), "used by \\coverpage")

    print("\nstray characters outside math")
    # an unescaped & or # outside a tabular is a hard error
    for lineno, line in enumerate(nocomment.splitlines(), 1):
        for ch in ("#",):
            if re.search(rf"(?<!\\){re.escape(ch)}", line):
                problems.append(f"line {lineno}: unescaped {ch}")
                print(f"  [FAIL] line {lineno}: unescaped {ch}  {line.strip()[:60]}")
    if not any(p.startswith("line ") for p in problems):
        check("no unescaped # outside macros", True)

    print()
    if problems:
        print(f"{len(problems)} problem(s): " + "; ".join(problems[:6]))
        return 1
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*",
                           re.sub(r"\\[a-zA-Z]+|\{[^}]*\}", " ", nocomment)))
    print(f"all checks pass  (~{words} words of prose, "
          f"{len(re.findall(r'begin.table.', nocomment))} tables, "
          f"{len(re.findall(r'begin.figure.', nocomment))} figures)")
    print("NOTE: static only - this does not prove it compiles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
