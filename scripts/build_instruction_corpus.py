#!/usr/bin/env python
"""Turn a measurement run into the committed instruction corpus.

    .venv/bin/python scripts/measure_instruction_corpus.py --out /tmp/m.json
    .venv/bin/python scripts/build_instruction_corpus.py --measurement /tmp/m.json

One row = one INSTRUCTION, in both languages, with the operation and operands it
reaches.  The spellings are lifted verbatim from ``docs/scenarios/*.md`` — they
are what a researcher actually typed, not spellings invented for a test (the
2026-08-22 lesson: the route was right and the INPUT was my own invention, so
three regressions walked past 5813 tests).

``status`` mirrors ``tests/corpus/notation_corpus.json``:

  ``pinned``  both languages reach the same operation with the same operands.
  ``gap``     they do not.  The divergence is recorded as data so that closing
              it FAILS the ratchet and forces this file to be updated — the same
              contract ``TestKnownGapsAreStillGaps`` has for notation.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "corpus" / "instruction_corpus.json"

#: Holds language-specific DISPLAY strings for the confirmation dialog, so it
#: differs between the two languages by design.  Comparing it would report 3
#: false divergences (measured 2026-08-23).
_NOISE_KEYS = ("candidates",)

_DOC = (
 "One INSTRUCTION per row, in both languages, with the operation and operands "
 "it reaches. Spellings are lifted verbatim from the ```scenario blocks in "
 "docs/scenarios/*_{jp,en}.md — real researcher input, not invented for a test. "
 "Regenerate with scripts/measure_instruction_corpus.py + "
 "scripts/build_instruction_corpus.py; the ratchet is "
 "tests/test_instruction_corpus.py. status=pinned means the two languages agree; "
 "status=gap records a measured divergence, and closing one must fail the "
 "ratchet so this file gets updated."
)


def _clean(args: dict) -> dict:
    return {k: v for k, v in (args or {}).items() if k not in _NOISE_KEYS}


def build(records: list) -> dict:
    by = collections.defaultdict(dict)
    for r in records:
        by[(r["scenario"], r["index"])][r["lang"]] = r
    rows = []
    for (scenario, index), v in sorted(by.items()):
        jp, en = v.get("jp"), v.get("en")
        row = {
            "id": f"{scenario.replace('scenario_', '')}:{index}",
            "scenario": scenario,
            "index": index,
            "kind": (jp or en)["kind"],
            "jp": jp["text"] if jp else None,
            "en": en["text"] if en else None,
        }
        if not (jp and en):
            row.update(status="gap", gap="unpaired",
                       operation=(jp or en)["operation"],
                       note="only one language has this line")
            rows.append(row)
            continue
        ja, ea = _clean(jp["args"]), _clean(en["args"])
        same_op = jp["operation"] == en["operation"]
        same_args = ja == ea
        # Compare the canonical AND the latex: agreeing on the value while
        # displaying it differently is a real divergence, and comparing only
        # the canonical hid one (see _created_exprs in the measurement script).
        same_created = ([c[1:] for c in jp["created"]]
                        == [c[1:] for c in en["created"]])
        # WHICH RULE claimed the line.  Pinning it is what makes the ~50 000
        # measured DECLINES a ratchet without listing them: every interceptor
        # ahead of the winner was offered this exact sentence and passed on it,
        # so a rule that loosens its premise and starts claiming a line it used
        # to decline shows up here as a changed route.  That is NL-GRAMMAR ②
        # ("a claim must carry its premise, and drop out when it fails") stated
        # as one field per row.
        same_route = (jp.get("route") == en.get("route")
                      and jp.get("routes") == en.get("routes"))
        row["operation"] = jp["operation"]
        # WHAT THE RESEARCHER SEES.  Operation, operands, created expressions and
        # route are all things a comparison can agree on while the SCREEN differs
        # — measured 2026-08-27, substitution answered identically in the two
        # languages and drew a labelled card in one and an unlabelled one in the
        # other, and every solve line drew an unlabelled card in BOTH.  Neither
        # was visible to any field above.  `True` means the operation's label
        # reached AT LEAST ONE of the cards this line draws — ANY, not ALL,
        # because an operation can answer with several expressions (solve
        # returns every root) and those later cards are further answers to the
        # same operation, not further operations.
        row["labelled_jp"] = any(bool(c) for c in jp["cards"]) if jp.get("cards") else None
        row["labelled_en"] = any(bool(c) for c in en["cards"]) if en.get("cards") else None
        row["route"] = jp.get("route")
        if jp.get("routes") and jp.get("routes") != [jp.get("route")]:
            row["routes"] = jp["routes"]
        # The screen is a comparison axis of its own now.  0 divergences the day
        # it was added (2026-08-27) — that is the point: it went in right after
        # the one it would have caught was fixed, so it starts as a ratchet
        # rather than a backlog.
        same_labelled = row["labelled_jp"] == row["labelled_en"]
        if same_op and same_args and same_created and same_route and same_labelled:
            row["status"] = "pinned"
            if ja:
                row["operands"] = ja
        else:
            row["status"] = "gap"
            row["gap"] = ("operation" if not same_op else
                          "operands" if not same_args else
                          "created_expressions" if not same_created else
                          "route" if not same_route else "card_label")
            if not same_route:
                row["route_jp"] = jp.get("route")
                row["route_en"] = en.get("route")
                row["routes_jp"] = jp.get("routes")
                row["routes_en"] = en.get("routes")
            if not same_op:
                row["operation_en"] = en["operation"]
            row["operands_jp"] = ja
            row["operands_en"] = ea
            if not same_created:
                row["created_jp"] = jp["created"]
                row["created_en"] = en["created"]
        rows.append(row)
    return {"_doc": _DOC, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    data = build(json.loads(a.measurement.read_text(encoding="utf-8")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8")
    c = collections.Counter(r["status"] for r in data["rows"])
    g = collections.Counter(r.get("gap") for r in data["rows"] if r["status"] == "gap")
    print(f"--- {len(data['rows'])} row(s) -> {a.out}")
    print(f"    {dict(c)}   gaps by kind: {dict(g)}")
