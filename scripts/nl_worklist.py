r"""NL-GRAMMAR ① の worklist —— 次の一問を機械的に選ぶ。

`i18n/nl_patterns.json` のパターンを literal 単位に割り、**3つ以上のキーが
自前で書き下している語**を、指示コーパスの行数で並べる。

    .venv/bin/python scripts/nl_worklist.py [表示件数]

**fragment は展開しない。** fragment に入っている語は既に「1箇所」であり、
畳む対象ではない —— ここが探すのは、畳めるのに各キーが別々に書いている語である。
`frag` 欄の「済」は「その語を含む fragment は在るのに、このキーたちは自前で
書いている」という意味で、**むしろ強い信号**（畳み先が在るのに合流していない）。

**この道具は3回、自分の誤りで嘘をついた**（2026-08-27・2026-09-05 に実測）。
同じ穴を掘り直さないために書き残す:

1. `\(\?P?[<:=!#][^)]*\)?` で正規表現の構文を消していたら、**グループの中身
   （alternation そのもの）まで食べていた**。上位の並びが丸ごと変わった。
2. `_comment` 欄まで読んでいて、**散文の断片が literal として出た**（697件の
   大半がそれ）。読む欄は `ja` / `en` だけ。
3. コーパスの日本語欄は `ja` ではなく **`jp`**。取り違えると日本語の語が
   すべて「0行」と出て、**英語しか候補に上がらない**。
"""
import json, re, collections, sys

D = json.load(open("i18n/nl_patterns.json"))
FRAG = D.get("_fragments", {})
KEYS = {k: v for k, v in D.items() if not k.startswith("_")}

_SYNTAX = [
    r"\{\{[A-Za-z0-9_]+\}\}",
    r"\(\?P<[A-Za-z0-9_]+>",
    r"\(\?P=[A-Za-z0-9_]+\)",
    r"\(\?[:=!#i]",
    r"\(\?<[=!]",
    r"\\\\[A-Za-z]+",   # LaTeX コマンド。先に消さないと `\t` が食われ 'heta' が残る
    r"\\[A-Za-z]",      # \s \b \d \w ...
    r"\[[^\]]*\]",
    r"\{\d+(?:,\d*)?\}",
]
_SYNTAX_RE = re.compile("|".join(_SYNTAX))
_PUNCT_RE = re.compile(r"[()|?*+^$.\\/\-\s,:;=<>!&#%@~\"']")

_JA = re.compile(r"[ぁ-んァ-ン一-龥]")
_TOK = re.compile(r"[ぁ-んァ-ン一-龥ー]+|[A-Za-z]+")

def literals(pat: str):
    s = _PUNCT_RE.sub("\x00", _SYNTAX_RE.sub("\x00", pat))
    out = []
    for run in s.split("\x00"):
        for tok in _TOK.findall(run):
            if _JA.search(tok):
                if len(tok) >= 2:
                    out.append(tok)
            elif len(tok) >= 4:
                out.append(tok.lower())
    return out

def pats_of(v):
    """正規表現の欄だけ集める（`_comment` は散文なので除く）。"""
    acc = []
    def walk(x):
        if isinstance(x, str):
            acc.append(x)
        elif isinstance(x, list):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for kk, y in x.items():
                if not kk.startswith("_"):
                    walk(y)
    walk(v)
    return acc

lit2keys = collections.defaultdict(set)
for k, v in KEYS.items():
    seen = set()
    for p in pats_of(v):
        seen.update(literals(p))
    for lit in seen:
        lit2keys[lit].add(k)

frag_lits = set()
for fv in FRAG.values():
    for p in pats_of(fv):
        frag_lits.update(literals(p))

corpus = json.load(open("tests/corpus/instruction_corpus.json"))
rows = corpus["rows"] if isinstance(corpus, dict) else corpus
def row_texts(r):
    out = []
    for lang in ("jp", "ja", "en"):
        v = r.get(lang)
        if isinstance(v, str): out.append(v)
        elif isinstance(v, list): out.extend(x for x in v if isinstance(x, str))
        elif isinstance(v, dict):
            out.extend(x for x in v.values() if isinstance(x, str))
    return out
texts = [" \x01 ".join(row_texts(r)) for r in rows]
low = [t.lower() for t in texts]
print(f"corpus rows: {len(rows)}   keys: {len(KEYS)}   fragments: {len(FRAG)}")

res = []
for lit, ks in lit2keys.items():
    if len(ks) < 3:
        continue
    n = sum(1 for t, tl in zip(texts, low) if (lit in tl if lit.isascii() else lit in t))
    res.append((n, len(ks), lit, sorted(ks)))
res.sort(key=lambda x: (-x[0], -x[1]))
print(f"\n3キー以上で共有される literal: {len(res)}\n")
print(f"{'行':>4} {'鍵':>3} {'frag':>4}  語")
top = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for n, nk, lit, ks in res[:top]:
    mark = "済" if lit in frag_lits else "  "
    print(f"{n:>4} {nk:>3}  {mark}   {lit}   {ks[:5]}{'…' if len(ks)>5 else ''}")
