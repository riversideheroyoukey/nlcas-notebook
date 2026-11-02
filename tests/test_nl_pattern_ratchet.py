"""NL-pattern externalization ratchet (docs/27 §所見G, docs/28 §5 — Step 4).

The third audit found NL trigger regexes written with hardcoded Japanese
literals in ``ui/interpret_*.py`` instead of going through the
``i18n/nl_patterns.json`` loader (``nlp(key, lang)``).  The (a) real gaps were
fixed and the (b) true duplicates were migrated (docs/28 §2); the remaining
(c)/(d)/(n) sites plus ``interpret_f1.py``'s 3 (b) sites were deliberately
DEFERRED under the policy "freeze the count, migrate when the file is touched
anyway" (docs/28 §5.1).

This test IS that freeze.  Each ``ui/interpret_*.py`` file has a ceiling on how
many hardcoded-Japanese regex lines it may contain.  Exceeding one FAILS — the
two valid responses mirror the R0 hygiene ratchet:

  1. Migrate the offending pattern to ``i18n/nl_patterns.json`` and call it via
     ``nlp(key, lang)`` (preferred — the count drops, LOWER the ceiling here), or
  2. If a new hardcoded pattern is genuinely unavoidable, raise the ceiling in a
     commit that says why.

New NL patterns should be written through ``nl_patterns.json`` from the start
(see the ``feedback_nl_input_i18n`` memory).  A new ``ui/interpret_*.py`` file
with hardcoded JA regexes that isn't listed below also fails, so growth can't
sneak in untracked.

Counting method is the reproducible physical-line count from docs/28 §5.2
(NOT the coarser logical-pattern count in §5.3).  Measured 2026-07-19.
"""
import glob
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Japanese code points (hiragana/katakana/CJK ideographs) — a raw JA char in a
# regex literal means the pattern is hardcoded, not going through nlp().
_JP_RE = re.compile(r"[぀-ヿ一-鿿]")

# Signals that a line carries a regex literal (docs/28 §5.2).
_REGEX_MARKERS = (
    "re.compile", "re.search", "re.match", "re.sub", "re.findall",
)


def _docstring_lines(rel):
    """Every line number occupied by a DOCSTRING in *rel*.

    A Japanese docstring is prose, not an NL trigger pattern — but this file's
    "is it a regex?" test accepts any line starting with ``r"``, and a raw
    docstring (``r\"\"\"``) starts exactly that way.  Measured 2026-08-23, that
    made the count meaningless for the files with the most Japanese prose:
    ``spin_value_set.py`` and ``operand_boundary.py`` scored 8 and 5 with **no
    NL pattern in either**.

    Read from the AST rather than by spotting ``r\"\"\"``, so the exclusion cannot
    quietly drop a triple-quoted REGEX; checked when this landed — every
    r-triple-quoted line in the counted files is a real docstring.
    """
    import ast
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = set()
    for node in ast.walk(tree):
        # Any bare string EXPRESSION is documentation — the module/def/class
        # docstring, and equally the "attribute docstring" convention (a string
        # written under an assignment), which is how `symbol_context.py`
        # documents its fields.  Restricting this to `body[0]` counted those as
        # patterns (measured 2026-08-23).
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def _strip_trailing_comment(line):
    r"""The code part of *line* — a Japanese TRAILING COMMENT is not a pattern.

    ``r"(?:\\[A-Za-z]+\\s*)?"  # \hat, \vec …（省略可）`` is an ASCII regex
    with a Japanese note beside it; counting it said five of
    ``backend_handler.py``'s hits were hardcoded Japanese when none of them were.
    """
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _hardcoded_ja_regex_count(rel):
    """Count physical lines in ``rel`` that hold a hardcoded-JA regex literal.

    Follows docs/28 §5.2, with two exclusions added 2026-08-23 after measuring
    what the original spelling actually counted: DOCSTRINGS and TRAILING
    COMMENTS.  Both are prose, and counting them made the ceilings noise — a
    file could approach its ceiling by gaining Japanese *documentation*, which
    is the opposite of what this ratchet is for.  Measured impact on the
    existing ceilings: two files drop (``interpret_pipeline`` 6→3,
    ``interpret_equation_intercepts`` 2→1), both still under, and no file rises.
    """
    docstrings = _docstring_lines(rel)
    hits = 0
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if s.startswith("#") or lineno in docstrings:
                continue
            if not _JP_RE.search(_strip_trailing_comment(line)):
                continue
            has_regex = (
                any(m in line for m in _REGEX_MARKERS)
                or ("_RE" in line and ('r"' in line or "r'" in line))
                or s.startswith('r"')
                or s.startswith("r'")
            )
            if has_regex:
                hits += 1
    return hits


# --- Per-file ceilings, pinned at the 2026-07-19 measured value (docs/28 §5.2).
# LOWER a ceiling whenever you migrate one of that file's patterns to
# nl_patterns.json.  interpret_f1.py's 60 includes its 3 deferred (b) sites,
# frozen here because migrating only them would conflict with its lang-less
# merged-pattern design (docs/28 §4).
NL_HARDCODED_JA_CEILING = {
    "ui/interpret_f1.py": 1,       # f1 fully externalized (60→1): dispersion/spin/graph/energy/neighbour-sum/bond grammars → nl_patterns.json. The 1 residual is the content-based の-particle strip in _parse_neighbor_sum body extraction (docs/28 §4 content-based exception, like l1 old/new-basis)
    "ui/interpret_rules.py": 0,    # stage 5: combine / component-substitute / E→E dainyu / diff-order → nl_patterns.json (stages 2-3: notation kw, wrt-q̇)
    "ui/interpret_l1.py": 4,       # stage 1: canon/verify/apply/metric-constraint + transform-subst → nl_patterns.json; the 4 left are the content-based old/new-basis matrix path (docs/28 §4 exception)
    "ui/interpret_pipeline.py": 8,  # stage 2: imperative name_assign → nl_patterns.json
    "ui/interpret_plan.py": 0,     # stage 4: eigensystem/heisenberg/site/var/candidates/freeform/auto-select → nl_patterns.json
    "ui/interpret_common.py": 2,   # (n) infrastructure: variable-separator + CJK-detection utilities, not NL trigger patterns — permanently exempt
    "ui/interpret_equation_intercepts.py": 2,  # (n) infrastructure: fullwidth-digit conversion + kanji-numeral parse, not NL trigger patterns — permanently exempt
    "ui/interpret_integral.py": 0,  # stage 3: single-variable definite integral → nl_patterns.json

    # --- Brought inside the ratchet 2026-08-23, frozen at the measured value.
    # Policy is docs/28 §5.1's, unchanged: freeze the count, migrate when the
    # file is touched anyway.  These carry the NL grammar the interceptors do
    # NOT hold — declaration sentences, the operator-algebra substitutions, the
    # verb suffixes — and none of it was counted before.
    "notebook/runtime/llm_supervisor.py": 2,           # **LOWERED 25→2 (2026-08-27)**: 動詞語尾 `verb_tail_strip` / 序数接頭 `ordinal_prefix_strip` / 微分変数 `diff_var_particle` / 数の領域と符号8キー / ∇族4パターン（fragment `vec_op_kw`・`vec_op_kw_chain`）/ 曖昧語4キー を nl_patterns.json へ（19キー＋fragment 2）。**移す前に1行ずつ「引き金か基盤か」を分類した**——線は「操作が増えると増える語彙か、言語の機能語・活用か」で、新しい操作は動詞（抽出して）・フィールドの語（次）・型の語（実数）・演算子名（発散）を増やすが、助詞も する の活用も増やさない。残る2はその基盤側（`_JP_PARTICLES_STRIP` の助詞集合と `_JP_COMMAND_SUFFIX` の して/する/しろ/せよ/しなさい）で、`interpret_common._VARS_SEP_RE` と同じ理由で恒久免除。実在760綴り＋トークン979＋probe 68 の 1807 入力で新旧一致（span・全グループまで比較）、コーパス523行で差分0
    "notebook/runtime/julia_backend_handler.py": 1,     # **LOWERED 23→2 (2026-08-23)**: Heisenberg 方程式・演算子代入の6綴り・指標の付け替え・同類項まとめを nl_patterns.json へ（11キー＋fragment `julia_op_call`——2026-09-11 に `operator_operand` へ改名して広げた）。呼び出し側の**試す順序はそのまま**で、パターンの出所だけを外に出した——順序は構造であって言語の内容ではない。移す前に AST で14個の regex を厳密に取り出し、実在760綴りで**マッチ結果もグループも**新旧一致することを確認。残る2は `フェルミ|fermion` の分類（両言語を見る guard）と、`_normalize_latex_subscripts` の先読み `[にをで\\s]`（NL の引き金ではなく文字変換の境界）
    "notebook/runtime/backend_handler.py": 1,          # **LOWERED 13→1 (2026-08-23)**: 宣言文の文法（`decl_stmt` / `decl_stmt_wa` / `decl_suffix` / 型語彙5キー / `redeclare` / `time_dependent` / `spin_operator_decl`）を nl_patterns.json へ。**呼び出し側は言語を絞らず両方を見る**（`_nl_any`）——ここは読みではなく「宣言かどうか」「どの型か」を訊く側で、外すと define_scalar_parameter に落ちて silent-wrong になる。移す前にコーパスの実在760綴りで12パターンとも新旧が一致することを確認。残る1は CJK 判定のユーティリティ（基盤）
    "notebook/runtime/julia_handler_common.py": 0,     # **LOWERED 1→0 (2026-08-27)**: 残っていたスピンの型語（`スピン演算子|パウリ演算子|パウリ行列`）を `spin_def_op_text` へ。訊くのは唯一の呼び出し側（`julia_backend_handler.should_bypass_auto_register_formula`）で、**guard なので両言語を無条件に見る**（`nlp_any`）。`spin_operator_decl` はこの真の上位集合（裸の「パウリ」等も持つ）だが、畳むと guard が広がるので**畳んでいない**——実測して別に決める一問。/ 以前: **LOWERED 7→1 (2026-08-23)**: the particle-operator declaration grammar moved to nl_patterns.json (`particle_operator_decl`, `bare_quantum_op_decl`, `particle_def_op_text`), and with it four try-in-order branches collapsed into one named-group lookup. Checked first over the 14 real spellings the instruction corpus holds: the new patterns read the same (op, particle, kind). The 1 left is the spin type-word list, whose reading lives in backend_handler
    "ui/supervisor_alias_rewrite.py": 0,               # **LOWERED 6→0 (2026-08-27)**: 「この記号は微分の変数か、H-9 の解く対象か」——2a-sym はこれを**同一のバイト列で2度**書いていた（35行離れた lookahead と全文検査）。語は fragment `ja_op_var_tail` / `en_op_var_head` へ、**係り方（前か後ろか、否定先読みか全文検索か）はコードに残す**——語順は言語、係り方は構造である。実行時に研究者自身の記号名を挟み込むので `nlp()` ではなく新しい `nl_patterns.fragment()` を使う。コーパス760綴り×439記号の**341,542対**で新旧一致（先読みは全位置で比較）、523行で差分0。英語の固定長 lookbehind だけは `re` の制約で fragment を使えず手で同期する（第二の防壁として意図的に残置——理由はコード註）
    "notebook/runtime/token_field_mapping.py": 1,      # **LOWERED 2→1 (2026-08-27)**: `^(\d+)次` の order 予備読みを `order_suffix_token` へ。残る1は CJK 判定のユーティリティ（基盤・恒久免除）。この予備は order 一族で**いちばん狭い綴り**で、`4階まで`・`四次まで` は落ちる——揃えるかは ① の一問（理由は JSON の註）
    "ui/routes/export.py": 2,                          # (n) infrastructure: CJK detection for the TeX preamble + a bullet-character regex — not NL triggers
    "ui/debug_repl.py": 1,                             # (n) not a regex at all: a Japanese TeX comment line in generated output, caught by the r-string rule
    "ui/timeline_serializer.py": 0,                    # **LOWERED 1→0 (2026-08-27)**: `^E\d+ (に対して|について|の|…)` を `expr_ref_prefix` へ。この接頭辞は研究者が打つものではなく `interpret_pipeline` が**セッション言語に関わらず日本語で**付けるので、剥がす側に英語の兄弟が無いのは抜けではない（理由は JSON の註）
}


#: Directories whose sources may carry NL trigger patterns.  Discovery used to
#: be `ui/interpret_*.py` alone, which is where the interceptors live — but the
#: BACKEND HANDLERS read declaration sentences too, and NL-GRAMMAR was raised
#: partly because two silent-wrongs came from exactly there (2026-08-22).
#: Measured 2026-08-23 when the glob was widened: **80 hardcoded-JA lines across
#: 9 files were outside the ratchet**, against 11 inside it — the two largest
#: (`llm_supervisor.py` 25, `julia_backend_handler.py` 23) being bigger than any
#: file the ratchet had ever tracked.
_SOURCE_DIRS = ("ui", "notebook")


def _interpret_files():
    for d in _SOURCE_DIRS:
        for full in sorted(glob.glob(os.path.join(REPO_ROOT, d, "**", "*.py"),
                                     recursive=True)):
            rel = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
            if "/__pycache__/" in rel:
                continue
            yield rel


def test_nl_patterns_within_ceilings():
    over = []
    for rel, ceiling in NL_HARDCODED_JA_CEILING.items():
        n = _hardcoded_ja_regex_count(rel)
        if n > ceiling:
            over.append(f"{rel}: {n} > {ceiling}")
    assert not over, (
        "Hardcoded-Japanese NL regex count grew past its ceiling. Migrate the "
        "new pattern to i18n/nl_patterns.json (call it via nlp(key, lang)), or "
        "raise the ceiling in tests/test_nl_pattern_ratchet.py with a reason:\n  "
        + "\n  ".join(over))


def test_no_untracked_hardcoded_nl_file():
    """A source file with hardcoded-JA regexes but no ceiling entry is untracked
    growth — list it (with a ceiling) or externalize its patterns."""
    untracked = []
    for rel in _interpret_files():
        n = _hardcoded_ja_regex_count(rel)
        if n > 0 and rel not in NL_HARDCODED_JA_CEILING:
            untracked.append(f"{rel}: {n}")
    assert not untracked, (
        "New file(s) with hardcoded-JA NL regexes not tracked by the ratchet — "
        "add to NL_HARDCODED_JA_CEILING or externalize to nl_patterns.json:\n  "
        + "\n  ".join(untracked))
