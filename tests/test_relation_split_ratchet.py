r"""Ratchet: raw `=` splitting, per file.

ONE-QUESTION-MANY-ANSWERS (docs/open_items.md) is one sentence — *the same
question is answered in several places* — and "where does this text split into a
relation" is its worst instance.  `shared_parser.find_top_level_relation` (with
`strip_lhs` / `split_relation` on top of it) is the single definition: bracket-
aware, `:=`/`==`-aware, and blind to whether the `=` was written with spaces.
Every `text.split("=")` is a second answer, and each one that has been measured
was measured wrong:

* `\sum_{i=1}^{3} i^2 = 14` reached Maxima as `1}^3*i^2 = 14`, whose scratch
  reply `_res_` was registered as the result (2026-08-11, three separate sites).
* `canonical_variables` reported the variables of `1}^N*a_i*x`, losing the dummy
  index the partial-vs-total test depends on (2026-08-11).
* `x + x = y` simplified published `x + x = 2 x=y`, because the "is this already
  an equation" test was spelled `" = "` while Maxima's `tex1` writes `2 x=y`
  (2026-08-13 — the one this ratchet was written with).

Two prior sweeps show why a count is the right instrument.  The first estimated
ten sites and found seventy-one; the second measured eighteen dynamically and
could say nothing about the other forty, because the taint that identified an
Expression string does not survive the f-string that rebuilds `LHS = RHS`.  A
static count survives both problems: it cannot be optimistic.

**The rule is the same as the two ratchets beside it** (`test_code_hygiene.py`,
`test_nl_pattern_ratchet.py`): a number may DROP freely (migrate a site to
`split_relation`/`strip_lhs` → lower it here), and may only RISE in a commit
that says why.  An unlisted file with any raw split fails outright, so a new one
cannot appear untracked — which is exactly how the count reached 71 unnoticed.

Not every site is a defect.  Some split CAS output (`parse_solve_result` reads
`[[x = 2]]`), some split a `key=value` token from a natural-language request,
some split a transform spec the researcher typed.  Those are not Expression text
and have no bracketed `=` to lose; they are counted anyway, because the point of
a ratchet is that nobody has to re-derive that judgement to notice growth.
"""
import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_DIRS = ("notebook", "ui")
_SPLIT_ATTRS = frozenset({"split", "rsplit", "partition", "rpartition",
                          "find", "index", "count"})

# --- Per-file ceilings.  First measured 2026-08-13 at 60 CALLS on 58 lines (AST —
# a docstring that describes the defect is not a site, which a grep cannot tell;
# and `canon.split("=",1)[0], canon.split("=",1)[1]` on one line is two answers,
# not one, which a per-line count cannot tell).  Now 53, after engine.py's seven
# moved to shared_parser the same day.
# Lower a number when a site migrates to shared_parser; never raise one silently.
CEILINGS = {
    # Six files reached ZERO on 2026-08-13 and are no longer listed:
    # engine_{finance,fourier,numeric}_mixin, maxima_{game,signal}_mixin,
    # spin_algebra.  All six asked ONE question — "strip the H-1 `LHS = `" —
    # and `_strip_h1_lhs` was defined twice, VERBATIM, with a comment calling
    # duplication of small helpers this codebase's precedent.  Both copies
    # answered `\sum_{i=1}^{3} c_i = u*x + v*y` with `1}^{3} c_i = u*x + v*y`.
    # They now call shared_parser.strip_lhs and the copies are deleted.

    # notebook/engine.py: ZERO since 2026-08-13 (was 7). It held the ONE correct
    # answer twice over while disagreeing with itself seven times; all seven now
    # ask shared_parser. Deliberately kept in the table AT ZERO — an entry that
    # reads 0 is a stronger statement than an absent one, and any regrowth is
    # caught by the ceiling rather than by the unlisted-file check.
    "notebook/engine.py": 0,
    # free_params: ZERO since 2026-08-13 (was 2). Both fed the CAS; one
    # registered `_res_ = _res_`, the adapter's own scratch name.
    # tests/test_mixin_relation_split.py
    "notebook/engine_free_params_mixin.py": 0,
    # laplace: ZERO since 2026-08-13 (was 2). The body lands inside
    # `integrate(body*exp(-s t))`; both sides of the linear form are registered.
    "notebook/engine_laplace_mixin.py": 0,
    # persistence: ZERO since 2026-08-13 (was 2). Load-time repair migrations,
    # gated on _WRONG_FUNC_RE too; no input was found that changes their
    # outcome, so this is the one definition rather than a measured fix.
    "notebook/engine_persistence_mixin.py": 0,    # load(): latex repair migrations
    # Backend handlers: a mix of Expression text (via expr_lookup) and CAS
    # OUTPUT (`parse_solve_result` on `[[x = 2, y = 1]]`).
    # julia_backend_handler.py: ZERO since 2026-08-13 (was 5). All five landed
    # in a Julia command string and four were broken; one interpolates into a
    # STRING LITERAL, where the fragment `\sum_{i` also made `\s` an invalid
    # escape. tests/test_julia_builder_relation_split.py
    "notebook/runtime/julia_backend_handler.py": 0,
    # maxima_backend_handler.py: ZERO since 2026-08-13 (was 6). All six landed in
    # CAS COMMAND TEXT and four were measurably broken; the sixth was safe only
    # because the lowering pass runs before it. tests/test_maxima_builder_relation_split.py
    "notebook/runtime/maxima_backend_handler.py": 0,
    # econ: ZERO since 2026-08-13 (was 2). Both built CAS text for the
    # Lagrangian. tests/test_lagrangian_bondsum_split.py
    "notebook/runtime/maxima_econ_mixin.py": 0,
    # NL request text, not Expression text: `key=value` field extraction.
    # KEPT AT 2 DELIBERATELY (reviewed 2026-08-13).  "where does a `key=value`
    # token split" is a DIFFERENT QUESTION from "where does a mathematical
    # relation split", and answering it with `split_relation` would be the very
    # conflation this ratchet exists to prevent.  Measured safe as written:
    # `partition` keeps everything after the FIRST `=` in the value (so
    # `upper=\sum_{i=1}^{3}` survives whole), the key is guarded by
    # `k in required_fields`, and the second site splits text a regex has already
    # bounded to `x=1`.  Migrating these would be motion, not repair.
    # MOVED 2026-08-15: both sites now live in token_field_mapping.py, which
    # the EN-ALIAS-MULTIWORD fix split out of the supervisor.  The reason is
    # unchanged and travels with them; the supervisor itself is now at ZERO.
    "notebook/runtime/llm_supervisor.py": 0,
    "notebook/runtime/token_field_mapping.py": 2,
    # Interceptors: raw researcher text before an Expression exists, plus
    # canonical/latex of referenced E-IDs.
    # interpret_early.py: ZERO since 2026-08-13 (was 3). One of them fed
    # spin_algebra's parser — CAS-bypass Python, so no backend stood between
    # the mangled string and the answer. tests/test_spin_early_relation_split.py
    "ui/interpret_early.py": 0,
    # interpret_equation_intercepts.py: ZERO since 2026-08-13 (was 4). Two of
    # the four were H-5z's registration halves — the only sites in this ratchet
    # whose mis-split was PERSISTED into the session file rather than used once
    # in a command. tests/test_rhs_set_relation_split.py
    "ui/interpret_equation_intercepts.py": 0,
    # interpret_f1: ZERO since 2026-08-13 (was 1). `_SYM_LHS_RE` accepts `\` and
    # `{`, so the naive split's `\sum_{i` looked like a simple name and this
    # route CLAIMED an input it should have declined.
    "ui/interpret_f1.py": 0,
    # interpret_integral: ZERO since 2026-08-13 (was 1). Named-vs-positional
    # transform spec; no realistic input separates the two readings, so it is
    # the one definition rather than a measured fix.
    "ui/interpret_integral.py": 0,
    # ui/interpret_l1.py: ZERO since 2026-08-13 (was 9). Counted first:
    # EIGHT of the nine fed a CAS argument (build_inverse_transform vars /
    # new_basis, substitute lhs+rhs, operator_substitute op_rules) and only
    # _forward_has_operators chose a prompt. Unlike engine.py this one had a
    # LIVE defect: tests/test_l1_relation_split.py.
    "ui/interpret_l1.py": 0,
    # interpret_pipeline.py: ZERO since 2026-08-13 (was 3). One registered what
    # it split (`E1 = 0`); the other two decide the differentiation variable and
    # the eq-diff dialog — their MATERIAL was wrong, no input was found that
    # flips the decision. tests/test_expr_id_equation.py
    "ui/interpret_pipeline.py": 0,
    # ui/interpret_rules.py: ZERO since 2026-08-13 (was 5). All five were in
    # _try_rule_dainyu and all five became substitute's lhs/rhs — and that lhs
    # is ALSO the timeline label, so a mis-split was published.
    # tests/test_dainyu_relation_split.py
    "ui/interpret_rules.py": 0,
}

TOTAL_CEILING = 2


def _sites_in_source(source):
    """``[(line, "split('=')"), …]`` for one module's text.

    AST rather than grep, deliberately: the two matches in `shared_parser.py` and
    `lhs_formatting.py` sit inside DOCSTRINGS that describe this very defect, and
    counting a warning as an instance of what it warns about is how a ratchet
    starts lying.
    """
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SPLIT_ATTRS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        sep = node.args[0].value
        if "=" not in sep or sep in ("==", ":="):
            continue
        out.append((node.lineno, f"{node.func.attr}({sep!r})"))
    return out


def _watched_files():
    """Every .py the ratchet is responsible for."""
    out = []
    for directory in WATCH_DIRS:
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, directory)):
            if "__pycache__" in root:
                continue
            out.extend(os.path.join(root, f) for f in sorted(files)
                       if f.endswith(".py"))
    return out


def _raw_split_sites():
    """``{repo-relative path: [(line, "split('=')"), …]}`` over the watched dirs."""
    found: dict[str, list[tuple[int, str]]] = {}
    for path in _watched_files():
        sites = _sites_in_source(open(path, encoding="utf-8").read())
        if sites:
            found[os.path.relpath(path, REPO_ROOT)] = sites
    return found


def test_no_unlisted_file_splits_on_equals():
    """A new raw split must be a visible decision, not a quiet addition."""
    sites = _raw_split_sites()
    unlisted = {f: v for f, v in sites.items() if f not in CEILINGS}
    assert not unlisted, (
        "these files split text on `=` without being in CEILINGS — use "
        "shared_parser.strip_lhs / split_relation, or add the file here with "
        "the reason:\n"
        + "\n".join(f"  {f}: {v}" for f, v in sorted(unlisted.items())))


def test_no_file_exceeds_its_ceiling():
    sites = _raw_split_sites()
    over = {f: (len(v), CEILINGS[f]) for f, v in sites.items()
            if f in CEILINGS and len(v) > CEILINGS[f]}
    assert not over, (
        "raw `=` splits grew; migrate to shared_parser or raise the ceiling "
        "here with a reason:\n"
        + "\n".join(f"  {f}: {n} > {c}" for f, (n, c) in sorted(over.items())))


def test_total_does_not_grow():
    total = sum(len(v) for v in _raw_split_sites().values())
    assert total <= TOTAL_CEILING, (
        f"{total} raw `=` splits, ceiling {TOTAL_CEILING}")


def test_the_scan_finds_the_known_sites():
    """Non-vacuity: a broken walk would make every assertion above pass empty.

    The anchor is a SYNTHETIC module, not a file that happens to still have a
    raw split today — this test named `notebook/engine.py` until 2026-08-13,
    when that file reached zero and the check that proves the scan works became
    the check that fails for succeeding.
    """
    assert [w for _, w in _sites_in_source(
        's.split("=", 1)\n'
        's.partition(" = ")\n'
        's.split("==")\n'          # equality TEST, not a relation split
        's.split(":=", 1)\n'       # a DEFINITION
        's.split(",")\n'
    )] == ["split('=')", "partition(' = ')"]
    # The floor is on FILES WALKED, not on sites found.  A floor on sites is
    # self-defeating — it fails for succeeding, which this test learned twice:
    # first it named `notebook/engine.py` (which reached zero), then it required
    # 30 sites (the count passed below that the same day).
    assert len(_watched_files()) > 100


def test_the_single_definition_is_not_itself_counted():
    """`shared_parser` and `lhs_formatting` describe this defect in prose.

    Both quote `t.split("=", 1)[1]` inside a docstring.  They must stay out of
    the count — otherwise removing the explanation would "improve" the ratchet.
    """
    sites = _raw_split_sites()
    assert "notebook/runtime/shared_parser.py" not in sites
    assert "notebook/runtime/lhs_formatting.py" not in sites


class TestTheReachabilityAssumption:
    r"""Why the 53 remaining sites are not currently wrong — pinned, because it
    is nowhere decided.

    engine.py's seven splits were migrated on 2026-08-13 and three of them fed
    the CAS.  For those three the old spelling could only go wrong on an
    expression whose LEFT side carries a bracketed `=` before the equation's
    own — and no such expression is ever registered as a symbol, so the defect
    had no reachable path.  That is not a rule anybody wrote: it falls out of
    `_extract_simple_symbol_name`'s regexes refusing every spelling containing
    `=`.  The same accident is what protects the 53 sites still splitting
    naively (`maxima_backend_handler._resolve_operand`, the `_*_source_body`
    helpers, `interpret_rules`' substitution path …), and those have NOT been
    migrated.

    **Nothing depends on this any more, as of the same day it was written.**
    All 53 of those sites were migrated to `shared_parser` on 2026-08-13, so the
    splits are bracket-aware and spacing-blind and the property could change
    tomorrow without breaking them.  The class is kept because it records a
    measured fact cheaply — but a future reader must not think anything is
    resting on it (open_items archive, SPLIT-REACH-ASSUMPTION).  If it fails
    because H-1 grew to accept a decorated indexed name, updating it is fine.

    The other half of the assumption — that nothing whose latex comes wholly
    from the CAS (Maxima's `tex1`, which writes `x=e^{-t\gamma}` unspaced) ever
    enters `lhs_alias_map` — is NOT pinned: asserting it needs an inventory of
    every path that creates an alias, which is that open item's work.
    """

    def test_no_lhs_spelling_containing_an_equals_becomes_a_symbol_name(self):
        from notebook.runtime.lhs_formatting import _extract_simple_symbol_name
        for spelling in (r"\sum_{i = 1}^{3} i^2", r"H_{n = 0}", r"\bar{H}_{i = 1}",
                         r"\hat{H}_{k = 0}", r"\frac{dH_{i = 1}}{dt}",
                         r"\sum_{i=1}^{3} i^2", "H + V"):
            assert _extract_simple_symbol_name(spelling) is None, spelling

    def test_the_spellings_that_DO_become_names_carry_no_equals(self):
        """The other side of it — otherwise the test above passes on a stub."""
        from notebook.runtime.lhs_formatting import _extract_simple_symbol_name
        for spelling, name in ((r"\bar{H}", "H_bar"), ("H_0", "H_0"),
                               (r"\langle ab \rangle", "ab_exp"),
                               (r"\frac{dH}{dt}", "H_dt")):
            assert _extract_simple_symbol_name(spelling) == name, spelling
            assert "=" not in spelling
