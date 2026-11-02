"""Every name the product LOADS must be bound somewhere in its module.

ONE-QUESTION-MANY-ANSWERS 残作業 2 (docs/open_items.md): the corpus (第1期) and
the route attribution (第2期) catch the "one notation, two answers" type — proven
by mutation.  Neither catches the type the worker introduces WHILE editing: both
slips made during that work (a missing ``UnknownNotationError`` import, a dropped
parenthesis in the ``\\partial_x`` rewrite) were found by probe and grep, not by a
test.  This file is the mechanism for the first half of that class — a name used
but bound nowhere resolves to ``NameError`` at runtime, and a static scan sees it
without executing the path.

**It found two live ones on the day it was written (2026-08-13), both left by
the very refactors this project calls its structural work:**

* ``julia_sympy_mixin`` was split out of ``julia_backend_handler`` (AUD-4) and
  kept a ``_logger.debug`` call whose logger stayed behind, so the "best-effort,
  keep the original" fallback in ``prepare_sympy_wick_inputs`` raised
  ``NameError`` instead of falling back.  The module's own docstring asserted it
  referenced no module-level state — the assertion no code checked.
* ``semantic_resolver._resolve_subscript`` kept ten lines of the pre-
  SUBSCRIPT-POLICY-SPLIT implementation AFTER the new ``raise``, referring to a
  local (``_sub_name``) that the refactor had deleted.  Unreachable, so nothing
  failed; a reader has no way to tell dead code from a live branch.

**It was deliberately CONSERVATIVE — file-level, not scope-accurate — and that
cost it the very class it exists for (FUNC-LOCAL-IMPORT-SCOPE, fixed
2026-09-10).**  Function-local imports are the norm in this codebase, and a name
imported inside one function was counted as bound for the whole module: when 24
lines were extracted from ``backend_handler.extra_pre_phase1`` into a new method,
``InterpretResult`` raised ``NameError`` at runtime while this scan passed all
170 files.  The reading is now scope-accurate
(``scripts/scope_name_resolution.py``), which is strictly stronger — a name bound
nowhere resolves in no scope chain either.

**Measured before adopting it**, because the ticket asked whether today's zero was
a hole or the truth: the scope-accurate scan reports **0 on the 170 product
files**.  It was the CHECK that had the hole, not the code.  The instrument was
calibrated first (4 defect shapes reported, 12 legal shapes silent — including
the three the ticket predicted would trip it: ``global``/``nonlocal``,
comprehensions, and ``try/except ImportError`` aliases); the details are in that
script's docstring.

What it never does is report a name that resolves: a hit is always a real one,
which is why the bar is zero rather than a ratchet.

Annotations are skipped only when the module has ``from __future__ import
annotations`` (they are strings then).  Without it an unresolvable annotation is
evaluated at def time and IS a live error, so it is reported.
"""
import ast
import builtins
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_DIRS = ("notebook", "ui")
WATCH_FILES = ("cli.py",)

_SAFE = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__debug__", "__builtins__", "__path__",
}


#: 解決は `scripts/scope_name_resolution.py` に1つだけ置く——**人が走らせられる
#: 測定器**であり、`nl_spelling_coverage.py` と同じ形（道具は scripts、規則はここ）。
from scripts.scope_name_resolution import scan as unresolved_names  # noqa: E402


def _product_files():
    paths = [os.path.join(REPO_ROOT, f) for f in WATCH_FILES]
    for directory in WATCH_DIRS:
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, directory)):
            if "__pycache__" in root:
                continue
            paths.extend(os.path.join(root, f) for f in files if f.endswith(".py"))
    return sorted(paths)


@pytest.mark.parametrize("path", _product_files(),
                         ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_no_unresolved_names(path):
    hits = unresolved_names(open(path, encoding="utf-8").read())
    rel = os.path.relpath(path, REPO_ROOT)
    assert not hits, "\n".join(
        f"{rel}:{ln}  name {nm!r} is used but bound nowhere in this module "
        f"(NameError when this line runs)" for ln, nm in hits)


def test_the_scan_itself_catches_the_two_shapes_it_exists_for():
    """A silent checker is worse than none — pin that it still reports.

    The two sources are the two live defects of 2026-08-13, reduced: a logger
    left behind by a mixin extraction, and a leftover branch referring to a
    local the refactor deleted.
    """
    extracted_mixin = (
        "class M:\n"
        "    def f(self):\n"
        "        try:\n"
        "            g()\n"
        "        except Exception:\n"
        "            _logger.debug('kept the original')\n"
        "    def g(self):\n"
        "        return 1\n"
    )
    leftover_branch = (
        "def f(expr):\n"
        "    raise UnknownNotationError('called_subscript')\n"
        "    if _sub_name is None:\n"
        "        return None\n"
    )
    # `g` も報告される——**メソッドから兄弟メソッドを素の名前では呼べない**
    # （`self.g()` が要る）ので、これは本物の `NameError` である。
    # 期待値が `["_logger"]` だけだったのは、旧いファイル単位の走査が
    # クラスのスコープを見分けなかったから（2026-09-10、スコープ版に移行）。
    assert [nm for _, nm in unresolved_names(extracted_mixin)] == ["g", "_logger"]
    assert sorted(nm for _, nm in unresolved_names(leftover_branch)) == [
        "UnknownNotationError", "_sub_name"]


def test_lazy_annotations_are_not_reported_but_evaluated_ones_are():
    """The one distinction that decides whether an annotation can fail.

    ``def load() -> NotebookEngine`` appears three times in this product with no
    import of that name, and all three are safe — the modules carry
    ``from __future__ import annotations``, so the annotation is never
    evaluated.  Reporting them would make the scan noise; skipping the same
    shape WITHOUT that import would miss a real def-time NameError.
    """
    body = "def load(x: Missing) -> Other:\n    return x\n"
    assert unresolved_names("from __future__ import annotations\n" + body) == []
    assert sorted(nm for _, nm in unresolved_names(body)) == ["Missing", "Other"]


#: **報告しなければならない形。** 先頭は起票 FUNC-LOCAL-IMPORT-SCOPE そのもの
#: ——関数内 import を別のメソッドが使う。これを黙って通したせいで、
#: `backend_handler` から切り出した24行が実行時 `NameError` を出しながら
#: 検査は170件通過した（2026-09-06）。
MUST_REPORT = [
    ("関数内 import を別のメソッドが使う", "InterpretResult", """
class H:
    def a(self, t):
        from notebook.runtime.models import InterpretResult
        return InterpretResult(t)
    def b(self, t):
        return InterpretResult(t)
"""),
    ("関数内の代入を別の関数が読む", "helper", """
def a():
    helper = 1
    return helper
def b():
    return helper
"""),
    ("クラス属性をメソッドが素の名前で読む", "TABLE", """
class C:
    TABLE = {1: 2}
    def f(self):
        return TABLE
"""),
    ("内包表記の変数を外で読む", "y", """
def f(xs):
    ys = [y for y in xs]
    return y
"""),
    # 既定値とデコレータは **def 実行時に外側のスコープで**評価されるので、
    # 解決しなければ本物の def 時 NameError である。訪ねるのを止めても
    # 何も落ちなかった（変異試験、2026-09-10）ので足した。
    ("既定値に未束縛の名前", "Missing", "def f(x=Missing):\n    return x\n"),
    ("キーワード既定値に未束縛の名前", "Missing", "def f(*, x=Missing):\n    return x\n"),
    ("デコレータに未束縛の名前", "Missing", "@Missing\ndef f():\n    return 1\n"),
    ("注釈に未束縛の名前（遅延評価なし）", "Missing", "def f(x: Missing):\n    return x\n"),
]

#: **沈黙しなければならない形。** 起票が「踏むだろう」と名指した3つ
#: （`global`/`nonlocal`・内包表記・`try/except ImportError` の別名）は
#: ここに入っている——**予測された偽陽性を、採用前に測って潰してある**。
MUST_BE_SILENT = [
    # **モジュール側に代入を置かない**——置くと `global` を無視しても通ってしまい、
    # 何も測っていないことになる（変異試験で生き残って分かった、2026-09-10）。
    ("global", "def s():\n    global _c\n    _c = 1\ndef g():\n    return _c\n"),
    ("nonlocal", "def o():\n    n = 0\n    def i():\n        nonlocal n\n        n += 1\n        return n\n    return i\n"),
    ("try/except ImportError の別名",
     "try:\n    import ujson as json\nexcept ImportError:\n    import json\ndef f():\n    return json.dumps({})\n"),
    ("デコレータと既定値は外側で評価",
     "import functools\nD = 3\n@functools.cache\ndef f(x=D):\n    return x\n"),
    ("walrus", "def f(xs):\n    if (n := len(xs)) > 2:\n        return n\n    return 0\n"),
    ("内包表記が外側の名前を読む", "F = 2\ndef f(xs):\n    return [x * F for x in xs]\n"),
    ("クラス本体が外側を読む", "B = object\nclass C(B):\n    X = B\n"),
    ("入れ子関数が外側のローカルを読む",
     "def o(xs):\n    t = 0\n    def i(x):\n        return x + t\n    return [i(x) for x in xs]\n"),
    ("0引数 super()", "class A:\n    def f(self):\n        return 1\nclass B(A):\n    def f(self):\n        return super().f()\n"),
    ("f-string が外側を読む", 'N = "x"\ndef f():\n    return f"{N}!"\n'),
    ("match の捕捉名",
     "def f(v):\n    match v:\n        case {'a': x, **rest}:\n            return x, rest\n        case [a, *b]:\n            return a, b\n        case _:\n            return None\n"),
    ("既定値がクラス属性を読む", "class C:\n    N = 3\n    def f(self, n=N):\n        return n\n"),
]


class TestTheInstrumentIsCalibrated:
    r"""**極端な数字はまず測定器を疑う。** 製品170本で 0 件という結果は、
    走査が何も見ていなくても同じ数字になる——だから採用の前に、報告する形と
    沈黙する形を両方固定してある（起票 FUNC-LOCAL-IMPORT-SCOPE、2026-09-10）。
    """

    @pytest.mark.parametrize("label,name,src", MUST_REPORT,
                             ids=[c[0] for c in MUST_REPORT])
    def test_it_reports_a_name_that_does_not_resolve(self, label, name, src):
        assert name in [nm for _, nm in unresolved_names(src)], (label, src)

    @pytest.mark.parametrize("label,src", MUST_BE_SILENT,
                             ids=[c[0] for c in MUST_BE_SILENT])
    def test_it_stays_silent_on_legal_code(self, label, src):
        assert unresolved_names(src) == [], (label, unresolved_names(src))

    def test_it_is_strictly_stronger_than_the_file_level_reading(self):
        """どこにも束縛が無い名前は、どのスコープ鎖でも解決しない——だから
        スコープ版は元の版が報告するものを全部報告する。移行が「取りこぼしを
        増やさない」ことの根拠がこれである。"""
        nowhere = "def f():\n    return CompletelyUnbound\n"
        assert [nm for _, nm in unresolved_names(nowhere)] == ["CompletelyUnbound"]


def test_no_module_compiles_with_a_syntax_warning():
    r"""An invalid escape sequence is a defect Python has announced it will break.

    Same family as the unresolved names above — a mechanical slip a static scan
    sees and the suite does not, because warnings are not failures.  Added
    2026-08-13 with a live instance, and the instance was MINE: writing the
    ONE-QUESTION-MANY-ANSWERS ratchet's own comments, a docstring quoting
    `\sum_{i=1}` went in without an `r` prefix, and 4404 tests passed over it.
    It surfaced only because an unrelated probe printed stderr.

    That is the residual-2 shape exactly: the corpus and the route attribution
    catch "one notation, two answers"; they cannot catch what the worker types
    while fixing something else.
    """
    import warnings
    offenders = []
    for path in _product_files():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(open(path, encoding="utf-8").read(), path, "exec")
        offenders += [f"{os.path.relpath(path, REPO_ROOT)}: {w.message}"
                      for w in caught if issubclass(w.category, SyntaxWarning)]
    assert not offenders, "\n".join(offenders)


def test_the_scan_actually_sees_the_product():
    """Non-vacuity: a path typo would make every case above pass on nothing."""
    files = _product_files()
    assert len(files) > 100, files[:5]
    assert any(p.endswith("notebook/engine.py") for p in files)
