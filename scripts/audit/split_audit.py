"""Which of the remaining raw `=` splits ever RECEIVE a dangerous string?

Instrumenting the 50-odd call sites would mean editing 50 places (and this
session has already shown what hand-editing costs).  Instrument the SOURCE
instead: every dangerous string originates as an ``Expression.canonical`` or
``Expression.latex``.  Hand those out as a ``str`` subclass whose ``split`` /
``partition`` / ``find`` compare the naive answer with the bracket-aware one and
record the CALLER whenever they disagree.

Behaviour is unchanged — the naive answer is still returned — so the suite must
pass exactly as before.  A disagreement is a live defect at that call site;
silence over the whole suite is evidence (not proof) that the site is never fed
a nested `=`.

**Silence has two causes, and the 2026-08-11 run could not tell them apart**
(open_items ONE-QUESTION-MANY-ANSWERS 残作業 1: "残り約32箇所は…何も言っていない").
A site is silent either because it never ran, or because the string reaching it
was DERIVED from ``Expression.canonical`` (``.strip()``, a slice, an f-string)
and ``str`` methods return plain ``str``, dropping the taint.  The second is the
dangerous one — the site really is fed Expression text and the audit simply
could not see it.  Two additions separate them:

* taint PROPAGATES through the string operations that derive one expression
  string from another (``strip``/``replace``/slice/``+``/f-string/split halves);
* the plugin ENUMERATES the raw split sites itself (AST, so docstrings do not
  count) and reports which of them were never fed a tainted string, so the
  silent set is named rather than inferred.

Whether a named-silent site ran at all is a coverage question — run the suite
with ``--cov=notebook --cov=ui`` alongside this plugin and compare.

Usage:  PYTHONPATH=scripts/audit pytest -p split_audit ...
        SPLIT_AUDIT_JSON=/path/out.json  to also dump the raw records.
"""
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

_hits = Counter()
_examples = {}
_calls = Counter()

_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_ATTRS = {"split", "rsplit", "partition", "rpartition", "find", "index"}


def _enumerate_sites():
    """Every `x.split("=")`-shaped call in the product, by AST (not grep).

    AST is what makes the count trustworthy: the two matches in
    ``shared_parser.py`` / ``lhs_formatting.py`` are DOCSTRINGS describing the
    defect, and a grep counts them as sites.
    """
    sites = {}
    for pkg in ("notebook", "ui"):
        for path in sorted((_ROOT / pkg).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                      # pragma: no cover
                continue
            rel = str(path.relative_to(_ROOT))
            for node in ast.walk(tree):
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
                sites[(rel, node.func.value.end_lineno or node.lineno)] = (
                    f"{node.func.attr}({sep!r})")
    return sites


def _norm(parts):
    """Compare MEANING, not spacing.

    A site splitting on ``" = "`` keeps no surrounding blanks while the
    bracket-aware scan splits on the bare ``=``; the 2026-08-11 run reported two
    such pairs as disagreements and both were the audit's own artefact.
    """
    if isinstance(parts, (list, tuple)):
        return type(parts)(str(p).strip() for p in parts)
    return parts


def _record(kind, value, naive, aware):
    frame = sys._getframe(2)
    where = f"{frame.f_code.co_filename.split('Research_Notebook/')[-1]}:{frame.f_lineno}"
    if where.startswith(("/", "<")) or "site-packages" in where:
        return
    _calls[where] += 1
    if _norm(naive) != _norm(aware):
        _hits[f"{where}  [{kind}]"] += 1
        _examples.setdefault(f"{where}  [{kind}]",
                             (str(value)[:70], str(naive)[:40], str(aware)[:40]))


def _wrap(v):
    return AuditStr(v) if type(v) is str else v


def _relation(s):
    from notebook.runtime.shared_parser import find_top_level_relation
    return find_top_level_relation(str(s))


class AuditStr(str):
    """A str that notices when a naive `=` split disagrees with the real one.

    It also stays attached to the strings DERIVED from it: an expression string
    that has been ``.strip()``ed is still expression text, and the call site that
    splits it is exactly as exposed as one splitting the attribute directly.
    """

    def split(self, sep=None, maxsplit=-1):
        out = str.split(self, sep, maxsplit) if sep is not None else str.split(self)
        if sep is not None and "=" in sep:
            rel = _relation(self)
            aware = ([self[:rel[0]], self[rel[0] + len(rel[1]):]]
                     if rel and rel[1] == "=" else [str(self)])
            naive = [str(x) for x in out]
            _record("split", self, naive[:2], aware[:2])
        return [_wrap(x) for x in out]

    def rsplit(self, sep=None, maxsplit=-1):
        out = str.rsplit(self, sep, maxsplit) if sep is not None else str.rsplit(self)
        if sep is not None and "=" in sep:
            rel = _relation(self)
            aware = ([self[:rel[0]], self[rel[0] + len(rel[1]):]]
                     if rel and rel[1] == "=" else [str(self)])
            naive = [str(x) for x in out]
            _record("rsplit", self, naive[:2], aware[:2])
        return [_wrap(x) for x in out]

    def partition(self, sep):
        out = str.partition(self, sep)
        if "=" in sep:
            rel = _relation(self)
            aware = ((self[:rel[0]], "=", self[rel[0] + 1:])
                     if rel and rel[1] == "=" else (str(self), "", ""))
            _record("partition", self, tuple(str(x) for x in out),
                    tuple(str(x) for x in aware))
        return tuple(_wrap(x) for x in out)

    def rpartition(self, sep):
        out = str.rpartition(self, sep)
        if "=" in sep:
            rel = _relation(self)
            aware = ((self[:rel[0]], "=", self[rel[0] + 1:])
                     if rel and rel[1] == "=" else ("", "", str(self)))
            _record("rpartition", self, tuple(str(x) for x in out),
                    tuple(str(x) for x in aware))
        return tuple(_wrap(x) for x in out)

    def find(self, sub, *a):
        out = str.find(self, sub, *a)
        if "=" in sub:
            rel = _relation(self)
            aware = rel[0] if rel and rel[1] == "=" else -1
            # `.find(" = ")` looks for a SPACED equals, one character left of the
            # bracket-aware answer; only a bigger gap (or one side finding
            # nothing) is a real disagreement.
            same = (out >= 0 and aware >= 0 and abs(out - aware) <= 1) or \
                   (out < 0 and aware < 0)
            _record("find", self, out, aware if not same else out)
        return out

    def index(self, sub, *a):
        out = str.index(self, sub, *a)
        if "=" in sub:
            rel = _relation(self)
            aware = rel[0] if rel and rel[1] == "=" else -1
            same = aware >= 0 and abs(out - aware) <= 1
            _record("index", self, out, aware if not same else out)
        return out

    # --- taint propagation ------------------------------------------------
    # Only the operations that DERIVE one expression string from another.
    # `__getitem__` wraps slices but not single characters: the parsers walk
    # these strings a character at a time, and wrapping every char makes the
    # suite pay for the audit without telling it anything.
    def __getitem__(self, key):
        out = str.__getitem__(self, key)
        return _wrap(out) if isinstance(key, slice) else out

    def __add__(self, other):
        return _wrap(str.__add__(self, other))

    def __radd__(self, other):
        return _wrap(str.__add__(other, self))

    def __format__(self, spec):
        return _wrap(str.__format__(self, spec))

    def __str__(self):
        return self


def _derive(name):
    def method(self, *a, **kw):
        return _wrap(getattr(str, name)(self, *a, **kw))
    method.__name__ = name
    return method


for _name in ("strip", "lstrip", "rstrip", "replace", "removeprefix",
              "removesuffix", "lower", "upper", "casefold", "expandtabs",
              "title", "swapcase", "center", "ljust", "rjust", "zfill",
              "format", "join"):
    setattr(AuditStr, _name, _derive(_name))


def pytest_configure(config):
    from notebook.models.expression import Expression
    orig = Expression.__getattribute__

    def patched(self, name):
        v = orig(self, name)
        if name in ("canonical", "latex", "input_canonical") and type(v) is str:
            return AuditStr(v)
        return v

    Expression.__getattribute__ = patched


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    w = terminalreporter.write_line
    sites = _enumerate_sites()
    seen = {}
    for where, n in _calls.items():
        f, _, ln = where.rpartition(":")
        seen.setdefault(f, []).append((int(ln), n))

    def exercised(rel, line):
        return any(abs(ln - line) <= 2 for ln, _ in seen.get(rel, []))

    silent = [(rel, ln, what) for (rel, ln), what in sorted(sites.items())
              if not exercised(rel, ln)]

    w("")
    w("=== raw `=` split audit (source-instrumented, taint-propagating) ===")
    w(f"raw split sites in notebook/+ui/ (AST): {len(sites)}")
    w(f"…fed an Expression-derived string during this run: {len(sites) - len(silent)}")
    w(f"…never fed one (silent — coverage says whether they ran): {len(silent)}")
    if not _hits:
        w("DISAGREEMENTS: none — no site was fed a string whose naive `=` split "
          "differs from the bracket-aware one, over the whole suite.")
    else:
        w(f"DISAGREEMENTS at {len(_hits)} site(s):")
        for site, n in _hits.most_common():
            val, naive, aware = _examples[site]
            w(f"  {site}  ×{n}")
            w(f"      value {val!r}")
            w(f"      naive {naive!r}   aware {aware!r}")
    w("")
    w("silent sites (never handed a tainted string):")
    for rel, ln, what in silent:
        w(f"  {rel}:{ln}  {what}")
    w("")
    w("sites exercised (top 15 by call count):")
    for site, n in _calls.most_common(15):
        w(f"  {n:6d}  {site}")

    out = os.environ.get("SPLIT_AUDIT_JSON")
    if out:
        Path(out).write_text(json.dumps({
            "sites": {f"{r}:{l}": w_ for (r, l), w_ in sites.items()},
            "silent": [f"{r}:{l}" for r, l, _ in silent],
            "calls": dict(_calls),
            "hits": dict(_hits),
            "examples": _examples,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
