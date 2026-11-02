r"""One question, one function: stripping an H-1 `LHS = ` prefix.

ONE-QUESTION-MANY-ANSWERS ratchet, final lowering (60 → 2).  The last six sites
were the SAME question in six places — "the researcher's expression arrived as
`LHS = RHS`; give me the body" — and two of them were the same three lines
copied verbatim, under a comment that called duplicating small helpers *this
codebase's precedent*.

Measured 2026-08-13, both copies on `\sum_{i=1}^{3} c_i = u*x + v*y`:

    maxima_game._strip_h1_lhs     1}^{3} c_i = u*x + v*y
    maxima_signal._strip_h1_lhs   1}^{3} c_i = u*x + v*y
    shared_parser.strip_lhs       u*x + v*y

Where each answer went: the game builders' matrix parsing, convolve's f/g, the
Fourier/Itô transform body, `spin_algebra`'s term parser (which has NO CAS
behind it — CLAUDE.md F-13 — so nothing downstream could object), and the
NUMERIC ASSUMPTION guard, which decides whether a sandboxed run is allowed to
start.  That last one also used the spaced `" = "`, so it missed a CAS-produced
equation as well.

This file pins the deletion, not just the fix: if a private copy comes back,
`test_no_module_defines_its_own_lhs_stripper` fails.
"""
import ast
import os

import pytest

from notebook.runtime.shared_parser import strip_lhs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM_EQUATION = r"\sum_{i=1}^{3} c_i = u*x + v*y"


@pytest.mark.parametrize("text, expected", [
    (SUM_EQUATION, "u*x + v*y"),
    ("A = [2,0;0,1]", "[2,0;0,1]"),          # a payoff matrix, the game case
    ("x^2 + 1", "x^2 + 1"),                  # not an equation → unchanged
    ("f(x) := x^2", "f(x) := x^2"),          # a DEFINITION is not a relation
    ("D=4*omega^2", "4*omega^2"),            # CAS spelling: no spaces
])
def test_the_one_definition(text, expected):
    assert strip_lhs(text) == expected


def test_every_caller_agrees_because_there_is_only_one():
    """The former sites that still exist, exercised through what each feeds.

    `convolve` and the two Fourier builders were among the six and are gone
    (SCENARIO-MACRO cases 5 / 2026-08-14): the researcher writes those integrals
    directly, so there is no builder left to hand an H-1 label to.  Their
    removal does not weaken this file — its load-bearing half is
    `test_no_module_defines_its_own_lhs_stripper`, a SHAPE check over every
    module, which does not depend on any particular caller existing.
    """
    from notebook.runtime.spin_algebra import parse_spin_expression

    # spin_algebra's parser (no CAS behind it)
    terms = parse_spin_expression(r"H = \omega S^z_1")
    assert terms and len(terms) == 1


def test_no_module_defines_its_own_lhs_stripper():
    """The copies are gone; a new one must not appear.

    Not a name check — a SHAPE check: any module-level function whose body is
    "if `=` in x: return x.split(...)" is this question being answered again.
    The name `_strip_h1_lhs` was the one used twice, but the next copy will be
    called something else.
    """
    offenders = []
    for directory in ("notebook", "ui"):
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, directory)):
            if "__pycache__" in root:
                continue
            for name in sorted(f for f in files if f.endswith(".py")):
                path = os.path.join(root, name)
                tree = ast.parse(open(path, encoding="utf-8").read())
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    src = ast.dump(node)
                    if ("attr='split'" in src and "value='='" in src
                            and "Return" in src and len(node.body) <= 4):
                        offenders.append(
                            f"{os.path.relpath(path, REPO_ROOT)}:{node.lineno} "
                            f"{node.name}")
    assert not offenders, (
        "these look like a private re-implementation of shared_parser.strip_lhs:\n  "
        + "\n  ".join(offenders))
