"""pytest plugin: which interceptor WINS during a test run.

    ROUTE_COVERAGE_OUT=/tmp/won.json \
      .venv/bin/python -m pytest tests/ -q -p scripts.pytest_route_coverage

Why this exists.  The instruction corpus (``tests/corpus/instruction_corpus.json``)
pins the route of every scenario line, which answers "did a rule start claiming
something it used to decline?".  It cannot answer the other half — "is this
rule's WINNING behaviour measured anywhere at all?" — because a rule the corpus
never triggers looks identical to one that does not exist.

Measured 2026-08-23 with this plugin, over the whole suite:

    registered interceptors            76
    win at least once in the corpus    47
    win at least once in the tests     67
    win in EITHER                      69
    NEVER win anywhere                  7

and of those 7, five (``numeric_fork_nl``, the four ``plan_*``) turned out to
have handler-level tests that bypass the registry, leaving TWO with no
measurement of any kind — ``set_latex_name`` and ``pending_followup``, now
covered by ``tests/test_unreached_interceptors.py``.

Counting mentions of the name in test files is NOT a substitute and was tried
first: it reported all 29 corpus-absent rules as covered, when the only file
mentioning several of them pins the registry's ORDER and never runs them.
"""
import atexit
import json
import os
import pathlib

import ui.interpret_registry as _reg

_WINNERS: set = set()
_ORIGINAL = _reg.run_interceptors


async def _record(ctx, interceptors):
    out = await _ORIGINAL(ctx, interceptors)
    route = getattr(ctx, "route", None)
    if route:
        _WINNERS.add(route)
    return out


_reg.run_interceptors = _record
# The dispatchers imported the symbol BY VALUE, so patching the module alone
# would record nothing — measured: the first run reported zero winners.
for _mod in ("ui.interpret_rules", "ui.interpret_early",
             "ui.interpret_equation_intercepts", "ui.interpret_pipeline"):
    try:
        _m = __import__(_mod, fromlist=["run_interceptors"])
        if hasattr(_m, "run_interceptors"):
            _m.run_interceptors = _record
    except Exception:                                  # noqa: BLE001 — a probe
        pass


@atexit.register
def _dump() -> None:
    out = os.environ.get("ROUTE_COVERAGE_OUT")
    if out:
        pathlib.Path(out).write_text(json.dumps(sorted(_WINNERS), indent=1),
                                     encoding="utf-8")
    else:
        print(f"\n--- interceptors that won: {len(_WINNERS)}")
