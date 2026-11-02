"""SCALAR-HANDOFF, the button route (docs/34 §5.4, slice 4).

A button cannot be read the way a typed sentence can — the researcher who typed
`E5 を Maxima セッションに渡して` said both the target and the intent, the one who
clicked said nothing.  So the button PREVIEWS first, and these pin the two
properties that makes it worth having:

* the preview leaves NO record (a look is not an act), and
* the preview shows what the create would actually carry — one body, two
  callers, because a second implementation of "what crosses and what travels
  with it" is the surest way to make the two disagree.

Also pinned: a refusal arrives as the translated reason at PREVIEW time, so the
create button is never offered for a crossing that cannot happen.
"""
import asyncio

import ui.notebook_bridge as nb
from notebook.models.quantity import Quantity
from notebook.models.step import StepKind
from ui.routes.handoff import HandoffRequest, handoff_create, handoff_preview
from ui.session_io import sessions_dir


def _session(lang="ja"):
    nb.start_engine(title="handoff_route", mode="standard", language=lang,
                    domains=[])
    eng = nb.get_engine()
    eng.symbol_context.register("Delta", Quantity(
        name="Delta", kind="scalar", domain="real", sign="positive"))
    eng.symbol_context.register("omega_D", Quantity(
        name="omega_D", kind="scalar", domain="real", sign="positive"))
    return eng


def test_preview_records_nothing_and_creates_no_file():
    async def _run():
        eng = _session()
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            before_steps = len(eng.steps)
            before_files = set(sessions_dir().glob("*.json"))

            res = await handoff_preview(HandoffRequest(expr_id=e.display_id))

            assert res["ok"] is True, res
            assert len(eng.steps) == before_steps
            assert not any(s.kind == StepKind.scalar_handoff
                           for s in eng.steps.values())
            assert set(sessions_dir().glob("*.json")) == before_files
            assert not getattr(eng, "scalar_handoffs", {})
        finally:
            nb.stop_engine()

    asyncio.run(_run())


def test_preview_lists_what_create_actually_carries():
    """The declaration list is the whole point of showing a preview: it is what
    decides whether the target session can continue (§5.2 measured an uncarried
    `omega_D` turning the answer into `asinh(omega(D)/Delta)`)."""
    async def _run():
        eng = _session()
        created = None
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            prev = await handoff_preview(HandoffRequest(expr_id=e.display_id))
            shown = {(d["name"], d["domain"], d["sign"])
                     for d in prev["declarations"]}
            # Delta and omega_D both cross, signs included — omega_D appears
            # nowhere in the expression and must still be listed.
            assert ("Delta", "real", "positive") in shown
            assert ("omega_D", "real", "positive") in shown

            res = await handoff_create(HandoffRequest(expr_id=e.display_id))
            assert res["ok"] is True, res
            created = sessions_dir() / res["filename"]
            handoff = eng.scalar_handoffs[res["handoff_id"]]
            carried = {(d.name, d.domain, d.sign) for d in handoff.declarations}
            assert carried == shown
        finally:
            if created:
                created.unlink(missing_ok=True)
            nb.stop_engine()

    asyncio.run(_run())


def test_create_records_the_crossing_and_leaves_the_session_loaded():
    async def _run():
        eng = _session()
        created = None
        try:
            e = eng.register_expression("1/sqrt(xi^2 + Delta^2)")
            res = await handoff_create(HandoffRequest(expr_id=e.display_id))
            assert res["ok"] is True, res
            created = sessions_dir() / res["filename"]
            assert created.exists()
            # a file is written, not opened — the /fork shape
            assert nb.get_engine() is eng
            assert any(s.kind == StepKind.scalar_handoff
                       for s in eng.steps.values())
            # the note says what was verified, or plainly that it was not
            assert res["note"]
            assert res["verification"].get("status") in {
                "agreed", "differed", "not_checkable"}
        finally:
            if created:
                created.unlink(missing_ok=True)
            nb.stop_engine()

    asyncio.run(_run())


def test_operator_expression_is_refused_at_preview_time():
    """The refusal must arrive BEFORE the confirm button is offered, as words."""
    async def _run():
        eng = _session()
        try:
            eng.symbol_context.register("c", Quantity(
                name="c", kind="operator", statistics="fermi"))
            e = eng.register_expression("c(1) + Delta")
            res = await handoff_preview(HandoffRequest(expr_id=e.display_id))
            assert res["ok"] is False
            assert "c" in res["error"]
            assert "HandoffRefused" not in res["error"]      # words, not a class
            assert "handoff." not in res["error"]            # translated, not a key
            assert not any(s.kind == StepKind.scalar_handoff
                           for s in eng.steps.values())
        finally:
            nb.stop_engine()

    asyncio.run(_run())


def test_refusal_is_translated_in_english_too():
    async def _run():
        eng = _session("en")
        try:
            eng.symbol_context.register("c", Quantity(
                name="c", kind="operator", statistics="fermi"))
            e = eng.register_expression("c(1) + Delta")
            res = await handoff_preview(HandoffRequest(expr_id=e.display_id))
            assert res["ok"] is False
            assert "handoff." not in res["error"]
            # the English message must not be the Japanese one
            assert not any("぀" <= ch <= "ヿ" for ch in res["error"]), \
                res["error"]
        finally:
            nb.stop_engine()

    asyncio.run(_run())
