"""Dialogue-driven spec proposal (spec_adviser.py, docs/25 §3).

Everything deterministic is covered here with a FAKE chat callable — the JSON
contract, strict parsing (broken replies never become a spec), the
validate_numeric_spec gatekeeper (issues are surfaced, not silently dropped —
unlike plan_adviser), and the refine-turn prompt wiring. What remains
untested by design is the real adviser LLM's proposal quality (same policy as
tests/test_plan_candidates.py for plan_adviser).
"""
import json

from notebook.models.numeric import IntentSpec, InputVar, MethodSpec, NumericSpec, OutputSpec
from notebook.numeric.spec_adviser import build_spec_prompt, parse_spec, propose_spec

GOOD_REPLY = json.dumps({
    "intent": {"kind": "parameter_sweep", "goal": "sweep x"},
    "inputs": [
        {"name": "x", "role": "sweep", "spec": {"min": 0, "max": 10, "points": 100}},
        {"name": "k", "role": "fixed", "spec": {"value": 2}},
        {"name": "a", "role": "fixed", "spec": {"value": 0.3}},
    ],
    "outputs": [{"name": "y", "shape": "series", "over": ["x"], "role": "primary"}],
    "method": {"name": "direct_eval", "params": {}},
    "presentation": {"kind": "line", "options": {}},
})

CONVERGENCE_REPLY = json.dumps({
    "intent": {"kind": "root_finding", "goal": "iterate until |dx|<tol"},
    "inputs": [{"name": "x", "role": "free", "spec": {"initial": 0.5}},
               {"name": "k", "role": "fixed", "spec": {"value": 2}}],
    "outputs": [{"name": "x_star", "shape": "scalar"},
                {"name": "trace", "shape": "trace", "over": ["iteration"], "role": "diagnostic"}],
    "method": {"name": "fixed_point", "params": {"tol": 1e-8, "max_iter": 1000}},
})


def _fake_chat(reply):
    return lambda system, user: reply


class TestParseSpec:
    def test_good_reply_parses(self):
        spec = parse_spec(GOOD_REPLY)
        assert spec is not None
        assert spec.intent.kind == "parameter_sweep"
        assert [v.name for v in spec.inputs] == ["x", "k", "a"]

    def test_convergence_reply_parses(self):
        spec = parse_spec(CONVERGENCE_REPLY)
        assert spec is not None
        assert spec.method.name == "fixed_point"
        assert spec.method.params["tol"] == 1e-8

    def test_empty_reply_is_none(self):
        assert parse_spec("") is None
        assert parse_spec(None) is None

    def test_prose_only_reply_is_none(self):
        assert parse_spec("Sure, here's a plan for you...") is None

    def test_malformed_json_is_none(self):
        assert parse_spec("{intent: not valid json,}") is None

    def test_json_array_of_scalars_is_none(self):
        # no {..} object anywhere in the reply
        assert parse_spec("[1, 2, 3]") is None

    def test_missing_required_field_is_none(self):
        # OutputSpec.name is required; a reply omitting it must not become a spec
        bad = json.dumps({"outputs": [{"shape": "scalar"}]})
        assert parse_spec(bad) is None

    def test_invalid_role_is_none(self):
        # InputRole is a closed set (docs/24 §8-2)
        bad = json.dumps({"inputs": [{"name": "x", "role": "wiggle"}]})
        assert parse_spec(bad) is None

    def test_fenced_json_extracted(self):
        fenced = f"Here you go:\n```json\n{GOOD_REPLY}\n```\nLet me know."
        spec = parse_spec(fenced)
        assert spec is not None and spec.intent.kind == "parameter_sweep"


class TestBuildSpecPrompt:
    def test_first_turn_has_no_prior_spec(self):
        prompt = build_spec_prompt("sweep x", "y = sin(k*x)*exp(-a*x)", {"x", "k", "a"})
        assert "Previous proposal" not in prompt
        assert "Propose the spec now" in prompt

    def test_refine_turn_includes_prior_and_correction(self):
        prior = NumericSpec(intent=IntentSpec(kind="parameter_sweep"))
        prompt = build_spec_prompt("sweep x", "y = sin(k*x)", {"x", "k"},
                                   prior_spec=prior, refinement="sweep k instead")
        assert "Previous proposal" in prompt
        assert "sweep k instead" in prompt
        assert "Revise the previous proposal" in prompt

    def test_none_free_symbols_omits_the_line(self):
        prompt = build_spec_prompt("goal", "zeta(x)", None)
        assert "Free symbols" not in prompt

    def test_narrative_included_when_given(self):
        narrative = [{"kind": "user_input", "text": "define k as scalar"}]
        prompt = build_spec_prompt("goal", "y", {"y"}, narrative=narrative)
        assert "define k as scalar" in prompt


class TestProposeSpec:
    def test_good_reply_yields_clean_spec_no_issues(self):
        spec, issues = propose_spec(_fake_chat(GOOD_REPLY), "sweep x",
                                    "y = sin(k*x)*exp(-a*x)", {"x", "k", "a"})
        assert spec is not None
        assert issues == []

    def test_unparseable_reply_yields_none_spec(self):
        spec, issues = propose_spec(_fake_chat("not json at all"), "goal",
                                    "y", {"y"})
        assert spec is None
        assert issues == []

    def test_spec_with_gatekeeper_issues_is_still_returned(self):
        # x is not among the expression's free symbols -> gatekeeper flags it,
        # but the spec itself parsed fine and must be shown, not discarded
        # (docs/25 §3: the researcher refines from here, unlike plan_adviser).
        spec, issues = propose_spec(_fake_chat(GOOD_REPLY), "sweep x",
                                    "y = sin(k*x)*exp(-a*x)", free_symbols={"k", "a"})
        assert spec is not None
        codes = {i["code"] for i in issues}
        assert "numeric.spec.unknown_symbol" in codes

    def test_convergence_reply_has_no_membership_issue_for_free_role(self):
        # "free" (initial value) inputs still get symbol-membership checked
        spec, issues = propose_spec(_fake_chat(CONVERGENCE_REPLY), "converge",
                                    "g(x)=x", free_symbols={"x", "k"})
        assert spec is not None
        assert issues == []

    def test_system_prompt_lists_known_kinds(self):
        seen = {}
        def spy_chat(system, user):
            seen["system"] = system
            return GOOD_REPLY
        propose_spec(spy_chat, "goal", "y", {"y"})
        assert "parameter_sweep" in seen["system"]
        assert "eigenvalue_problem" in seen["system"]

    def test_none_free_symbols_skips_membership_check(self):
        # expression not whitelist-lowerable -> free_symbols=None -> the
        # gatekeeper's symbol check is skipped, same policy as
        # validate_numeric_spec itself when called directly with None.
        spec, issues = propose_spec(_fake_chat(GOOD_REPLY), "sweep x",
                                    "y = zeta(x)", free_symbols=None)
        assert spec is not None
        assert issues == []

    def test_refine_turn_passes_prior_spec_through(self):
        prior = NumericSpec(intent=IntentSpec(kind="parameter_sweep"))
        seen = {}
        def spy_chat(system, user):
            seen["user"] = user
            return GOOD_REPLY
        propose_spec(spy_chat, "goal", "y", {"x", "k", "a"},
                    prior_spec=prior, refinement="use k as the sweep var")
        assert "use k as the sweep var" in seen["user"]
