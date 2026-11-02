"""Tests for NotationProfileLoader (B-4)."""
import pytest
from notebook.runtime.notation_profile_loader import NotationProfileLoader


@pytest.fixture
def loader():
    return NotationProfileLoader()


class TestProfileResolution:
    def test_calculus_base_prime_is_derivative(self, loader):
        exclusive, meaning = loader.is_exclusive("prime", ["high_school_math"])
        assert exclusive is True
        assert meaning == "derivative"

    def test_second_quantization_prime_is_creation_operator(self, loader):
        exclusive, meaning = loader.is_exclusive("prime", ["condensed_matter"])
        assert exclusive is True
        assert meaning == "creation_operator"

    def test_no_domain_returns_not_exclusive(self, loader):
        exclusive, meaning = loader.is_exclusive("prime", [])
        assert exclusive is False
        assert meaning is None

    def test_unknown_domain_returns_not_exclusive(self, loader):
        exclusive, meaning = loader.is_exclusive("prime", ["unknown_domain_xyz"])
        assert exclusive is False

    def test_classical_physics_bracket_square_not_commutator(self, loader):
        """classical_physics inherits calculus_base: [A,B] = interval, not commutator."""
        rule = loader.get_rule("bracket_square", ["analytical_mechanics"])
        assert rule is not None
        assert rule.get("meaning") == "interval_or_list"

    def test_quantum_standard_bracket_square_is_commutator(self, loader):
        """quantum_standard overrides: [A,B] = commutator."""
        exclusive, meaning = loader.is_exclusive("bracket_square", ["quantum_mechanics"])
        assert exclusive is True
        assert meaning == "commutator"

    def test_second_quantization_anticommutator(self, loader):
        exclusive, meaning = loader.is_exclusive("bracket_curly", ["condensed_matter"])
        assert exclusive is True
        assert meaning == "anticommutator"

    def test_statistics_E_bracket_expectation(self, loader):
        exclusive, meaning = loader.is_exclusive("E_bracket", ["probability_statistics"])
        assert exclusive is True
        assert meaning == "expectation_value"


class TestInheritance:
    def test_second_quantization_inherits_quantum_standard_commutator(self, loader):
        """second_quantization_profile inherits bracket_square=commutator from quantum_standard."""
        exclusive, meaning = loader.is_exclusive("bracket_square", ["spin_systems"])
        assert exclusive is True
        assert meaning == "commutator"

    def test_second_quantization_inherits_calculus_base_Delta(self, loader):
        """Deep inheritance: second_quantization → quantum_standard → classical_physics → calculus_base."""
        _, rules = loader.profile_for_domains(["condensed_matter"])
        assert "Delta" in rules

    def test_child_overrides_parent_prime(self, loader):
        """second_quantization overrides calculus_base's prime=derivative with creation_operator."""
        _, rules = loader.profile_for_domains(["condensed_matter"])
        assert rules["prime"]["meaning"] == "creation_operator"

    def test_most_specific_profile_wins(self, loader):
        """When both condensed_matter (depth 4) and high_school_math (depth 0) active,
        second_quantization_profile wins."""
        _, meaning = loader.is_exclusive("prime", ["condensed_matter", "high_school_math"])
        assert meaning == "creation_operator"


class TestNotationHintText:
    def test_hint_contains_profile_id(self, loader):
        hint = loader.notation_hint_text(["condensed_matter"])
        assert "second_quantization_profile" in hint

    def test_hint_contains_exclusive_tag(self, loader):
        hint = loader.notation_hint_text(["condensed_matter"])
        assert "[exclusive]" in hint

    def test_hint_empty_for_no_domains(self, loader):
        hint = loader.notation_hint_text([])
        assert hint == ""

    def test_hint_contains_prime_creation_operator(self, loader):
        hint = loader.notation_hint_text(["quantum_optics"])
        assert "creation_operator" in hint


class TestProfileForDomains:
    def test_returns_profile_id(self, loader):
        pid, _ = loader.profile_for_domains(["differential_equations"])
        assert pid == "calculus_base"

    def test_returns_none_for_unknown_domain(self, loader):
        pid, rules = loader.profile_for_domains(["no_such_domain"])
        assert pid is None
        assert rules == {}
