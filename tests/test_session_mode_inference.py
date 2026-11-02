"""Tests for _infer_session_mode in notebook_bridge.

Verifies that selecting any domain whose category has default_backend=='julia_full'
causes second_quantization mode, even when mixed with non-Julia domains.
"""
import pytest
from ui.notebook_bridge import _infer_session_mode
from notebook.models.session import SessionMode


class TestInferSessionMode:
    def test_empty_domains_gives_standard(self):
        assert _infer_session_mode([]) == SessionMode.standard_symbolic

    def test_physics_domain_gives_standard(self):
        assert _infer_session_mode(["quantum_mechanics"]) == SessionMode.standard_symbolic

    def test_math_domain_gives_standard(self):
        assert _infer_session_mode(["linear_algebra"]) == SessionMode.standard_symbolic

    def test_julia_domain_alone_gives_second_quantization(self):
        assert _infer_session_mode(["condensed_matter"]) == SessionMode.second_quantization

    def test_spin_systems_gives_second_quantization(self):
        assert _infer_session_mode(["spin_systems"]) == SessionMode.second_quantization

    def test_quantum_chemistry_sq_gives_second_quantization(self):
        assert _infer_session_mode(["quantum_chemistry_sq"]) == SessionMode.second_quantization

    def test_julia_domain_mixed_with_physics_gives_second_quantization(self):
        """Julia wins when combined with a standard-symbolic domain."""
        assert _infer_session_mode(["quantum_mechanics", "condensed_matter"]) == SessionMode.second_quantization

    def test_julia_domain_mixed_with_math_gives_second_quantization(self):
        assert _infer_session_mode(["linear_algebra", "spin_systems"]) == SessionMode.second_quantization

    def test_multiple_julia_domains_gives_second_quantization(self):
        assert _infer_session_mode(["condensed_matter", "quantum_optics", "spin_systems"]) == SessionMode.second_quantization

    def test_many_non_julia_domains_gives_standard(self):
        assert _infer_session_mode(["quantum_mechanics", "linear_algebra", "electromagnetism"]) == SessionMode.standard_symbolic

    def test_unknown_domain_is_ignored(self):
        assert _infer_session_mode(["nonexistent_domain"]) == SessionMode.standard_symbolic

    def test_unknown_domain_mixed_with_julia_gives_second_quantization(self):
        assert _infer_session_mode(["nonexistent_domain", "condensed_matter"]) == SessionMode.second_quantization
