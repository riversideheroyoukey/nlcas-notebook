"""MaximaBackendHandler mixin: F-8 definition-based transform command
builders (signal_processing / control_theory domains).

Mirrors MaximaBackendHandler.laplace_definition_command's design exactly
(assume/forget convergence guard around a raw integrate() — never Maxima's
opaque builtins) for the Fourier transform and its inverse, plus a Z-transform
built the same way but as a sum instead of an integral (nusum, first use of
that Maxima function in this codebase — no prior template exists to reuse),
and convolution (continuous via integrate, discrete via sum).

Extracted as its own mixin (not appended directly to
notebook/runtime/maxima_backend_handler.py) for the same hygiene-ratchet
reason as MaximaEconMixin/MaximaGameMixin.
"""
from __future__ import annotations

import re

from notebook.runtime.backend_handler import BuildResult, MATH_FUNC_NAMES
from notebook.runtime.shared_parser import strip_lhs   # was a verbatim copy


class MaximaSignalMixin:
    """F-8: definition-based transform command builders (see module docstring)."""

    @staticmethod
    def _positive_params_assume(expr: str, exclude: "set[str]") -> "list[str]":
        params = sorted({x for x in re.findall(r"[A-Za-z_]\w*", expr)
                         if x not in exclude and x not in MATH_FUNC_NAMES})
        return [f"{p}>0" for p in params]

    def z_transform_definition_command(self, x: str, n: str, z: str) -> str:
        """Sum_{n=0}^{inf} x[n] z^{-n}, via Maxima's nusum (closed-form
        summation) under the same convergence guard as
        laplace/fourier_definition_command.

        Deliberately uses PLAIN ``nusum(...,0,inf)``, not a
        ``limit(nusum(...,0,N),N,inf)`` wrapper: the latter resolves more
        cleanly for a CONCRETE numeric base (confirmed live:
        nusum((1/2)^n*z^(-n),n,0,inf) leaves an uncancelled
        ``(1/2)^inf/z^inf`` boundary artifact; the limit form resolves it to
        2z/(2z-1)) but for a SYMBOLIC base it triggers an interactive Maxima
        sign question ("Is z-1 positive...") that HANGS the persistent
        adapter even when the caller has already assumed the natural
        ordering (confirmed live: `assume(z>a,a>0)` does not prevent the
        prompt — Maxima specifically wants `z>1`, an assumption about the
        transform variable itself that this generic command has no business
        guessing, per the no-hardcoded-modeling-choices rule). The plain
        nusum form never hangs; it may leave an unresolved boundary term for
        a symbolic base, which the researcher's own follow-up simplification
        (with their own region-of-convergence assumption) can resolve — a
        visible, correct-but-unsimplified result is preferred over a command
        that risks freezing the session."""
        al = self._positive_params_assume(x, {n, z, "exp", "e"})
        assume_str = f"assume({', '.join(al)}), " if al else ""
        forget_str = f"forget({', '.join(al)}), " if al else ""
        return (f"block([_r], {assume_str}"
                f"_r: nusum(({x})*({z})^(-{n}), {n}, 0, inf), "
                f"{forget_str}_r)")
