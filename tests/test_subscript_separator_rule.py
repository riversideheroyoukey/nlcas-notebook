"""Multi-indices must be separated by a comma or a space (AUDIT-SUBSCRIPT-SPLIT).

Before this rule, any lowercase alphabetic run in a subscript was split per
character, so ordinary physics notation silently became a different quantity:

    E_{rot}   → E(r, o, t)      rotational energy, now a three-index object
    T_{kin}   → T(k, i, n)
    c_{k,up}  → c(k, u, p)      spin "up" split into u·p
    bessel_jn → bessel(j, n)

No error was raised.  The discriminator was letter CASE — ``E_{HF}`` survived,
``E_{rot}`` did not — and appeared in no document.

The rule is the one CLAUDE.md's Math Input Conventions already state for
expression BODIES, and for the same stated reason: "kx, ky, nk are common
two-letter variable names — splitting them would produce silent wrong results".
Subscripts were the one place still auto-splitting.

Trade-off, accepted deliberately: the conventional unseparated tensor spelling
``A_{ij}`` now means ONE index.  Write ``A_{i,j}`` or ``A_{i j}`` for two.  The
two carve-outs are ``\\delta_{ij}`` (its own Kronecker path) and
einstein-summation sessions, whose whole premise is that a repeated index is
implicit and which are opted into per session.
"""
import pytest

from notebook.engine import NotebookEngine


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="subscript rule") as e:
        yield e


def _lower(engine, expr):
    return engine._handler.to_backend_code(expr).code


class TestUnseparatedIsOneIndex:
    """The bug cases.  Each of these was a different quantity before."""

    @pytest.mark.parametrize("expr,expected", [
        ("E_{rot}", "E(rot)"),
        ("E_{lab}", "E(lab)"),
        ("T_{kin}", "T(kin)"),
        ("bessel_jn", "bessel(jn)"),
        ("A_{ij}", "A(ij)"),
        ("T_{ijk}", "T(ijk)"),
    ])
    def test_no_implicit_split(self, engine, expr, expected):
        assert _lower(engine, expr) == expected

    def test_a_multi_char_index_in_a_list_survives(self, engine):
        """``c_{k,up}`` is momentum k and spin up — two indices, the second of
        which is two letters.  It used to come out as three."""
        assert _lower(engine, "c_{k,up}") == "c(k, up)"

    def test_uppercase_was_never_split_and_still_is_not(self, engine):
        assert _lower(engine, "E_{HF}") == "E(HF)"
        assert _lower(engine, "H_{BCS}") == "H(BCS)"


class TestBothSeparators:
    """"Comma OR space" — the space form used not to resolve at all, which made
    half the rule untrue."""

    @pytest.mark.parametrize("expr,expected", [
        ("A_{i,j}", "A(i, j)"),
        ("A_{i j}", "A(i, j)"),
        ("A_{i j k}", "A(i, j, k)"),
        ("A_{i,j,k}", "A(i, j, k)"),
        (r"a_{i,\sigma}", "a(i, sigma)"),
        (r"a_{i \sigma}", "a(i, sigma)"),
    ])
    def test_separated_indices(self, engine, expr, expected):
        assert _lower(engine, expr) == expected

    def test_comma_and_space_agree(self, engine):
        assert _lower(engine, "A_{i j}") == _lower(engine, "A_{i,j}")


class TestCarveOuts:

    def test_kronecker_delta_keeps_its_own_path(self, engine):
        """``\\delta_{ij}`` is universal notation and resolves through the
        dedicated Kronecker branch, so the separator rule does not reach it."""
        assert _lower(engine, "delta_{ij}") == "kron_delta(i, j)"

    def test_greek_index_names_are_one_index(self, engine):
        """``\\mu`` is not m·u — this was already true and must stay true."""
        assert _lower(engine, "a_{mu}") == "a(mu)"
        assert _lower(engine, "a_{sigma}") == "a(sigma)"

    def test_einstein_summation_sessions_still_split(self):
        """That mode exists for implicit repeated indices; ``A_{ij}B_{jk}`` is
        how it is written, and it is opted into per session."""
        with NotebookEngine(title="einstein") as e:
            e.symbol_context.einstein_summation = True
            assert _lower(e, "A_{ij}") == "A(i, j)"
            assert _lower(e, "T_{ijk}") == "T(i, j, k)"


class TestSingleIndexUnaffected:

    @pytest.mark.parametrize("expr,expected", [
        ("a_i", "a(i)"),
        ("x_1", "x(1)"),
        ("a_{i}", "a(i)"),
        ("a_{12}", "a(12)"),
    ])
    def test_one_index_is_unchanged(self, engine, expr, expected):
        assert _lower(engine, expr) == expected


class TestTheSpaceSeparatorSurvivesRegistration:
    r"""Found reopening AUDIT-INDEXED-CALL, 2026-08-04.

    The rule is "a comma **or a space**", and `to_backend_code` honoured both —
    which is what the tests above check.  The path a researcher actually takes
    did not:

        A_{i j}(x)  registered → canonical `A_i*j(x)` → CAS `A(i) * j(x)`
        A_{i j}     registered → canonical `A_i*j`    → CAS `A(i) * j`

    A tensor component silently became a product, with no error.  Two passes in
    `LatexConverter` undid the separator in turn — `_remove_remaining_braces`
    dropped the braces (a comma is in its exclusion class, a space was not) and
    `_implicit_multiplication` then read the bare `i j` as multiplication.

    The 3-layer pipeline was never at fault: handed the original text it answers
    `A[i, j](x)`.  So the two spellings are made one at the entrance, and these
    tests check the two paths now agree — the property that was missing, not
    just the value.
    """

    SPELLINGS = [
        ("A_{i j}",        "A_{i,j}"),
        ("A_{i j k}",      "A_{i,j,k}"),
        ("A_{i j}(x)",     "A_{i,j}(x)"),
        (r"g_{\mu \nu}",   r"g_{\mu,\nu}"),
        (r"g_{\mu \nu}(x)", r"g_{\mu,\nu}(x)"),
    ]

    @pytest.mark.parametrize("spaced,comma", SPELLINGS)
    def test_space_and_comma_agree_through_the_converter(self, engine, spaced, comma):
        conv = engine._handler
        assert conv.latex_to_canonical(spaced) == conv.latex_to_canonical(comma)

    @pytest.mark.parametrize("spaced,comma", SPELLINGS)
    def test_space_and_comma_agree_through_the_lowering_door(self, engine, spaced, comma):
        assert _lower(engine, spaced) == _lower(engine, comma)

    def test_a_space_outside_a_subscript_is_still_multiplication(self, engine):
        """The reason the fix is scoped to `_{…}`: everywhere else the space
        between two letters really is a product, and that must keep working."""
        conv = engine._handler
        for expr in (r"\frac{a b}{c}", r"\sqrt{a b}", "m (l*omega^2)"):
            got = conv.latex_to_canonical(expr)
            assert "*" in got, (expr, got)

    def test_unseparated_ascii_is_still_one_index(self, engine):
        """The separator rule's other half is unchanged: with no separator,
        `ij` is ONE index — you cannot tell a two-letter name from two
        indices."""
        assert _lower(engine, "A_{ij}") == "A(ij)"

    def test_adjacent_commands_are_two_indices(self, engine):
        r"""`\mu\nu` is not the same case: the backslash IS a boundary, so
        these are unambiguously two tokens.  The parser had always read them
        that way; only the converter merged them, and the two paths disagreed
        (`g_munu` vs `g(mu, nu)`)."""
        assert _lower(engine, r"g_{\mu\nu}") == "g(mu, nu)"
        assert engine._handler.latex_to_canonical(r"g_{\mu\nu}") == "g_{mu,nu}"

    @pytest.mark.parametrize("latex,expected", [
        (r"\alpha\beta", "alpha*beta"),
        (r"\mu\nu + 1", "mu*nu + 1"),
        (r"\alpha\beta\gamma", "alpha*beta*gamma"),
    ])
    def test_adjacent_commands_outside_a_subscript_are_a_product(
            self, engine, latex, expected):
        r"""Measured 2026-08-04: the commands were replaced by their ASCII names
        one after another with nothing inserted, so `\alpha\beta` — the product
        αβ — became the single symbol `alphabeta`, silently.  Outside a
        subscript the boundary must become multiplication."""
        assert engine._handler.latex_to_canonical(latex) == expected


class TestDelimiterCommandsAreNotFactors:
    r"""`\left` / `\right` are syntax, not symbols.

    The boundary rule above inserts a space in front of a command glued to what
    precedes it.  Applied to a delimiter that is wrong twice over:
    `G_{0}\left(k,\omega\right)` is a function CALL, and `G_0 (k, omega)` is
    then read as multiplication by the space-then-paren rule — the silent-wrong
    `test_a_digit_subscript_reads_back_as_a_call_not_a_product` exists to
    prevent.  Measured when that test failed on the first version of the fix.
    """

    @pytest.mark.parametrize("latex,expected", [
        (r"\left(x+y\right)", "(x+y)"),
        (r"G_{0}\left(k , \omega\right)", "G_0(k , omega)"),
        (r"\left(\frac{a}{b}\right)^2", "(((a)/(b)))^2"),   # the fraction is
        # ONE unit `((a)/(b))` (LatexConverter._frac, 2026-08-14) inside the
        # delimiter pair's own parentheses — redundant, not wrong.
        (r"\left|x\right|", "|x|"),
    ])
    def test_delimiters_keep_their_spacing(self, engine, latex, expected):
        assert engine._handler.latex_to_canonical(latex) == expected

    @pytest.mark.parametrize("latex,expected", [
        (r"x_\alpha", "x_alpha"),
        (r"x^\alpha", "x^alpha"),
        (r"\alpha_\beta", "alpha_beta"),
    ])
    def test_a_command_after_sub_or_superscript_belongs_to_it(
            self, engine, latex, expected):
        """`_` and `^` are excluded from the rule: there the command IS part of
        what precedes it, and a space would break the subscript or exponent."""
        assert engine._handler.latex_to_canonical(latex) == expected
