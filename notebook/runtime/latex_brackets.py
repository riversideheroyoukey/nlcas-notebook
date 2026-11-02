r"""Balanced-delimiter scanning over LaTeX text.

Nothing here knows about a CAS.  It lives in its own module because BOTH the
shared display layer and a backend's own LaTeX repairs need it, and a shared
layer importing a module named after one backend is exactly what the hygiene
guard in ``tests/test_code_hygiene.py`` exists to stop — Core Principle #10.

Why balanced scanning rather than a regex, every time: a LaTeX operand
routinely contains delimiters of its own.  ``\left(f(t)+g(t)\right)`` has
ordinary parentheses inside it, and ``f(g(x))^{\ast}`` has a nested call — a
regex cuts at the first closer and produces something that still parses, which
is how these turn into wrong answers rather than errors.
"""

#: The two spellings an argument list arrives in.  Maxima's ``tex1`` writes the
#: ``\left…\right`` pair; the H-1 LHS formatter writes bare parentheses.  Order
#: matters: the long spelling has to be tried first.
PAREN_PAIRS = ((r"\left(", r"\right)"), ("(", ")"))


def balanced(s: str, i: int, open_tok: str, close_tok: str) -> "tuple[str, int] | None":
    r"""Read a balanced *open_tok* … *close_tok* run starting at *i*.

    Returns ``(contents, index_after)``, or ``None`` when *s* does not open
    with *open_tok* at *i* or the run never closes.
    """
    if not s.startswith(open_tok, i):
        return None
    depth, j = 0, i
    while j < len(s):
        if s.startswith(open_tok, j):
            depth += 1
            j += len(open_tok)
        elif s.startswith(close_tok, j):
            depth -= 1
            j += len(close_tok)
            if depth == 0:
                return s[i + len(open_tok):j - len(close_tok)], j
        else:
            j += 1
    return None


def brace_group(s: str, i: int) -> "tuple[str, int] | None":
    """Read a balanced ``{...}`` starting at *i*; return (content, index after)."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return None
