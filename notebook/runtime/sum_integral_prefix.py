r"""Reading a Σ or ∫ written as a PREFIX — `\sum_k f_k`, `\int f dx`.

Extracted from `semantic_resolver.py` 2026-08-07, when the hygiene ratchet fired
(1386 > 1315).  Same move as `subscript_policy` and `julia_sympy_serialize`, and
the same reason: the growth was all one concern.

The concern is narrow and easy to state, which is why it deserves its own file:
**a sum's subscript is an index list, and must be read in that position.**
Handing it to the generic resolver first — which is what the caller did — asks a
question that has no business being asked here, and the generic answer was wrong
twice, silently (measured 2026-08-07):

* ``\sum_{m,n}`` — the same curly braces are the ANTICOMMUTATOR ``{A,B}``
  everywhere else, and with exactly two symbols that is what came back, so the
  sum was not recognised at all and the expression degraded to a product.
* ``\sum_{m,n,p}`` — three or more stayed a BracketExpr and only ``args[0]``
  was taken, so a triple sum silently became a single one.

`resolve_node_fn` and `binder` are passed in rather than imported, so this
module knows nothing about which backend is resolving.
"""
from __future__ import annotations

from contextlib import nullcontext

from notebook.models.ir import (
    Add, Expr, BracketExpr, Equation, FunctionCall, IndexSum, Integral, Juxtaposition,
    Limit, Mul, Number, Pow, Subscript, Superscript, Symbol,
)
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.shared_parser import UnknownNotationError


def _flatten_mul_juxt(expr: Expr) -> list[Expr]:
    args = []
    def flatten(e):
        if isinstance(e, (Mul, Juxtaposition)):
            if isinstance(e, Mul):
                for a in e.args: flatten(a)
            else:
                flatten(e.left)
                flatten(e.right)
        else:
            args.append(e)
    flatten(expr)
    return args


#: How a Σ / ∫ prefix is spelled once `latex_to_canonical` has run.
_SUM_HEADS = frozenset({"sum"})
#: Π.  Read by the SAME machinery as Σ — a binder that names dummies and takes
#: a body — because that is all the two have to agree on; only the operator
#: applied to the terms differs.  Left untaught, `\prod_i x_i` lowered to
#: `prod(i) * x(i)`: "the function prod at i" times `x(i)`, a plausible product
#: returned as a success (measured 2026-08-09).  A partition function
#: `Z = \prod_i (1 + e^{-\beta \epsilon_i})` is ordinary statistical mechanics.
_PROD_HEADS = frozenset({"prod", "product"})
_BINDER_HEADS = _SUM_HEADS | _PROD_HEADS
_INT_HEADS = frozenset({"int", "%int", "\\int"})
#: `\lim`.  Read by the SAME machinery as Σ and ∫ — a binder written as a
#: PREFIX, naming a variable and taking a body — because that is all they have
#: to agree on.  Left untaught, the plainest spellings there are came back as
#: something else, and as SUCCESSES (measured 2026-09-11):
#:
#:     \lim_{x \to 0} \frac{\sin x}{x}  →  lim[x, to, 0](sin(x) * x^-1)
#:     \lim_{n \to \infty} a_n          →  lim(n, to, inf) * a(n)      ← a PRODUCT
#:     \lim_{t' \to t^+} G(t,t')        →  one undefined symbol, raw LaTeX
#:
#: `\liminf` / `\limsup` are DIFFERENT binders and are deliberately absent.
_LIM_HEADS = frozenset({"lim"})



def _prefix_head(node) -> "str | None":
    r"""The name a prefix is BUILT ON — `sum` for `sum`, `sum_i`, `sum_{i=1}^{N}`.

    Walked down `.base` only: a subscript and a superscript DECORATE the head,
    they never replace it.  This is the whole discriminator against an ordinary
    applied subscript — `f_i(x)` is built on `f`, not on `sum`.
    """
    while isinstance(node, (Subscript, Superscript)):
        node = node.base
    return node.name if isinstance(node, Symbol) else None


def _give_back_bracket(node: Expr, heads) -> "tuple[Expr, Expr | None]":
    r"""A bracket written after a Σ/∫ prefix is APPLIED to it.  Hand it back.

    The parser is semantics-free, so `\sum_i (x_i + y_i)` and `f_i(x)` reach it
    as the same shape and it reads both as an application (CLAUDE.md, "A
    subscript that is APPLIED is part of the name").  For a binder that reading
    swallows the BODY, and what the body is applied to differs by spelling —
    measured 2026-08-08, each one silently:

    * ``\sum_i (x_i+y_i)`` — the subscript's own content is applied:
      ``Subscript(sum, FunctionCall(i, [body]))`` → the name ``sum_i``, Σ gone.
    * ``\sum_{m,n} (h+g)`` — same, with a braced index list, so it folded to a
      Maxima SUBSCRIPTED FUNCTION ``sum[m,n](…)`` — which the CAS accepts.
    * ``\int (x+1) dx`` — nothing decorates ``int``, so the integrand is applied
      to ``int`` ITSELF and the prefix stopped being the first factor.  Maxima
      then evaluated the product of three symbols: ``dx*int*x+dx*int``, returned
      as a SUCCESS.  An indefinite integral of a parenthesised integrand is
      ordinary writing, which makes this the worst of the three.
    * ``\sum_{i=1}^{N} (x_i+y_i)`` — the superscript's content is applied, so the
      upper bound became ``N(body)``.

    **A LIMIT is not a swallowed body, and telling them apart is exact**
    (2026-08-16).  A limit that is itself a one-argument call has the same shape
    as "the script applied to a bracketed body", and reading it the second way
    is silently wrong::

        \int_0^{\sqrt{2}} x dx   → integrate(2 * x, x, 0, sqrt)
        \int_{\sqrt{2}}^{2} x dx → integrate(2^2 * x, x)   ← bounds GONE
        \sum_{i=0}^{\sqrt{2}} i  → sum(2 * i, i, 0, sqrt)
        \int_0^{f(a)} x dx       → integrate(a * x, x, 0, f)

    ``\sqrt{2}`` as a limit of integration needs no defending, and the name
    became the bound while its argument moved into the integrand.  Two
    arguments (``g(a,b)``) or a coefficient (``2 f(a)``) escaped, which is what
    kept it hidden.

    The discriminator needs no guessing: **a SUPERSCRIPT is always a limit**
    (the case above is unreachable — the converter inserts a ``*`` after
    ``^{N}``), and a SUBSCRIPT is an index only for Σ/Π; on an ∫ it is a limit
    too.  So the give-back applies to a Σ/Π subscript and nowhere else.

    A decoration written on the BODY sits outside the application, so the two
    directions have to be told apart rather than merged — ``\int (x+1)^2 dx``
    parses as ``Superscript(FunctionCall(int, [x+1]), 2)``, and reading its
    superscript the way ``\sum_i^{N}(…)`` is read would give the PREFIX an
    exponent.  The shapes decide: a swallow in ``.sub``/``.script`` decorates the
    prefix, a swallow in ``.base`` decorates the body.  Left unread, that one
    stayed silently wrong after the rest were fixed: ``(int*(x+1))^2*dx``.

    Returns ``(prefix, body)``, or ``(node, None)`` when *node* is not a Σ/∫
    prefix carrying a swallowed body — including every ordinary ``f_i(x)``.
    """
    if isinstance(node, FunctionCall) and len(node.args) == 1:
        if _prefix_head(node.func) in heads:
            return node.func, node.args[0]
        return node, None
    if isinstance(node, Subscript):
        if (isinstance(node.sub, FunctionCall) and len(node.sub.args) == 1
                and _prefix_head(node.base) in heads
                and _prefix_head(node.base) not in _INT_HEADS):
            return Subscript(base=node.base, sub=node.sub.func), node.sub.args[0]
        prefix, body = _give_back_bracket(node.base, heads)
        if body is not None:
            return prefix, Subscript(base=body, sub=node.sub)
    elif isinstance(node, Superscript):
        prefix, body = _give_back_bracket(node.base, heads)
        if body is not None:
            return prefix, Superscript(base=body, script=node.script)
    return node, None


def _count_prefixes(args: list[Expr], heads) -> int:
    """How many Σ/∫ signs stand among these factors."""
    return sum(1 for node in args
               if _prefix_head(_give_back_bracket(node, heads)[0]) in heads)


def _first_prefix_index(args: list[Expr], heads) -> "int | None":
    r"""Where the first Σ/∫ prefix sits in a flattened product — not just at 0.

    Only the first factor used to be examined, so a binder anywhere else was
    never read: the second sum of `\sum_i x_i \sum_j y_j` collapsed to the bare
    symbol `sum(j)` and Maxima refused the command (open_items B-10 ②-d,
    measured 2026-08-08).  A product of sums is ordinary physics writing.

    A candidate in the LAST position with nothing applied to it does not count.
    A binder with nothing to bind is not a binder, and reading it as a sum over
    an empty body would answer `sum(1, j, …)` — plausible, and invented — where
    today the CAS refuses out loud.
    """
    for k, node in enumerate(args):
        stripped, body = _give_back_bracket(node, heads)
        if _prefix_head(stripped) in heads and (body is not None or k < len(args) - 1):
            return k
    return None


def _split_at_prefix(args: list[Expr], heads):
    """`[a, b, prefix(body), c]` → `([a, b], prefix, [body, c])`, or None.

    A body handed back is FLATTENED into the tail, so what follows the prefix is
    one list of factors however it was bracketed.  `\\int \\frac{d\\omega}{2\\pi}
    G(\\omega)` is why: the bracket puts `d`·`omega` inside one Mul, and left
    nested the measure reader saw a single node that was not a differential.
    """
    at = _first_prefix_index(args, heads)
    if at is None:
        return None
    prefix, body = _give_back_bracket(args[at], heads)
    tail = (_flatten_mul_juxt(body) if body is not None else []) + args[at + 1:]
    return args[:at], prefix, tail


def _as_one(nodes: list[Expr], resolve_node_fn, context: SymbolContext) -> Expr:
    """Resolve several factors AS ONE product, so a binder among them is read."""
    if len(nodes) == 1:
        return resolve_node_fn(nodes[0], context)
    return resolve_node_fn(Mul(args=list(nodes)), context)


def _resolve_body(body_args: list[Expr], resolve_node_fn,
                  context: SymbolContext) -> Expr:
    r"""A binder's body — resolved as ONE thing when it contains another binder.

    Factor by factor (which is what this did) an inner Σ/∫ has nothing to bind:
    `\sum_i x_i \sum_j y_j` came out as `sum(x(i) * sum(j) * y(j), i, …)`.
    Handing the tail back whole lets the inner binder read its own body exactly
    the way this one did — the conventional reading, where a Σ binds everything
    written to its right.
    """
    if not body_args:
        return Number(value=1)
    if _first_prefix_index(body_args, _BINDER_HEADS | _INT_HEADS) is not None:
        return _as_one(body_args, resolve_node_fn, context)
    if len(body_args) == 1:
        return resolve_node_fn(body_args[0], context)
    return Mul(args=[resolve_node_fn(a, context) for a in body_args])


def may_be_prefix(node: Expr) -> bool:
    r"""Could *node* be carrying a Σ/∫ prefix?  The resolver's gate.

    A prefix normally sits in a PRODUCT — which is the shape the readers below
    flatten, and how they have always been reached.  A BRACKETED body is read as
    an application instead, and then there is no product left to arrive on:
    `\sum_i (x_i+y_i)` IS the whole expression.  So that shape needs its own way
    in — and only that shape does.

    A BARE prefix — `\sum_j` with nothing applied AND nothing after it —
    deliberately does NOT qualify, here or in `_first_prefix_index`.  A binder
    with nothing to bind is not a binder, and reading it as a sum over an empty
    body would answer `sum(1, j, …)`: plausible, and invented.  As it stands the
    CAS refuses the command out loud.
    """
    return (isinstance(node, (Juxtaposition, Mul))
            or _give_back_bracket(node, _BINDER_HEADS | _INT_HEADS | _LIM_HEADS)[1]
            is not None)


def _differential_var(node: Expr) -> "Symbol | None":
    r"""`dx` → `x`.  One written token, so `d` and the variable are joined.

    **A SUBSCRIPTED differential is one too** (`dx_1` → `x_1`). It arrives as
    `Subscript(base=Symbol('dx'), sub=…)` because the subscript splits the
    token, so the symbol test above never saw it and the measure was not read
    at all — measured 2026-09-17, `\int f dx_1` came back as the PRODUCT
    `dx(1)*f*int`, `completed`: `\int` left as an ordinary symbol named `int`,
    and a differential turned into a factor. `\int f dx` / `\int x^2 dx` /
    `\int f(x) dx` were all correct, so **only the indexed spelling** fell.

    Only a NUMBER or a plain SYMBOL subscript is read (`dx_1`, `dx_i`): those
    are the spellings this system writes for an indexed name. Anything else is
    left alone rather than guessed at.
    """
    if isinstance(node, Symbol) and node.name.startswith("d") and len(node.name) > 1:
        name = node.name[1:]
        return Symbol(name=name[1:] if name.startswith("\\") else name)
    if isinstance(node, Subscript):
        base = _differential_var(node.base)
        if base is None:
            return None
        # **添字は付け替えるのであって、名前に畳み込まない。** 畳むと
        # `Symbol("x_1")` になり、被積分関数の側の `x_1`（未宣言なら添字として
        # `x(1)` に降りる）と**別の CAS 物**になる——実測 2026-09-17、
        # `\int x_1 dx_1` が `x(1)*x_1` を `completed` で返した。見える積
        # （`dx(1)*f*int`）を**見えない積**に変えただけで、前より悪い。
        # 書かれたままの部分木を返し、**被積分関数と同じ解決器**に通す。
        if isinstance(node.sub, (Number, Symbol)):
            return Subscript(base=base, sub=node.sub)
    return None


def _is_bare_d(node: Expr) -> bool:
    return isinstance(node, Symbol) and node.name == "d"


def _read_measure(rest: list[Expr]) -> "tuple[list[Symbol], list[Expr], int | None] | None":
    r"""Which variables an ∫ integrates over, and what is left as the integrand.

    Returns ``(vars OUTERMOST-first, body, measure_dim)`` or ``None`` when *rest*
    carries no differential at all.  *measure_dim* is set only for a VOLUME
    measure (``d^3k``); an ordinary integral leaves it ``None``.  Four
    spellings, all ordinary and three of them silently wrong before
    2026-08-08:

    * ``\int f dx`` — trailing, and SEVERAL may trail (``\int x y dx dy``).  The
      first one written is the INNERMOST, which is the convention the two-∫ form
      ``\int\int f dx dy`` already followed.
    * ``\int dx f`` — leading, and several may lead (``\int dx dy f``).  There
      the first one written is the OUTERMOST.  Only ONE was read: ``\int dx dy x
      y`` integrated over `x` and kept `dy` as a FACTOR — ``(dy·x²·y)/2``,
      returned as a success.
    * ``\int d\omega f`` — the differential of a Greek or command-spelled
      variable.  ``\frac{dk}{2\pi}`` works because `dk` is one token, but
      `_separate_adjacent_commands` (CLAUDE.md) has to split `d\omega` into
      `d`·`omega`, so the measure of every frequency integral in Green's
      function work arrived as a product of two symbols named `d` and `omega`:
      ``\int \frac{d\omega}{2\pi} G(\omega)`` came back as
      ``(d·int·omega·G(omega))/(2π)``.

    A bare ``d`` is read as a differential only when a SYMBOL follows it, and
    only after the trailing form has been tried — so ``\int d x^2 dx``, where
    `d` is an ordinary constant, still integrates over `x` with `d` in the
    integrand.
    """
    if not rest:
        return None

    # Trailing: `\int f dx dy` — scan back while differentials keep coming.
    # A differential is ONE node (`dx`) or TWO (`d`·`\theta`): the leading form
    # below has always read the split pair, but the trailing form did not — and
    # the trailing form is the one THIS SYSTEM PRINTS.  `\int_0^{2\pi} r\,dr\,
    # d\theta` is in the published coordinate-jacobian examples, and re-reading
    # it produced a Mul: the integral silently became a product, and with an
    # infinite bound it raised a raw pydantic ValidationError at the researcher
    # (measured 2026-08-14, SCENARIO-MACRO case 5).  Every Greek integration
    # variable is written this way — dθ, dφ, dτ, dω.
    end = len(rest)
    while end > 0:
        if _differential_var(rest[end - 1]) is not None:
            end -= 1
            continue
        if (end >= 2 and isinstance(rest[end - 1], Symbol)
                and _is_bare_d(rest[end - 2])):
            end -= 2
            continue
        break
    if end < len(rest):
        # ONE trailing differential belongs to each ∫ still standing in the
        # body; this ∫ takes what is left.  Taking them all greedily broke
        # `\int\int x y dx dy`, whose inner sign was then left in the integrand
        # (`integrate(integrate(int*x*y, x), y)`) — measured while fixing the
        # single-∫ form, and the reason the old one-differential slice worked.
        # Read the trailing run into (variable, width) pairs so a split
        # `d`·`\theta` counts as ONE differential, not two.
        trailing: list[tuple[Symbol, int]] = []
        at = end
        while at < len(rest):
            var = _differential_var(rest[at])
            if var is not None:
                trailing.append((var, 1))
                at += 1
            else:
                trailing.append((rest[at + 1], 2))
                at += 2
        inner = _count_prefixes(rest[:end], _INT_HEADS)
        take = max(1, len(trailing) - inner)
        mine_pairs = trailing[len(trailing) - take:]
        cut = len(rest) - sum(w for _v, w in mine_pairs)
        # First written is innermost, so reverse into outermost-first order.
        return [v for v, _w in reversed(mine_pairs)], rest[:cut], None

    # Leading: `\int dx dy f`, `\int d\omega f`.  First written is outermost.
    leading: list[Symbol] = []
    at = 0
    while at < len(rest):
        var = _differential_var(rest[at])
        if var is not None:
            leading.append(var)
            at += 1
            continue
        if _is_bare_d(rest[at]) and at + 1 < len(rest):
            if isinstance(rest[at + 1], Symbol):
                leading.append(rest[at + 1])
                at += 2
                continue
        # `d^3` is a Superscript at the surface and a Pow once resolved; this
        # runs on the surface, but both are accepted so the check does not
        # depend on which side of the resolver it is called from.
        if (isinstance(rest[at], (Pow, Superscript))
                and _is_bare_d(getattr(rest[at], "base", None))
                and at + 1 < len(rest) and isinstance(rest[at + 1], Symbol)):
            # `d^3k` — the volume element of field theory, and the measure of
            # `\int \frac{d^3k}{(2\pi)^3}`.  Left unread it came back as the
            # PRODUCT `(d³·int·k·f(k))/(8π³)`, a plausible answer to a question
            # nobody asked (measured 2026-08-08).  It is READ here, dimension
            # and all; there is no d-dimensional integral to hand the CAS, so
            # the refusal happens at code generation — by which point the
            # researcher has been able to WRITE the expression, see it, and
            # expand the vector into components.
            _dim = getattr(rest[at], "exponent", None) or getattr(rest[at], "script", None)
            _n = getattr(_dim, "value", None)
            if isinstance(_n, int) and _n > 0:
                return [rest[at + 1]], rest[at + 2:], _n
            break
        break
    if leading:
        return leading, rest[at:], None
    return None


def try_resolve_integral(expr: Expr, resolve_node_fn, context: SymbolContext) -> Expr | None:
    split = _split_at_prefix(_flatten_mul_juxt(expr), _INT_HEADS)
    if split is None: return None
    lead, int_op, rest = split

    read = _read_measure(rest)
    if read is None:
        return None
    var_syms, body_args, measure_dim = read

    lower = None
    upper = None
    if isinstance(int_op, Subscript):
        lower = resolve_node_fn(int_op.sub, context)
    elif isinstance(int_op, Superscript):
        upper = resolve_node_fn(int_op.script, context)
        if isinstance(int_op.base, Subscript):
            lower = resolve_node_fn(int_op.base.sub, context)

    # `var_syms` is written OUTERMOST first, so the innermost integral is built
    # first and the written range decorates the outermost — the one the ∫ sign
    # it was written on actually carries.
    result = _resolve_body(body_args, resolve_node_fn, context)
    # **積分変数は被積分関数と同じ扉を通す。** 素の `dx` では恒等だが、
    # 添字つき `dx_1` では効く: 通さないと測度だけが `x_1`（原子）に、
    # 被積分関数は `x(1)`（添字）になり、CAS は被積分関数を定数と見る。
    var_syms = [resolve_node_fn(v, context) for v in var_syms]
    for _v in var_syms:
        if not isinstance(_v, Symbol):
            # **IR が「積分変数は記号である」と言っている**（`Integral.var`）。
            # 未宣言の添字つきはここで `IndexedSymbol` になるので、そのまま
            # 渡すと生の pydantic ValidationError が研究者に返る——このコード
            # ベースが繰り返し直してきた形である。断りは既に在るので、それを
            # 使う（文の綴り「… を x_1 で 0 から 1 まで定積分して」が返すのと
            # **同じ理由・同じ2つの出口**——一つの問いに一つの答え）。
            _base = getattr(getattr(_v, "base", None), "name", None) or \
                getattr(_v, "name", "")
            _idx = ", ".join(str(getattr(i, "name", getattr(i, "value", i)))
                             for i in (getattr(_v, "indices", None) or []))
            _written = f"{_base}_{_idx}" if _idx else _base
            raise UnknownNotationError("integral.var_is_indexed",
                                       var=_written, base=_base,
                                       lowered=f"{_base}({_idx})")
    for var_sym in reversed(var_syms):
        _outermost = var_sym is var_syms[0]
        result = Integral(expr=result, var=var_sym,
                          lower=lower if _outermost else None,
                          upper=upper if _outermost else None,
                          measure_dim=measure_dim)
    # Factors written BEFORE the ∫ are not under it: `a \int f dx` is
    # `a·∫f dx`.  They used to stop the integral being recognised at all.
    if lead:
        return Mul(args=[_as_one(lead, resolve_node_fn, context), result])
    return result

#: 片側の印。`_limit_direction`（変換器の入口）が `^+` / `^-` をこの語に開いている
#: ので、ここは1つの綴りだけを知っていればよい。
_LIMIT_SIDES = {"plus": "plus", "minus": "minus"}
#: 矢印。同じく入口で1つに畳んである。
_LIMIT_ARROW_TOKENS = frozenset({"to"})


def _read_limit_spec(sub, resolve_node_fn, context):
    r"""``{x, to, 0}`` / ``{t', to, t, plus}`` → ``(var, point, direction)``.

    ``None`` when the subscript is not a limit's — a `\lim` whose subscript says
    something else is left alone rather than guessed at.
    """
    if not isinstance(sub, BracketExpr):
        return None
    parts = list(sub.args)
    direction = None
    if parts and isinstance(parts[-1], Symbol) and parts[-1].name in _LIMIT_SIDES:
        direction = _LIMIT_SIDES[parts.pop().name]
    if len(parts) != 3:
        return None
    var_node, arrow, point_node = parts
    if not (isinstance(arrow, Symbol) and arrow.name in _LIMIT_ARROW_TOKENS):
        return None
    var = resolve_node_fn(var_node, context)
    if not isinstance(var, Symbol):
        return None
    return var, resolve_node_fn(point_node, context), direction


def try_resolve_limit(expr: Expr, resolve_node_fn, context) -> "Expr | None":
    r"""Read a `\lim` written as a PREFIX — `\lim_{x \to 0} f(x)`.

    Mirrors :func:`try_resolve_integral`: the binder takes the following TERM,
    not the whole sum, which is the convention this project already applies to
    Σ (`\sum_i x_i + y` is `sum(x(i), i) + y`).  Factors written BEFORE the
    `\lim` are not under it.
    """
    lead, lim_op, rest = (_split_at_prefix(_flatten_mul_juxt(expr), _LIM_HEADS)
                          or (None, None, None))
    if lim_op is None:
        return None
    if not isinstance(lim_op, Subscript):
        return None          # a bare `\lim` binds nothing; not a limit
    spec = _read_limit_spec(lim_op.sub, resolve_node_fn, context)
    if spec is None or not rest:
        return None
    var, point, direction = spec
    result = Limit(expr=_resolve_body(rest, resolve_node_fn, context),
                   var=var, point=point, direction=direction)
    if lead:
        return Mul(args=[_as_one(lead, resolve_node_fn, context), result])
    return result


def _unwrap_group(node):
    r"""`^{N}` arrives as a one-element brace group; the bound is the element.

    Left wrapped, the generator renders it as the backend's LIST literal —
    `sum(x(i), i, 1, [N])` (measured 2026-08-08).  Braces here are LaTeX
    grouping, not a collection.
    """
    if isinstance(node, BracketExpr) and len(node.args) == 1:
        return node.args[0]
    return node


def _one_sum_index(node, resolve_node_fn, context):
    """Read ONE dummy index (and an optional lower bound) from a sum subscript.

    This is today's single-index logic, unchanged, factored out so the
    multi-index case can reuse it instead of restating it.
    """
    res = resolve_node_fn(node, context)
    lower = None
    if isinstance(res, Equation):
        idx, lower = res.lhs, res.rhs
    else:
        idx = res
    # Robust extraction for a mis-parsed `=`, e.g. Juxtaposition(i, 1) -> Mul(i, 1)
    if isinstance(idx, Mul) and len(idx.args) > 0:
        idx = idx.args[0]
    if isinstance(idx, Juxtaposition):
        idx = idx.left
    return ([idx] if idx is not None else []), lower


#: A CONSTRAINT token that may stand among a sum's indices, and how it reads
#: back.  `_subscript_constraint` in the LaTeX converter has already made every
#: spelling one of these, so nothing here has to know that `<` and `\lt` and
#: `\langle` are the same question.
_CONSTRAINT_MARKS = {
    "ne": r"\ne", "neq": r"\ne", "lt": "<", "gt": ">",
    "le": r"\le", "leq": r"\le", "ge": r"\ge", "geq": r"\ge",
    "langle": r"\langle\,\rangle", "rangle": r"\langle\,\rangle",
}


def _sum_dummy_indices(sub, resolve_node_fn, context):
    r"""The dummy indices of a sum, read IN THE POSITION WHERE THEY APPEAR.

    A sum's subscript is an index list and nothing else.  Resolving it with the
    generic resolver first — which is what this did — asks a question that has
    no business being asked here, and the generic answer is wrong twice
    (measured 2026-08-07, both silent):

    * ``\sum_{m,n}`` — the SAME curly braces mean the ANTICOMMUTATOR ``{A,B}``
      elsewhere, and with exactly two symbols that is what came back.  An
      Anticommutator matched none of the caller's branches, so the sum was not
      recognised at all and the whole expression degraded to a product:
      ``\sum_{m,n} h_{m,n}`` reached Maxima as ``sum(m, n) * h(m, n)``.
    * ``\sum_{m,n,p}`` — three or more args stayed a BracketExpr, and the
      caller took ``args[0]``, so the second and third indices were dropped in
      silence: a triple sum became ``sum(h(m, n), m)``.  That one is the worse
      shape of the two, because a plausible answer comes back.

    Returns ``(index_nodes, lower, constraint)``; ``lower`` is only meaningful
    for the single-index ``\sum_{i=1}`` form, exactly as before, and
    ``constraint`` is the condition written among the indices (``None`` for an
    ordinary sum).
    """
    if isinstance(sub, BracketExpr) and sub.args:
        names = [getattr(a, "name", None) for a in sub.args]
        marks = [n for n in names if n in _CONSTRAINT_MARKS]
        if marks:
            # `\sum_{j>i}^{M}` — a relation whose right side is the LOWER limit.
            # For a summation index (an integer by construction) `j > i` IS
            # `j` from `i+1`, so this is a rewrite, not a guess.  Only the
            # three-token shape `index REL bound` qualifies; a chain
            # (`i<j<k`) has more and stays refused.
            if len(sub.args) == 3 and names[1] in ("gt", "ge"):
                idxs, _ = _one_sum_index(sub.args[0], resolve_node_fn, context)
                bound = resolve_node_fn(sub.args[2], context)
                if names[1] == "gt":
                    bound = Add(args=[bound, Number(value=1)])
                return idxs, bound, None
            out = []
            for arg, name in zip(sub.args, names):
                if name in _CONSTRAINT_MARKS:
                    continue
                idxs, _ = _one_sum_index(arg, resolve_node_fn, context)
                out.extend(idxs)
            return out, None, _CONSTRAINT_MARKS[marks[0]]
        if len(sub.args) == 1:
            idxs, lower = _one_sum_index(sub.args[0], resolve_node_fn, context)
            return idxs, lower, None
        out = []
        for arg in sub.args:
            idxs, _ = _one_sum_index(arg, resolve_node_fn, context)
            out.extend(idxs)
        return out, None, None
    idxs, lower = _one_sum_index(sub, resolve_node_fn, context)
    return idxs, lower, None


def try_resolve_sum(expr: Expr, resolve_node_fn, context: SymbolContext,
                    binder=None) -> Expr | None:
    _args = _flatten_mul_juxt(expr)
    split = _split_at_prefix(_args, _BINDER_HEADS)
    if split is None: return None
    lead, sum_op, body_args = split
    _op = "product" if _prefix_head(sum_op) in _PROD_HEADS else "sum"
    _constraint = None
    # Is this Σ INSIDE an ∫?  Ask the integral reader rather than guessing from
    # position — `\int_0^1 x dx \sum_{i=1}^{3} i` also has an ∫ to the left, and
    # there the two are a PRODUCT (that ∫ already took its measure).  Deferring
    # on position alone broke it.
    #
    # Resolved first, a Σ that really is inside swallowed everything after it —
    # the integrand AND the differential — and left the ∫ standing as an
    # ordinary symbol (measured 2026-08-09):
    #
    #   \int \sum_i f_i(x) dx   →  int * sum(f[i](x) * dx, i)
    #   \int dx \sum_i f_i(x)   →  integrate(1, x) * sum(...)   ← a valid `∫1dx`
    #
    # The second is the dangerous one: a well-formed integral of the wrong
    # thing.  Only `\int \left(\sum_i …\right) dx` came out right, and nobody
    # brackets an expansion written under an integral.
    _int_split = _split_at_prefix(_args, _INT_HEADS)
    if _int_split is not None:
        _read = _read_measure(_int_split[2])
        if _read is not None and _count_prefixes(_read[1], _BINDER_HEADS):
            return None
    idx_syms: list = []
    lower = None
    upper = None
    if isinstance(sum_op, Subscript):
        idx_syms, lower, _constraint = _sum_dummy_indices(
            sum_op.sub, resolve_node_fn, context)
    elif isinstance(sum_op, Superscript):
        upper = _unwrap_group(resolve_node_fn(sum_op.script, context))
        if isinstance(sum_op.base, Subscript):
            idx_syms, lower, _constraint = _sum_dummy_indices(
                sum_op.base.sub, resolve_node_fn, context)

    if idx_syms:
        # This path already read the indices before the body; the binding just
        # makes that fact available to the subscript policy (docs/33 SS9.1).
        # EVERY dummy is bound, not only the first — inside `\sum_{m,n}` both
        # `m` and `n` are indices because the sum says so.
        from contextlib import nullcontext
        _names = [getattr(s, "name", None) for s in idx_syms]
        with (binder(_names) if binder is not None else nullcontext()):
            body = _resolve_body(body_args, resolve_node_fn, context)

        from notebook.models.ir import IndexSymbol
        dummies = [_as_dummy(s) for s in idx_syms]
        # The generator already nests one `sum(...)` per dummy, outermost
        # first — the researcher's own suggestion, and the same iterated shape
        # `evaluate_integral_form` uses for a multiple integral.  Measured:
        # nesting order does not change the result (`ratsimp(diff) = 0`).
        #
        # A range the researcher WROTE is carried through.  It used to be read
        # into these two locals and then dropped on the floor.  Attaching it
        # needs nothing invented — Maxima's `sum` takes four arguments, and a
        # written range is what makes the four-argument form available.
        #
        # It does not help yet: `\sum_{i=1}^{N}` delivers the UPPER bound only.
        # The lower one never arrives, because the lexer has no `=` token and a
        # non-top-level `=` is silently skipped (`{i=1}` parses as
        # `Juxtaposition(i, 1)`) — the same failure the parser's own comment
        # names for a stray `<`/`>`, which IS guarded.  Deciding that is a
        # separate, wider change.
        #
        # Not per-index on purpose: a superscript bound written over several
        # indices cannot be attributed to one of them without guessing, and a
        # sum with NO range stays rangeless rather than acquiring a default —
        # a range is a physics choice, not something to fill in here.
        # A CONSTRAINED pair sum is the one multi-index case where a written
        # bound CAN be attributed: `\sum_{i<j}^{N}` means both indices run to
        # N, because the constraint is precisely what ties them into ONE range.
        # Without this the superscript was read and then dropped on the floor —
        # `\sum_{i<j}^{N} V_{i,j}` reached the generator with `upper=None`,
        # silently (measured 2026-08-17).
        from notebook.runtime.constrained_sum import is_relational
        _single = len(dummies) == 1 or (
            len(dummies) == 2 and is_relational(_constraint))
        result = IndexSum(
            body=body,
            dummy_indices=dummies,
            lower=lower if _single else None,
            upper=upper if _single else None,
            op=_op,
            constraint=_constraint,
        )
        # Factors written BEFORE the Σ are not under it: `a \sum_i x_i` is
        # `a·Σ`, and `x_i \sum_j y_j` is a product of a factor and a sum.
        # They used to stop the sum being recognised at all, because only the
        # first factor was examined.
        if lead:
            return Mul(args=[_as_one(lead, resolve_node_fn, context), result])
        return result
    return None


def _as_dummy(node):
    r"""束縛添字を1つの ``IndexSymbol`` にする。

    ``\sum_{\mathbf k'}`` —— **ベクトル値の添字**は、走らせる名前が中の記号である。
    変換器がベクトルを ``vec(k')`` と包むので、包みの外側では和の添字にならず、
    ``IndexSum.dummy_indices`` の型検査で**生の pydantic ValidationError** が
    研究者に出ていた（実測 2026-08-20、1次の自己エネルギー
    ``\Sigma^{(1)}(\mathbf k)=\sum_{\mathbf k'} v(\mathbf k-\mathbf k')n_{\mathbf k'}``）。
    名前を走らせれば本体の ``vec(k')`` の中に届く（Maxima で
    ``subst(k'=1, f(vec(k')))`` は ``f(vec(1))``）ので、包みを剥がすのが読みになる。
    """
    from notebook.models.ir import FunctionCall, IndexSymbol, Symbol
    if isinstance(node, IndexSymbol):
        return node
    if isinstance(node, Symbol):
        return IndexSymbol(name=node.name)
    if (isinstance(node, FunctionCall) and isinstance(node.func, Symbol)
            and node.func.name == "vec" and len(node.args) == 1
            and isinstance(node.args[0], Symbol)):
        return IndexSymbol(name=node.args[0].name)
    return node
