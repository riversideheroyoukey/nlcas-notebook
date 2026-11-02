"""F-1 / spin-algebra / graph early intercepts (lattice sums, dispersion,
crystal descriptors, spectra, matrix transforms).

Extracted verbatim from interpret_pipeline.py (refactor Stage 2,
2026-07-04).  run_interpret_pipeline imports this back; behaviour is
unchanged.
"""
from __future__ import annotations

import logging
import re
from functools import partial
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.shared_parser import split_relation, strip_lhs
from ui.interpret_common import (  # noqa: F401
    _EXPR_ID_RE,
)
from ui.interpret_f1 import (  # noqa: F401
    _is_save_lattice_figure,
    _parse_bond_sum_commutator,
    _parse_bond_sum_substitute,
    _parse_concretize,
    _parse_crystal,
    _parse_dispersion,
    _parse_energy_moment,
    _parse_graph_spec,
    _parse_neighbor_sum,
    _parse_neighbor_sum_concretize,
    _parse_specific_heat,
    _parse_spectral_function,
    _parse_spin_commutator,
    field_unstated_refusal,
    _parse_spin_spectral,
    _parse_symbolic_bond_sum,
    _relabel_sites,
    _token_sub,
)
from ui.interpret_l1 import (  # noqa: F401
    _new_basis_symbols,
    _parse_inner_product,
    _parse_matrix_transform,
)
from ui.interpret_pair_rep import (
    _try_heisenberg_operator, _try_pair_bracket)
from ui.interpret_registry import (
    Interceptor, InterceptorContext, run_interceptors,
)
from ui.interpret_spin_species import (  # noqa: F401
    _try_declare_site_spins,
    _try_spin_expectation,
)

_logger = logging.getLogger(__name__)


async def _try_early_intercepts(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    """Early F-1 / L-1 / spin-algebra rule-based intercepts (run on the raw text
    before token registration).  Returns a response dict to short-circuit the
    pipeline, or None to continue.  R1c (docs/27): the former giant if/return
    chain is now an ordered registry — the firing order and each entry's
    rationale are the data in _EARLY_INTERCEPTORS, not code position.
    Adding an early intercept is "write a _try_early_* + one list entry"."""
    return await run_interceptors(
        InterceptorContext(engine=engine, loop=loop, lang=lang,
                           text=_f1_raw, registered=registered),
        _EARLY_INTERCEPTORS)


# ---------------------------------------------------------------------------
# R1c batch 4 (docs/27): the head interceptors (inner product, explicit-matrix
# inverse transform, lattice figure save, neighbour/bond sums), extracted verbatim
# from _try_early_intercepts so that function is now a thin registry dispatcher.
# These fire FIRST (before the spin workflow), so they head the list.
# ---------------------------------------------------------------------------
async def _try_early_inner_product(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-ip. L-1 (non-operator metric): declare an inner product ⟨e_i,e_j⟩ = v into
    # the session Gram matrix.  Intercepted on the raw text BEFORE H-1 would register
    # it as an ordinary equation.
    _ip = _parse_inner_product(_f1_raw)
    if _ip:
        _ipa, _ipb, _ipv = _ip
        if not hasattr(engine, "gram_inner_products"):
            engine.gram_inner_products = {}
        engine.gram_inner_products[(_ipa, _ipb)] = _ipv
        return {"status": "completed", "operation": "declare_inner_product",
                "message": t("interpreter.inner_product_declared", lang),
                "registered_expressions": registered}
    return None


async def _try_early_matrix_transform(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-l1m. L-1 explicit-matrix inverse transform: "M = [[…]]、旧基底 …、新基底 … の
    # 逆変換".  Intercepted on the raw text BEFORE inline-formula splitting would
    # register "M = [[…]]" as a standalone equation.
    _l1m = _parse_matrix_transform(_f1_raw)
    if _l1m:
        _m_eids: list[str] = []
        for _i, _newsym in enumerate(_l1m["new"]):
            _terms_c, _terms_l = [], []          # canonical (CAS *) vs display (spaces)
            for _j, _oldsym in enumerate(_l1m["old"]):
                _coef = _l1m["M"][_i][_j].strip()
                if _coef in ("0", "-0", ""):
                    continue
                if _coef == "1":
                    _terms_c.append(_oldsym); _terms_l.append(_oldsym)
                else:
                    _terms_c.append(f"{_coef}*{_oldsym}")
                    _terms_l.append(f"{_coef} {_oldsym}")
            _rhs_c = " + ".join(_terms_c).replace("+ -", "- ") or "0"
            _rhs_l = " + ".join(_terms_l).replace("+ -", "- ") or "0"
            # canonical keeps '*' for the CAS; the user-facing latex uses the
            # space-separated math-input convention (no backend '*').
            _mexpr = await loop.run_in_executor(
                None, partial(engine.register_expression,
                              f"{_newsym} = {_rhs_c}", latex=f"{_newsym} = {_rhs_l}"))
            _m_eids.append(_mexpr.display_id)
        _m_args = {
            "equations": ", ".join(_m_eids),
            "vars": ", ".join(_l1m["old"]),
            "new_basis": ", ".join(_new_basis_symbols(engine, _m_eids)),
        }
        try:
            _m_step, _ = await loop.run_in_executor(
                None, partial(engine.run_operation, "build_inverse_transform", _m_args))
        except Exception as _m_exc:
            return {"status": "failed", "error": str(_m_exc),
                    "registered_expressions": registered}
        _m_ids = _m_step.output_refs or []
        for e in (engine.expressions[i] for i in _m_ids if i in engine.expressions):
            registered.append({"expr_id": e.display_id,
                               "canonical": e.canonical, "latex": e.latex})
        return {"status": "completed", "operation": "build_inverse_transform",
                "expr_id": _m_ids[0] if _m_ids else None,
                "registered_expressions": registered}
    return None


async def _try_early_save_lattice_figure(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-fig. "格子図を保存して" → save a standalone figure of the latest lattice.
    if _is_save_lattice_figure(_f1_raw, lang):
        from notebook.runtime.lattice_figure import save_lattice_figure_for_engine
        from ui import session_io
        _fig = await loop.run_in_executor(
            None, partial(save_lattice_figure_for_engine, engine,
                          out_dir=session_io.sessions_dir()))
        if not _fig.get("ok"):
            return {"status": "failed",
                    "error": t("lattice.no_lattice", lang),
                    "registered_expressions": registered}
        _paths = ", ".join(p for p in (_fig.get("svg"), _fig.get("pdf"),
                                       _fig.get("tex")) if p)
        _msg = t("lattice.figure_saved", lang, paths=_paths)
        if _fig.get("pdf_error"):
            _msg += t("lattice.pdf_skipped", lang, reason=_fig["pdf_error"])
        return {"status": "completed", "operation": "save_lattice_figure",
                "message": _msg, "registered_expressions": registered}
    return None


async def _try_early_neighbor_sum(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # Concretize a stored symbolic bond sum ("En を N サイト…で展開") OR expand a
    # fresh neighbor sum — both run expand_neighbor_sum through the same dispatch.
    _f1_parsed = _parse_concretize(_f1_raw, engine, lang) or _parse_neighbor_sum(_f1_raw, lang)
    if _f1_parsed:
        _f1_op, _f1_args = _f1_parsed
        try:
            _f1_step, _f1_raw_out = await loop.run_in_executor(
                None, partial(engine.run_operation, _f1_op, _f1_args)
            )
        except Exception as _f1_exc:
            return {"status": "failed", "error": str(_f1_exc),
                    "registered_expressions": registered}
        _f1_out = _f1_step.output_refs or []
        for _f1_id in _f1_out:
            _f1_e = engine.expressions.get(_f1_id)
            if _f1_e:
                registered.append({
                    "expr_id":   _f1_e.display_id,
                    "canonical": _f1_e.canonical,
                    "latex":     _f1_e.latex,
                })
        return {
            "status":    "completed",
            "operation": _f1_op,
            "expr_id":   _f1_out[0] if _f1_out else None,
            "registered_expressions": registered,
        }
    return None


async def _try_early_neighbor_sum_concretize(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-nconc. B stage 3b: concretize a neighbour-sum EOM at a site — build the
    # local Hamiltonian (bonds touching the site) via pin_site, then commute it
    # with the operator at that site (live, normal-ordered result).
    _nconc = _parse_neighbor_sum_concretize(_f1_raw, engine, lang)
    if _nconc:
        try:
            _hs_step, _ = await loop.run_in_executor(
                None, partial(engine.run_operation, "expand_neighbor_sum",
                              _nconc["expand_args"],
                              # scaffolding for the commutator below; keyed _Tn
                              # so it takes no E number (INTERNAL-INTERMEDIATE-
                              # EID-GAP).  Still reachable until hidden below.
                              internal=True))
            _hs_id = (_hs_step.output_refs or [None])[0]
            _cc_step, _ = await loop.run_in_executor(
                None, partial(engine.run_operation, "commutator",
                              {"a": _hs_id, "b": _nconc["op"]}))
        except Exception as _ncexc:
            return {"status": "failed", "error": str(_ncexc),
                    "registered_expressions": registered}
        if _hs_id and _hs_id in engine.expressions:
            engine.expressions[_hs_id].hidden = True
        _cc_id = (_cc_step.output_refs or [None])[0]
        if _cc_id:
            _cc_e = engine.expressions[_cc_id]
            registered.append({"expr_id": _cc_e.display_id,
                               "canonical": _cc_e.canonical, "latex": _cc_e.latex})
        return {"status": "completed", "operation": "neighbor_sum_concretize",
                "expr_id": _cc_id, "registered_expressions": registered}
    return None


async def _try_early_bond_sum_commutator(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-bcomm. B stage 3b: commutator of a symbolic bond sum with a local
    # operator → neighbour sum Σ_{j∈nn(n)} [body(n,j), O_n].  The representative
    # bond commutator [body(1,2), O_1] is computed by the engine and relabelled
    # 1→n, 2→j (subscripts only).  Requires a Julia spin session.
    _bc = _parse_bond_sum_commutator(_f1_raw, engine, lang)
    if _bc:
        from notebook.runtime.backend_handler import build_neighbor_sum_terms as _bnst
        _bc_desc = _bc["descriptor"]
        _bc_body = _bc_desc.get("body", "")
        _bc_i1, _bc_i2 = _bc_desc.get("index1"), _bc_desc.get("index2")
        _bc_n = _bc["site"]
        _bc_j = "j" if _bc_n != "j" else "k"
        _bc_op = f"S{_bc['comp']}_1"
        _rep, _rerr = _bnst(_bc_body, _bc_i1, _bc_i2, None, "open", "custom", None, "(1,2)")
        if _rerr or not _rep:
            return {"status": "failed", "error": _rerr or "bad bond sum",
                    "registered_expressions": registered}
        _rep = _rep.strip()
        if _rep.startswith("(") and _rep.endswith(")"):
            _rep = _rep[1:-1]
        try:
            _rep_ex = await loop.run_in_executor(
                None, partial(engine.register_expression, _rep, internal=True))
            _cstep, _ = await loop.run_in_executor(
                None, partial(engine.run_operation, "commutator",
                              {"a": _rep_ex.display_id, "b": _bc_op},
                              internal=True))
        except Exception:
            return {"status": "failed",
                    "error": t("error.operator_missing_index", lang),
                    "registered_expressions": registered}
        _cout = engine.expressions[_cstep.output_refs[0]]
        _rep_ex.hidden = True   # hide intermediates AFTER the commutator runs
        _cout.hidden = True
        _relabeled = _relabel_sites(_cout.latex, {1: _bc_n, 2: _bc_j})
        _ns_latex = rf"\sum_{{{_bc_j} \in \mathrm{{nn}}({_bc_n})}} {_relabeled}"
        _ns_desc = {
            "kind": "neighbor_sum", "orig_body": _bc_body,
            "index1": _bc_i1, "index2": _bc_i2, "comp": _bc["comp"],
            "fixed_site": _bc_n, "neighbor_index": _bc_j,
            "pair_constraint": _bc_desc.get("pair_constraint", "nearest_neighbor"),
        }
        _ns_ex = await loop.run_in_executor(
            None, partial(engine.register_bond_sum, _ns_latex, _ns_desc))
        registered.append({"expr_id": _ns_ex.display_id,
                           "canonical": _ns_ex.canonical, "latex": _ns_ex.latex})
        return {"status": "completed", "operation": "bond_sum_commutator",
                "expr_id": _ns_ex.display_id,
                "registered_expressions": registered}
    return None


# ---------------------------------------------------------------------------
# R1c batch 3 (docs/27): the general single-spin workflow — (a) register a custom
# spin H, (c) effective-field magnitude, (d) general commutator (the precession
# EOM) — extracted from _try_early_intercepts.  These fire before the batch-2
# spectral blocks, so they head the list.  The four helpers the blocks shared
# were function-local closures over engine/loop/registered; they become module
# helpers taking those explicitly.  The rotating-frame derivation is rebuilt on
# these GENERAL primitives (no canned per-scenario helper).
# ---------------------------------------------------------------------------
def _spin_terms(engine, _eid):
    from notebook.runtime import spin_algebra as _spa
    _e = engine.expressions.get(_eid)
    if not _e or _eid not in engine._active_expr_ids():
        return None
    try:
        return _spa.parse_spin_expression(_e.latex or _e.canonical)
    except Exception:
        return None


async def _reg_spin(engine, loop, registered, _latex):
    _ex = await loop.run_in_executor(
        None, partial(engine.register_dispersion, _latex, {"kind": "spin"}))
    registered.append({"expr_id": _ex.display_id,
                       "canonical": _ex.canonical, "latex": _ex.latex})
    return {"status": "completed", "operation": "spin",
            "expr_id": _ex.display_id, "latex": _ex.latex,
            "registered_expressions": registered}


def _lhs_label(engine, _eid):
    r"""The expression's own left side, for `[A, B] = …` displays.

    Bracket-aware because the label is REGISTERED: measured 2026-08-13, an
    expression stored as `\sum_{i=1}^{2} S^z_i = H` labelled itself `\sum_{i`.
    """
    _e = engine.expressions.get(_eid)
    _l = (_e.latex if _e else "") or ""
    _rel = split_relation(_l)
    return _rel[0] if (_rel is not None and "=" in _rel[1]) else _eid


def _expand_spin_eids(engine, _rhs):
    # Replace an E-id reference by the referenced spin expression's RHS, so a
    # spin H can be built by SUMMING expressions ("H_rot = E1 + ω S^z_1").
    def _sub(_m):
        _e = engine.expressions.get(_m.group(0))
        _l = (_e.latex or _e.canonical or "") if _e else ""
        # Handed to spin_algebra's parser, so it decides the VALUE — and there
        # is no CAS between it and the answer, so nothing downstream can object.
        # Measured: `\sum_{i=1}^{2} S^z_i = H` expanded to `1}^{2} S^z_i = H`.
        _rel = split_relation(_l)
        _rhs = _rel[2] if (_rel is not None and "=" in _rel[1]) else _l
        return _rhs or _m.group(0)
    return re.sub(r"E\d+", _sub, _rhs)


async def _try_early_spin_register(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # (a) register a custom spin Hamiltonian — directly, OR as a SUM that adds a
    # term to a referenced spin expression.  The rotating-frame step is exactly this:
    # "H_{rot} = E1 + \omega S^z_1" ADDS the frame term ω S^z to the lab H (E1) — the
    # honest general operation, not a black-box "rotating frame" transform.
    from notebook.runtime import spin_algebra as _spa
    # The RAW request text, split before any expression exists.  `"=" in text`
    # read a NAME's own subscript as the equation: measured 2026-08-13,
    # `H_{n=1} = \omega S^z_1` registered — completed, no error — as
    # `H_{n = \omega S^{z}_{1}`, the researcher's `1}` gone.
    _rel_raw = split_relation(_f1_raw)
    if _rel_raw is not None and "=" in _rel_raw[1] and "S^" in _f1_raw:
        _lhs_sp, _rhs_sp = _rel_raw[0], _rel_raw[2]
        try:
            _terms = _spa.parse_spin_expression(_expand_spin_eids(engine, _rhs_sp))
        except Exception:
            _terms = None
        if _terms:
            return await _reg_spin(engine, loop, registered,
                                   rf"{_lhs_sp.strip()} = {_spa.to_latex(_terms)}")
    return None


async def _try_early_spin_field_magnitude(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # (c) magnitude of the effective field acting on the spin = the Euclidean norm
    # √(Σ_a c_a²) of the registered H's coefficient vector (H is already the
    # component-form inner product −B·S, so compute it straight from E2 — no vector
    # is materialised).  General coeff_vector_norm, not a physics-named helper.
    from notebook.runtime import spin_algebra as _spa
    _ef_m = nlp("spin_field_magnitude", lang).search(_f1_raw)
    if _ef_m:
        _terms = _spin_terms(engine, _ef_m.group("eid"))
        if _terms is not None:
            return await _reg_spin(
                engine, loop, registered,
                rf"\omega_{{\mathrm{{eff}}}} = {_spa.coeff_vector_norm(_terms)}")
    return None



def _declared_half_spins(engine) -> "dict | None":
    """F-13 で宣言されたサイト別スピン（`spin_algebra` の簡約に渡す用）。

    渡すと ½ のサイトの同一サイト積が畳まれ、交換子を重ねても項数が頭打ちになる
    （4サイト鎖で 2→12→76→474→2758→打ち切り が 2→6→14→21→24→28→30 になる）。
    **宣言が無ければ None** ——一般スピンでは $(S^a)^2$ は定数ではないので、
    畳んでよいかどうかは研究者が宣言した内容でしか決まらない。

    **交換子を「表示する」経路専用**（A-5d）。モーメント・連分数・スペクトル関数は
    内積が spin-½ のトレースなので宣言によらず畳む——そちらへ渡すと、
    計算が既に仮定していることを研究者に宣言させることになる。
    """
    from fractions import Fraction
    spins = getattr(getattr(engine, "symbol_context", None), "site_spins", None)
    if not spins:
        return None
    out = {}
    for site, mag in spins.items():
        try:
            out[int(site)] = Fraction(str(mag))
        except (TypeError, ValueError):
            return None            # 読めないものが混ざるなら畳まない
    return out or None


async def _try_early_spin_commutator_general(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # (d) general commutator [E_a, b] where b is a spin operator (S^z_1) or an E-id
    from notebook.runtime import spin_algebra as _spa
    _cg_m = nlp("spin_commutator_general", lang).search(_f1_raw)
    if _cg_m:
        _ta = _spin_terms(engine, _cg_m.group("a"))
        _b_raw = _cg_m.group("b").strip()
        _tb = (_spin_terms(engine, _b_raw) if _EXPR_ID_RE.match(_b_raw)
               else (lambda: (_spa.parse_spin_expression(_b_raw)))())
        if _ta is not None and _tb:
            _comm = _spa.commutator(_ta, _tb)
            _b_disp = _lhs_label(engine, _b_raw) if _EXPR_ID_RE.match(_b_raw) else _b_raw
            return await _reg_spin(
                engine, loop, registered,
                rf"[{_lhs_label(engine, _cg_m.group('a'))}, {_b_disp}] = {_spa.to_latex(_comm)}")
    return None


# ---------------------------------------------------------------------------
# R1c batch 2 (docs/27): spectral / spin / graph tail interceptors, extracted
# verbatim from _try_early_intercepts (same wrap-in-async-def transform as batch
# 1).  They fire BEFORE the fourier/bond-sum batch — hence they head the list.
# ---------------------------------------------------------------------------
async def _try_early_spin_commutator(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-spin. F-1 stage 3c: nested spin commutator on a finite lattice.  The
    # composable spin algebra holds the result as structured data, so a 2nd
    # commutator [H,[H,O]] composes (the EOM/moment hierarchy) — breaking problem B.
    _spc = _parse_spin_commutator(_f1_raw, lang)
    if _spc:
        if _fr := field_unstated_refusal(_spc, lang, registered):
            return _fr
        from notebook.runtime.backend_handler import build_adjacency_matrix
        from notebook.runtime import spin_algebra as _spa
        _spA, _sperr = build_adjacency_matrix(**_spc["spec"])
        if _sperr:
            return {"status": "failed", "error": _sperr,
                    "registered_expressions": registered}
        _n = len(_spA)
        _edges = [(i + 1, j + 1) for i in range(_n) for j in range(i + 1, _n)
                  if _spA[i][j]]
        _H = _spa.build_hamiltonian(_edges, kind=_spc["kind"], n_sites=_n,
                                    field=_spc.get("field"))
        _O = _spa.spin_operator(_spc["comp"], _spc["site"])
        _res = _spa.nested_commutator(_H, _O, _spc["order"],
                                      _declared_half_spins(engine))
        _lhs = rf"S^{{{_spc['comp']}}}_{{{_spc['site']}}}"
        for _ in range(_spc["order"]):
            _lhs = rf"[H, {_lhs}]"
        _sp_latex = rf"{_lhs} = {_spa.to_latex(_res)}"
        _sp_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _sp_latex,
                          {"kind": "spin_commutator"}))
        registered.append({"expr_id": _sp_ex.display_id,
                           "canonical": _sp_ex.canonical, "latex": _sp_ex.latex})
        return {"status": "completed", "operation": "spin_commutator",
                "expr_id": _sp_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_specific_heat(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-heat. F-1 stage 3c: finite-T specific heat C(β) as a high-temperature
    # series from the ∞-T energy cumulants (the lattice loops appear order by order).
    _sh = _parse_specific_heat(_f1_raw, lang)
    if _sh:
        from notebook.runtime import spin_algebra as _spa
        if _sh["general"]:
            _sh_body = _spa.specific_heat_general_latex(_sh["kind"])
        else:
            from notebook.runtime.backend_handler import build_adjacency_matrix
            _shA, _sherr = build_adjacency_matrix(**_sh["spec"])
            if _sherr:
                return {"status": "failed", "error": _sherr,
                        "registered_expressions": registered}
            _shn = len(_shA)
            _sh_edges = [(i + 1, j + 1) for i in range(_shn)
                         for j in range(i + 1, _shn) if _shA[i][j]]
            _sh_body = _spa.specific_heat_concrete_latex(_sh["kind"], _sh_edges)
        _sh_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _sh_body,
                          {"kind": "specific_heat"}))
        registered.append({"expr_id": _sh_ex.display_id,
                           "canonical": _sh_ex.canonical, "latex": _sh_ex.latex})
        return {"status": "completed", "operation": "specific_heat",
                "expr_id": _sh_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_energy_moment(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-energy. F-1 stage 3c: ∞-T energy moments ⟨H^n⟩ (high-T expansion).  The
    # bridge to 3c-graph: ⟨H²⟩∝|E|, ⟨H³⟩ carries the triangle count N△.
    _enm = _parse_energy_moment(_f1_raw, lang)
    if _enm:
        from notebook.runtime import spin_algebra as _spa
        from notebook.runtime.requested_order import energy_moment_refusal
        # 範囲外は `max(2, min(4, …))` で黙って丸められていた——`6次まで` が
        # 4次の答えを `completed` で返す（実測 2026-09-06）。
        if _enm.get("upto_written") is not None:
            _en_why = energy_moment_refusal(_enm["upto_written"], lang)
            if _en_why:
                return {"status": "failed", "error": _en_why,
                        "registered_expressions": registered}
        if _enm["general_spin"]:
            _en_body = _spa.general_spin_energy_moment_latex(_enm["kind"], upto=_enm["upto"])
        elif _enm["general"]:
            _en_body = _spa.energy_moment_general_latex(_enm["kind"], upto=_enm["upto"])
        else:
            from notebook.runtime.backend_handler import (
                build_adjacency_matrix, graph_edge_count, graph_triangle_count,
                graph_four_cycle_count)
            _enA, _enerr = build_adjacency_matrix(**_enm["spec"])
            if _enerr:
                return {"status": "failed", "error": _enerr,
                        "registered_expressions": registered}
            _en = len(_enA)
            _en_edges = [(i + 1, j + 1) for i in range(_en) for j in range(i + 1, _en)
                         if _enA[i][j]]
            _en_moms = r",\ ".join(
                rf"\langle H^{n} \rangle = "
                rf"{_spa.energy_moment(_enm['kind'], _en_edges, n).to_latex()}"
                for n in range(2, _enm["upto"] + 1))
            _en_inv = (rf"|E| = {graph_edge_count(_enA)},\ "
                       rf"N_\triangle = {graph_triangle_count(_enA)}")
            if _enm["upto"] >= 4:
                _en_inv += rf",\ N_\square = {graph_four_cycle_count(_enA)}"
            _en_body = rf"{_en_moms} \quad ({_en_inv})"
        _en_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _en_body,
                          {"kind": "energy_moment"}))
        registered.append({"expr_id": _en_ex.display_id,
                           "canonical": _en_ex.canonical, "latex": _en_ex.latex})
        return {"status": "completed", "operation": "energy_moment",
                "expr_id": _en_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_spectral_terminator(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-term. F-1 stage 3c: infinite-system continuous band via the √-terminator
    # (analytic closed form: terminator t(z) + Wigner-semicircle spectral density).
    if nlp("f1_terminator", lang).search(_f1_raw):
        from notebook.runtime import spin_algebra as _spa
        _term_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _spa.terminator_spectral_latex(),
                          {"kind": "spectral_terminator"}))
        registered.append({"expr_id": _term_ex.display_id,
                           "canonical": _term_ex.canonical, "latex": _term_ex.latex})
        return {"status": "completed", "operation": "spectral_terminator",
                "expr_id": _term_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_spectral_function(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-specfn. F-1 stage 3c: recursion-method Green's function / spectral function
    # — the continued fraction from the Lanczos b_n², and (single-mode) the S(ω) peaks.
    _spf = _parse_spectral_function(_f1_raw, lang)
    if _spf:
        if _fr := field_unstated_refusal(_spf, lang, registered):
            return _fr
        from notebook.runtime.backend_handler import build_adjacency_matrix
        from notebook.runtime import spin_algebra as _spa
        _spfA, _spferr = build_adjacency_matrix(**_spf["spec"])
        if _spferr:
            return {"status": "failed", "error": _spferr,
                    "registered_expressions": registered}
        _spfn = len(_spfA)
        _spf_edges = [(i + 1, j + 1) for i in range(_spfn)
                      for j in range(i + 1, _spfn) if _spfA[i][j]]
        _spfH = _spa.build_hamiltonian(_spf_edges, kind=_spf["kind"], n_sites=_spfn,
                                       field=_spf.get("field"))
        _spfO = _spa.spin_operator(_spf["comp"], _spf["site"])
        _b2 = _spa.continued_fraction(_spfH, _spfO, 8)
        _spf_body = _spa.spectral_latex(_b2)
        _spf_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _spf_body,
                          {"kind": "spectral_function"}))
        registered.append({"expr_id": _spf_ex.display_id,
                           "canonical": _spf_ex.canonical, "latex": _spf_ex.latex})
        return {"status": "completed", "operation": "spectral_function",
                "expr_id": _spf_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_spin_spectral(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-spinspec. F-1 stage 3c: ∞-T spin-½ moments / continued fraction (recursion
    # method) — built on the composable algebra (inner product + Lanczos).
    _sps = _parse_spin_spectral(_f1_raw, lang)
    if _sps:
        if _fr := field_unstated_refusal(_sps, lang, registered):
            return _fr
        from notebook.runtime import spin_algebra as _spa
        from notebook.runtime.requested_order import general_spin_moment_refusal
        if _sps.get("general"):                       # un-expanded formula in z
            # **書かれた次数が結果を変えないなら、黙らない。** この枝は depth を
            # 読んだうえで捨てる（`general_low_moments` / `general_b1_squared`）
            # ので、1〜8 のどれを書いても同じ式が `completed` で返っていた
            # （実測 2026-09-06）。理由は使う地点で言い、行ける道（格子を具体的に
            # 書く）を名指す。
            _sps_why = (general_spin_moment_refusal(
                _sps["mode"], _sps["requested"], lang)
                if _sps.get("requested") is not None else None)
            if _sps_why:
                return {"status": "failed", "error": _sps_why,
                        "registered_expressions": registered}
            if _sps["mode"] == "contfrac":
                _b1 = _spa.general_b1_squared(_sps["kind"], _sps["comp"])
                _body = rf"b_1^2 = {_b1.to_latex()}"
            else:
                _m0, _m2 = _spa.general_low_moments(_sps["kind"], _sps["comp"])
                _body = rf"M_0 = {_m0.to_latex()},\ M_2 = {_m2.to_latex()}"
        else:
            from notebook.runtime.backend_handler import build_adjacency_matrix
            _spsA, _spserr = build_adjacency_matrix(**_sps["spec"])
            if _spserr:
                return {"status": "failed", "error": _spserr,
                        "registered_expressions": registered}
            _n = len(_spsA)
            _edges = [(i + 1, j + 1) for i in range(_n) for j in range(i + 1, _n)
                      if _spsA[i][j]]
            _H = _spa.build_hamiltonian(_edges, kind=_sps["kind"], n_sites=_n,
                                        field=_sps.get("field"))
            _O = _spa.spin_operator(_sps["comp"], _sps["site"])
            if _sps["mode"] == "contfrac":
                _bs = _spa.continued_fraction(_H, _O, _sps["depth"])
                _body = r",\ ".join(rf"b_{{{i + 1}}}^2 = {b.to_latex()}"
                                    for i, b in enumerate(_bs)) or "b_1^2 = 0"
            else:
                _ms = _spa.moments(_H, _O, _sps["depth"])
                _body = r",\ ".join(rf"M_{{{2 * k}}} = {m.to_latex()}"
                                    for k, m in enumerate(_ms))
        _sps_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _body,
                          {"kind": "spin_spectral", "mode": _sps["mode"]}))
        registered.append({"expr_id": _sps_ex.display_id,
                           "canonical": _sps_ex.canonical, "latex": _sps_ex.latex})
        return {"status": "completed", "operation": f"spin_{_sps['mode']}",
                "expr_id": _sps_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_graph_identity(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-gid. F-1 stage 3c-graph: spectral-graph identities (tr A=0, tr A²=2|E|,
    # tr A³=6N△).  With a graph spec the theorem is verified on that A (both sides
    # evaluated); without one the general symbolic statements are shown.
    if nlp("f1_identity", lang).search(_f1_raw):
        from notebook.runtime.backend_handler import (
            build_adjacency_matrix, graph_identity_statements, graph_identity_check)
        _gid_spec = _parse_graph_spec(_f1_raw, lang)
        if _gid_spec:
            _gidA, _giderr = build_adjacency_matrix(**_gid_spec)
            if _giderr:
                return {"status": "failed", "error": _giderr,
                        "registered_expressions": registered}
            _gid_latex, _ = graph_identity_check(_gidA)
        else:
            _gid_latex = graph_identity_statements()
        _gid_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _gid_latex,
                          {"kind": "graph_identity"}))
        registered.append({"expr_id": _gid_ex.display_id,
                           "canonical": _gid_ex.canonical, "latex": _gid_ex.latex})
        return {"status": "completed", "operation": "graph_identity",
                "expr_id": _gid_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_graph(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-graph. F-1 stage 3c-graph: finite-graph adjacency matrix A and its
    # integer invariants (degree sequence, triangle count).  A is registered as an
    # ordinary matrix expression, so eigenvalues/determinant/trace chain for free.
    if nlp("f1_adjacency", lang).search(_f1_raw) or nlp("f1_degree", lang).search(_f1_raw) \
            or nlp("f1_triangle", lang).search(_f1_raw):
        _gspec = _parse_graph_spec(_f1_raw, lang)
        if _gspec:
            from notebook.runtime.backend_handler import (
                build_adjacency_matrix, adjacency_to_canonical, adjacency_to_latex,
                graph_degree_sequence, graph_triangle_count)
            _A, _gerr = build_adjacency_matrix(**_gspec)
            if _gerr:
                return {"status": "failed", "error": _gerr,
                        "registered_expressions": registered}
            if nlp("f1_degree", lang).search(_f1_raw):
                _seq = ", ".join(map(str, graph_degree_sequence(_A)))
                _glatex = rf"\deg = ({_seq})"
            elif nlp("f1_triangle", lang).search(_f1_raw):
                _glatex = rf"\#\triangle = {graph_triangle_count(_A)}"
            elif nlp("f1_spectrum", lang).search(_f1_raw):        # 隣接行列の固有値 → chain eigenvalues
                _gA = await loop.run_in_executor(
                    None, partial(engine.register_expression,
                                  adjacency_to_canonical(_A),
                                  latex=adjacency_to_latex(_A),
                                  internal=True))
                _gstep, _ = await loop.run_in_executor(
                    None, partial(engine.run_operation, "eigenvalues",
                                  {"expr": _gA.display_id}))
                _gA.hidden = True       # hide the intermediate AFTER the op runs
                _gid = (_gstep.output_refs or [None])[0]
                _ge = engine.expressions[_gid]
                registered.append({"expr_id": _ge.display_id,
                                   "canonical": _ge.canonical, "latex": _ge.latex})
                return {"status": "completed", "operation": "adjacency_spectrum",
                        "expr_id": _gid, "registered_expressions": registered}
            else:                                     # adjacency matrix
                _gex = await loop.run_in_executor(
                    None, partial(engine.register_expression,
                                  adjacency_to_canonical(_A),
                                  latex=adjacency_to_latex(_A)))
                registered.append({"expr_id": _gex.display_id,
                                   "canonical": _gex.canonical, "latex": _gex.latex})
                return {"status": "completed", "operation": "adjacency_matrix",
                        "expr_id": _gex.display_id,
                        "registered_expressions": registered}
            _gex = await loop.run_in_executor(
                None, partial(engine.register_dispersion, _glatex,
                              {"kind": "graph_invariant"}))
            registered.append({"expr_id": _gex.display_id,
                               "canonical": _gex.canonical, "latex": _gex.latex})
            return {"status": "completed", "operation": "graph_invariant",
                    "expr_id": _gex.display_id,
                    "registered_expressions": registered}
    return None


# ---------------------------------------------------------------------------
# R1c batch 1 (docs/27): fourier / bond-sum tail interceptors, extracted verbatim
# from _try_early_intercepts.  Each is the original if-block wrapped in an async
# def that returns None on fall-through; the ordered list below IS the firing
# order.  Adding a lattice model is "write a _try_early_* + one list entry".
# ---------------------------------------------------------------------------
async def _try_early_crystal(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-crystal. F-1 stage 3c-fourier: abstract crystal descriptor — arbitrary
    # bonds (α,β,ΔR,amp) → Bloch matrix H̃(k) + closed-form bands (N≤2).
    _cry = _parse_crystal(_f1_raw, lang)
    if _cry:
        from notebook.runtime.backend_handler import build_crystal_bloch
        _cry_mat, _cry_band, _cry_err = build_crystal_bloch(_cry)
        if _cry_err:
            return {"status": "failed", "error": _cry_err,
                    "registered_expressions": registered}
        _cry_last = None
        for _cl in (_cry_mat, _cry_band):
            if not _cl:
                continue
            _cry_ex = await loop.run_in_executor(
                None, partial(engine.register_dispersion, _cl,
                              {"kind": "dispersion", "model": "crystal"}))
            registered.append({"expr_id": _cry_ex.display_id,
                               "canonical": _cry_ex.canonical, "latex": _cry_ex.latex})
            _cry_last = _cry_ex.display_id
        return {"status": "completed", "operation": "dispersion_relation",
                "expr_id": _cry_last, "registered_expressions": registered}
    return None


async def _try_early_dispersion(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-disp. F-1 stage 3c-fourier: tight-binding dispersion ε(k) from a stored
    # symbolic bond sum.  "H の [正方格子で] 分散関係を求めて" → ε(k)=Σ 2t cos(k·ΔR).
    # Built directly from the unit-cell hopping ΔR (no commutator → bypasses the
    # opaque-canonical ceiling), so it is self-contained.
    _disp = _parse_dispersion(_f1_raw, engine, lang)
    if _disp:
        from notebook.runtime.backend_handler import build_dispersion
        _disp_coeff, _disp_pc, _disp_nnn, _disp_onsite, _disp_nnn_b, _disp_name_err = _disp
        if _disp_name_err:
            return {"status": "failed", "error": _disp_name_err,
                    "registered_expressions": registered}
        _disp_latex, _disp_err = build_dispersion(
            _disp_coeff, _disp_pc, _disp_nnn, _disp_onsite, _disp_nnn_b)
        if _disp_err or not _disp_latex:
            return {"status": "failed", "error": _disp_err or "dispersion failed",
                    "registered_expressions": registered}
        _disp_desc = {"kind": "dispersion", "pair_constraint": _disp_pc,
                      "coeff": _disp_coeff, "nnn_coeff": _disp_nnn,
                      "onsite": _disp_onsite, "nnn_b": _disp_nnn_b}
        _disp_ex = await loop.run_in_executor(
            None, partial(engine.register_dispersion, _disp_latex, _disp_desc))
        registered.append({"expr_id": _disp_ex.display_id,
                           "canonical": _disp_ex.canonical, "latex": _disp_ex.latex})
        return {"status": "completed", "operation": "dispersion_relation",
                "expr_id": _disp_ex.display_id,
                "registered_expressions": registered}
    return None


async def _try_early_bond_sum_substitute(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-bsub. B stage 2: substitution into a symbolic bond sum, keeping the sum
    # symbolic — Op(Σ body) = Σ Op(body).  "H の J を 2t に置き換え".
    _bsub = _parse_bond_sum_substitute(_f1_raw, engine, lang)
    if _bsub:
        _bs_eid, _bs_lhs, _bs_rhs = _bsub
        _bs_ex = engine.expressions[_bs_eid]
        _bs_desc = dict(_bs_ex.lattice_descriptor)
        _bs_desc["body"] = _token_sub(_bs_desc.get("body", ""), _bs_lhs, _bs_rhs)
        _bs_latex = _token_sub(_bs_ex.latex, _bs_lhs, _bs_rhs)
        _bs_new = await loop.run_in_executor(
            None, partial(engine.register_bond_sum, _bs_latex, _bs_desc))
        registered.append({
            "expr_id":   _bs_new.display_id,
            "canonical": _bs_new.canonical,
            "latex":     _bs_new.latex,
        })
        return {
            "status":    "registered",
            "operation": "bond_sum_substitute",
            "expr_id":   _bs_new.display_id,
            "registered_expressions": registered,
        }
    return None


async def _try_early_symbolic_bond_sum(engine, loop, lang, _f1_raw, registered) -> "dict | None":
    # 0c-sym. B stage 1: a bond-sum notation with no expand verb / site count is
    # registered UNEXPANDED as a symbolic expression (display round-trip), to be
    # concretized later with "En を N サイト…で展開".
    _sym = _parse_symbolic_bond_sum(_f1_raw, lang)
    if _sym:
        _sym_latex, _sym_desc = _sym
        _sym_expr = await loop.run_in_executor(
            None, partial(engine.register_bond_sum, _sym_latex, _sym_desc))
        registered.append({
            "expr_id":   _sym_expr.display_id,
            "canonical": _sym_expr.canonical,
            "latex":     _sym_expr.latex,
        })
        return {
            "status":    "registered",
            "operation": "register_bond_sum",
            "expr_id":   _sym_expr.display_id,
            "registered_expressions": registered,
        }
    return None


_EARLY_INTERCEPTORS: "list[Interceptor]" = [
    # R1c batch 4: head interceptors (fire first, before the spin workflow).
    Interceptor(
        "inner_product",
        lambda c: _try_early_inner_product(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1: declare an inner product ⟨e_i,e_j⟩=v into the Gram matrix; "
             "before H-1 would register it as an ordinary equation."),
    Interceptor(
        "matrix_transform",
        lambda c: _try_early_matrix_transform(c.engine, c.loop, c.lang, c.text, c.registered),
        note="L-1: explicit-matrix inverse transform 'M=[[…]], 旧基底…, 新基底…'; "
             "before inline-formula splitting would register 'M=[[…]]' alone."),
    Interceptor(
        "heisenberg_operator",
        lambda c: _try_heisenberg_operator(c.engine, c.loop, c.lang, c.text,
                                           c.registered),
        note="PAIR-REP: 「A を En のもとで時間発展させて」 → the Heisenberg-picture "
             "operator as (coefficient, QA operator) pairs."),
    Interceptor(
        "pair_bracket",
        lambda c: _try_pair_bracket(c.engine, c.loop, c.lang, c.text,
                                    c.registered),
        note="PAIR-REP: 「E5 と E6 の反交換関係」 when either operand is a "
             "time-evolved pair — coefficient product × equal-time algebra. "
             "Claims nothing when neither is a pair, so the generic bracket "
             "route is untouched."),
    Interceptor(
        "save_lattice_figure",
        lambda c: _try_early_save_lattice_figure(c.engine, c.loop, c.lang, c.text, c.registered),
        note="「格子図を保存して」 → save a standalone figure of the latest lattice."),
    Interceptor(
        "neighbor_sum",
        lambda c: _try_early_neighbor_sum(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Concretize a stored symbolic bond sum OR expand a fresh neighbour "
             "sum — both dispatch expand_neighbor_sum."),
    Interceptor(
        "neighbor_sum_concretize",
        lambda c: _try_early_neighbor_sum_concretize(c.engine, c.loop, c.lang, c.text, c.registered),
        note="B stage 3b: concretize a neighbour-sum EOM at a site (pin_site + "
             "commutator)."),
    Interceptor(
        "bond_sum_commutator",
        lambda c: _try_early_bond_sum_commutator(c.engine, c.loop, c.lang, c.text, c.registered),
        note="B stage 3b: commutator of a symbolic bond sum with a local operator "
             "→ neighbour sum (representative bond commutator relabelled)."),
    # R1c batch 3: general single-spin workflow (fires before the spectral tail).
    Interceptor(
        "declare_site_spins",
        lambda c: _try_declare_site_spins(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-13: per-site spin magnitudes (electron + nucleus, heteronuclear "
             "NMR). Heads the spin group: it is a DECLARATION, so it must not be "
             "read as an attempt to register a spin Hamiltonian."),
    Interceptor(
        "spin_register",
        lambda c: _try_early_spin_register(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Register a custom spin H from 'LHS = RHS' when the RHS parses as a "
             "spin expression (the rotating-frame step = adding a frame term to a "
             "referenced spin H)."),
    Interceptor(
        "spin_field_magnitude",
        lambda c: _try_early_spin_field_magnitude(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Effective-field magnitude = coeff-vector norm of the registered H."),
    Interceptor(
        "spin_commutator_general",
        lambda c: _try_early_spin_commutator_general(c.engine, c.loop, c.lang, c.text, c.registered),
        note="General commutator [E_a, b] (spin operator or E-id) — the precession "
             "EOM; before the lattice spin_commutator so 'S^z_1 と H の交換子' is not "
             "grabbed as a nested-lattice commutator."),
    # R1c batch 2: spectral / spin / graph family (fire before the fourier tail).
    Interceptor(
        "spin_commutator",
        lambda c: _try_early_spin_commutator(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Nested spin commutator [H,[H,O]] on a finite lattice."),
    Interceptor(
        "spin_expectation",
        lambda c: _try_spin_expectation(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-13: ⟨A^n⟩ of a REGISTERED spin expression with the declared "
             "per-site spins. Before energy_moment, which reads 'モーメント' too "
             "but only together with a model name and no E-id."),
    Interceptor(
        "specific_heat",
        lambda c: _try_early_specific_heat(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Finite-T specific heat C(β) high-T series from ∞-T cumulants."),
    Interceptor(
        "energy_moment",
        lambda c: _try_early_energy_moment(c.engine, c.loop, c.lang, c.text, c.registered),
        note="∞-T energy moments ⟨H^n⟩ (bridge to the graph invariants)."),
    Interceptor(
        "spectral_terminator",
        lambda c: _try_early_spectral_terminator(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Infinite-system continuous band via the √-terminator."),
    Interceptor(
        "spectral_function",
        lambda c: _try_early_spectral_function(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Recursion-method Green's function / spectral function (Lanczos b_n²)."),
    Interceptor(
        "spin_spectral",
        lambda c: _try_early_spin_spectral(c.engine, c.loop, c.lang, c.text, c.registered),
        note="∞-T spin-½ moments / continued fraction (recursion method)."),
    Interceptor(
        "graph_identity",
        lambda c: _try_early_graph_identity(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Spectral-graph identities (tr A=0, tr A²=2|E|, tr A³=6N△); before "
             "the adjacency-matrix block so 恒等式 is not grabbed as a plain graph."),
    Interceptor(
        "graph",
        lambda c: _try_early_graph(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Finite-graph adjacency matrix A + integer invariants "
             "(degree/triangle/spectrum)."),
    # R1c batch 1: fourier / bond-sum family.
    Interceptor(
        "crystal",
        lambda c: _try_early_crystal(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Abstract crystal descriptor → Bloch matrix + closed-form bands."),
    Interceptor(
        "dispersion",
        lambda c: _try_early_dispersion(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Tight-binding ε(k) from a stored symbolic bond sum."),
    Interceptor(
        "bond_sum_substitute",
        lambda c: _try_early_bond_sum_substitute(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Substitute into a symbolic bond sum, keeping the sum symbolic; "
             "before the bare register so a substitution is not stored as a new "
             "unexpanded sum."),
    Interceptor(
        "symbolic_bond_sum",
        lambda c: _try_early_symbolic_bond_sum(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Bond-sum notation with no expand verb → register UNEXPANDED; the "
             "catch-all fallback of the tail."),
]
