"""F-1: lattice neighbour-sums, dispersion, crystal descriptors, and the
spin-algebra parsers (commutator / moment / spectral / specific-heat).

Pure rule-based parsers/helpers extracted from interpret_pipeline.py (Phase 1
refactor). run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re

from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.requested_order import recursion_units
from ui.interpret_common import _HAS_CJK_RE, number_word
from notebook.runtime.shared_parser import split_relation


# F-1: "<body> の最近接和を N サイト [周期境界]で展開して" → expand_neighbor_sum.
# Trigger / body / all-pairs / N-site / 2D-lattice / figure-save grammar →
# nl_patterns.json (f1_ns_trigger / f1_ns_body / f1_ns_allpairs_kw /
# f1_ns_allpairs_body / f1_ns_nsite / f1_ns_lattice / f1_save_fig_noun /
# f1_save_fig_verb).  English word order reverses the JP "<body> の 最近接和"
# (the sum-type noun comes first, body follows "of"); f1_ns_body carries both,
# and the の-presence heuristic in _parse_neighbor_sum routes between them
# (content-based, docs/28 §4).
_NS_COMPONENT_RE = re.compile(r'\^\{?[A-Za-z]\}?', re.U)
_NS_SUBSCRIPT_RE = re.compile(r'_\{?([A-Za-z])\}?', re.U)
_NS_ZENKAKU = {ord('０') + i: ord('0') + i for i in range(10)}
# \sum_{<ij>} bond-sum notation: \sum / \Sigma / Σ / ∑ with an angle-bracketed
# index pair ⟨ij⟩ / \langle i j \rangle / <ij>.  Captures the two index letters.
# (Language-neutral notation, kept in code.)
_NS_BOND_SUB_RE = re.compile(
    r'(?:\\sum|\\Sigma|Σ|∑)\s*_\s*\{?\s*'
    r'(?:\\langle|⟨|<)\s*([A-Za-z])\s*,?\s*([A-Za-z])\s*(?:\\rangle|⟩|>)\s*\}?',
    re.U)
# all-pairs notation: \sum_{i<j} (molecular / quantum-chemistry pairwise sums).
_NS_ALLPAIRS_SUB_RE = re.compile(
    r'(?:\\sum|\\Sigma|Σ|∑)\s*_\s*\{?\s*'
    r'([A-Za-z])\s*(?:<|\\lt)\s*([A-Za-z])\s*\}?',
    re.U)
# ORDERED pairs: Σ_{i≠j}.  A SEPARATE constraint from Σ_{i<j}, not a spelling of
# it — N(N-1) terms against N(N-1)/2.  All four spellings used to be listed in
# the regex above, so `\sum_{i \ne j} V_{i,j}` over 3 sites expanded to THREE
# terms and its display came back rewritten as `\sum_{i<j}`: half the sum, in
# notation the researcher had not written (measured 2026-08-10).
# `\neq` BEFORE `\ne`, or the longer spelling is matched as the shorter one and
# a stray `q` is left behind.
_NS_ORDERED_SUB_RE = re.compile(
    r'(?:\\sum|\\Sigma|Σ|∑)\s*_\s*\{?\s*'
    r'([A-Za-z])\s*(?:≠|\\neq|\\ne\b|!=)\s*([A-Za-z])\s*\}?',
    re.U)
_NS_SHAPE_RE = re.compile(r'(\d+)\s*[x×]\s*(\d+)', re.U)
def _keep(m):
    """The match, or None when this route cannot carry what was written."""
    if m is None or not _whole_subscript(m):
        return None
    # A BOUND written on the Σ (`\sum_{i<j}^{N}`) is not something a bond
    # descriptor can hold — it expands by a SITE COUNT given later, not by a
    # symbolic range.  Claimed anyway, the superscript was left dangling in the
    # display (`\sum_{i<j} ^{N} V_{i,j}`) and the CAS then failed on it
    # (measured 2026-08-10).  Falling through, the general pipeline refuses it
    # by name.
    return None if m.string[m.end():m.end() + 1] == "^" else m


def _whole_subscript(m) -> bool:
    r"""Did the bond-sum regex consume the WHOLE subscript?

    The two-index pattern ends with an optional ``}``, so a CHAINED constraint
    matches its first pair and stops there: `\sum_{i<j<k<l}` matched
    `\sum_{i<j` and left `<k<l}` to be taken as the summand.  It was registered
    as an ordinary Σ_{i<j} over (i,j), displayed as
    `\sum_{i<j} <k<l} V_{i,j,k,l}` — broken LaTeX with a dangling brace — and
    would have expanded to pair terms with `<k<l}` glued into each one
    (measured 2026-08-10).

    A chained constraint is beyond what a two-index bond descriptor can say, so
    it must not be claimed here.  Falling through, the general pipeline refuses
    it BY NAME (`parser.unsupported_notation.constrained_sum`), which is the
    honest answer.

    The test: if the match opened a brace it must also close it.
    """
    text = m.group(0)
    return "{" not in text or text.rstrip().endswith("}")



def _is_save_lattice_figure(text: str, lang: str = "ja") -> bool:
    t = text or ""
    return bool(nlp("f1_save_fig_noun", lang).search(t) and nlp("f1_save_fig_verb", lang).search(t))


def _ns_extract_body_after(rest: str, lang: str = "ja") -> str:
    """Body of a sum-notation instruction = the text after the \\sum_{<ij>}
    construct, up to the first instruction boundary (particle を/の, the
    English preposition over/on/for, the N-site phrase, or 展開/expand)."""
    cuts = [len(rest)]
    pm = nlp("f1_body_boundary", lang).search(rest)
    if pm:
        cuts.append(pm.start())
    nm = nlp("f1_ns_nsite", lang).search(rest)
    if nm:
        cuts.append(nm.start())
    em = nlp("f1_expand", lang).search(rest)
    if em:
        cuts.append(em.start())
    return rest[:min(cuts)].strip()


def _ja_body_end(t: str, lang: str, mention_start: int) -> int:
    """日本語の本体はどこで終わるか——**指定が始まる手前**（位置ではなく文法）。

    格子名・和の語・サイト数のうち、最初に現れるもの。直す前は「格子名の手前」
    固定で、日本語の2通りの語順の片方でしか正しくなかった——測定と症状は
    `tests/test_neighbor_sum_body_boundary.py`。英語は語順が逆なので
    `f1_ns_body[en]` が読み、ここには来ない。
    """
    ends = [mention_start]
    for key in ("f1_ns_trigger", "f1_ns_allpairs_kw", "f1_ns_nsite"):
        m = nlp(key, lang).search(t)
        if m:
            ends.append(m.start())
    return min(ends)


def _parse_neighbor_sum(text: str, lang: str = "ja") -> "tuple[str, dict] | None":
    """Parse a 1D-chain nearest-neighbor sum instruction into an
    expand_neighbor_sum operation (F-1).  Returns (op, args) or None.

    Trigger forms (each → a pair_constraint):
      - nearest-neighbor keyword:  "S^z_i S^z_j の最近接和を 4 サイト 周期境界で展開して"
      - nearest-neighbor notation: "\\sum_{⟨ij⟩} S^z_i S^z_j を 4 サイト …で展開して"
      - all-pairs keyword:         "q_i q_j の全ペア和を 3 サイトで展開して"
      - all-pairs notation:        "\\sum_{i<j} q_i q_j を 3 サイトで展開して"
      - custom graph (arbitrary bonds, no shape/site-count needed):
        "S^z_i S^z_j のグラフ (1,2),(2,3),(1,3) の最近接和を展開して" /
        "the nearest-neighbor sum of S^z_i S^z_j over the graph (1,2),(2,3),(1,3)"
    Named-lattice / 1D / all-pairs forms require an explicit N-site count and
    the 展開/expand verb.  N (and, for nearest-neighbor, the boundary) are
    taken from the accompanying instruction because the ⟨ij⟩ / i<j notation
    itself carries neither.  ``all_pairs`` (Σ_{i<j}) serves molecular /
    quantum-chemistry pairwise sums.  An explicit but UNRECOGNIZED lattice
    name (e.g. "ダイヤモンド格子"/"diamond lattice") is forced through as the
    literal (invalid) ``pair_constraint`` value rather than silently falling
    back to the 1D chain, so the existing ``_BOND_GENERATORS`` lookup in
    ``build_neighbor_sum_terms`` rejects it with a clear error BEFORE the
    body is substituted — see the silent-lattice-fallback bugfix (the same
    bug class already fixed for _parse_dispersion).
    """
    t = (text or "").strip()
    bond_m = _keep(_NS_BOND_SUB_RE.search(t))
    ap_m = _keep(_NS_ALLPAIRS_SUB_RE.search(t))
    nn_kw = nlp("f1_ns_trigger", lang).search(t)
    ap_kw = nlp("f1_ns_allpairs_kw", lang).search(t)
    lat_m = nlp("f1_ns_lattice", lang).search(t)
    graph_m = nlp("f1_graph_edge", lang).search(t)
    if not (bond_m or ap_m or nn_kw or ap_kw or lat_m or graph_m):
        return None
    if not nlp("f1_expand", lang).search(t):
        return None

    # Custom graph: an arbitrary bond list, e.g. "グラフ (1,2),(2,3),(1,3)" /
    # "graph (1,2),(2,3),(1,3)" — reuses the same edge-list mini-grammar as
    # the 3c-graph command family's _parse_graph_spec.  No lattice shape or
    # site count is needed: the bonds fully determine the sites.
    custom_edges = None
    mention_start = None
    if graph_m:
        from notebook.runtime.backend_handler import _parse_edges
        custom_edges = _parse_edges(t)
        if not custom_edges:
            return None
        mention_start = graph_m.start()

    # 2D lattice: keyword picks the generator; the Lx×Ly shape gives the size.
    pc_2d = None
    lattice_shape = None
    unrecognized_mention = None
    if custom_edges is not None:
        pass
    elif lat_m:
        if lat_m.group("hc"):
            pc_2d = "honeycomb_2d"
        elif lat_m.group("tri"):
            pc_2d = "triangular_2d"
        else:
            pc_2d = "square_2d"
        shape_m = _NS_SHAPE_RE.search(t)
        if shape_m:
            lattice_shape = f"{shape_m.group(1)}x{shape_m.group(2)}"
        mention_start = lat_m.start()
    else:
        unrecognized_mention = _find_unrecognized_lattice_mention(t, lang)
        if unrecognized_mention:
            _um_gd = unrecognized_mention.groupdict()
            pc_2d = (_um_gd.get("ja") or _um_gd.get("en") or "").strip()
            mention_start = unrecognized_mention.start()

    if custom_edges is not None:
        n_sites = None
    elif lattice_shape:
        n_sites = int(shape_m.group(1)) * int(shape_m.group(2))
    else:
        nm = nlp("f1_ns_nsite", lang).search(t)
        if not nm:
            return None
        n_sites = int(nm.group(1).translate(_NS_ZENKAKU))
    if nlp("f1_boundary_cylinder", lang).search(t):
        boundary = "cylinder"
    elif nlp("f1_boundary_periodic", lang).search(t):
        boundary = "periodic"
    else:
        boundary = "open"
    # a 2D/custom lattice is nearest-neighbor, not all-pairs
    all_pairs = bool(ap_m or ap_kw) and not pc_2d and custom_edges is None
    notation_m = ap_m if ap_m else bond_m

    if notation_m:
        # Notation form: indices come from the bracket; body follows it.
        idx1, idx2 = notation_m.group(1), notation_m.group(2)
        body = _ns_extract_body_after(t[notation_m.end():], lang)
        if not body or idx1 == idx2:
            return None
    else:
        if mention_start is not None:
            # Custom graph / 2D keyword / unrecognized-lattice form: body
            # precedes the mention in Japanese (e.g. "Sz_i Sz_j の 正方格子
            # 3x3 の最近接和を展開して" / "…のグラフ (1,2),(2,3) の最近接和…");
            # English reverses this ("the nearest-neighbor sum of Sz_i Sz_j
            # over a 3x3 honeycomb lattice" / "… over the graph (1,2),(2,3)")
            # so body instead follows "sum of".
            _en_body_m = (None if "の" in t[:mention_start]
                          else nlp("f1_ns_body", "en").search(t))
            if _en_body_m:
                body = _en_body_m.group("body").strip()
            elif _HAS_CJK_RE.search(t[:mention_start]):
                # 末尾の助詞は本体ではない——`… を 4 サイトの正方格子で…` は
                # `Sz_i Sz_j を` を残していた（数式が助詞で終わることは無い）。
                body = re.sub(r"[のをは]$", "",
                              t[:_ja_body_end(t, lang, mention_start)].strip()).strip()
            else:
                # **英語の文に日本語の位置規則を当てない。** ここは 1D の枝が
                # 既に正しく答えている問いで（読めなければ `return None`）、
                # 2D の枝だけが「読めなかったら手前を全部取る」に落ちていた。
                # 全 ASCII の文でそれをやると、`Expand the neighbor sum of
                # Sz_i Sz_j over a 3x3 honeycomb lattice` の**英単語が被和関数の
                # 因子**になり `completed` が返る（実測 2026-09-24、五十五問目）。
                # 日本語の位置規則が答えてよいのは日本語の文だけである。
                return None
        else:
            # 1D keyword form: body precedes "の最近接和"/"の全ペア和" in
            # Japanese, or follows "sum of" in English.
            bm = (nlp("f1_ns_allpairs_body", "ja").match(t) if all_pairs
                  else nlp("f1_ns_body", "ja").match(t))
            if bm:
                body = bm.group("body").strip()
            else:
                _en_body_m = nlp("f1_ns_body", "en").search(t)
                if not _en_body_m:
                    return None
                body = _en_body_m.group("body").strip()
        if not body:
            return None
        body_no_comp = _NS_COMPONENT_RE.sub("", body)
        idxs: list[str] = []
        for m in _NS_SUBSCRIPT_RE.finditer(body_no_comp):
            c = m.group(1)
            if c not in idxs:
                idxs.append(c)
        if len(idxs) < 2:
            return None
        idx1, idx2 = idxs[0], idxs[1]

    args = {
        "body": body,
        "index1": idx1,
        "index2": idx2,
        "boundary": boundary,
    }
    if n_sites is not None:
        args["n_sites"] = n_sites
    pair_constraint = ("custom" if custom_edges is not None
                        else pc_2d or ("all_pairs" if all_pairs else None))
    if pair_constraint:
        args["pair_constraint"] = pair_constraint
    if custom_edges is not None:
        args["edges"] = custom_edges
    if lattice_shape:
        args["lattice_shape"] = lattice_shape
    return ("expand_neighbor_sum", args)


_SYM_LHS_RE = re.compile(r'^[A-Za-z\\][\w{}\\^]*$')


def _parse_symbolic_bond_sum(text: str, lang: str = "ja") -> "tuple[str, dict] | None":
    """B stage 1: a bond-sum NOTATION with no expand verb and no site count is
    kept symbolic.  ``[-J] \\sum_{⟨ij⟩} S^z_i S^z_j`` → (display_latex,
    descriptor).  The equation form ``H = -J \\sum_{⟨ij⟩} …`` is also accepted
    (1b): the display keeps the LHS so the sum can be concretized by name.
    Returns None for the expand path (展開/N present).
    """
    t = (text or "").strip()
    if not t:
        return None
    if nlp("f1_expand", lang).search(t) or nlp("f1_ns_nsite", lang).search(t):
        return None
    # Equation form: a simple name on the left, a bond sum on the right.
    lhs_prefix = ""
    # `_SYM_LHS_RE` accepts `\` and `{`, so the naive split's `\sum_{i` LOOKED
    # like a simple name and this route claimed an input it should have declined.
    # Measured 2026-08-13 on `\sum_{i=1}^{3} S_i = \sum_{\langle i,j \rangle} V_{i,j}`:
    #   body       (1}^{3} S_i =)*(V_{i,j})      ← a fragment, with its `=`, as a factor
    #   lhs_prefix \sum_{i = 1}^{3} S_i = …      ← written into the display
    _rel = split_relation(t)
    if _rel is not None and "=" in _rel[1]:
        _lhs, _rhs = _rel[0], _rel[2]
        if not _lhs or not _SYM_LHS_RE.match(_lhs):
            return None  # complex LHS → not a simple named bond sum
        lhs_prefix = f"{_lhs} = "
        t = _rhs
    ap_m = _keep(_NS_ALLPAIRS_SUB_RE.search(t))
    ne_m = _keep(_NS_ORDERED_SUB_RE.search(t))
    bond_m = _keep(_NS_BOND_SUB_RE.search(t))
    m = ap_m or ne_m or bond_m
    if not m:
        return None
    idx1, idx2 = m.group(1), m.group(2)
    if idx1 == idx2:
        return None
    prefix = t[:m.start()].strip()
    body = t[m.end():].strip()
    if not body:
        return None
    # The label is written back into the DISPLAY, so it has to say what the
    # researcher said: `\sum_{i \ne j}` redisplayed as `\sum_{i<j}` is a
    # different statement, and it was being made silently.
    if ap_m:
        pair_constraint, label = "all_pairs", f"{idx1}<{idx2}"
    elif ne_m:
        pair_constraint, label = "ordered_pairs", rf"{idx1} \ne {idx2}"
    else:
        pair_constraint, label = "nearest_neighbor", rf"\langle {idx1}{idx2} \rangle"
    display = lhs_prefix + (f"{prefix} " if prefix else "") + rf"\sum_{{{label}}} {body}"
    body_for_expand = f"({prefix})*({body})" if prefix else body
    descriptor = {
        "body": body_for_expand,
        "index1": idx1,
        "index2": idx2,
        "pair_constraint": pair_constraint,
    }
    return (display, descriptor)


_CONCRETIZE_EID_RE = re.compile(r'(?<![A-Za-z0-9_])E(\d+)(?![A-Za-z0-9_])')


def _parse_concretize(text: str, engine, lang: str = "ja") -> "tuple[str, dict] | None":
    """B stage 1: "En を N サイト […]で展開" where En is a stored symbolic bond
    sum (has a lattice_descriptor) → concretize it via expand_neighbor_sum.
    The command may override the lattice (正方格子 3x3 等) and boundary.
    """
    t = (text or "").strip()
    if not nlp("f1_expand", lang).search(t):
        return None
    def _is_bond_sum(ex):
        d = getattr(ex, "lattice_descriptor", None)
        return bool(d) and d.get("kind", "bond_sum") == "bond_sum"

    desc = None
    for m in _CONCRETIZE_EID_RE.finditer(t):
        ex = getattr(engine, "expressions", {}).get(f"E{m.group(1)}")
        if ex is not None and _is_bond_sum(ex):
            desc = ex.lattice_descriptor
            break
    # Named reference (1b): "H を …で展開" where H = a symbolic bond sum.
    if desc is None:
        for _name, _eid in getattr(engine, "lhs_alias_map", {}).items():
            if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(_name)
                             + r'(?![A-Za-z0-9_])', t):
                continue
            ex = getattr(engine, "expressions", {}).get(_eid)
            if ex is not None and _is_bond_sum(ex):
                desc = ex.lattice_descriptor
                break
    if desc is None:
        return None

    args = {
        "body": desc.get("body"),
        "index1": desc.get("index1"),
        "index2": desc.get("index2"),
        "pair_constraint": desc.get("pair_constraint", "nearest_neighbor"),
    }
    # boundary
    if nlp("f1_boundary_cylinder", lang).search(t):
        args["boundary"] = "cylinder"
    elif nlp("f1_boundary_periodic", lang).search(t):
        args["boundary"] = "periodic"
    else:
        args["boundary"] = "open"
    # 2D lattice override (正方格子/三角格子/ハニカム + LxL)
    lat_m = nlp("f1_ns_lattice", lang).search(t)
    shape_m = _NS_SHAPE_RE.search(t) if lat_m else None
    if lat_m:
        if lat_m.group("hc"):
            args["pair_constraint"] = "honeycomb_2d"
        elif lat_m.group("tri"):
            args["pair_constraint"] = "triangular_2d"
        else:
            args["pair_constraint"] = "square_2d"
        if shape_m:
            args["lattice_shape"] = f"{shape_m.group(1)}x{shape_m.group(2)}"
            args["n_sites"] = int(shape_m.group(1)) * int(shape_m.group(2))
    nm = nlp("f1_ns_nsite", lang).search(t)
    if "n_sites" not in args:
        if not nm:
            return None
        args["n_sites"] = int(nm.group(1).translate(_NS_ZENKAKU))
    return ("expand_neighbor_sum", args)


# F-1 stage 3c-fourier: tight-binding dispersion ε(k) from a symbolic bond sum.
# Dispersion / Lieb / kagome / lattice-name-mention grammar → nl_patterns.json
# (f1_dispersion / f1_lieb / f1_kagome / f1_lattice_mention[_safe]).  The
# lattice-mention "safe" list excludes fixed multi-word phrases that legitimately
# contain "格子"/"lattice" without naming a crystal structure: 格子和/lattice sum
# (F-1 neighbour-sum trigger), 格子図/lattice figure (figure-save trigger), and
# 副格子/サブ格子/sublattice (the staggered onsite-potential keyword).
# Named self-contained model: SSH dimerised chain (its own amplitudes v, w).
_SSH_RE = re.compile(r'SSH|ＳＳＨ', re.U | re.I)
_SSH_PARAM_RE = re.compile(
    r"(?P<name>[vw])\s*=\s*(?P<val>-?\\?[A-Za-z0-9][\w'^{}\\.]*)", re.U)


# Abstract crystal descriptor: bonds (α, β, (ΔR…), amplitude), each in parens.
# Crystal keyword → nl_patterns.json["f1_crystal_kw"].
_BOND_TOK_RE = re.compile(
    r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*\(([-\d,\s]*)\)\s*,\s*([^(),]+?)\s*\)', re.U)


# F-1 stage 3c-graph: finite-graph adjacency matrix and its invariants.  Graph-
# invariant keywords (adjacency / degree / triangle / spectrum / identity) and the
# complete-graph / graph-edge triggers → nl_patterns.json (f1_adjacency / f1_degree
# / f1_triangle / f1_spectrum / f1_identity / f1_graph_allpairs / f1_graph_edge).


# F-1 stage 3c: nested spin commutators on a finite lattice (composable — breaks
# "problem B": the result feeds back into the next commutator).  Spin-model /
# commutator / order / field grammar → nl_patterns.json (f1_spin_model / f1_commut
# / f1_order_n / f1_order2 / f1_field[_longitudinal]).  Model names are proper nouns
# kept in both ja/en fields.
_SPIN_OP_RE = re.compile(r'S\^?\{?([xyz])\}?_\{?(\d+)\}?', re.U)


# (The canned rotating-frame recognizer was removed: the rotating-frame scenario
# is rebuilt on the general single-spin operations in interpret_pipeline.)


def _spin_model_kind(token: str) -> str:
    t = token.lower()
    if "xxz" in t or "ＸＸＺ" in token:
        return "xxz"
    if "xy" in t or "ＸＹ" in token:
        return "xy"
    if "イジング" in token or "ising" in t:
        return "ising"
    return "heisenberg"


#: 「磁場と書いてあるが、向きが書かれていない」——値でも `None` でもない第三の状態。
FIELD_DIRECTION_UNSTATED = ("h", None)


def read_field(text: str, lang: str = "ja"):
    """磁場の `(記号, 向き)` ／ `None`（書かれていない）／ `FIELD_DIRECTION_UNSTATED`。

    **「磁場があるか」と「どの向きか」は別の問い**で、訊いていたのは向きだけだった
    ——`f1_field` は `横磁場|縦磁場` しか知らないので、**向きの書かれていない磁場は
    磁場なしと区別できなかった**。実測 2026-09-07:

        磁場中のイジング模型 2サイトで S^z_1 の交換子を求めて  → [H, S^z_1] = 0
        Ising model in a magnetic field, 2 sites, …            → 0

    磁場を書かなかった場合と**同じ答え**で、しかも `[H,S^z]=0`（S^z は保存する）は
    横磁場なら偽である。両方 `completed`。

    向きを既定しないのは、それが**物理の選択**だから（CLAUDE.md「問題依存の物理の
    選択を共有コードに焼き込まない」）。微分の種別（∂ か d か）とまったく同じ形で、
    読めなければ訊く——`ui/interpret_early.py` の `field_unstated_refusal` が
    その1箇所。

    直す前はこの4行が `_parse_spin_commutator` / `_parse_spectral_function` /
    `_parse_spin_spectral` の**3箇所に写して**あった。第三の状態を足すと3箇所とも
    直すことになるので、読みをここへ出した。
    """
    if nlp("f1_field_longitudinal", lang).search(text):
        return ("h", "z")
    if nlp("f1_field_transverse", lang).search(text):
        return ("h", "x")
    if nlp("f1_field_mention", lang).search(text):
        return FIELD_DIRECTION_UNSTATED
    return None


def field_unstated_refusal(spec, lang, registered) -> "dict | None":
    """磁場と書いてあるのに**向きが読めない**なら、理由つきの失敗を返す。

    理由は `read_field` に。**`build_hamiltonian(field=…)` を呼ぶ手前で必ず訊く**
    こと——3つの呼び出し側があり、`tests/test_field_direction.py` が AST で数えている。
    """
    from notebook.runtime.i18n import t as _t
    if spec.get("field") != FIELD_DIRECTION_UNSTATED:
        return None
    return {"status": "failed",
            "error": _t("interpreter.field_direction_unstated", lang),
            "registered_expressions": registered}


def _parse_spin_commutator(text: str, lang: str = "ja") -> "dict | None":
    """F-1 3c: "<模型> <格子>で S^α_n の[N段]交換子を求めて" / "the Nth-order
    commutator of S^α_n in <model> on <lattice>" → a nested spin commutator
    [H,[…[H,S^α_n]…]] on a finite lattice.  Supports Heisenberg / Ising / XXZ /
    XY, an optional transverse (横磁場→x) / longitudinal (縦磁場→z) field, and
    an arbitrary nesting order (二重/double=2, "3段交換子"/"3rd-order
    commutator"=3, …)."""
    t = (text or "").strip()
    if not nlp("f1_commut", lang).search(t):
        return None
    mm = nlp("f1_spin_model", lang).search(t)
    op_m = _SPIN_OP_RE.search(t)
    if not mm or not op_m:
        return None
    spec = _parse_graph_spec(t, lang)
    if not spec:
        return None
    om = nlp("f1_order_n", lang).search(t)
    # 変換は `number_word` 1つ（全角も漢数字も語の序数も、そこが読む）。
    order = (number_word(om.group(1)) if om
             else 2 if nlp("f1_order2", lang).search(t) else 1) or 1
    field = read_field(t, lang)
    return {"kind": _spin_model_kind(mm.group()), "comp": op_m.group(1),
            "site": int(op_m.group(2)), "order": order, "spec": spec, "field": field}


# F-1 3c: infinite-temperature spin-½ moments / continued fraction (recursion
# method).  moment / continued-fraction / depth grammar → nl_patterns.json
# (f1_moment / f1_contfrac / f1_spec_depth, depth groups d1/d2/d3).
def _spec_depth_written(m: "re.Match") -> "tuple[str | None, int | None]":
    """`(単位, 数)` —— **`次`（次数）と `段`/`深さ`（段数）は別の量である。**

    どちらが書かれたかは `f1_spec_depth` の**群名**が言い（`order` / `depth`）、
    量への換算は `requested_order.recursion_units` が持つ。枝の接尾辞は
    `pick_group` が読むので並べない。
    """
    o = pick_group(m, "order")
    if o is not None:
        return "order", number_word(o)
    d = pick_group(m, "depth")
    if d is not None:
        return "depth", number_word(d)
    return None, None

# general-formula / general-spin keywords → nl_patterns.json (f1_general /
# f1_general_spin); spectral-function / square-root-terminator triggers →
# f1_specfn / f1_terminator.
_SPIN_OP_NOSITE_RE = re.compile(r'S\^?\{?([xyz])\}?', re.U)


def _parse_spectral_function(text: str, lang: str = "ja") -> "dict | None":
    """F-1 3c: "<模型> <格子>で S^α_n のスペクトル関数／グリーン関数を求めて" → the
    recursion-method Green's function (continued fraction) and, for a single-mode
    terminating chain, the spectral function S(ω)."""
    t = (text or "").strip()
    if not nlp("f1_specfn", lang).search(t):
        return None
    mm = nlp("f1_spin_model", lang).search(t)
    om = _SPIN_OP_RE.search(t)
    if not mm or not om:
        return None
    spec = _parse_graph_spec(t, lang)
    if not spec:
        return None
    field = read_field(t, lang)
    return {"kind": _spin_model_kind(mm.group()), "comp": om.group(1),
            "site": int(om.group(2)), "spec": spec, "field": field}


def _parse_spin_spectral(text: str, lang: str = "ja") -> "dict | None":
    """F-1 3c: "<模型> <格子>で S^α_n の m 次までのモーメント／連分数を求めて" →
    infinite-temperature spin-½ moments M_{2k} or recursion-method b_n².  With
    "一般"/"配位数" the un-expanded general formula in the coordination number z is
    returned instead (no concrete lattice needed)."""
    t = (text or "").strip()
    moment, cf = nlp("f1_moment", lang).search(t), nlp("f1_contfrac", lang).search(t)
    if not (moment or cf):
        return None
    mm = nlp("f1_spin_model", lang).search(t)
    if not mm:
        return None
    general = bool(nlp("f1_general", lang).search(t))
    om = _SPIN_OP_RE.search(t)
    if om:
        comp, site = om.group(1), int(om.group(2))
    elif general and _SPIN_OP_NOSITE_RE.search(t):
        comp, site = _SPIN_OP_NOSITE_RE.search(t).group(1), 0
    else:
        return None
    spec = _parse_graph_spec(t, lang)
    if not spec and not general:
        return None
    dm = nlp("f1_spec_depth", lang).search(t)
    # 一般式の枝は depth を**読んで捨てる**（`general_low_moments` は名前どおり
    # 低次だけ）ので、書かれたかどうかを持ち回って使う地点で断る。
    # `depth` は計算に渡す**段数**、`requested` は研究者が求めた量を**その枝の
    # 単位で**言い直したもの（モーメントなら次数、連分数なら段数）。換算の理由は
    # `requested_order.recursion_units` にある。
    _unit, _n = _spec_depth_written(dm) if dm else (None, None)
    depth, requested = recursion_units(
        "contfrac" if cf else "moments", _unit, _n, 3 if cf else 2)
    field = read_field(t, lang)
    return {"mode": "contfrac" if cf else "moments", "kind": _spin_model_kind(mm.group()),
            "comp": comp, "site": site, "depth": depth, "spec": spec,
            "field": field, "general": general, "requested": requested}


# F-1 3c: infinite-temperature energy moments ⟨H^n⟩ (high-T expansion) — the
# bridge to 3c-graph: ⟨H²⟩∝|E|=½tr A², ⟨H³⟩ carries the triangle count N△=⅙tr A³.
# energy-moment / specific-heat triggers → nl_patterns.json (f1_energy / f1_heat).


def _parse_specific_heat(text: str, lang: str = "ja") -> "dict | None":
    """F-1 3c: "<模型> [<格子>] の比熱の高温展開を[一般式で]求めて" → the
    high-temperature series of the specific heat C(β) (a finite-temperature
    observable built from the ∞-T energy moments)."""
    t = (text or "").strip()
    if not nlp("f1_heat", lang).search(t):
        return None
    mm = nlp("f1_spin_model", lang).search(t)
    if not mm:
        return None
    spec = _parse_graph_spec(t, lang)
    general = bool(nlp("f1_general", lang).search(t)) or spec is None
    return {"kind": _spin_model_kind(mm.group()), "general": general, "spec": spec}


def _parse_energy_moment(text: str, lang: str = "ja") -> "dict | None":
    """F-1 3c: "<模型> [<格子>] の高温展開／エネルギーモーメントを求めて" → the ∞-T
    energy moments ⟨H²⟩,⟨H³⟩.  With "一般" (or no lattice) the general formula in the
    graph invariants |E|, N△ is returned; with a lattice the concrete values."""
    t = (text or "").strip()
    if not nlp("f1_energy", lang).search(t):
        return None
    mm = nlp("f1_spin_model", lang).search(t)
    if not mm:
        return None
    spec = _parse_graph_spec(t, lang)
    general = bool(nlp("f1_general", lang).search(t)) or spec is None
    dm = nlp("f1_spec_depth", lang).search(t)
    # **書かれた生の値も持ち回る。** `max(2, min(4, …))` は範囲外を黙って丸める
    # ——`6次まで` が4次の答えを `completed` で返していた（実測 2026-09-06）。
    # 丸めた値は下流がそのまま使うので残し、判断材料として生の値を足す。
    # ここは ⟨H^n⟩ で **recursion が無い**ので、`次` と `段` は同じ n を指す
    # （モーメント／連分数の枝だけが段数と次数を区別する）。
    raw = _spec_depth_written(dm)[1] if dm else None
    upto = max(2, min(4, raw)) if raw is not None else 3
    return {"kind": _spin_model_kind(mm.group()), "general": general,
            "general_spin": bool(nlp("f1_general_spin", lang).search(t)),
            "spec": spec, "upto": upto, "upto_written": raw}


def _parse_graph_spec(text: str, lang: str = "ja") -> "dict | None":
    """Parse a finite-graph specification shared by the 3c-graph commands:
    a named lattice (正方/三角/ハニカム + LxL), an explicit custom edge list, a
    complete graph, or an N-site 1D chain — plus the boundary."""
    t = (text or "").strip()
    args: dict = {}
    if nlp("f1_graph_edge", lang).search(t):
        from notebook.runtime.backend_handler import _parse_edges
        edges = _parse_edges(t)
    else:
        edges = []
    lat_m = nlp("f1_ns_lattice", lang).search(t)
    if edges:
        args["pair_constraint"] = "custom"
        args["edges"] = edges
        return args
    if nlp("f1_boundary_cylinder", lang).search(t):
        args["boundary"] = "cylinder"
    elif nlp("f1_boundary_periodic", lang).search(t):
        args["boundary"] = "periodic"
    else:
        args["boundary"] = "open"
    if lat_m:
        _lgd = lat_m.groupdict()
        args["pair_constraint"] = ("honeycomb_2d" if _lgd.get("hc")
                                   else "triangular_2d" if _lgd.get("tri")
                                   else "square_2d")
        shape_m = _NS_SHAPE_RE.search(t)
        if not shape_m:
            return None
        args["lattice_shape"] = f"{shape_m.group(1)}x{shape_m.group(2)}"
        args["n_sites"] = int(shape_m.group(1)) * int(shape_m.group(2))
        return args
    nm = nlp("f1_ns_nsite", lang).search(t)
    if not nm:
        return None
    args["n_sites"] = int(nm.group(1).translate(_NS_ZENKAKU))
    args["pair_constraint"] = ("all_pairs" if nlp("f1_graph_allpairs", lang).search(t)
                               else "nearest_neighbor")
    return args


def _parse_crystal(text: str, lang: str = "ja") -> "list | None":
    """F-1 3c: "次のボンドで結晶の分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)" →
    a list of (α, β, ΔR-tuple, amplitude) bonds for the abstract Bloch builder."""
    t = (text or "").strip()
    if not nlp("f1_dispersion", lang).search(t) or not nlp("f1_crystal_kw", lang).search(t):
        return None
    bonds = []
    for m in _BOND_TOK_RE.finditer(t):
        dr = tuple(int(x) for x in m.group(3).split(",") if x.strip())
        bonds.append((int(m.group(1)), int(m.group(2)), dr, m.group(4).strip()))
    return bonds or None
# Optional next-nearest (diagonal) hopping and staggered sublattice (mass)
# potential keywords → nl_patterns.json["f1_nnn"/"f1_onsite"] (each captures an
# optional amplitude token, groups amp/amp2).  "next-nearest" swallows an
# optional "-neighbor" suffix so the trailing text isn't mis-captured as the
# amplitude.


def _bond_sum_coeff(body: str) -> str:
    """Extract the scalar prefactor of a bond-sum body ``(-t)*(c'_i c_j)`` → ``-t``.

    A body with no leading parenthesised coefficient yields ``""`` (amplitude 1).
    """
    m = re.match(r"^\s*\(([^()]*)\)\s*\*\s*\(", body or "")
    return m.group(1).strip() if m else ""


def _find_unrecognized_lattice_mention(t: str, lang: str = "ja") -> "re.Match | None":
    """Detect an explicit lattice-name mention (JP "<name>格子" / EN "<name>
    lattice") that none of the recognized keyword checks
    (_NS_LATTICE_RE/_KAGOME_RE/_LIEB_RE, used by both _parse_dispersion and
    _parse_neighbor_sum) recognized, so an unsupported lattice name (e.g.
    "ダイヤモンド格子"/"diamond lattice", "パイロクロア格子"/"pyrochlore
    lattice") can be rejected explicitly instead of silently falling back to
    a default lattice — see the silent-lattice-fallback bugfix.  Only
    meaningful when called after those keyword checks have already failed to
    match, so any mention found here is by construction unrecognized.
    Excludes "格子和"/"lattice sum" (the F-1 neighbour-sum trigger),
    "格子図"/"lattice figure" (the standalone-figure-save trigger), and
    "副格子"/"サブ格子"/"sublattice" (the staggered onsite-potential keyword),
    which legitimately contain "格子"/"lattice" without naming a crystal
    structure.  Returns the ``re.Match`` (so callers can use both the name,
    via ``m.group("ja") or m.group("en")``, and its position in the text,
    via ``m.start()``), or None if no such mention is found.
    """
    for m in nlp("f1_lattice_mention", lang).finditer(t):
        window = t[max(0, m.start() - 4): m.end() + 4]
        if nlp("f1_lattice_mention_safe", lang).search(window):
            continue
        gd = m.groupdict()
        name = (gd.get("ja") or gd.get("en") or "").strip()
        if name:
            return m
    return None


def _find_unrecognized_lattice_name(t: str, lang: str = "ja") -> "str | None":
    """Convenience wrapper around _find_unrecognized_lattice_mention that
    returns just the matched name (used by _parse_dispersion, which only
    needs the name for its error message, not its position)."""
    m = _find_unrecognized_lattice_mention(t, lang)
    if not m:
        return None
    gd = m.groupdict()
    return (gd.get("ja") or gd.get("en") or "").strip()


def _parse_dispersion(text: str, engine, lang: str = "ja"):
    """F-1 3c: "En/H の[正方格子で][次近接 t'で][質量 Δ で]分散関係を求めて" where the
    target is a stored symbolic bond sum → build the tight-binding dispersion.

    The hopping amplitude is the bond body's scalar prefactor; the lattice is the
    descriptor's ``pair_constraint`` unless overridden (正方格子/三角格子/ハニカム).
    A 次近接/NNN keyword adds a same-sublattice hopping (default ``-t'``; a second
    comma-separated amplitude gives per-sublattice t_A′,t_B′); a 質量/副格子ポテンシャル
    keyword adds a staggered onsite term (default ``\\Delta``).
    Returns ``(coeff, pair_constraint, nnn, onsite, nnn_b, error)`` or None
    (``None`` only for the two guard cases: no dispersion verb, or no
    resolvable bond-sum target — unchanged from before).  ``error`` is
    non-None when the text names an explicit lattice that none of the known
    keywords recognized, instead of silently defaulting to the bond sum's
    existing lattice.
    """
    t = (text or "").strip()
    if not nlp("f1_dispersion", lang).search(t):
        return None

    def _is_bond_sum(ex):
        d = getattr(ex, "lattice_descriptor", None)
        return bool(d) and d.get("kind", "bond_sum") == "bond_sum"

    desc = None
    for m in _CONCRETIZE_EID_RE.finditer(t):
        ex = getattr(engine, "expressions", {}).get(f"E{m.group(1)}")
        if ex is not None and _is_bond_sum(ex):
            desc = ex.lattice_descriptor
            break
    if desc is None:
        for _name, _eid in getattr(engine, "lhs_alias_map", {}).items():
            if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(_name)
                             + r'(?![A-Za-z0-9_])', t):
                continue
            ex = getattr(engine, "expressions", {}).get(_eid)
            if ex is not None and _is_bond_sum(ex):
                desc = ex.lattice_descriptor
                break
    if desc is None:
        return None

    pc = desc.get("pair_constraint", "nearest_neighbor")
    lat_m = nlp("f1_ns_lattice", lang).search(t)        # optional lattice override
    unrecognized_name = None
    if lat_m:
        _lgd = lat_m.groupdict()
        if _lgd.get("hc"):
            pc = "honeycomb_2d"
        elif _lgd.get("tri"):
            pc = "triangular_2d"
        elif _lgd.get("sq"):
            pc = "square_2d"
    elif nlp("f1_kagome", lang).search(t):              # non-bipartite ring (kagome)
        pc = "kagome_2d"
    elif nlp("f1_lieb", lang).search(t):               # star-structured (Lieb)
        pc = "lieb_2d"
    else:
        unrecognized_name = _find_unrecognized_lattice_name(t, lang)
    nnn = nnn_b = None
    nnn_m = nlp("f1_nnn", lang).search(t)
    if nnn_m:
        nnn = nnn_m.group("amp") or "-t'"
        nnn_b = nnn_m.group("amp2")
        if nnn_b == nnn:               # two equal amplitudes → symmetric
            nnn_b = None
    onsite = None
    on_m = nlp("f1_onsite", lang).search(t)
    if on_m:
        onsite = on_m.group("amp") or r"\Delta"
    error = None
    if unrecognized_name:
        error = (f"dispersion_relation: unrecognized lattice name {unrecognized_name!r}; "
                 f"known names: square, triangular, honeycomb, kagome, Lieb "
                 f"(for other lattices, use the abstract crystal-descriptor form: "
                 f"\"the dispersion relation of the crystal with the following bonds: …\")")
    return (_bond_sum_coeff(desc.get("body", "")), pc, nnn, onsite, nnn_b, error)


# B stage 2: substitution into a symbolic bond sum, keeping the sum symbolic
# ("H の J を 2t に置き換え" / "replace J with 2t in H" → H = -2t Σ_{⟨ij⟩} …).
# Grammar → nl_patterns.json["f1_bond_subst"] (groups ref/lhs/rhs).
def _token_sub(s: str, lhs: str, rhs: str) -> str:
    """Replace the whole token *lhs* with *rhs* in *s* (TeX-command-aware)."""
    if lhs.startswith("\\"):
        pat = re.escape(lhs) + r"(?![A-Za-z])"
    else:
        pat = r"(?<![A-Za-z0-9_\\])" + re.escape(lhs) + r"(?![A-Za-z0-9_])"
    return re.sub(pat, lambda _m: rhs, s)


# B stage 3b: commutator of a symbolic bond sum with a local operator
# [H, S^x_n] / "H と S^x_n の交換子"  → neighbour sum Σ_{j∈nn(n)} [body(n,j), O_n].
# The bracket form is language-neutral (kept in code); the JP keyword form →
# nl_patterns.json["f1_bond_comm_kw"] (ja-only; EN uses the bracket form).
_BOND_COMM_BRACKET_RE = re.compile(
    r'\[\s*(?P<ref>E\d+|[A-Za-z][A-Za-z0-9_]*)\s*,\s*'
    r'S\^?(?P<comp>[xyzpm])_(?P<site>[a-z])\s*\]', re.U)


def _relabel_sites(latex: str, mapping: dict) -> str:
    """Relabel SITE subscripts only (``Sy_1 → Sy_n``), never coefficients.

    ``mapping`` maps a concrete site number (int) to a label (str).
    """
    for site, label in mapping.items():
        latex = re.sub(r"_\{" + str(site) + r"\}", f"_{{{label}}}", latex)
        latex = re.sub(r"_" + str(site) + r"(?![0-9])", f"_{label}", latex)
    return latex


def _parse_bond_sum_commutator(text: str, engine, lang: str = "ja") -> "dict | None":
    """Detect ``[H, S^α_n]`` where H is a symbolic bond sum.  Returns a dict
    {ref_eid, comp, site, descriptor} or None."""
    m = _BOND_COMM_BRACKET_RE.search(text or "") or nlp("f1_bond_comm_kw", lang).search(text or "")
    if not m:
        return None
    ref = m.group("ref")
    eid = ref if ref in getattr(engine, "expressions", {}) else \
        getattr(engine, "lhs_alias_map", {}).get(ref)
    if not eid:
        return None
    ex = getattr(engine, "expressions", {}).get(eid)
    desc = getattr(ex, "lattice_descriptor", None) if ex else None
    if not desc or desc.get("kind", "bond_sum") != "bond_sum":
        return None
    return {"eid": eid, "comp": m.group("comp"), "site": m.group("site"),
            "descriptor": desc}


def _parse_neighbor_sum_concretize(text: str, engine, lang: str = "ja") -> "dict | None":
    """B stage 3b: concretize a neighbour-sum EOM at a concrete site on a lattice.
    "この近傍和を 正方格子 3x3 のサイト5で展開" → build the local Hamiltonian
    (bonds touching site 5) via pin_site, then commute with the operator at 5.
    """
    t = (text or "").strip()
    if not nlp("f1_concretize_verb", lang).search(t):
        return None
    sm = nlp("f1_ns_site", lang).search(t)
    if not sm:
        return None
    site = int(sm.group(1).translate(_NS_ZENKAKU))
    desc = None
    for m in _CONCRETIZE_EID_RE.finditer(t):
        ex = getattr(engine, "expressions", {}).get(f"E{m.group(1)}")
        if ex and (getattr(ex, "lattice_descriptor", None) or {}).get("kind") == "neighbor_sum":
            desc = ex.lattice_descriptor
            break
    if desc is None:  # fall back to the most recent neighbour sum
        best = -1
        for eid, ex in getattr(engine, "expressions", {}).items():
            d = getattr(ex, "lattice_descriptor", None)
            if d and d.get("kind") == "neighbor_sum":
                n = int(eid[1:]) if eid[1:].isdigit() else 0
                if n > best:
                    best, desc = n, d
    if desc is None:
        return None

    pair_constraint = desc.get("pair_constraint", "nearest_neighbor")
    lattice_shape = None
    n_sites = None
    lat_m = nlp("f1_ns_lattice", lang).search(t)
    if lat_m:
        if lat_m.group("hc"):
            pair_constraint = "honeycomb_2d"
        elif lat_m.group("tri"):
            pair_constraint = "triangular_2d"
        else:
            pair_constraint = "square_2d"
    shape_m = _NS_SHAPE_RE.search(t)
    if shape_m:
        lattice_shape = f"{shape_m.group(1)}x{shape_m.group(2)}"
        n_sites = int(shape_m.group(1)) * int(shape_m.group(2))
    else:
        nm = nlp("f1_ns_nsite", lang).search(t)
        if nm:
            n_sites = int(nm.group(1).translate(_NS_ZENKAKU))
    boundary = ("cylinder" if nlp("f1_boundary_cylinder", lang).search(t)
                else "periodic" if nlp("f1_boundary_periodic", lang).search(t)
                else "open")
    return {
        "expand_args": {
            "body": desc.get("orig_body"), "index1": desc.get("index1"),
            "index2": desc.get("index2"), "pin_site": site,
            "pair_constraint": pair_constraint, "lattice_shape": lattice_shape,
            "n_sites": n_sites, "boundary": boundary,
        },
        "op": f"S{desc.get('comp')}_{site}",
    }


def _parse_bond_sum_substitute(text: str, engine, lang: str = "ja") -> "tuple[str, str, str] | None":
    """Detect a substitution targeting a symbolic bond sum (B stage 2).
    Returns (expr_id, lhs, rhs) or None (so a normal substitution still works).
    """
    m = nlp("f1_bond_subst", lang).search(text or "")
    if not m:
        return None
    ref = m.group("ref")
    eid = None
    if ref in getattr(engine, "expressions", {}):
        eid = ref
    elif ref in getattr(engine, "lhs_alias_map", {}):
        eid = engine.lhs_alias_map[ref]
    if not eid:
        return None
    ex = getattr(engine, "expressions", {}).get(eid)
    if ex is None or not getattr(ex, "lattice_descriptor", None):
        return None
    return (eid, m.group("lhs").strip(), m.group("rhs").strip())
