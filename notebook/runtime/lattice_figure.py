"""Lattice figure generation (F-1).

Turns a bond generator's output into a drawable figure so the user can confirm
that a generated lattice is the shape they intended, and embed it in a paper:

  - ``lattice_layout`` — site coordinates + bond list for a (pair_constraint,
    size, boundary) combination, reusing the same ``_BOND_GENERATORS`` registry
    that ``expand_neighbor_sum`` uses (so the figure matches the computation).
  - ``lattice_to_tikz`` — a ``tikzpicture`` for LaTeX papers.
  - ``lattice_to_svg`` — a standalone SVG for quick viewing.
  - ``save_lattice_figure`` — writes .svg + standalone .tex, and compiles a .pdf
    when a LaTeX engine is available.

Coordinates are stored in math orientation (y up).  Periodic wrap-around bonds
(those much longer than a unit cell) are drawn dashed so the boundary condition
is visible.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from notebook.runtime.backend_handler import _BOND_GENERATORS, _shape_or_square

_WRAP_FACTOR = 1.5  # a bond longer than this × the unit spacing is a wrap edge


def lattice_layout(pair_constraint, n_sites, boundary="open", lattice_shape=None,
                   edges=None):
    """Return ``(positions, bonds)`` where positions maps a site index to an
    (x, y) coordinate (math orientation) and bonds is the list of (a, b) pairs.

    Raises ValueError for an unknown pair_constraint or invalid geometry.
    """
    # custom: arbitrary graph from an explicit edge list, drawn on a circle.
    if pair_constraint == "custom":
        from notebook.runtime.backend_handler import _parse_edges
        bonds = _parse_edges(edges)
        if not bonds:
            raise ValueError("custom layout needs an 'edges' list")
        n = max(max(a, b) for a, b in bonds)
        positions = {idx: (math.cos(2 * math.pi * (idx - 1) / n),
                           math.sin(2 * math.pi * (idx - 1) / n))
                     for idx in range(1, n + 1)}
        return positions, bonds

    gen = _BOND_GENERATORS.get(pair_constraint)
    if gen is None:
        raise ValueError(
            f"unknown pair_constraint {pair_constraint!r}; "
            f"known: {sorted(_BOND_GENERATORS)}")
    shape = None
    if lattice_shape:
        if isinstance(lattice_shape, (tuple, list)):
            shape = (int(lattice_shape[0]), int(lattice_shape[1]))
        else:
            import re
            m = re.match(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*$", str(lattice_shape))
            if not m:
                raise ValueError(f"invalid lattice_shape {lattice_shape!r}")
            shape = (int(m.group(1)), int(m.group(2)))
    n = shape[0] * shape[1] if shape else int(n_sites)
    bonds = gen(n, boundary, shape)

    positions: dict[int, tuple[float, float]] = {}
    if pair_constraint in ("square_2d", "triangular_2d", "honeycomb_2d"):
        Lx, Ly = _shape_or_square(shape, n)
        shear = 0.5 if pair_constraint == "triangular_2d" else 0.0
        dy = math.sqrt(3) / 2 if pair_constraint == "triangular_2d" else 1.0
        for idx in range(1, n + 1):
            x = (idx - 1) % Lx
            y = (idx - 1) // Lx
            positions[idx] = (x + shear * y, y * dy)
    elif pair_constraint in ("all_pairs", "ordered_pairs"):
        # arrange on a circle so every chord is visible (both pair sets are
        # complete graphs; ordered_pairs draws each chord twice)
        for idx in range(1, n + 1):
            ang = 2 * math.pi * (idx - 1) / n
            positions[idx] = (math.cos(ang), math.sin(ang))
    else:  # nearest_neighbor / 1D chain
        for idx in range(1, n + 1):
            positions[idx] = (idx - 1, 0.0)
    return positions, bonds


def _bond_is_wrap(positions, bonds, a, b) -> bool:
    pa, pb = positions[a], positions[b]
    dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    units = [math.hypot(positions[u][0] - positions[v][0],
                        positions[u][1] - positions[v][1]) for u, v in bonds]
    unit = min(units) if units else 1.0
    return dist > _WRAP_FACTOR * unit


def lattice_to_tikz(positions, bonds, scale=1.2) -> str:
    """A ``tikzpicture`` (requires ``\\usepackage{tikz}`` or standalone[tikz])."""
    lines = [
        f"\\begin{{tikzpicture}}[scale={scale}, "
        "site/.style={circle,draw,fill=white,inner sep=1pt,"
        "minimum size=5mm,font=\\scriptsize}]",
    ]
    for a, b in bonds:
        (xa, ya), (xb, yb) = positions[a], positions[b]
        style = "[dashed]" if _bond_is_wrap(positions, bonds, a, b) else ""
        lines.append(f"  \\draw{style} ({xa:.3f},{ya:.3f}) -- ({xb:.3f},{yb:.3f});")
    for idx in sorted(positions):
        x, y = positions[idx]
        lines.append(f"  \\node[site] at ({x:.3f},{y:.3f}) {{{idx}}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def tikz_figure_for_operation(operation, args) -> "str | None":
    """Centered ``tikzpicture`` LaTeX for an ``expand_neighbor_sum`` step, or
    None when the operation is not a lattice sum or the geometry is invalid.

    Used by the TeX export to drop the lattice figure next to its expanded
    Hamiltonian.  Requires ``tikz`` in the document preamble.
    """
    if operation != "expand_neighbor_sum":
        return None
    args = args or {}
    try:
        positions, bonds = lattice_layout(
            args.get("pair_constraint", "nearest_neighbor"),
            args.get("n_sites"), args.get("boundary", "open"),
            args.get("lattice_shape"), args.get("edges"))
    except (ValueError, TypeError):
        return None
    return "\\begin{center}\n" + lattice_to_tikz(positions, bonds) + "\n\\end{center}"


def lattice_to_standalone_tex(tikz: str) -> str:
    return ("\\documentclass[tikz,border=4pt]{standalone}\n"
            "\\begin{document}\n" + tikz + "\n\\end{document}\n")


def lattice_to_svg(positions, bonds, px_per_unit=70, margin=28, radius=14) -> str:
    """A standalone SVG (with xmlns) for quick viewing in a browser/image tool."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w = (maxx - minx) * px_per_unit + 2 * margin
    h = (maxy - miny) * px_per_unit + 2 * margin

    def sx(x):
        return margin + (x - minx) * px_per_unit

    def sy(y):  # flip: math y-up → screen y-down
        return margin + (maxy - y) * px_per_unit

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" '
        f'height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for a, b in bonds:
        dash = ' stroke-dasharray="5,4"' if _bond_is_wrap(positions, bonds, a, b) else ""
        parts.append(
            f'<line x1="{sx(positions[a][0]):.1f}" y1="{sy(positions[a][1]):.1f}" '
            f'x2="{sx(positions[b][0]):.1f}" y2="{sy(positions[b][1]):.1f}" '
            f'stroke="#444" stroke-width="2"{dash}/>')
    for idx in sorted(positions):
        x, y = positions[idx]
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" '
            f'fill="#e1f5ee" stroke="#0f6e56" stroke-width="2"/>')
        parts.append(
            f'<text x="{sx(x):.1f}" y="{sy(y) + 4:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#04342c">{idx}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _latest_lattice_args(engine):
    """Return the operation_args of the most recent ``expand_neighbor_sum`` step
    in *engine*, or None when there is no lattice to draw."""
    best, best_n = None, -1
    for step in getattr(engine, "steps", {}).values():
        if getattr(step, "operation", None) != "expand_neighbor_sum":
            continue
        if not step.operation_args:
            continue
        sid = getattr(step, "step_id", "")
        n = int(sid[1:]) if sid[1:].isdigit() else 0
        if n >= best_n:
            best_n, best = n, step.operation_args
    return best


def save_lattice_figure_for_engine(engine, out_dir=None, name=None,
                                   compile_pdf=True) -> dict:
    """Save a standalone figure for the engine's most recent lattice sum.

    Returns the ``save_lattice_figure`` result with ``ok=True``, or
    ``{"ok": False, "error": "no_lattice"}`` when no lattice has been built.
    """
    args = _latest_lattice_args(engine)
    if not args:
        return {"ok": False, "error": "no_lattice"}
    if name is None:
        import re as _re
        title = getattr(getattr(engine, "session", None), "title", "") or "lattice"
        name = _re.sub(r"[^\w.-]+", "_", str(title)).strip("_") + "_lattice"
    result = save_lattice_figure(
        args.get("pair_constraint", "nearest_neighbor"),
        args.get("n_sites"), args.get("boundary", "open"),
        args.get("lattice_shape"), out_dir=out_dir, name=name,
        compile_pdf=compile_pdf, edges=args.get("edges"))
    result["ok"] = True
    return result


def _find_latex_engine():
    for eng in ("latexmk", "pdflatex", "lualatex"):
        if shutil.which(eng):
            return eng
    return None


def save_lattice_figure(pair_constraint, n_sites, boundary="open",
                        lattice_shape=None, out_dir=None, name="lattice",
                        compile_pdf=True, edges=None) -> dict:
    """Write ``<name>.svg`` and a standalone ``<name>.tex`` to *out_dir*, and
    compile ``<name>.pdf`` when a LaTeX engine is available.

    Returns a dict with the produced paths (``svg``/``tex``/``pdf``), the raw
    ``tikz`` string, and a ``pdf_error`` note when compilation was skipped/failed.
    """
    out_dir = Path(out_dir) if out_dir else Path("sessions")
    out_dir.mkdir(parents=True, exist_ok=True)
    positions, bonds = lattice_layout(pair_constraint, n_sites, boundary,
                                      lattice_shape, edges)
    tikz = lattice_to_tikz(positions, bonds)
    svg_path = out_dir / f"{name}.svg"
    tex_path = out_dir / f"{name}.tex"
    svg_path.write_text(lattice_to_svg(positions, bonds), encoding="utf-8")
    tex_path.write_text(lattice_to_standalone_tex(tikz), encoding="utf-8")

    result = {"svg": str(svg_path), "tex": str(tex_path), "tikz": tikz,
              "pdf": None, "pdf_error": None}
    if not compile_pdf:
        return result
    engine = _find_latex_engine()
    if engine is None:
        result["pdf_error"] = "no LaTeX engine (latexmk/pdflatex/lualatex) found"
        return result
    cmd = ([engine, "-pdf", "-interaction=nonstopmode", tex_path.name]
           if engine == "latexmk"
           else [engine, "-interaction=nonstopmode", tex_path.name])
    try:
        proc = subprocess.run(cmd, cwd=out_dir, capture_output=True,
                              timeout=120, text=True)
        pdf_path = out_dir / f"{name}.pdf"
        if pdf_path.exists():
            result["pdf"] = str(pdf_path)
        else:
            result["pdf_error"] = f"{engine} did not produce a PDF"
    except (subprocess.SubprocessError, OSError) as exc:
        result["pdf_error"] = f"{engine} failed: {exc}"
    return result
