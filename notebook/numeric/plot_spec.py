"""Declarative Cartesian plots: expression references, never executable text.

Independent of a domain or a solver. A marker may refer to a CAS-computed
coordinate; plotting it does not claim that the point is an optimum.
"""
from __future__ import annotations

import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from notebook.runtime.shared_parser import UnknownNotationError, split_relation
from notebook.numeric.expr_to_numpy import canonical_to_numpy


class ExpressionRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expr_id: str = Field(pattern=r'^E[0-9]+$')


Coordinate = FiniteFloat | ExpressionRef


class PlotLayer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['line', 'contour', 'region', 'point']
    expr_id: str | None = Field(default=None, pattern=r'^E[0-9]+$')
    constraints: list[str] = Field(default_factory=list, max_length=32)
    x: Coordinate | None = None
    y: Coordinate | None = None
    levels: list[Coordinate] = Field(default_factory=lambda: [0.0], max_length=32)
    label: str = Field(default='', max_length=300)
    color: str | None = Field(default=None, max_length=40)
    linestyle: Literal['-', '--', ':', '-.'] = '-'
    alpha: FiniteFloat = Field(default=0.25, gt=0, le=1)
    marker: Literal['o', 's', '*', 'x', '+', '^', 'D', '.'] = '*'
    markersize: FiniteFloat = Field(default=12, gt=0, le=40)
    fixed_values: dict[str, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_kind(self):
        if self.kind in ('line', 'contour') and not self.expr_id:
            raise ValueError('expr_id required')
        if self.kind == 'region' and not self.constraints:
            raise ValueError('constraints required')
        if self.kind == 'point' and (self.x is None or self.y is None):
            raise ValueError('point coordinates required')
        if self.kind != 'region' and self.constraints:
            raise ValueError('constraints only apply to regions')
        if self.kind not in ('line', 'contour') and self.expr_id:
            raise ValueError('expr_id only applies to lines and contours')
        if self.kind != 'point' and (self.x is not None or self.y is not None):
            raise ValueError('coordinates only apply to points')
        if self.kind == 'contour' and not self.levels:
            raise ValueError('contour levels required')
        return self


class CartesianPlot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    x: str
    y: str | None = None
    layers: list[PlotLayer] = Field(min_length=1, max_length=32)
    title: str = Field(default='', max_length=300)
    xlabel: str = Field(default='', max_length=300)
    ylabel: str = Field(default='', max_length=300)
    formats: list[Literal['png', 'svg', 'pdf']] = Field(default_factory=lambda: ['png', 'svg'])
    dpi: int = Field(default=200, ge=72, le=600)
    legend: bool = True


def is_cartesian(spec):
    return spec is not None and spec.presentation.kind == 'cartesian'


def read_plot(spec):
    from pydantic import ValidationError
    try:
        if spec.presentation.kind != "cartesian":
            raise ValueError("Cartesian presentation required")
        plot = CartesianPlot.model_validate(spec.presentation.options)
        if spec.method.name not in ('', 'direct_eval'):
            raise ValueError('plotting does not run a solver')
        sweeps = {v.name: v for v in spec.inputs if v.role == 'sweep'}
        if not plot.formats or len(plot.formats) != len(set(plot.formats)):
            raise ValueError('formats')
        if not plot.x or plot.y == plot.x or set(sweeps) != {v for v in (plot.x, plot.y) if v}:
            raise ValueError('axes')
        if len({v.name for v in spec.inputs}) != len(spec.inputs):
            raise ValueError('duplicate inputs')
        if any(v.role not in ('sweep', 'fixed') for v in spec.inputs):
            raise ValueError('input roles')
        counts = []
        for v in sweeps.values():
            lo, hi, n = v.spec.get('min'), v.spec.get('max'), v.spec.get('points')
            if (not all(math.isfinite(float(z)) for z in (lo, hi)) or float(lo) >= float(hi)
                    or isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 2000):
                raise ValueError('grid')
            counts.append(n)
        if math.prod(counts) * len(plot.layers) > 2_000_000:
            raise ValueError('grid budget')
        if not plot.y and any(l.kind in ('region', 'contour', 'point') for l in plot.layers):
            raise ValueError('two axes required')
        return plot
    except (ValidationError, ValueError, TypeError, OverflowError) as exc:
        raise UnknownNotationError('numeric.plot.invalid_spec') from exc


def plot_reference_ids(plot):
    refs = []
    for layer in plot.layers:
        refs += ([layer.expr_id] if layer.expr_id else []) + layer.constraints
        refs += [v.expr_id for v in [layer.x, layer.y, *layer.levels] if isinstance(v, ExpressionRef)]
    return list(dict.fromkeys(refs))


def scalar_parts(canonical, *, relation=False):
    parts = split_relation(canonical)
    if not parts:
        if relation:
            raise UnknownNotationError('numeric.plot.need_inequality')
        return canonical, None
    lhs, op, rhs = parts
    if relation:
        if op not in ('<', '<=', '>', '>='):
            raise UnknownNotationError('numeric.plot.need_inequality')
    elif op != '=':
        raise UnknownNotationError('numeric.plot.need_scalar')
    return f'({lhs})-({rhs})', op


def snapshot_plot_sources(engine, spec):
    plot = read_plot(spec)
    sources, free = [], {n for n in (plot.x, plot.y) if n}
    for eid in plot_reference_ids(plot):
        expr = engine.expressions.get(eid)
        if expr is None:
            raise UnknownNotationError('numeric.task_source_missing', expr_id=eid)
        scalar = expr.canonical
        # Inequalities are meaningful only inside a region, where both sides
        # are lowered separately by the same scalar whitelist.
        parts = split_relation(expr.canonical)
        if parts:
            scalar = f'({parts[0]})-({parts[2]})'
        _, symbols = canonical_to_numpy(scalar, engine.symbol_context)
        free |= symbols
        sources.append({'expr_id':eid, 'canonical':expr.canonical,
                        'latex':expr.latex, 'branch_id':expr.branch_id})
    return sources, free


def plot_result_text(task_id, summary, diagnostics, lang):
    from notebook.runtime.i18n import t
    message = t('numeric.plot.result', lang, task_id=task_id, layers=summary['plot_layers'])
    for warning in diagnostics.get('warnings', []):
        if isinstance(warning, dict) and warning.get('code') in ('empty_sampled_region', 'levels_without_crossing'):
            message += ' ' + t('numeric.plot.' + warning['code'], lang, layer=warning['layer'] + 1)
    return message
