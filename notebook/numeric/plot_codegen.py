"""Lower referenced scalar expressions through the numpy whitelist to a plot.

No eval, arbitrary Python, domain constants, optimization or LLM code here.
Each expression is frozen at fork time. Symbol names are replaced with
private axis arguments or validated finite literals, avoiding Python-name
collisions and injection through untrusted specs.
"""
from __future__ import annotations

import ast
import inspect
import math
from notebook.numeric.expr_to_numpy import canonical_to_numpy
from notebook.numeric.plot_spec import ExpressionRef, read_plot, scalar_parts
from notebook.numeric.plot_runtime import render_cartesian
from notebook.runtime.shared_parser import UnknownNotationError, split_relation
from notebook.runtime.i18n import t


def generate_plot_code(task, context=None, lang='en'):
    plot = read_plot(task.spec)
    sources = {s['expr_id']:s['canonical'] for s in task.plot_sources}
    fixed = {k:float(v) for k, v in task.fixed_values.items()}
    if not all(math.isfinite(v) for v in fixed.values()):
        raise UnknownNotationError('numeric.plot.invalid_spec')
    sweeps = {v.name:v.spec for v in task.spec.inputs if v.role == 'sweep'}
    axes = [[float(sweeps[n]['min']), float(sweeps[n]['max']), int(sweeps[n]['points'])]
            for n in (plot.x, plot.y) if n]

    def compile_value(value, bindings, mode='scalar'):
        if isinstance(value, ExpressionRef):
            value = value.expr_id
        if isinstance(value, (int, float)):
            return f'lambda _x, _y: {float(value)!r}', None
        if value not in sources:
            raise UnknownNotationError('numeric.task_source_missing', expr_id=value)
        canonical = sources[value]
        if mode == 'line' and split_relation(canonical):
            raise UnknownNotationError('numeric.plot.need_scalar')
        scalar, relation = scalar_parts(canonical, relation=(mode == 'region'))
        expression, free = canonical_to_numpy(scalar, context)
        layer_free.update(free)
        all_free.update(free)
        allowed_axes = {plot.x:'_x'} if mode == 'line' else {plot.x:'_x', plot.y:'_y'}
        if mode == 'scalar':
            allowed_axes = {}
        unbound = free - set(allowed_axes) - set(bindings)
        if unbound:
            raise UnknownNotationError('numeric.missing_values', symbols=', '.join(sorted(unbound)))
        class Bind(ast.NodeTransformer):
            def visit_Attribute(self, node):
                # A variable named np is distinct from the whitelisted np.sin.
                return node if isinstance(node.value, ast.Name) and node.value.id == "np" else self.generic_visit(node)
            def visit_Name(self, node):
                if node.id in free:
                    return (ast.Name(id=allowed_axes[node.id], ctx=ast.Load()) if node.id in allowed_axes
                            else ast.Constant(float(bindings[node.id])))
                return node
        tree = ast.fix_missing_locations(Bind().visit(ast.parse(expression, mode='eval')))
        return f'lambda _x, _y: {ast.unparse(tree.body)}', relation

    lines = []
    all_free = set()
    for index, layer in enumerate(plot.layers):
        layer_free = set()
        if set(layer.fixed_values) & {plot.x, plot.y}:
            raise UnknownNotationError('numeric.plot.invalid_spec')
        bindings = {**fixed, **layer.fixed_values}
        if layer.kind == 'point':
            values = [compile_value(v, bindings) for v in (layer.x, layer.y)]
        elif layer.kind == 'region':
            values = [compile_value(e, bindings, 'region') for e in layer.constraints]
        else:
            values = [compile_value(layer.expr_id, bindings, layer.kind)]
            if layer.kind == 'contour':
                values += [compile_value(v, bindings) for v in layer.levels]
        unused = set(layer.fixed_values) - layer_free
        if unused:
            raise UnknownNotationError("numeric.spec.unknown_symbol", name=sorted(unused)[0])
        data = layer.model_dump(mode='json')
        data['relations'] = [r for _, r in values]
        lines += [f'layers.append({data!r})',
                  f'layers[{index}]["functions"] = [{", ".join(s for s, _ in values)}]']
    unused = set(fixed) - all_free
    if unused:
        raise UnknownNotationError('numeric.spec.unknown_symbol', name=sorted(unused)[0])
    config = plot.model_dump(mode='json', exclude={'layers', 'x', 'y'})
    config.update(xlabel=plot.xlabel or plot.x, ylabel=plot.ylabel or plot.y or task.target_quantity,
                  target=task.target_quantity or 'plot',
                  nonreal_message=t('numeric.plot.nonreal', lang),
                  nonfinite_message=t('numeric.plot.nonfinite', lang))
    # PNG is always available for the existing timeline; other formats are
    # additional downloadable products, never replacements for the preview.
    config['formats'] = list(dict.fromkeys(['png', *plot.formats]))
    source = ('# Deterministic Cartesian plot; frozen sources: ' + repr(task.source_expression_ids) + '\n'
              + 'import numpy as np\n' + inspect.getsource(render_cartesian) + '\nlayers = []\n' + '\n'.join(lines)
              + f'\nrender_cartesian({config!r}, layers, {axes!r})\n')
    return source, {'source':'cartesian_template', 'plot_sources':task.plot_sources,
                    'grid':axes, 'layers':len(plot.layers), 'sampled_regions':True}
