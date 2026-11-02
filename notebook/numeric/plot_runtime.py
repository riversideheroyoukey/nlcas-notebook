"""Standalone renderer embedded in generated scripts (no project imports).

Regions are sampled on the declared finite grid, not exact polygons or a
proof of feasibility/optimality. Invalid function domains stay transparent.
"""


def render_cartesian(config, layers, axes):
    import json
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out = os.environ['NB_NUMERIC_OUT']
    x = np.linspace(*axes[0])
    y = np.linspace(*axes[1]) if len(axes) == 2 else None
    X, Y = np.meshgrid(x, y) if y is not None else (x, None)
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    handles, artifacts, diagnostics = [], [], []
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def real(value, shape):
        array = np.asarray(value, dtype=complex)
        if np.any((np.abs(array.imag) > 1e-12) | np.isnan(array.imag)):
            raise ValueError(config['nonreal_message'])
        return np.broadcast_to(array.real, shape)

    def scalar(fn):
        value = real(fn(0., 0.), ())
        if not np.isfinite(value):
            raise ValueError(config['nonfinite_message'])
        return float(value)

    for index, layer in enumerate(layers):
        kind, functions = layer['kind'], layer['functions']
        color = layer['color'] or palette[index % len(palette)]
        label = layer['label']
        file = f'layer_{index + 1:02d}.csv'
        path = os.path.join(out, file)
        with np.errstate(all='ignore'):
            if kind == 'point':
                px, py = [scalar(fn) for fn in functions]
                artist, = ax.plot([px], [py], marker=layer['marker'], markersize=layer['markersize'], linestyle='none', color=color)
                if label:
                    ax.annotate(label, (px, py), xytext=(6, 8), textcoords='offset points',
                                bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.8})
                data, header = np.array([[px, py]]), 'x,y'
            elif kind == 'line':
                values = real(functions[0](x, 0.), x.shape)
                values = np.where(np.isfinite(values), values, np.nan)
                artist, = ax.plot(x, values, color=color, linestyle=layer['linestyle'])
                data, header = np.column_stack([x, values]), 'x,value'
                if not np.any(np.isfinite(values)):
                    raise ValueError(config['nonfinite_message'])
            elif kind == 'contour':
                values = real(functions[0](X, Y), X.shape)
                levels = sorted(set(scalar(fn) for fn in functions[1:]))
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    raise ValueError(config['nonfinite_message'])
                active = [v for v in levels if finite.min() < v < finite.max()]
                if active:
                    ax.contour(X, Y, np.ma.masked_invalid(values), levels=active,
                               colors=[color], linestyles=layer['linestyle'])
                if len(active) != len(levels):
                    diagnostics.append({'layer':index, 'code':'levels_without_crossing'})
                artist = Line2D([], [], color=color, linestyle=layer['linestyle'])
                data, header = np.column_stack([X.ravel(), Y.ravel(), values.ravel()]), 'x,y,value'
            else:
                valid = np.ones(X.shape, dtype=bool)
                mask = np.ones(X.shape, dtype=bool)
                for fn, relation in zip(functions, layer['relations']):
                    residual = real(fn(X, Y), X.shape)
                    valid &= np.isfinite(residual)
                    comparisons = {'<':np.less, '<=':np.less_equal, '>':np.greater, '>=':np.greater_equal}
                    mask &= comparisons[relation](residual, 0.)
                mask &= valid
                if not np.any(valid):
                    raise ValueError(config['nonfinite_message'])
                ax.contourf(X, Y, np.ma.masked_where(~mask, mask.astype(float)),
                            levels=[.5, 1.5], colors=[color], alpha=layer['alpha'])
                if not np.any(mask):
                    diagnostics.append({'layer':index, 'code':'empty_sampled_region'})
                artist = Patch(facecolor=color, alpha=layer['alpha'])
                data = np.column_stack([X.ravel(), Y.ravel(), mask.ravel().astype(int), valid.ravel().astype(int)])
                header = 'x,y,inside,valid'
            np.savetxt(path, data, delimiter=',', header=header, comments='')
        artifacts.append({'name':f'layer_{index + 1:02d}', 'value_kind':'series',
                          'value_type':'real_float64', 'file':file})
        if label:
            handles.append((artist, label))
    ax.set(xlim=axes[0][:2], xlabel=config['xlabel'], ylabel=config['ylabel'], title=config['title'])
    if y is not None:
        ax.set_ylim(axes[1][:2])
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    if handles and config["legend"]:
        ax.legend([h for h, _ in handles], [s for _, s in handles])
    for extension in config['formats']:
        fig.savefig(os.path.join(out, 'figure.' + extension), dpi=config['dpi'], facecolor='white')
    plt.close(fig)
    summary = {'target':config['target'], 'plot_layers':len(layers), 'grid':axes}
    with open(os.path.join(out, 'plot.json'), 'w') as stream:
        json.dump({'summary':summary, 'diagnostics':diagnostics}, stream, indent=2)
    print('NB_RESULT ' + json.dumps(summary))
    print('NB_MANIFEST ' + json.dumps({'artifacts':artifacts, 'diagnostics':{
        'success':True, 'method':'cartesian_sampling', 'warnings':diagnostics}}))
