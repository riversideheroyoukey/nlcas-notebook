"""Plots must preserve mathematics across domains, snapshots and reloads."""
import json
from pathlib import Path
import pytest
from notebook.engine import NotebookEngine
from notebook.models.numeric import NumericSpec
from notebook.numeric.intent_registry import codegen_route_for_spec
from notebook.numeric.templates import generate_code
from notebook.numeric.sandbox import SandboxRunner
from notebook.runtime.shared_parser import UnknownNotationError


def make_spec(layers, two_axes=True, fixed=None, points=41):
    axes = ['x', 'y'] if two_axes else ['x']
    return NumericSpec.model_validate({
        'intent': {'kind':'cartesian_plot'},
        'inputs':[{'name':n, 'role':'sweep', 'spec':{'min':-2,'max':2,'points':points}} for n in axes]
                 + [{'name':k,'role':'fixed','spec':{'value':v}} for k,v in (fixed or {}).items()],
        'outputs':[{'name':'plot','shape':'field','over':axes}],
        'presentation':{'kind':'cartesian','options':{
            'x':'x', **({'y':'y'} if two_axes else {}), 'layers':layers,
            'formats':['png','svg','pdf']}}})


def register(engine, *expressions):
    return [engine.register_expression(e).display_id for e in expressions]


def run_plot(engine, root, source, spec):
    engine.save(root/'session.json')
    task = engine.fork_numeric_task(source, spec=spec)
    art = engine.generate_numeric_code(task.task_id)
    assert art.metadata['source'] == 'cartesian_template'
    assert 'eval(' not in art.content
    result = engine.run_numeric_task(task.task_id)
    assert result.ok, result.stderr
    paths = [Path(engine.numeric_artifacts[a].path) for a in result.result_artifact_ids
             if engine.numeric_artifacts[a].path]
    assert {'figure.png','figure.svg','figure.pdf'} <= {p.name for p in paths}
    return task, paths[0].parent


def csv_rows(file):
    import csv
    with open(file) as f:
        return list(csv.DictReader(f))


def test_nonlinear_region_hole_and_boundary(tmp_path):
    e = NotebookEngine()
    disk, outside, boundary, px, py = register(e, 'x^2+y^2<=1', 'x^2+y^2>=1/4',
                                              'x^2+y^2=1', '0', '1/2')
    spec = make_spec([
        {'kind':'region','constraints':[disk,outside],'label':'annulus'},
        {'kind':'contour','expr_id':boundary,'label':'outer boundary'},
        {'kind':'point','x':{'expr_id':px},'y':{'expr_id':py},'label':'reference'}])
    task, root = run_plot(e, tmp_path, boundary, spec)
    assert set(task.source_expression_ids) == {disk,outside,boundary,px,py}
    rows = csv_rows(root/'layer_01.csv')
    center = next(r for r in rows if abs(float(r['x']))<1e-12 and abs(float(r['y']))<1e-12)
    assert center['inside'].startswith('0.')
    assert any(float(r['inside']) == 1 for r in rows)
    e.expressions[disk].canonical = 'x^2+y^2<=4'
    assert e.numeric_task_stale_info(task.task_id)['stale']
    # Regeneration uses the snapshot, not the now-edited source.
    source, _ = generate_code(task, e.symbol_context, lang=e.lang)
    assert source == e.numeric_artifacts[next(a for a in e.numeric_artifacts
        if e.numeric_artifacts[a].artifact_type=='code')].content
    e.save(tmp_path/'session.json')
    loaded = NotebookEngine.load(tmp_path/'session.json')
    assert loaded.numeric_tasks[task.task_id].plot_sources == task.plot_sources
    assert loaded.numeric_task_stale_info(task.task_id)['stale']


def test_parameter_comparison_and_constant_curve(tmp_path):
    e = NotebookEngine()
    expr, const = register(e, 'k*x^2', '3')
    spec = make_spec([{'kind':'line','expr_id':expr,'label':'k=1'},
                      {'kind':'line','expr_id':expr,'fixed_values':{'k':3},'label':'k=3'},
                      {'kind':'line','expr_id':const}], two_axes=False, fixed={'k':1})
    task, root = run_plot(e,tmp_path,expr,spec)
    a, b, c = [csv_rows(root/f'layer_0{i}.csv') for i in (1,2,3)]
    for ar, br, cr in zip(a,b,c):
        assert float(br['value']) == pytest.approx(3*float(ar['value']))
        assert float(cr['value']) == 3
    assert codegen_route_for_spec(task.spec) == 'deterministic'


def test_partial_domain_is_not_filled_as_feasible(tmp_path):
    e = NotebookEngine()
    relation, = register(e, 'log(x)<=y')
    _, root = run_plot(e,tmp_path,relation,make_spec([{'kind':'region','constraints':[relation]}]))
    for r in csv_rows(root/'layer_01.csv'):
        if float(r['x']) <= 0:
            assert float(r['inside']) == 0 and float(r['valid']) == 0


def test_contour_level_comes_from_computed_expression(tmp_path):
    e = NotebookEngine()
    expr, level = register(e,'x^2+y^2','1/2')
    task, root = run_plot(e,tmp_path,expr,make_spec([{'kind':'contour','expr_id':expr,
                                                'levels':[{'expr_id':level}]}]))
    assert level in task.source_expression_ids
    assert json.loads((root/'plot.json').read_text())['diagnostics'] == []


def test_rejects_complex_values_instead_of_silently_taking_real_part(tmp_path):
    e = NotebookEngine()
    expr, = register(e,'%i*x')
    e.save(tmp_path/'complex.json')
    task = e.fork_numeric_task(expr, spec=make_spec([{'kind':'line','expr_id':expr}],False))
    result = e.run_numeric_task(task.task_id)
    assert not result.ok and ('complex' in result.stderr or '複素' in result.stderr)


@pytest.mark.parametrize('change', ['missing_ref','huge_grid','duplicate_axis','nan_range','equality_region'])
def test_invalid_requests_are_refused(change):
    e = NotebookEngine()
    expr, eq = register(e,'x+y','x+y=1')
    spec = make_spec([{'kind':'contour','expr_id':expr}])
    if change == 'missing_ref': spec.presentation.options['layers'][0]['expr_id'] = 'E999'
    if change == 'huge_grid': spec.inputs[0].spec['points'] = 1000000
    if change == 'duplicate_axis': spec.presentation.options['y'] = 'x'
    if change == 'nan_range': spec.inputs[0].spec['min'] = float('nan')
    if change == 'equality_region': spec.presentation.options['layers'] = [{'kind':'region','constraints':[eq]}]
    with pytest.raises((ValueError, UnknownNotationError)):
        task = e.fork_numeric_task(expr,spec=spec)
        e.generate_numeric_code(task.task_id)


def test_marker_cannot_guess_an_axis_dependent_coordinate():
    e = NotebookEngine()
    expr, = register(e,'x+y')
    task = e.fork_numeric_task(expr,spec=make_spec([
        {'kind':'point','x':{'expr_id':expr},'y':0}]))
    with pytest.raises(ValueError):
        e.generate_numeric_code(task.task_id)


def test_llm_prompt_contains_full_spec_and_plot_sources():
    from notebook.numeric.llm_codegen import build_codegen_prompt
    e = NotebookEngine()
    expr, = register(e,'sin(x)')
    task = e.fork_numeric_task(expr,spec=make_spec([{'kind':'line','expr_id':expr}],False))
    prompt = build_codegen_prompt(task,'')
    assert 'cartesian' in prompt and 'layers' in prompt and 'sin(x)' in prompt


def test_adviser_receives_catalog_and_parser_rejects_multiple_specs():
    from notebook.numeric.spec_adviser import build_spec_prompt, parse_spec, propose_spec
    spec = make_spec([{'kind':'line','expr_id':'E2'}],False)
    catalog = [{'expr_id':'E2','canonical':'sin(x)'}]
    prompt = build_spec_prompt('draw','0',set(),expression_catalog=catalog)
    assert 'E2' in prompt and 'sin(x)' in prompt
    assert parse_spec(spec.model_dump_json() + '\n' + spec.model_dump_json()) is None
    proposal, issues = propose_spec(lambda s,u: spec.model_dump_json(),'draw','0',set(),expression_catalog=catalog)
    assert proposal and not issues


def test_downloads_are_confined_and_survive_timeline_reload(tmp_path):
    from notebook.models.numeric import NumericArtifact
    from ui.numeric_products import product_path, numeric_downloads
    from ui.timeline_serializer import build_timeline_entries
    e = NotebookEngine()
    expr, = register(e,'sin(x)')
    _, root = run_plot(e,tmp_path,expr,make_spec([{'kind':'line','expr_id':expr}],False))
    entries = build_timeline_entries(e)
    result = next(v for v in entries if v['type']=='numeric_result')
    assert {'figure.svg','figure.pdf','layer_01.csv'} <= {v['name'] for v in result['downloads']}
    outside = tmp_path/'private.json'; outside.write_text('{}')
    for name in ('outside.json','link.json'):
        target = outside
        if name == 'link.json':
            target = root/name; target.symlink_to(outside)
        art = NumericArtifact(task_id='N001',artifact_type='log',path=str(target))
        e.numeric_artifacts[art.artifact_id] = art
        with pytest.raises(ValueError): product_path(e,art.artifact_id)
        assert not numeric_downloads(e,[art.artifact_id])


def test_connection_failure_is_distinct_from_invalid_model_reply(monkeypatch):
    import asyncio
    from ui.routes.numeric import numeric_propose_spec, NumericProposeSpecRequest
    import ui.notebook_bridge as bridge
    import ui.interpret_plan as plan
    e = NotebookEngine(); expr, = register(e,'x')
    monkeypatch.setattr(bridge,'get_engine',lambda:e)
    def failed(system,user): return ''
    failed.last_error='connection refused'
    failed.last_reply=''
    monkeypatch.setattr(plan,'_resolve_adviser_chat',lambda:failed)
    response = asyncio.run(numeric_propose_spec(NumericProposeSpecRequest(expr_id=expr,goal='plot')))
    assert response['stage']=='adviser' and 'connection refused' in response['detail']
    failed.last_error=''
    response = asyncio.run(numeric_propose_spec(NumericProposeSpecRequest(expr_id=expr,goal='plot')))
    assert response['stage']=='parse'


def test_axis_named_np_does_not_shadow_numpy_functions(tmp_path):
    e = NotebookEngine(); expr, = register(e,'sin(np)')
    spec = make_spec([{'kind':'line','expr_id':expr}],False)
    spec.inputs[0].name = 'np'; spec.outputs[0].over = ['np']; spec.presentation.options['x'] = 'np'
    _, root = run_plot(e,tmp_path,expr,spec)
    rows = csv_rows(root/'layer_01.csv')
    assert all(abs(float(row['value'])) <= 1 for row in rows)


def test_unused_scenario_parameter_is_not_silently_ignored():
    e = NotebookEngine(); expr, = register(e,'x^2')
    spec = make_spec([{'kind':'line','expr_id':expr,'fixed_values':{'k':2}}],False)
    task = e.fork_numeric_task(expr,spec=spec)
    with pytest.raises(ValueError): e.generate_numeric_code(task.task_id)


def test_plot_intent_never_silently_becomes_a_scalar_evaluation():
    from notebook.numeric.spec_validate import validate_numeric_spec
    e = NotebookEngine(); expr, = register(e,'1')
    spec = NumericSpec.model_validate({'intent':{'kind':'cartesian_plot'},
         'outputs':[{'name':'plot'}], 'presentation':{'kind':'scalar'}})
    assert validate_numeric_spec(spec)
    with pytest.raises(ValueError): e.fork_numeric_task(expr,spec=spec)


def test_empty_sampled_region_is_explained_in_timeline(tmp_path):
    from ui.timeline_serializer import build_timeline_entries
    e = NotebookEngine(); expr, = register(e,'x^2+y^2<0')
    task, root = run_plot(e,tmp_path,expr,make_spec([{'kind':'region','constraints':[expr]}]))
    entries = build_timeline_entries(e)
    result = next(v for v in entries if v['type']=='numeric_result')
    assert 'Layer 1' in result['text'] or '層1' in result['text']
    assert json.loads((root/'plot.json').read_text())['diagnostics'][0]['code']=='empty_sampled_region'
