// ui/notebook_numeric.js — numeric-layer front-end (N-1): the fork / ChatBot /
// graph modals and the numeric timeline-card renderers (R6, docs/27).
//
// Extracted verbatim from notebook_main.js.  The single-closure design is kept
// via dependency injection: initNumeric(deps) receives read accessors for the
// currently selected expression and the shared appendEntry(); it returns the two
// render functions the timeline needs.  All bodies are byte-for-byte the
// original (they were already one indent deep inside the DOMContentLoaded
// closure, matching initNumeric's body).  _t / katex resolve as globals.

import { escapeHtml, prepareLaTeX } from './notebook_utils.js';

export function initNumeric({ getSelectedExprId, getSelectedExprLatex, appendEntry }) {
    // ── Numeric-layer entries (N-1) ──────────────────────────────────────
    function renderNumericForkHtml(entry) {
        return `<span class="note-text">${escapeHtml(entry.text || '')}</span>`;
    }

    function renderNumericResultHtml(entry) {
        const cls = entry.ok ? 'numeric-summary' : 'numeric-summary numeric-failed';
        let html = `<div class="${cls}">${escapeHtml(entry.text || '')}</div>`;
        if (entry.ok && entry.figure_artifact_id) {
            const url = `/api/notebook/numeric/figure/${encodeURIComponent(entry.figure_artifact_id)}`;
            html += `<img class="numeric-fig" src="${url}" alt="${_t('ui.numeric_fig_alt')}">`;
        }
        if (entry.ok && entry.downloads && entry.downloads.length) {
            html += '<div class="numeric-downloads">' + entry.downloads.map(file =>
                `<a href="/api/notebook/numeric/file/${encodeURIComponent(file.artifact_id)}" download>${escapeHtml(file.name)}</a>`
            ).join(' · ') + '</div>';
        }
        return html;
    }

    // ── Numeric-layer fork modal (N-1) ────────────────────────────────────
    const numericModal       = document.getElementById('numeric-modal');
    const numericModalClose  = document.getElementById('numeric-modal-close');
    const numericModalExpr   = document.getElementById('numeric-modal-expr');
    const numericModalError  = document.getElementById('numeric-modal-error');
    const numericTarget      = document.getElementById('numeric-target');
    const numericSweepVar    = document.getElementById('numeric-sweep-var');
    const numericSweepMin    = document.getElementById('numeric-sweep-min');
    const numericSweepMax    = document.getElementById('numeric-sweep-max');
    const numericSweepPoints = document.getElementById('numeric-sweep-points');
    const numericFixedRows   = document.getElementById('numeric-fixed-rows');
    const numericGenerateBtn = document.getElementById('numeric-generate-btn');
    const numericCodeArea    = document.getElementById('numeric-code-area');
    const numericCode        = document.getElementById('numeric-code');
    const numericRunBtn      = document.getElementById('numeric-run-btn');
    const numericTimeout     = document.getElementById('numeric-timeout');
    const numericForkBtn     = document.getElementById('numeric-fork-btn');

    let _numericExprId = null;
    let _numericParams = [];
    let _numericDisplay = {};   // param name -> LaTeX display form (e.g. "omega_0" -> "\omega_{0}")
    let _numericTaskId = null;
    // Remembers the last successful (non-NaN) run's inputs per source
    // expression, so re-opening the modal for the same expression (the
    // "tweak a parameter, look at the graph, tweak again" loop) starts from
    // where the researcher left off instead of blank fields.
    let _lastNumericByExpr = {};
    let _pendingNumericInput = null;

    function _numericError(msg) {
        numericModalError.textContent   = msg || '';
        numericModalError.style.display = msg ? 'block' : 'none';
    }

    // One fixed-value input per free symbol EXCEPT the swept one (which gets a
    // range instead). Rebuilt whenever the sweep variable changes.  If the
    // last successful run for this expression fixed this same param, prefill
    // it — the common workflow is nudging one value and re-running.
    function _rebuildFixedRows() {
        const sweep = numericSweepVar.value;
        const last = _lastNumericByExpr[_numericExprId];
        const lastFixed = (last && last.sweepVar === sweep) ? last.fixedValues : {};
        numericFixedRows.innerHTML = '';
        _numericParams.filter(p => p !== sweep).forEach(p => {
            const row = document.createElement('div');
            row.className = 'numeric-modal-row';
            const lab = document.createElement('label');
            const disp = _numericDisplay[p];
            if (disp && disp !== p) {
                try { katex.render(disp, lab, { displayMode: false, throwOnError: false }); }
                catch (_) { lab.textContent = p; }
            } else {
                lab.textContent = p;
            }
            const inp = document.createElement('input');
            inp.type = 'text'; inp.dataset.param = p; inp.style.flex = '1';
            if (lastFixed[p] !== undefined) inp.value = lastFixed[p];
            row.appendChild(lab); row.appendChild(inp);
            numericFixedRows.appendChild(row);
        });
    }

    function closeNumericModal() { numericModal.style.display = 'none'; }
    numericModalClose.addEventListener('click', closeNumericModal);
    numericModal.addEventListener('click', e => { if (e.target === numericModal) closeNumericModal(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && numericModal.style.display === 'flex') closeNumericModal();
    });
    numericSweepVar.addEventListener('change', _rebuildFixedRows);

    async function openNumericModal() {
        if (!getSelectedExprId()) return;
        _numericExprId = getSelectedExprId();
        _numericTaskId = null;
        _numericDisplay = {};
        _numericError('');
        numericCodeArea.style.display = 'none';
        numericCode.value = '';
        numericSweepVar.innerHTML = '';
        numericFixedRows.innerHTML = '';
        numericGenerateBtn.disabled = true;
        numericModalExpr.innerHTML = '';
        if (getSelectedExprLatex()) {
            try { katex.render(prepareLaTeX(getSelectedExprLatex()), numericModalExpr, { displayMode: true, throwOnError: false }); }
            catch (_) { numericModalExpr.textContent = getSelectedExprLatex(); }
        } else { numericModalExpr.textContent = getSelectedExprId(); }
        // Default target = the equation's LHS if present, else a neutral name.
        let target = '';
        if (getSelectedExprLatex() && getSelectedExprLatex().includes(' = ')) {
            target = getSelectedExprLatex().split(' = ')[0].trim();
        }
        numericTarget.value = target || 'y';
        numericModal.style.display = 'flex';
        // Fetch the free symbols the deterministic template will demand.
        try {
            const data = await fetch(`/api/notebook/numeric/params/${encodeURIComponent(_numericExprId)}`).then(r => r.json());
            if (!data.ok) { _numericError(data.detail || 'error'); return; }
            _numericParams = data.parameters || [];
            _numericDisplay = data.display || {};
            if (_numericParams.length === 0) { _numericError(_t('ui.numeric_no_params')); return; }
            _numericParams.forEach(p => {
                const opt = document.createElement('option'); opt.value = p; opt.textContent = p;
                numericSweepVar.appendChild(opt);
            });
            // Re-run convenience: default the sweep variable and its range to
            // the last successful run's choice for this expression.
            const last = _lastNumericByExpr[_numericExprId];
            if (last && _numericParams.includes(last.sweepVar)) {
                numericSweepVar.value = last.sweepVar;
                numericSweepMin.value = last.sweepMin;
                numericSweepMax.value = last.sweepMax;
                numericSweepPoints.value = last.sweepPoints;
            }
            _rebuildFixedRows();
            numericGenerateBtn.disabled = false;
        } catch (err) { _numericError(String(err)); }
    }

    numericForkBtn.addEventListener('click', openNumericModal);

    function _collectFixedValues() {
        const out = {};
        numericFixedRows.querySelectorAll('input[data-param]').forEach(inp => {
            const v = inp.value.trim();
            if (v !== '') out[inp.dataset.param] = v;
        });
        return out;
    }

    numericGenerateBtn.addEventListener('click', async () => {
        _numericError('');
        const sweepVar = numericSweepVar.value;
        if (!sweepVar) { _numericError(_t('ui.numeric_pick_sweep')); return; }
        const body = {
            expr_id: _numericExprId,
            problem_type: 'parameter_sweep',
            target_quantity: numericTarget.value.trim(),
            sweep_parameters: { [sweepVar]: {
                min: parseFloat(numericSweepMin.value),
                max: parseFloat(numericSweepMax.value),
                points: parseInt(numericSweepPoints.value, 10),
            }},
            fixed_values: _collectFixedValues(),
        };
        _pendingNumericInput = {
            sweepVar,
            sweepMin: numericSweepMin.value,
            sweepMax: numericSweepMax.value,
            sweepPoints: numericSweepPoints.value,
            fixedValues: body.fixed_values,
        };
        numericGenerateBtn.disabled = true;
        const prev = numericGenerateBtn.textContent;
        numericGenerateBtn.textContent = _t('ui.numeric_generating');
        try {
            const fr = await fetch('/api/notebook/numeric/fork', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }).then(r => r.json());
            if (!fr.ok) { _numericError(fr.detail || 'fork failed'); return; }
            _numericTaskId = fr.task.task_id;
            const gr = await fetch(`/api/notebook/numeric/generate/${encodeURIComponent(_numericTaskId)}`, {
                method: 'POST',
            }).then(r => r.json());
            if (!gr.ok) { _numericError(gr.detail || 'generate failed'); return; }
            numericCode.value = gr.code || '';
            numericCodeArea.style.display = 'flex';
        } catch (err) { _numericError(String(err)); }
        finally {
            numericGenerateBtn.disabled = false;
            numericGenerateBtn.textContent = prev;
        }
    });

    numericRunBtn.addEventListener('click', async () => {
        if (!_numericTaskId) return;
        _numericError('');
        numericRunBtn.disabled = true;
        const prev = numericRunBtn.textContent;
        numericRunBtn.textContent = _t('ui.numeric_running');
        try {
            const rr = await fetch(`/api/notebook/numeric/run/${encodeURIComponent(_numericTaskId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ timeout_s: Number(numericTimeout.value) || undefined }),
            }).then(r => r.json());
            if (rr.ok === false && rr.detail) { _numericError(rr.detail); return; }
            // Re-run convenience: only remember this input as the default for
            // NEXT time if the run actually succeeded (not the NaN-flagged
            // case) — a broken parameter combination shouldn't become the
            // prefilled default.
            if (rr.ok && _pendingNumericInput) {
                _lastNumericByExpr[_numericExprId] = _pendingNumericInput;
            }
            // Append the SAME entry shape GET /timeline emits on reload, using
            // the summary text + figure id the run endpoint hands back.
            appendEntry('numeric_result', renderNumericResultHtml({
                text: rr.result_text || '',
                ok: rr.ok,
                figure_artifact_id: rr.figure_artifact_id || null,
                downloads: rr.downloads || [],
            }), {});
            closeNumericModal();
        } catch (err) { _numericError(String(err)); }
        finally {
            numericRunBtn.disabled = false;
            numericRunBtn.textContent = prev;
        }
    });

    // ── Numeric ChatBot modal (docs/25, B-γ propose-then-refine dialogue) ──
    const chatModal           = document.getElementById('numeric-chat-modal');
    const chatClose           = document.getElementById('numeric-chat-close');
    const chatExprEl          = document.getElementById('numeric-chat-expr');
    const chatLog             = document.getElementById('numeric-chat-log');
    const chatError           = document.getElementById('numeric-chat-error');
    const chatGoalRow         = document.getElementById('numeric-chat-goal-row');
    const chatGoalInput       = document.getElementById('numeric-chat-goal');
    const chatProposeBtn      = document.getElementById('numeric-chat-propose-btn');
    const chatRefineRow       = document.getElementById('numeric-chat-refine-row');
    const chatRefineInput     = document.getElementById('numeric-chat-refine');
    const chatRefineBtn       = document.getElementById('numeric-chat-refine-btn');
    const chatApproveBtn      = document.getElementById('numeric-chat-approve-btn');
    const chatCodeArea        = document.getElementById('numeric-chat-code-area');
    const chatCode            = document.getElementById('numeric-chat-code');
    const chatDiffBadge       = document.getElementById('numeric-chat-diffcheck-badge');
    const chatDiffBtn         = document.getElementById('numeric-chat-diffcheck-btn');
    const chatApproveCodeBtn  = document.getElementById('numeric-chat-approve-code-btn');
    const chatRunBtn          = document.getElementById('numeric-chat-run-btn');
    const chatTimeout         = document.getElementById('numeric-chat-timeout');
    const chatBtn             = document.getElementById('numeric-chat-btn');

    let _chatExprId = null;
    let _chatGoal = '';
    let _chatSpec = null;        // latest proposed/approved spec (plain JSON)
    let _chatTaskId = null;
    let _chatRoute = null;       // 'deterministic' | 'llm', from /numeric/fork
    let _chatCodeArtifactId = null;

    function _chatError(msg) {
        chatError.textContent = msg || '';
        chatError.style.display = msg ? 'block' : 'none';
    }

    function _appendChatTurn(cls, el) {
        const row = document.createElement('div');
        row.className = 'chat-turn ' + cls;
        row.appendChild(el);
        chatLog.appendChild(row);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function _appendChatText(cls, text) {
        const div = document.createElement('div');
        div.textContent = text;
        _appendChatTurn(cls, div);
    }

    // Faceted spec -> a readable card (docs/24 facets: intent/inputs/outputs/
    // method). Deliberately plain DOM, not a template string — spec content
    // (goal text, symbol names) comes from the LLM and must not be interpreted
    // as HTML.
    function _renderSpecCard(spec) {
        const card = document.createElement('div');
        card.className = 'spec-card';
        const row = (label, value) => {
            const r = document.createElement('div');
            r.className = 'spec-card-row';
            const l = document.createElement('span'); l.className = 'label'; l.textContent = label;
            const v = document.createElement('span'); v.textContent = value;
            r.appendChild(l); r.appendChild(v);
            return r;
        };
        const intent = spec.intent || {};
        card.appendChild(row(_t('ui.numeric_chat_spec_intent'),
            `${intent.kind || '?'} — ${intent.goal || ''}`));
        if (spec.inputs && spec.inputs.length) {
            const tbl = document.createElement('table');
            const thead = document.createElement('tr');
            [_t('ui.numeric_chat_spec_inputs'), 'role', 'spec'].forEach(h => {
                const th = document.createElement('th'); th.textContent = h; thead.appendChild(th);
            });
            tbl.appendChild(thead);
            spec.inputs.forEach(v => {
                const tr = document.createElement('tr');
                [v.name, v.role, JSON.stringify(v.spec || {})].forEach(c => {
                    const td = document.createElement('td'); td.textContent = c; tr.appendChild(td);
                });
                tbl.appendChild(tr);
            });
            card.appendChild(tbl);
        }
        if (spec.outputs && spec.outputs.length) {
            card.appendChild(row(_t('ui.numeric_chat_spec_outputs'),
                spec.outputs.map(o => `${o.name} (${o.shape}${
                    o.over && o.over.length ? ' over ' + o.over.join(',') : ''})`).join(', ')));
        }
        const method = spec.method || {};
        const params = method.params || {};
        card.appendChild(row(_t('ui.numeric_chat_spec_method'),
            `${method.name || ''}${Object.keys(params).length ? ' ' + JSON.stringify(params) : ''}`));
        const presentation = spec.presentation || {};
        const options = presentation.options || {};
        if (presentation.kind === 'cartesian') {
            card.appendChild(row(_t('ui.numeric_chat_spec_presentation'),
                [options.title, options.x, options.y].filter(Boolean).join(' / ')));
            const coordinate = v => v && v.expr_id ? v.expr_id : String(v);
            (options.layers || []).forEach(layer => {
                const refs = layer.constraints ? layer.constraints.join(' ∩ ') : layer.expr_id
                    || `(${coordinate(layer.x)}, ${coordinate(layer.y)})`;
                card.appendChild(row(_t('ui.numeric_plot_' + layer.kind),
                    [layer.label, refs].filter(Boolean).join(' — ')));
            });
        } else {
            card.appendChild(row(_t('ui.numeric_chat_spec_presentation'), presentation.kind || ''));
        }
        return card;
    }

    function _resetChatState() {
        _chatSpec = null; _chatTaskId = null; _chatRoute = null; _chatCodeArtifactId = null;
        chatLog.innerHTML = '';
        chatGoalInput.value = '';
        chatRefineInput.value = '';
        chatGoalRow.style.display = 'flex';
        chatRefineRow.style.display = 'none';
        chatCodeArea.style.display = 'none';
        chatCode.value = '';
        chatCode.readOnly = false;
        chatRunBtn.disabled = false;
        chatDiffBadge.style.display = 'none';
        chatDiffBtn.style.display = 'none';
        chatApproveCodeBtn.style.display = 'none';
        _chatError('');
    }

    async function openNumericChatModal() {
        if (!getSelectedExprId()) return;
        _chatExprId = getSelectedExprId();
        _resetChatState();
        chatExprEl.innerHTML = '';
        if (getSelectedExprLatex()) {
            try { katex.render(prepareLaTeX(getSelectedExprLatex()), chatExprEl, { displayMode: true, throwOnError: false }); }
            catch (_) { chatExprEl.textContent = getSelectedExprLatex(); }
        } else { chatExprEl.textContent = getSelectedExprId(); }
        chatModal.style.display = 'flex';
    }
    chatBtn.addEventListener('click', openNumericChatModal);

    function closeNumericChatModal() { chatModal.style.display = 'none'; }
    chatClose.addEventListener('click', closeNumericChatModal);
    chatModal.addEventListener('click', e => { if (e.target === chatModal) closeNumericChatModal(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && chatModal.style.display === 'flex') closeNumericChatModal();
    });

    // One B-γ turn: first proposal (no refinement) or a refine turn (prior
    // spec + the researcher's natural-language correction).  A spec with
    // gatekeeper issues is rendered anyway — the researcher refines from it.
    async function _requestSpecProposal(refinement) {
        _chatError('');
        const body = { expr_id: _chatExprId, goal: _chatGoal };
        if (_chatSpec) { body.prior_spec = _chatSpec; body.refinement = refinement || ''; }
        const btn = refinement ? chatRefineBtn : chatProposeBtn;
        const prevText = btn.textContent;
        btn.disabled = true; btn.textContent = _t('ui.numeric_chat_thinking');
        try {
            const r = await fetch('/api/notebook/numeric/propose_spec', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }).then(res => res.json());
            if (!r.ok) { _chatError(r.detail || 'error'); return; }
            _chatSpec = r.spec;
            _appendChatTurn('chat-turn-spec', _renderSpecCard(r.spec));
            if (r.issues && r.issues.length) {
                _appendChatText('chat-turn-issues', r.issues.join(' / '));
            }
            chatGoalRow.style.display = 'none';
            chatRefineRow.style.display = 'flex';
        } catch (err) { _chatError(String(err)); }
        finally { btn.disabled = false; btn.textContent = prevText; }
    }

    chatProposeBtn.addEventListener('click', () => {
        const goal = chatGoalInput.value.trim();
        if (!goal) return;
        _chatGoal = goal;
        _appendChatText('chat-turn-user', goal);
        _requestSpecProposal('');
    });

    chatRefineBtn.addEventListener('click', () => {
        const text = chatRefineInput.value.trim();
        if (!text) return;
        _appendChatText('chat-turn-user', text);
        chatRefineInput.value = '';
        _requestSpecProposal(text);
    });

    // Spec approval: fork with the faceted spec (server re-validates, defense
    // in depth), then dispatch on the server-decided codegen_route — the
    // frontend never re-derives deterministic-vs-LLM routing itself.
    chatApproveBtn.addEventListener('click', async () => {
        if (!_chatSpec) return;
        _chatError('');
        chatApproveBtn.disabled = true;
        const prevText = chatApproveBtn.textContent;
        chatApproveBtn.textContent = _t('ui.numeric_generating');
        try {
            const fr = await fetch('/api/notebook/numeric/fork', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ expr_id: _chatExprId, spec: _chatSpec }),
            }).then(res => res.json());
            if (!fr.ok) { _chatError(fr.detail || 'fork failed'); return; }
            _chatTaskId = fr.task.task_id;
            _chatRoute = fr.codegen_route;
            if (_chatRoute === 'deterministic') {
                const gr = await fetch(`/api/notebook/numeric/generate/${encodeURIComponent(_chatTaskId)}`,
                    { method: 'POST' }).then(res => res.json());
                if (!gr.ok) { _chatError(gr.detail || 'generate failed'); return; }
                chatCode.value = gr.code || '';
                chatCode.readOnly = true;       // deterministic: no review needed (slice 1)
                chatDiffBtn.style.display = 'none';
                chatApproveCodeBtn.style.display = 'none';
                chatRunBtn.disabled = false;
            } else {
                const gr = await fetch(`/api/notebook/numeric/generate_llm/${encodeURIComponent(_chatTaskId)}`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
                }).then(res => res.json());
                if (!gr.ok) { _chatError(gr.detail || 'generate failed'); return; }
                chatCode.value = gr.code || '';
                chatCode.readOnly = false;      // LLM path: editable, must be approved to run
                _chatCodeArtifactId = gr.artifact_id;
                chatDiffBtn.style.display = '';
                chatApproveCodeBtn.style.display = '';
                chatRunBtn.disabled = true;     // server also enforces this (numeric.code_not_reviewed)
            }
            chatRefineRow.style.display = 'none';
            chatCodeArea.style.display = 'flex';
        } catch (err) { _chatError(String(err)); }
        finally { chatApproveBtn.disabled = false; chatApproveBtn.textContent = prevText; }
    });

    // Code approval (LLM path, docs/25 §5): the edited textarea content
    // becomes the approved (and hence runnable) artifact.
    chatApproveCodeBtn.addEventListener('click', async () => {
        if (!_chatCodeArtifactId) return;
        _chatError('');
        chatApproveCodeBtn.disabled = true;
        try {
            const ar = await fetch(`/api/notebook/numeric/approve/${encodeURIComponent(_chatCodeArtifactId)}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ edited_source: chatCode.value }),
            }).then(res => res.json());
            if (!ar.ok) { _chatError(ar.detail || 'approve failed'); return; }
            _chatCodeArtifactId = ar.artifact_id;   // approval creates a NEW artifact
            chatApproveCodeBtn.textContent = _t('ui.numeric_chat_code_approved');
            chatRunBtn.disabled = false;
        } catch (err) { _chatError(String(err)); }
        finally { chatApproveCodeBtn.disabled = false; }
    });

    chatDiffBtn.addEventListener('click', async () => {
        if (!_chatTaskId) return;
        _chatError('');
        chatDiffBtn.disabled = true;
        try {
            const dr = await fetch(`/api/notebook/numeric/diffcheck/${encodeURIComponent(_chatTaskId)}`,
                { method: 'POST' }).then(res => res.json());
            if (!dr.ok) { _chatError(dr.detail || 'diffcheck failed'); return; }
            chatDiffBadge.style.display = 'block';
            chatDiffBadge.className = dr.verdict.match ? 'diffcheck-match' : 'diffcheck-mismatch';
            chatDiffBadge.textContent = dr.verdict.match
                ? _t('ui.numeric_chat_diff_match') : _t('ui.numeric_chat_diff_mismatch');
        } catch (err) { _chatError(String(err)); }
        finally { chatDiffBtn.disabled = false; }
    });

    chatRunBtn.addEventListener('click', async () => {
        if (!_chatTaskId) return;
        _chatError('');
        chatRunBtn.disabled = true;
        const prevText = chatRunBtn.textContent;
        chatRunBtn.textContent = _t('ui.numeric_running');
        try {
            const rr = await fetch(`/api/notebook/numeric/run/${encodeURIComponent(_chatTaskId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ timeout_s: Number(chatTimeout.value) || undefined }),
            }).then(res => res.json());
            if (rr.ok === false && rr.detail) { _chatError(rr.detail); return; }
            appendEntry('numeric_result', renderNumericResultHtml({
                text: rr.result_text || '',
                ok: rr.ok,
                figure_artifact_id: rr.figure_artifact_id || null,
                downloads: rr.downloads || [],
            }), {});
            closeNumericChatModal();
        } catch (err) { _chatError(String(err)); }
        finally { chatRunBtn.disabled = false; chatRunBtn.textContent = prevText; }
    });

    // ── Numeric graph modal (docs/26 §6: the DAG derived from refs) ───────
    // Simple layered layout: expressions and tasks are placed in columns by
    // longest-path depth; edges are drawn as SVG lines with a small label.
    // Plain-DOM/SVG building only — node text comes from the engine.

    const graphBtn = document.getElementById('numeric-graph-btn');
    const graphModal = document.getElementById('numeric-graph-modal');
    const graphClose = document.getElementById('numeric-graph-close');
    const graphWrap = document.getElementById('numeric-graph-svg-wrap');
    const graphEmpty = document.getElementById('numeric-graph-empty');
    const graphLegend = document.getElementById('numeric-graph-legend');

    const _SVG_NS = 'http://www.w3.org/2000/svg';
    const _NODE_W = 150, _NODE_H = 46, _COL_GAP = 90, _ROW_GAP = 22;

    function _graphLayout(graph) {
        // node id -> {kind:'expr'|'task', data}
        const nodes = new Map();
        graph.expressions.forEach(e => nodes.set(e.expr_id, { kind: 'expr', data: e }));
        graph.tasks.forEach(t => nodes.set(t.task_id, { kind: 'task', data: t }));
        // longest-path depth over the DAG
        const preds = new Map();
        graph.edges.forEach(e => {
            if (!preds.has(e.to)) preds.set(e.to, []);
            if (nodes.has(e.from)) preds.get(e.to).push(e.from);
        });
        const depth = new Map();
        function d(id, seen) {
            if (depth.has(id)) return depth.get(id);
            if (seen.has(id)) return 0;            // defensive (DAG by design)
            seen.add(id);
            const ps = preds.get(id) || [];
            const v = ps.length ? Math.max(...ps.map(p => d(p, seen))) + 1 : 0;
            depth.set(id, v);
            return v;
        }
        nodes.forEach((_v, id) => d(id, new Set()));
        // column buckets → coordinates
        const cols = new Map();
        nodes.forEach((_v, id) => {
            const c = depth.get(id) || 0;
            if (!cols.has(c)) cols.set(c, []);
            cols.get(c).push(id);
        });
        const pos = new Map();
        [...cols.keys()].sort((a, b) => a - b).forEach(c => {
            cols.get(c).sort().forEach((id, row) => {
                pos.set(id, { x: 16 + c * (_NODE_W + _COL_GAP),
                              y: 16 + row * (_NODE_H + _ROW_GAP) });
            });
        });
        const width = 32 + (Math.max(...depth.values(), 0) + 1) * (_NODE_W + _COL_GAP) - _COL_GAP;
        const rows = Math.max(...[...cols.values()].map(a => a.length), 1);
        const height = 32 + rows * (_NODE_H + _ROW_GAP) - _ROW_GAP;
        return { nodes, pos, width, height };
    }

    function _svgEl(tag, attrs) {
        const el = document.createElementNS(_SVG_NS, tag);
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
        return el;
    }

    function _renderGraph(graph) {
        graphWrap.textContent = '';
        graphLegend.textContent = '';
        if (!graph.tasks.length) {
            graphEmpty.style.display = 'block';
            return;
        }
        graphEmpty.style.display = 'none';
        const { nodes, pos, width, height } = _graphLayout(graph);
        const svg = _svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}` });

        graph.edges.forEach(e => {
            const a = pos.get(e.from), b = pos.get(e.to);
            if (!a || !b) return;
            const x1 = a.x + _NODE_W, y1 = a.y + _NODE_H / 2;
            const x2 = b.x, y2 = b.y + _NODE_H / 2;
            const color = e.kind === 'value' ? '#7c3aed'
                        : e.kind === 'reflux' ? '#0f766e' : '#94a3b8';
            svg.appendChild(_svgEl('line', {
                x1, y1, x2, y2, stroke: color, 'stroke-width': 1.5,
                'stroke-dasharray': e.kind === 'reflux' ? '5,3' : 'none',
            }));
            const label = e.kind === 'value'
                ? `${e.name || ''}${e.value_kind ? ` (${e.value_kind})` : ''}`
                : e.kind === 'reflux' ? (e.name ? `= ${e.name}` : '') : '';
            if (label) {
                const txt = _svgEl('text', {
                    x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 4,
                    'font-size': '10', fill: color, 'text-anchor': 'middle',
                });
                txt.textContent = label;
                svg.appendChild(txt);
            }
        });

        nodes.forEach((n, id) => {
            const p = pos.get(id);
            if (!p) return;
            const g = _svgEl('g', {});
            const isTask = n.kind === 'task';
            const stale = isTask && n.data.stale && n.data.stale.stale;
            const failed = isTask && n.data.status === 'failed';
            const fill = !isTask ? '#f1f5f9'
                       : failed ? '#fee2e2'
                       : n.data.status === 'executed' ? '#ecfdf5' : '#eff6ff';
            const stroke = stale ? '#d97706' : (!isTask ? '#94a3b8' : '#64748b');
            g.appendChild(_svgEl('rect', {
                x: p.x, y: p.y, width: _NODE_W, height: _NODE_H, rx: 6,
                fill, stroke, 'stroke-width': stale ? 2.5 : 1.2,
            }));
            const l1 = _svgEl('text', { x: p.x + 8, y: p.y + 18, 'font-size': '12',
                                        'font-weight': 'bold', fill: '#1e293b' });
            l1.textContent = isTask ? `${id} · ${n.data.kind}` : id;
            const l2 = _svgEl('text', { x: p.x + 8, y: p.y + 35, 'font-size': '10',
                                        fill: '#475569' });
            l2.textContent = isTask
                ? `${n.data.status}${stale ? ' · ' + _t('ui.numeric_graph_stale') : ''}`
                : (n.data.missing ? _t('ui.numeric_graph_missing')
                                  : (n.data.latex || '').slice(0, 24));
            g.appendChild(l1); g.appendChild(l2);
            svg.appendChild(g);
        });
        graphWrap.appendChild(svg);
        graphLegend.textContent = _t('ui.numeric_graph_legend');
    }

    const branchCompare = document.getElementById('numeric-branch-compare');
    const branchCompareBody = document.getElementById('numeric-branch-compare-body');

    function _renderBranchComparison(cmp) {
        branchCompareBody.textContent = '';
        const groups = (cmp && cmp.groups || []).filter(g => g.comparable);
        if (!groups.length) {
            branchCompare.style.display = 'none';
            return;
        }
        branchCompare.style.display = 'block';
        // columns = the branches that actually appear in comparable groups
        const branchIds = [...new Set(groups.flatMap(g => Object.keys(g.branches)))].sort();
        const table = document.createElement('table');
        table.style.cssText = 'border-collapse:collapse; font-size:.82em; min-width:100%;';
        const head = table.insertRow();
        const corner = head.insertCell();
        corner.textContent = _t('ui.numeric_branch_compare_group');
        branchIds.forEach(b => { const c = head.insertCell(); c.textContent = b; });
        [...head.cells].forEach(c => {
            c.style.cssText = 'border:1px solid #e2e8f0; padding:6px 10px; background:#f8fafc; font-weight:bold; text-align:left;';
        });
        groups.forEach(g => {
            const row = table.insertRow();
            const label = row.insertCell();
            label.textContent = `${g.kind}${g.target ? ' · ' + g.target : ''}`;
            branchIds.forEach(b => {
                const cell = row.insertCell();
                (g.branches[b] || []).forEach(entry => {
                    const box = document.createElement('div');
                    box.style.cssText = 'margin-bottom:6px;';
                    const line = document.createElement('div');
                    line.textContent = `${entry.task_id} · ${entry.status}`;
                    line.style.fontWeight = 'bold';
                    box.appendChild(line);
                    if (entry.summary) {
                        const s = document.createElement('div');
                        s.textContent = entry.summary;
                        s.style.cssText = 'color:#475569;';
                        box.appendChild(s);
                    }
                    if (entry.figure_artifact_id) {
                        const img = document.createElement('img');
                        img.src = `/api/notebook/numeric/figure/${encodeURIComponent(entry.figure_artifact_id)}`;
                        img.style.cssText = 'max-width:180px; display:block; margin-top:4px; border:1px solid #e2e8f0; border-radius:4px;';
                        box.appendChild(img);
                    }
                    cell.appendChild(box);
                });
            });
            [...row.cells].forEach(c => {
                c.style.cssText += 'border:1px solid #e2e8f0; padding:6px 10px; vertical-align:top;';
            });
        });
        branchCompareBody.appendChild(table);
    }

    async function openNumericGraphModal() {
        graphModal.style.display = 'flex';
        graphEmpty.style.display = 'none';
        branchCompare.style.display = 'none';
        graphWrap.textContent = '…';
        try {
            const r = await fetch('/api/notebook/numeric/graph').then(res => res.json());
            _renderGraph(r.graph || { expressions: [], tasks: [], edges: [] });
            _renderBranchComparison(r.branch_comparison);
        } catch (err) {
            graphWrap.textContent = String(err);
        }
    }

    if (graphBtn) graphBtn.addEventListener('click', openNumericGraphModal);
    if (graphClose) graphClose.addEventListener('click',
        () => { graphModal.style.display = 'none'; });
    if (graphModal) graphModal.addEventListener('click',
        (e) => { if (e.target === graphModal) graphModal.style.display = 'none'; });

    return { renderNumericForkHtml, renderNumericResultHtml };
}
