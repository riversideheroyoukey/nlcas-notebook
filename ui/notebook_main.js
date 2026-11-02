// ui/notebook_main.js — Main notebook screen logic
//
// R6 (docs/27): this is now an ES module (loaded via <script type="module">).
// Pure presentation helpers live in notebook_utils.js; feature families are being
// split into sibling modules incrementally.

import { escapeHtml, renderKatex } from './notebook_utils.js';
import { initNumeric } from './notebook_numeric.js';
import { initHandoff } from './notebook_handoff.js';
import {
    renderCasEntryHtml, renderUserEntryHtml,
    renderNoteHtml, renderAssumptionHtml, renderDepConfirmationHtml,
    renderConfirmationHtml,
} from './notebook_timeline.js';
import { initInput } from './notebook_input.js';
import { initAdvise } from './notebook_advise.js';
import { initPersistence } from './notebook_persistence.js';
import { initBranch } from './notebook_branch.js';
import { initPlan } from './notebook_plan.js';

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ─────────────────────────────────────────────────────────
    const timeline      = document.getElementById('timeline');
    const inputBox      = document.getElementById('input-box');
    const submitBtn     = document.getElementById('submit-btn');
    const stopBtn       = document.getElementById('cancel-btn');  // LLM processing stop button
    const hwIframe      = document.getElementById('hw-iframe');
    const opButtons     = document.getElementById('op-buttons');

    // ── State ─────────────────────────────────────────────────────────────
    let selectedExprId    = null;   // currently selected CAS expression in timeline
    let selectedExprLatex = null;   // latex of the selected expression
    let _currentAbortCtrl = null;   // AbortController for the in-flight /interpret request

    // ── Utilities ─────────────────────────────────────────────────────────
    // The pure presentation helpers live in notebook_utils.js (R6, docs/27);
    // main imports only escapeHtml + renderKatex — the rest are used solely by
    // the extracted sibling modules (input / numeric / timeline / advise).

    // ── Timeline ──────────────────────────────────────────────────────────

    const LABELS = { user: 'You', cas: 'CAS', supervisor: 'Notebook' };

    function appendEntry(type, htmlContent, { exprId = null, stepId = null, latex = null } = {}) {
        const entry = document.createElement('div');
        entry.className = `timeline-entry entry-${type}`;
        if (exprId)  entry.dataset.exprId  = exprId;
        if (stepId)  entry.dataset.stepId  = stepId;
        if (latex)   entry.dataset.latex   = latex;

        const label = document.createElement('div');
        label.className = 'entry-label';
        label.textContent = (type === 'note' || type === 'assumption'
                             || type === 'numeric_fork' || type === 'numeric_result'
                             || type === 'plan')
            ? _t(`ui.label_${type}`)
            : (LABELS[type] ?? type);
        entry.appendChild(label);

        const body = document.createElement('div');
        body.innerHTML = htmlContent;
        entry.appendChild(body);

        // KaTeX spans already rendered; but re-render any data-latex markers
        body.querySelectorAll('[data-latex-render]').forEach(el => {
            const lat = el.getAttribute('data-latex-render');
            try { katex.render(lat, el, { displayMode: el.dataset.display === '1', throwOnError: false }); }
            catch (_) { el.textContent = lat; }
        });

        // YOU card: add remove button
        if (type === 'user') {
            const removeBtn = document.createElement('button');
            removeBtn.className = 'user-remove-btn';
            removeBtn.title = _t('ui.user_remove_title');
            removeBtn.textContent = '✕';
            label.appendChild(removeBtn);
        }

        timeline.appendChild(entry);
        entry.scrollIntoView({ behavior: 'smooth', block: 'end' });

        if (type === 'cas') {
            entry.addEventListener('click', () => selectExpression(entry));
            entry.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                e.stopPropagation();   // prevent document contextmenu listener from closing the menu
                showCasContextMenu(entry, e.clientX, e.clientY);
            });
            attachForkButton(entry);
            _branch.attachLiveForkButton(entry);
        } else if (type === 'plan') {
            _plan.attachRetryButton(entry);
            _plan.attachReplanButton(entry);
        }
        return entry;
    }

    // ── Timeline card renderers (renderCasEntryHtml / renderUserEntryHtml /
    // normalizeMathDelimiters / renderNoteHtml / renderAssumptionHtml / the
    // confirmation renderers) live in ui/notebook_timeline.js (R6, docs/27);
    // appendEntry() above stays here as the DOM/state hub.

    // ── Numeric-layer entries (N-1) — the numeric card renderers live in
    // ui/notebook_numeric.js (R6); reached via _numeric.renderNumeric*Html below.

    // (confirmation-card renderers live in notebook_timeline.js — R6)

    // ── Expression selection ──────────────────────────────────────────────

    function selectExpression(entry, { insertId = true } = {}) {
        document.querySelectorAll('.entry-cas.selected')
            .forEach(e => e.classList.remove('selected'));
        entry.classList.add('selected');
        selectedExprId    = entry.dataset.exprId ?? null;
        selectedExprLatex = entry.dataset.latex  ?? null;
        opButtons.style.display = selectedExprId ? 'flex' : 'none';
        updateCExtButtons();

        // If input box has content, append the expr ID as plain text at the end.
        // Suppressed when called from the context menu (insertId=false) to avoid
        // double-insertion when the user then picks an operation from the menu.
        if (insertId && selectedExprId && !_input.isInputEmpty()) {
            appendExprIdToInput(selectedExprId);
        }
    }

    function deselectAll() {
        document.querySelectorAll('.entry-cas.selected')
            .forEach(e => e.classList.remove('selected'));
        selectedExprId    = null;
        selectedExprLatex = null;
        opButtons.style.display = 'none';
    }

    timeline.addEventListener('click', (e) => {
        if (!e.target.closest('.entry-cas')) deselectAll();
    });

    const _input = initInput({ sendDomainsToIframe: _sendDomainsToIframe });

    // ── Manual note ───────────────────────────────────────────────────────

    document.getElementById('note-btn')?.addEventListener('click', async () => {
        const text = inputBox.innerText.trim();
        if (!text) return;
        _input.clearInput();
        try {
            const res  = await fetch('/api/notebook/note', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text }),
            });
            const data = await res.json();
            appendEntry('note', renderNoteHtml(text), { stepId: data.step_id });
        } catch (e) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(e))}`);
        }
    });

    // ── Processing state helpers ──────────────────────────────────────────

    function _beginProcessing() {
        submitBtn.disabled = true;
        submitBtn.textContent = _t('ui.processing');
        stopBtn.style.display = 'inline-block';
    }

    function _endProcessing() {
        submitBtn.disabled = false;
        submitBtn.textContent = _t('ui.submit');
        stopBtn.style.display = 'none';
        _currentAbortCtrl = null;
    }

    stopBtn.addEventListener('click', () => {
        if (_currentAbortCtrl) {
            _currentAbortCtrl.abort();
        }
    });

    // ── Submit & interpret ────────────────────────────────────────────────

    async function handleSubmit() {
        const tokens = _input.serializeInput();
        if (_input.isInputEmpty()) return;

        // Capture selected expression BEFORE deselectAll clears it
        const activeExprId = selectedExprId;

        // Show user's input in timeline immediately
        const userHtml = renderUserEntryHtml(tokens);
        const userEntry = appendEntry('user', userHtml);
        _input.clearInput();
        deselectAll();

        _beginProcessing();

        try {
            const body = { tokens };
            if (activeExprId) body.selected_expr_id = activeExprId;
            _currentAbortCtrl = new AbortController();
            const res = await fetch('/api/notebook/interpret', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(body),
                signal:  _currentAbortCtrl.signal,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                appendEntry('supervisor', `❌ ${_t('ui.server_error')}: ${escapeHtml(err.detail ?? '')}`);
                return;
            }
            const data = await res.json();
            handleInterpretResult(data);
        } catch (e) {
            if (e.name === 'AbortError') {
                userEntry.remove();   // remove the YOU card on cancel
            } else {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(e))}`);
            }
        } finally {
            _endProcessing();
        }
    }

    function handleInterpretResult(data) {
        // Show any newly registered input expressions first
        for (const expr of data.registered_expressions ?? []) {
            appendEntry('cas', renderCasEntryHtml({ ...expr, deletable: true }), { exprId: expr.expr_id, stepId: expr.step_id, latex: expr.latex });
        }

        if (data.operation === 'execute_plan') {
            // C-Ext-5 fixed workflow: per-step checklist instead of only the
            // last step's expression (that expression still exists as its
            // own cas_operation entry and appears on reload).
            appendEntry('plan', _plan.renderPlanResultHtml(data), { stepId: data.step_id });
            triggerAutosave();
        } else if (data.status === 'completed') {
            // Don't duplicate if the result was already one of the registered exprs
            const alreadyShown = (data.registered_expressions ?? [])
                .some(e => e.expr_id === data.expr_id);
            if (!alreadyShown) {
                appendEntry('cas', renderCasEntryHtml({ ...data, deletable: true }),
                    { exprId: data.expr_id, stepId: data.step_id, latex: data.latex });
            }
            triggerAutosave();
        } else if (data.status === 'confirmed') {
            // Side-effect operation (e.g. define_scalar_parameter, define_fermion_operator).
            // When a rendered symbol is provided, format as:
            //   ✓ [rendered ∆] (Delta) をスカラーパラメータとして登録しました
            // Otherwise fall back to a plain text confirmation line.
            const confirmDesc = escapeHtml(data.message ?? _t('ui.op_completed'));
            let confirmLine;
            const removeBtn = data.step_id
                ? `<button class="define-remove-btn" data-step-id="${escapeHtml(data.step_id)}" title="${_t('ui.define_remove_title')}">✕</button>`
                : '';
            if (data.latex) {
                const renderedSym  = renderKatex(data.latex, false).outerHTML;
                const parenPart    = data.symbol_display
                    ? ` <span class="void-op-plain-name">(${escapeHtml(data.symbol_display)})</span>`
                    : '';
                confirmLine = `✓ ${renderedSym}${parenPart} ${confirmDesc}${removeBtn ? ' ' + removeBtn : ''}`;
            } else {
                confirmLine = confirmDesc + (removeBtn ? ' ' + removeBtn : '');
            }
            appendEntry('supervisor', confirmLine, { stepId: data.step_id });
            // Tag the most recent YOU card so cancellation can grey it out.
            if (data.step_id) {
                const userEntries = timeline.querySelectorAll('.entry-user');
                if (userEntries.length > 0) {
                    userEntries[userEntries.length - 1].dataset.defineStepId = data.step_id;
                }
            }
            // Re-render existing expression tiles when notation switches.
            if (data.operation === 'set_notation' && data.updated_exprs?.length) {
                data.updated_exprs.forEach(({ expr_id, latex }) => {
                    const entry = timeline.querySelector(`[data-expr-id="${CSS.escape(expr_id)}"]`);
                    if (!entry) return;
                    entry.dataset.latex = latex;
                    const mathEl = entry.querySelector('.cas-math');
                    if (mathEl) mathEl.replaceWith(renderKatex(latex, true));
                });
                triggerAutosave();
            }
        } else if (data.status === 'assumption_added') {
            appendEntry('assumption', renderAssumptionHtml(data.assumption_id, data.condition));
        } else if (data.status === 'assumption_removed') {
            // Remove the matching assumption entry from the timeline
            const el = timeline.querySelector(`[data-assumption-id="${CSS.escape(data.condition)}"]`);
            el?.closest('.timeline-entry')?.remove();
        } else if (data.status === 'registered') {
            // Input-only (no operation): expressions already shown above
        } else if (data.status === 'needs_confirmation') {
            // Attach the original supervisor_text for ambiguity_resolution recording
            data._user_text = data._user_text ?? '';
            appendEntry('supervisor', renderConfirmationHtml(data));
        } else if (data.status === 'needs_dep_confirmation') {
            appendEntry('supervisor', renderDepConfirmationHtml(data));
        } else {
            let errHtml = `❌ ${escapeHtml(data.error ?? _t('ui.unknown_error'))}`;
            const undef = data.undefined_symbols;
            if (undef && undef.length > 0) {
                const syms = undef.map(s => `<code>${escapeHtml(s)}</code>`).join(', ');
                const hint = _t('ui.undefined_symbols_hint').replace('{symbol}', escapeHtml(undef[0]));
                errHtml += `<br><div class="undefined-symbols-warning">` +
                    `⚠ <strong>${_t('ui.undefined_symbols_label')}:</strong> ${syms}` +
                    `<br><span class="undefined-hint">${escapeHtml(hint)}</span>` +
                    `</div>`;
            }
            appendEntry('supervisor', errHtml);
        }
    }

    // ── YOU card removal (event delegation on timeline) ──────────────────

    timeline.addEventListener('click', async (e) => {
        const removeBtn = e.target.closest('.user-remove-btn');
        if (!removeBtn) return;
        e.stopPropagation();
        const entry  = removeBtn.closest('.timeline-entry');
        const stepId = entry?.dataset.stepId;
        if (!stepId) {
            // No step_id: frontend-only card (e.g. cancelled input) — just remove
            entry?.remove();
            return;
        }
        try {
            const res = await fetch(`/api/notebook/step/${encodeURIComponent(stepId)}`, { method: 'DELETE' });
            if (res.ok) {
                entry.remove();
            } else {
                const err = await res.json().catch(() => ({}));
                appendEntry('supervisor', `❌ ${_t('ui.user_remove_title')}: ${escapeHtml(err.detail ?? '')}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    });

    // ── Note removal (event delegation on timeline) ──────────────────────

    timeline.addEventListener('click', async (e) => {
        const noteRemoveBtn = e.target.closest('.note-remove-btn');
        if (noteRemoveBtn) {
            const entry  = noteRemoveBtn.closest('.timeline-entry');
            const stepId = entry?.dataset.stepId;
            if (!stepId) return;
            try {
                const res = await fetch(`/api/notebook/note/${encodeURIComponent(stepId)}`, { method: 'DELETE' });
                if (res.ok) {
                    entry.remove();
                } else {
                    const err = await res.json().catch(() => ({}));
                    appendEntry('supervisor', `❌ ${_t('ui.note_remove_title')}: ${escapeHtml(err.detail ?? '')}`);
                }
            } catch (err) {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
            }
            return;
        }
    });

    // ── Define-step removal (event delegation on timeline) ───────────────

    timeline.addEventListener('click', async (e) => {
        const removeBtn = e.target.closest('.define-remove-btn');
        if (!removeBtn) return;
        const stepId = removeBtn.dataset.stepId;
        const entry  = removeBtn.closest('.timeline-entry');
        try {
            const res = await fetch(`/api/notebook/define/${encodeURIComponent(stepId)}`, { method: 'DELETE' });
            if (res.ok) {
                entry?.remove();
                // Grey out the YOU card that triggered this define operation.
                const userEntry = timeline.querySelector(`.entry-user[data-define-step-id="${CSS.escape(stepId)}"]`);
                if (userEntry) userEntry.classList.add('entry-cancelled');
            } else {
                const err = await res.json().catch(() => ({}));
                appendEntry('supervisor', `❌ ${_t('ui.define_remove_title')}: ${escapeHtml(err.detail ?? '')}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    });

    // ── Assumption removal (event delegation on timeline) ────────────────

    timeline.addEventListener('click', async (e) => {
        const removeBtn = e.target.closest('.assumption-remove-btn');
        if (!removeBtn) return;
        const assumptionId = removeBtn.dataset.assumptionId;
        try {
            const res = await fetch(`/api/notebook/assumption/${encodeURIComponent(assumptionId)}`, { method: 'DELETE' });
            if (res.ok) {
                removeBtn.closest('.timeline-entry')?.remove();
            } else {
                const err = await res.json().catch(() => ({}));
                appendEntry('supervisor', `❌ ${_t('ui.assumption_remove_title')}: ${escapeHtml(err.detail ?? '')}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    });

    // ── Expression removal (event delegation on timeline) ────────────────

    timeline.addEventListener('click', async (e) => {
        const removeBtn = e.target.closest('.cas-remove-btn');
        if (!removeBtn) return;
        const exprId = removeBtn.dataset.exprId;
        const entry  = removeBtn.closest('.timeline-entry');
        try {
            const res = await fetch(`/api/notebook/expression/${encodeURIComponent(exprId)}`, { method: 'DELETE' });
            if (res.ok) {
                if (selectedExprId === exprId) deselectAll();
                entry?.remove();
            } else {
                const err = await res.json().catch(() => ({}));
                appendEntry('supervisor', `❌ ${_t('ui.cas_remove_title')}: ${escapeHtml(err.detail ?? '')}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    });

    // ── Confirmation buttons (event delegation on timeline) ───────────────

    timeline.addEventListener('click', async (e) => {
        const confirmBtn = e.target.closest('.confirm-btn');
        const cancelBtn  = e.target.closest('.cancel-btn');

        if (cancelBtn) {
            cancelBtn.closest('.timeline-entry')?.remove();
            return;
        }

        const depConfirmBtn = e.target.closest('.dep-confirm-btn');
        if (depConfirmBtn) {
            const choice   = depConfirmBtn.dataset.choice;
            const op       = depConfirmBtn.dataset.op;
            const args     = JSON.parse(depConfirmBtn.dataset.args ?? '{}');
            const depVars  = JSON.parse(depConfirmBtn.dataset.depVars ?? '[]');
            depConfirmBtn.closest('.timeline-entry')?.remove();
            try {
                const body = {
                    operation: op,
                    args,
                    dep_confirmed: true,
                    remove_deps_for: choice === 'without' ? depVars : [],
                };
                const res  = await fetch('/api/notebook/operation', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(body),
                });
                const data = await res.json();
                if (data.ok) {
                    appendEntry('cas',
                        renderCasEntryHtml({ expr_id: data.expr_id, latex: data.latex, canonical: data.canonical, operation_label: data.operation_label, deletable: true }),
                        { exprId: data.expr_id, stepId: data.step_id, latex: data.latex }
                    );
                    triggerAutosave();
                } else {
                    appendEntry('supervisor', `❌ ${escapeHtml(data.detail ?? 'Error')}`);
                }
            } catch (err) {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
            }
            return;
        }

        if (!confirmBtn) return;

        const op              = confirmBtn.dataset.op;
        const args            = JSON.parse(confirmBtn.dataset.args ?? '{}');
        const userText        = JSON.parse(confirmBtn.dataset.userText ?? '""');
        const candidates      = JSON.parse(confirmBtn.dataset.candidates ?? '[]');
        const pendingDeps     = JSON.parse(confirmBtn.dataset.pendingDeps ?? '[]');
        confirmBtn.closest('.timeline-entry')?.remove();

        try {
            const res = await fetch('/api/notebook/operation', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ operation: op, args, user_text: userText, candidates, pending_dependencies: pendingDeps }),
            });
            const data = await res.json();
            if (data.needs_dep_confirmation) {
                appendEntry('supervisor', renderDepConfirmationHtml({
                    status:    'needs_dep_confirmation',
                    dep_vars:  data.dep_vars,
                    operation: data.operation,
                    args:      data.args,
                    diff_var:  data.diff_var,
                }));
            } else if (data.operation === 'execute_plan_candidate') {
                // C-Ext-5 candidate/free-form plan: per-step checklist
                // regardless of overall success/failure (a stopped plan's
                // partial results are meaningful, not just an error).
                appendEntry('plan', _plan.renderPlanResultHtml(data), { stepId: data.plan_id });
                triggerAutosave();
            } else if (data.operation === 'numeric_fork_nl') {
                // N-1 NL trigger: same numeric_result card the modal path and
                // GET /timeline reload produce.  A hard ValueError (bad target,
                // assumption violation) never reaches run_numeric_task, so
                // `text` is absent — fall back to a plain error message.
                if (data.text !== undefined) {
                    appendEntry('numeric_result',
                        _numeric.renderNumericResultHtml({ text: data.text, ok: data.ok, figure_artifact_id: data.figure_artifact_id }),
                        { stepId: data.step_id });
                } else {
                    appendEntry('supervisor', `❌ ${escapeHtml(data.detail ?? 'Error')}`);
                }
                triggerAutosave();
            } else if (data.ok && data.status === 'confirmed') {
                // Side-effect op (set_notation, declare_time_dependent, etc.)
                appendEntry('supervisor', escapeHtml(data.message ?? _t('ui.op_completed')));
                if (data.operation === 'set_notation' && data.updated_exprs?.length) {
                    data.updated_exprs.forEach(({ expr_id, latex }) => {
                        const entry = timeline.querySelector(`[data-expr-id="${CSS.escape(expr_id)}"]`);
                        if (!entry) return;
                        entry.dataset.latex = latex;
                        const mathEl = entry.querySelector('.cas-math');
                        if (mathEl) mathEl.replaceWith(renderKatex(latex, true));
                    });
                }
                triggerAutosave();
            } else if (data.ok && data.status === 'assumption_added') {
                appendEntry('assumption', renderAssumptionHtml(data.assumption_id, data.condition));
                triggerAutosave();
            } else if (data.ok && data.status === 'assumption_removed') {
                const el = timeline.querySelector(`[data-assumption-id="${CSS.escape(data.condition)}"]`);
                el?.closest('.timeline-entry')?.remove();
                triggerAutosave();
            } else if (data.ok) {
                appendEntry('cas',
                    renderCasEntryHtml({ expr_id: data.expr_id, latex: data.latex, canonical: data.canonical, operation_label: data.operation_label, deletable: true }),
                    { exprId: data.expr_id, stepId: data.step_id, latex: data.latex }
                );
                triggerAutosave();
            } else {
                appendEntry('supervisor', `❌ ${escapeHtml(data.detail ?? 'Error')}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    });

    // ── Operation buttons ─────────────────────────────────────────────────

    // Operations that can run without extra parameters (only need an expr)
    const INSTANT_OPS = new Set(['expand', 'factor', 'ratsimp']);

    document.querySelectorAll('.op-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!selectedExprId) return;
            const op       = btn.dataset.op;
            const opLabel  = _t(`ui.op_${op}`) || op;
            const needsParam = btn.dataset.needsParam === 'true';

            if (!needsParam && INSTANT_OPS.has(op)) {
                // Auto-submit: build token list and submit directly
                const instructText = _t('ui.op_instruction', {expr_id: selectedExprId, label: opLabel});
                const syntheticTokens = [{ type: 'text', value: instructText }];
                const userHtml = renderUserEntryHtml(syntheticTokens);
                const _autoUserEntry = appendEntry('user', userHtml);
                deselectAll();
                _beginProcessing();
                _currentAbortCtrl = new AbortController();
                fetch('/api/notebook/interpret', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ tokens: syntheticTokens }),
                    signal:  _currentAbortCtrl.signal,
                })
                    .then(r => r.json())
                    .then(handleInterpretResult)
                    .catch(err => {
                        if (err.name === 'AbortError') {
                            _autoUserEntry.remove();
                        } else {
                            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
                        }
                    })
                    .finally(_endProcessing);
            } else {
                // Insert hint text into input; user fills in the parameters
                inputBox.focus();
                const hint = _t('ui.op_instruction', {expr_id: selectedExprId, label: opLabel});
                document.execCommand('insertText', false, hint);
            }
        });
    });

    // ── CAS card context menu (right-click) ───────────────────────────────

    let _ctxMenu = null;

    function closeCasContextMenu() {
        if (_ctxMenu) { _ctxMenu.remove(); _ctxMenu = null; }
    }

    document.addEventListener('click',       closeCasContextMenu);
    document.addEventListener('contextmenu', closeCasContextMenu);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeCasContextMenu(); });

    /** Append exprId as plain text to inputBox end (shared by menu item and card click). */
    function appendExprIdToInput(exprId) {
        // Remove stray <br> that browsers insert into empty contenteditable divs.
        if (_input.isInputEmpty()) inputBox.innerHTML = '';
        const text = inputBox.textContent ?? '';
        const sep  = text.length > 0 && !text.endsWith(' ') ? ' ' : '';
        inputBox.appendChild(document.createTextNode(sep + exprId));
        const range = document.createRange();
        range.selectNodeContents(inputBox);
        range.collapse(false);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        inputBox.focus();
    }

    function showCasContextMenu(entry, x, y) {
        closeCasContextMenu();
        selectExpression(entry, { insertId: false });  // don't insert ID; menu items handle it
        if (!selectedExprId) return;
        const exprId = selectedExprId;

        const menu = document.createElement('div');
        menu.id = 'cas-context-menu';
        menu.className = 'cas-context-menu';

        // ── 1st item: send expr ID to input box ───────────────────────────
        const sendItem = document.createElement('button');
        sendItem.textContent = _t('ui.ctx_send_to_input');
        sendItem.addEventListener('click', (e) => {
            e.stopPropagation();
            closeCasContextMenu();
            appendExprIdToInput(exprId);
        });
        menu.appendChild(sendItem);

        // ── Separator ─────────────────────────────────────────────────────
        const sep = document.createElement('hr');
        sep.className = 'cas-context-menu-sep';
        menu.appendChild(sep);

        // ── Operation items: always insert hint text (never auto-execute) ─
        document.querySelectorAll('#op-buttons .op-btn, #op-buttons .op-btn-advise').forEach(btn => {
            if (btn.style.display === 'none') return;
            const item = document.createElement('button');
            item.textContent = btn.textContent.trim();
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                closeCasContextMenu();
                // Always insert hint text into input box regardless of INSTANT_OPS
                const op      = btn.dataset.op ?? '';
                const opLabel = btn.textContent.trim();
                const hint    = op
                    ? _t('ui.op_instruction', {expr_id: exprId, label: opLabel})
                    : opLabel;  // advise-btn has no data-op
                if (op === '' || btn.id === 'advise-btn') {
                    // For advise, just trigger it directly (it opens a modal)
                    btn.click();
                } else {
                    // Append at end (cursor is lost after menu interaction)
                    if (_input.isInputEmpty()) inputBox.innerHTML = '';
                    const cur = inputBox.textContent ?? '';
                    const sep = cur.length > 0 && !cur.endsWith(' ') ? ' ' : '';
                    inputBox.appendChild(document.createTextNode(sep + hint));
                    const range = document.createRange();
                    range.selectNodeContents(inputBox);
                    range.collapse(false);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                    inputBox.focus();
                }
            });
            menu.appendChild(item);
        });

        menu.style.left = `${x}px`;
        menu.style.top  = `${y}px`;
        document.body.appendChild(menu);
        _ctxMenu = menu;

        // Adjust if menu goes off-screen
        const rect = menu.getBoundingClientRect();
        if (rect.right  > window.innerWidth)  menu.style.left = `${x - rect.width}px`;
        if (rect.bottom > window.innerHeight) menu.style.top  = `${y - rect.height}px`;
    }

    // ── C-Ext Phase 4: Backend-agnostic operation buttons ─────────────────

    const reEvaluateBtn   = document.getElementById('re-evaluate-btn');
    const wickContractBtn = document.getElementById('wick-contract-btn');
    const opTaylorBtn     = document.getElementById('operator-taylor-btn');

    // SCALAR-HANDOFF (docs/34 §5): Julia-session-only button, wired the same
    // way — the module owns its modal, main owns the selection state.
    const _handoff = initHandoff({
        getSelectedExprId: () => selectedExprId,
        getSelectedExprLatex: () => selectedExprLatex,
    });

    /** Show/hide C-Ext buttons depending on session type and selected expr. */
    function updateCExtButtons() {
        const isJulia = (window._sessionPrimaryBackend === 'julia_full');
        // Re-evaluate: show when selected expr contains symbols from lhs_alias_map
        const hasDefinedSymbols = selectedExprId &&
            window._lhsAliasKeys &&
            window._lhsAliasKeys.some(k => {
                const canon = window._exprCanonicalMap &&
                              window._exprCanonicalMap[selectedExprId];
                return canon && new RegExp('(?<![A-Za-z0-9_])' +
                    k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') +
                    '(?![A-Za-z0-9_])').test(canon);
            });
        if (reEvaluateBtn)   reEvaluateBtn.style.display   = hasDefinedSymbols ? '' : 'none';
        if (wickContractBtn) wickContractBtn.style.display  = isJulia && selectedExprId ? '' : 'none';
        if (opTaylorBtn)     opTaylorBtn.style.display      = isJulia && selectedExprId ? '' : 'none';
        _handoff?.updateHandoffButton?.();
    }

    // Re-evaluate button: POST /api/notebook/re_evaluate with auto-detect
    if (reEvaluateBtn) {
        reEvaluateBtn.addEventListener('click', async () => {
            if (!selectedExprId) return;
            const exprId = selectedExprId;
            deselectAll();
            _beginProcessing();
            try {
                const resp = await fetch('/api/notebook/re_evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ expr_id: exprId, substitutions: {} }),
                }).then(r => r.json());
                if (resp.ok) {
                    await refreshTimeline();
                } else {
                    appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(resp.detail || 'error')}`);
                }
            } catch (err) {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
            } finally {
                _endProcessing();
            }
        });
    }

    // Wick contract button: open a dialog for contraction rules JSON
    if (wickContractBtn) {
        wickContractBtn.addEventListener('click', () => {
            if (!selectedExprId) return;
            const exprId = selectedExprId;
            const placeholder = JSON.stringify([
                {"op1": "c'", "op2": "c",
                 "value": "rho(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
                {"op1": "c",  "op2": "c",
                 "value": "kappa(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
                {"op1": "c'", "op2": "c'",
                 "value": "kappaStar(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"}
            ], null, 2);
            const rulesInput = prompt(
                `Wick 収縮規則を JSON で入力 (${exprId})\n\n例:\n${placeholder}`,
                placeholder
            );
            if (!rulesInput) return;
            let rules;
            try { rules = JSON.parse(rulesInput); }
            catch (e) { alert('JSON parse error: ' + e.message); return; }
            deselectAll();
            _beginProcessing();
            fetch('/api/notebook/wick_contract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    expr_id: exprId,
                    contraction_rules: rules,
                    include_constant: false,
                }),
            })
                .then(r => r.json())
                .then(async resp => {
                    if (resp.ok) {
                        await refreshTimeline();
                    } else {
                        appendEntry('supervisor',
                            `❌ Wick 収縮エラー: ${escapeHtml(resp.detail || 'error')}`);
                    }
                })
                .catch(err =>
                    appendEntry('supervisor',
                        `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`)
                )
                .finally(_endProcessing);
        });
    }

    // ── Advise modal ──────────────────────────────────────────────────────
    // The LLM adviser modal (status button, dropdown, submit, save-as-memo)
    // lives in ui/notebook_advise.js (R6, docs/27), wired via dependency
    // injection; loadAdviserList is returned so init can populate it at startup.
    const _advise = initAdvise({
        getSelectedExprId: () => selectedExprId,
        getSelectedExprLatex: () => selectedExprLatex,
        appendEntry,
    });

    // ── Numeric-layer modals (fork / ChatBot / graph) live in
    // ui/notebook_numeric.js (R6, docs/27), wired via dependency injection.
    const _numeric = initNumeric({
        getSelectedExprId: () => selectedExprId,
        getSelectedExprLatex: () => selectedExprLatex,
        appendEntry,
    });

    // ── Keyboard handling ─────────────────────────────────────────────────

    document.execCommand('defaultParagraphSeparator', false, 'br');

    inputBox.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            if (e.isComposing || e.keyCode === 229) return; // IME変換確定は無視
            e.preventDefault();
            handleSubmit();
            return;
        }
        // Backspace: delete an adjacent math-token as a unit
        if (e.key === 'Backspace') {
            const sel = window.getSelection();
            if (sel && sel.rangeCount > 0 && sel.getRangeAt(0).collapsed) {
                const range = sel.getRangeAt(0);
                const prevNode =
                    range.startOffset > 0
                        ? range.startContainer.childNodes?.[range.startOffset - 1]
                        : range.startContainer.previousSibling;
                if (prevNode?.classList?.contains('math-token')) {
                    e.preventDefault();
                    prevNode.remove();
                }
            }
        }
    });

    // Prevent paste from injecting styled HTML — accept plain text only
    inputBox.addEventListener('paste', (e) => {
        e.preventDefault();
        const text = e.clipboardData.getData('text/plain');
        if (text) document.execCommand('insertText', false, text);
    });

    // Normalize full-width characters as the user types (real-time visual feedback).
    // Skip during IME composition to avoid disturbing Japanese input mid-conversion.
    inputBox.addEventListener('input', (e) => {
        if (e.isComposing) return;
        _input.normalizeInputBoxText();
    });
    inputBox.addEventListener('compositionend', () => {
        _input.normalizeInputBoxText();
    });

    // ── Session domains (tracked for OCR hint injection) ──────────────────

    let _currentDomains = [];

    function _sendDomainsToIframe() {
        hwIframe?.contentWindow?.postMessage({ type: 'set_domains', domains: _currentDomains }, '*');
    }

    // ── Session restore ───────────────────────────────────────────────────

    async function loadInitialSession() {
        try {
            const sessionRes = await fetch('/api/notebook/session');
            const sessionData = await sessionRes.json();
            _currentDomains = sessionData.domains ?? [];
            _branch.refreshBranchList(sessionData);
            // Expose session metadata for C-Ext buttons
            // `/api/notebook/session` names this field `backend`; reading
            // `primary_backend` always got undefined, so this fell back to
            // 'maxima' in EVERY session — which meant the Julia-only buttons
            // (Wick 収縮 / 演算子展開) never appeared in a Julia session at all.
            // Found 2026-08-02 while wiring the handoff button, which is gated
            // the same way.  Both spellings accepted so a future rename of the
            // field cannot silently switch the buttons off again.
            window._sessionPrimaryBackend =
                sessionData.backend ?? sessionData.primary_backend ?? 'maxima';
            window._lhsAliasKeys = Object.keys(sessionData.lhs_alias_map ?? {});
            window._exprCanonicalMap = {};
            for (const expr of (sessionData.expressions ?? [])) {
                if (expr.expr_id && expr.canonical) {
                    window._exprCanonicalMap[expr.expr_id] = expr.canonical;
                }
            }
        } catch (_) {}
        try {
            const res  = await fetch('/api/notebook/timeline');
            const data = await res.json();
            const entries = data.entries ?? [];
            if (entries.length === 0) return;
            appendEntry('supervisor', _t('ui.session_restored'));
            for (const entry of entries) {
                if (entry.type === 'user_input') {
                    appendEntry('user', `<span>${escapeHtml(entry.text)}</span>`, { stepId: entry.step_id });
                } else if (entry.type === 'expression') {
                    appendEntry('cas', renderCasEntryHtml(entry), { exprId: entry.expr_id, stepId: entry.step_id, latex: entry.latex });
                } else if (entry.type === 'side_effect') {
                    appendEntry('cas',
                        renderCasEntryHtml({ operation_label: entry.text, expr_id: null, latex: entry.display_latex || entry.name || '', deletable: false }),
                        { stepId: entry.step_id });
                } else if (entry.type === 'note') {
                    appendEntry('note', renderNoteHtml(entry.text), { stepId: entry.step_id });
                } else if (entry.type === 'assumption') {
                    appendEntry('assumption', renderAssumptionHtml(entry.assumption_id, entry.display || entry.canonical));
                } else if (entry.type === 'numeric_fork') {
                    appendEntry('numeric_fork', _numeric.renderNumericForkHtml(entry), { stepId: entry.step_id });
                } else if (entry.type === 'numeric_result') {
                    appendEntry('numeric_result', _numeric.renderNumericResultHtml(entry), { stepId: entry.step_id });
                } else if (entry.type === 'plan_execution') {
                    appendEntry('plan', _plan.renderPlanResultHtml(entry), { stepId: entry.step_id });
                }
            }
        } catch (_) {}
    }

    // ── Fork (file-level branch) ──────────────────────────────────────────

    const forkModal    = document.getElementById('fork-modal');
    const forkModalLabel  = document.getElementById('fork-modal-label');
    const forkModalOk     = document.getElementById('fork-modal-ok');
    const forkModalCancel = document.getElementById('fork-modal-cancel');
    const forkModalCancel2= document.getElementById('fork-modal-cancel2');

    const closeForkModal = () => { forkModal.style.display = 'none'; };
    forkModalCancel?.addEventListener('click', closeForkModal);
    forkModalCancel2?.addEventListener('click', closeForkModal);

    let _forkStepId = null;
    function forkAtStep(stepId) {
        _forkStepId = stepId;
        forkModalLabel.value = '';
        forkModal.style.display = 'flex';
        forkModalLabel.focus();
    }

    forkModalOk?.addEventListener('click', async () => {
        const stepId = _forkStepId;
        const value  = forkModalLabel.value.trim();
        closeForkModal();
        try {
            const res  = await fetch('/api/notebook/fork', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ step_id: stepId, snapshot_filename: value }),
            });
            const data = await res.json();
            if (data.ok) {
                appendEntry('supervisor', _t('ui.fork_saved', {file: data.snapshot_file}));
            }
        } catch (_) {}
    });

    // Attach fork button to a timeline entry that has a step_id (hoisted function declaration)
    function attachForkButton(entry) {
        if (!entry.dataset.stepId) return;
        const btn = document.createElement('button');
        btn.className   = 'fork-btn';
        btn.title       = _t('ui.fork_step_title');
        btn.textContent = '💾';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            forkAtStep(entry.dataset.stepId);
        });
        const removeBtn = entry.querySelector('.cas-remove-btn');
        if (removeBtn) removeBtn.before(btn);
        else {
            const idBadge = entry.querySelector('.expr-id');
            if (idBadge) idBadge.after(btn);
            else entry.appendChild(btn);
        }
    }

    // ── Save / Load ───────────────────────────────────────────────────────

    // ── Branch summary ────────────────────────────────────────────────────

    // ── Status button ────────────────────────────────────────────────────
    function renderLlmList(items) {
        if (!items || items.length === 0) return '—';
        return items.map(it =>
            it.selected
                ? `<span style="font-weight:700; color:#1a6;">✓ ${escapeHtml(it.name)}</span>`
                : `<span style="color:#999;">${escapeHtml(it.name)}</span>`
        ).join('<br>');
    }

    document.getElementById('status-btn')?.addEventListener('click', async () => {
        const modal = document.getElementById('status-modal');
        const table = document.getElementById('status-table');
        modal.style.display = 'flex';
        table.innerHTML = '';
        try {
            await window._reloadI18n();
            const data = await fetch('/api/notebook/status').then(r => r.json());
            const rows = [
                [_t('ui.status_title'),       escapeHtml(data.title)],
                ...(data.current_filename ? [[_t('ui.status_file'), escapeHtml(data.current_filename)]] : []),
                [_t('ui.status_lang'),         escapeHtml(data.lang)],
                [_t('ui.status_backend'),      escapeHtml(data.backend)],
                ...(data.coordinate_system ? [[_t('ui.status_coord_system'), escapeHtml(data.coordinate_system)]] : []),
                [_t('ui.status_domains'),      escapeHtml((data.domains || []).join(', ') || '—')],
                [_t('ui.status_steps'),        escapeHtml(String(data.step_count ?? '—'))],
                [_t('ui.status_supervisor'),   renderLlmList(data.supervisors)],
                [_t('ui.status_advisers'),     renderLlmList(data.advisers)],
                [_t('ui.status_ocr_engine'),   renderLlmList(data.ocr_engines)],
                [_t('ui.status_ocr_refiner'),  renderLlmList(data.ocr_refiners)],
            ];
            table.innerHTML = rows.map(([k, v]) =>
                `<tr>
                  <td style="padding:4px 8px 4px 0; color:#666; white-space:nowrap; vertical-align:top;">${k}</td>
                  <td style="padding:4px 0; line-height:1.6; word-break:break-all;">${v}</td>
                 </tr>`
            ).join('');
        } catch (e) {
            table.innerHTML = `<tr><td colspan="2" style="color:#c00;">${escapeHtml(String(e))}</td></tr>`;
        }
    });
    document.getElementById('status-modal-close')?.addEventListener('click', () => {
        document.getElementById('status-modal').style.display = 'none';
    });
    document.getElementById('status-modal')?.addEventListener('click', e => {
        if (e.target === document.getElementById('status-modal'))
            document.getElementById('status-modal').style.display = 'none';
    });

    document.getElementById('summary-btn')?.addEventListener('click', async () => {
        const btn = document.getElementById('summary-btn');
        btn.disabled = true;
        btn.textContent = _t('ui.generating');
        try {
            const res  = await fetch('/api/notebook/summary', { method: 'POST' });
            const data = await res.json();
            if (data.ok) {
                appendEntry('note', renderNoteHtml(data.text), { stepId: data.step_id });
            } else {
                appendEntry('supervisor', `❌ ${_t('ui.summary_failed', {detail: data.detail ?? ''})}`);
            }
        } catch (e) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(e))}`);
        } finally {
            btn.disabled = false;
            btn.textContent = _t('ui.summary_btn');
        }
    });

    // ── Save / Load / Export / Shutdown / Autosave ─────────────────────────
    // Session persistence (save & load dialogs, LaTeX export, shutdown-with-save,
    // periodic autosave) lives in ui/notebook_persistence.js (R6, docs/27), wired
    // via dependency injection; triggerAutosave is returned for the interpret/
    // confirm handlers above that trigger a save after a successful operation.
    const _persistence = initPersistence({
        appendEntry,
        loadInitialSession,
        resetSelection: () => { selectedExprId = null; opButtons.style.display = 'none'; },
    });
    const triggerAutosave = _persistence.triggerAutosave;

    // ── Live branch (in-session fork) ──────────────────────────────────────
    // A second branch that coexists with others in this same session, distinct
    // from the file-snapshot fork above. Lives in ui/notebook_branch.js (R6).
    const _branch = initBranch({
        appendEntry,
        loadInitialSession,
        resetSelection: () => { selectedExprId = null; opButtons.style.display = 'none'; },
    });

    // ── Plan-result rendering (C-Ext-5) ─────────────────────────────────────
    // Per-step status checklist for a completed/stopped plan (fixed workflow
    // or Adviser/free-form candidate). Lives in ui/notebook_plan.js (R6).
    const _plan = initPlan({ appendEntry, triggerAutosave });

    // ── New session dialog ────────────────────────────────────────────────

    const newSessionBtn        = document.getElementById('new-session-btn');
    const newSessionModal      = document.getElementById('new-session-modal');
    const newSessionModalClose = document.getElementById('new-session-modal-close');
    const newSessionCancel     = document.getElementById('new-session-cancel');
    const newSessionOk         = document.getElementById('new-session-ok');
    const newSessionTitle      = document.getElementById('new-session-title');
    const newSessionLanguage   = document.getElementById('new-session-language');
    const newSessionSupervisor = document.getElementById('new-session-supervisor');
    const newSessionDomains    = document.getElementById('new-session-domains');

    // flat map of domain name → label, populated at modal open time
    let _domainLabelMap = {};  // {name: label}
    let _modalCategories = [];

    function _renderModalDomains(lang) {
        _domainLabelMap = {};
        _modalCategories.forEach(cat => cat.domains.forEach(d => {
            _domainLabelMap[d.name] = (lang === 'ja' ? d.label_ja : d.label_en) || d.label;
        }));
        if (!_modalCategories.length) {
            newSessionDomains.innerHTML = `<div style="color:#888;">${_t('ui.domains_none')}</div>`;
            return;
        }
        newSessionDomains.innerHTML = _modalCategories.map(cat => {
            const badge = cat.default_backend === 'julia_full'
                ? `<span style="font-size:.7em; background:#e3f2fd; color:#1565c0; padding:1px 6px; border-radius:3px; margin-left:6px; font-weight:normal;">Julia</span>`
                : '';
            const items = cat.domains.map(d => {
                const st = d.status ?? 'active';
                const isActive = (st === 'active');
                const statusBadge = st === 'placeholder'
                    ? `<span style="font-size:.68em; background:#fff3cd; color:#856404; padding:1px 5px; border-radius:3px; margin-left:4px;">${_t('ui.domain_status_placeholder')}</span>`
                    : st === 'future'
                    ? `<span style="font-size:.68em; background:#e8eaf6; color:#3949ab; padding:1px 5px; border-radius:3px; margin-left:4px;">${_t('ui.domain_status_future')}</span>`
                    : '';
                const dimStyle = isActive ? '' : 'opacity:0.45;';
                const titleAttr = isActive ? '' : (st === 'placeholder' ? _t('ui.domain_status_placeholder_title') : _t('ui.domain_status_future_title'));
                return `<label style="display:flex; align-items:center; gap:5px; cursor:${isActive ? 'pointer' : 'default'}; font-size:.9em; ${dimStyle}" title="${escapeHtml(titleAttr)}">
                    <input type="checkbox" value="${escapeHtml(d.name)}" style="width:14px; height:14px; flex-shrink:0;" ${isActive ? '' : 'disabled'}>
                    ${escapeHtml(_domainLabelMap[d.name] || d.label)}${statusBadge}
                </label>`;
            }).join('');
            return `<details style="border:1px solid #e8e8e8; border-radius:5px; overflow:hidden;">
                <summary style="cursor:pointer; padding:6px 10px; font-size:.85em; font-weight:600; color:#444; background:#f7f7f7; display:flex; align-items:center; user-select:none;">
                    <span style="flex:1;">${escapeHtml(cat.label)}${badge}</span>
                </summary>
                <div style="padding:8px 12px; display:flex; flex-wrap:wrap; gap:6px 18px;">
                    ${items}
                </div>
            </details>`;
        }).join('');
    }

    async function openNewSessionModal() {
        newSessionTitle.value = '';
        if (newSessionLanguage) newSessionLanguage.value = window._lang || 'ja';
        newSessionDomains.innerHTML = `<div style="color:#888;">${_t('ui.session_list_loading')}</div>`;
        newSessionModal.style.display = 'flex';
        // Fetch domains and supervisor list in parallel
        const [domainRes, supRes] = await Promise.allSettled([
            fetch('/api/notebook/domains'),
            fetch('/api/notebook/supervisor-list'),
        ]);
        try {
            const data = domainRes.status === 'fulfilled' ? await domainRes.value.json() : {};
            _modalCategories = data.categories ?? [];
            _renderModalDomains(newSessionLanguage?.value || 'ja');
        } catch (e) {
            newSessionDomains.innerHTML = `<div style="color:#c00;">${_t('ui.domains_load_failed')}</div>`;
        }
        try {
            const sdata = supRes.status === 'fulfilled' ? await supRes.value.json() : {};
            const sups = sdata.supervisors ?? [];
            if (newSessionSupervisor && sups.length) {
                newSessionSupervisor.innerHTML = sups.map(s =>
                    `<option value="${escapeHtml(s.name)}"${s.default ? ' selected' : ''}>${escapeHtml(s.name)}</option>`
                ).join('');
            }
        } catch (_) {}
    }

    async function startNewSession() {
        const title           = newSessionTitle.value.trim() || 'Notebook';
        const language        = newSessionLanguage?.value || 'ja';
        const supervisor_name = newSessionSupervisor?.value || null;
        const checked  = [...newSessionDomains.querySelectorAll('input[type=checkbox]:checked')];
        const domains  = checked.map(cb => cb.value);
        const coordSel = document.getElementById('new-session-coord-system');
        const coordinate_system = coordSel?.value || 'cartesian';
        newSessionModal.style.display = 'none';
        timeline.innerHTML = '';
        appendEntry('system', _t('ui.starting_session'));
        try {
            await fetch('/api/notebook/session/new', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, domains, language, supervisor_name, coordinate_system }),
            });
            _currentDomains = domains;
            // Reload UI language to match the new session's language
            if (window._reloadI18n) await window._reloadI18n();
            timeline.innerHTML = '';
            const labelOf = name => (_domainLabelMap[name] ?? name);
            const msg = domains.length
                ? _t('ui.session_started_domains', {domains: domains.map(labelOf).join('、')})
                : _t('ui.session_started');
            appendEntry('system', msg);
        } catch (e) {
            appendEntry('system', `❌ ${escapeHtml(String(e))}`);
        }
    }

    newSessionBtn?.addEventListener('click', openNewSessionModal);
    newSessionModalClose?.addEventListener('click', () => { newSessionModal.style.display = 'none'; });
    newSessionCancel?.addEventListener('click', () => { newSessionModal.style.display = 'none'; });
    newSessionOk?.addEventListener('click', startNewSession);
    newSessionLanguage?.addEventListener('change', () => _renderModalDomains(newSessionLanguage.value));

    // ── Init ──────────────────────────────────────────────────────────────

    submitBtn.addEventListener('click', handleSubmit);
    loadInitialSession();
    _advise.loadAdviserList();   // populate adviser dropdown and update button state
});
