// ui/notebook_timeline.js — timeline card / entry HTML renderers (R6, docs/27).
//
// Pure data→HTML functions extracted verbatim from notebook_main.js's Timeline
// section (dedented one level and exported).  No shared state: they use only the
// notebook_utils helpers below plus the globals _t and katex.  The DOM/state hub
// appendEntry() stays in notebook_main.js (it wires selectExpression / the CAS
// context menu / the fork button, which live there).

import { escapeHtml, escapeJsonAttr, renderKatex } from './notebook_utils.js';

export function renderCasEntryHtml(data) {
    const label = data.operation_label
        ? `<span class="cas-operation-label">${escapeHtml(data.operation_label)}</span>`
        : '';
    const idBadge = data.expr_id
        ? `<span class="expr-id">${escapeHtml(data.expr_id)}</span>`
        : '';
    const latex = data.latex || data.canonical || '';
    const mathEl = renderKatex(latex, true);
    const removeBtn = data.deletable
        ? `<button class="cas-remove-btn" data-expr-id="${escapeHtml(data.expr_id)}" title="${_t('ui.cas_remove_title')}">✕</button>`
        : '';
    const btnGroup = (idBadge || removeBtn)
        ? `<span class="cas-btn-group">${idBadge}${removeBtn}</span>`
        : '';

    let undefinedHtml = '';
    const undef = data.undefined_symbols;
    if (undef && undef.length > 0) {
        const syms = undef.map(s => `<code>${escapeHtml(s)}</code>`).join(', ');
        const hint = undef.length > 0
            ? _t('ui.undefined_symbols_hint').replace('{symbol}', escapeHtml(undef[0]))
            : '';
        undefinedHtml = `<div class="undefined-symbols-warning">` +
            `⚠ <strong>${_t('ui.undefined_symbols_label')}:</strong> ${syms}` +
            `<br><span class="undefined-hint">${escapeHtml(hint)}</span>` +
            `</div>`;
    }

    // Non-blocking notices: legitimate writing that may not be what was meant
    // (a bound index reusing a name the expression uses elsewhere).  Shown on
    // the SUCCESS card, because that is where the researcher is looking — the
    // undefined-symbol banner above is rendered on the error path only, so a
    // warning routed through it would never be seen.
    let warnHtml = '';
    const warns = data.warnings;
    if (warns && warns.length > 0) {
        const items = warns.map(w => escapeHtml(w)).join('<br>');
        warnHtml = `<div class="expr-warning">` +
            `⚠ <strong>${_t('ui.expr_warnings_label')}:</strong> ${items}` +
            `</div>`;
    }

    return label + btnGroup + mathEl.outerHTML + undefinedHtml + warnHtml;
}

export function renderUserEntryHtml(tokens) {
    return tokens.map(t => {
        if (t.type === 'text') return escapeHtml(t.value);
        const span = document.createElement('span');
        span.className = 'math-token';
        try { katex.render(t.latex, span, { displayMode: false, throwOnError: false }); }
        catch (_) { span.textContent = t.latex; }
        return span.outerHTML;
    }).join('');
}

// Normalize \[...\] → $$...$$ and \(...\) → $...$ (shared by notes and advise)
export function normalizeMathDelimiters(text) {
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `$$${m}$$`);
    text = text.replace(/\\\(([\s\S]*?)\\\)/g,  (_, m) => `$${m}$`);
    return text;
}

export function renderNoteHtml(text) {
    text = normalizeMathDelimiters(text);
    const segments = text.split(/(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)/);
    let inner = '';
    for (const seg of segments) {
        if ((seg.startsWith('$$') && seg.endsWith('$$') && seg.length > 4) ||
            (seg.startsWith('$')  && seg.endsWith('$')  && seg.length > 2)) {
            const math = seg.startsWith('$$') ? seg.slice(2, -2) : seg.slice(1, -1);
            const span = document.createElement('span');
            try { katex.render(math, span, { displayMode: false, throwOnError: false }); }
            catch (_) { span.textContent = seg; }
            inner += span.outerHTML;
        } else {
            inner += escapeHtml(seg).replace(/\n/g, '<br>');
        }
    }
    return `<span class="note-text">${inner}</span>`
         + `<button class="note-remove-btn" title="${_t('ui.note_remove_title')}">✕</button>`;
}

export function renderAssumptionHtml(assumptionId, canonical) {
    const mathEl = renderKatex(canonical, false);
    return `<span class="assumption-condition" data-assumption-id="${escapeHtml(assumptionId)}">${mathEl.outerHTML}</span>`
        + `<button class="assumption-remove-btn" data-assumption-id="${escapeHtml(assumptionId)}" title="${_t('ui.assumption_remove_title')}">✕</button>`;
}

export function renderDepConfirmationHtml(data) {
    const vars    = (data.dep_vars ?? []).join('、');
    const diffVar = data.diff_var ?? '';
    const opAttr  = `data-op="${escapeHtml(data.operation)}"`;
    const argsAttr = `data-args='${escapeJsonAttr(data.args ?? {})}'`;
    const depVarsAttr = `data-dep-vars='${escapeJsonAttr(data.dep_vars ?? [])}'`;
    return `<div>⚠️ ${escapeHtml(_t('ui.dep_warning', {vars, diffVar}))}</div>`
        + `<div class="confirm-actions">`
        + `<button class="dep-confirm-btn" data-choice="with" ${opAttr} ${argsAttr} ${depVarsAttr}>${_t('ui.dep_total')}</button>`
        + `<button class="dep-confirm-btn" data-choice="without" ${opAttr} ${argsAttr} ${depVarsAttr}>${escapeHtml(_t('ui.dep_partial', {vars}))}</button>`
        + `<button class="cancel-btn">${_t('ui.cancel')}</button>`
        + `</div>`;
}

export function renderConfirmationHtml(data) {
    const promptText  = escapeHtml(data.confirmation_prompt ?? _t('ui.confirm_default'));
    const userTextAttr = `data-user-text='${escapeJsonAttr(data._user_text ?? '')}'`;
    const candidatesJson = JSON.stringify(data.candidates ?? []);
    const lang = window._lang || 'ja';
    let actionsHtml = '';

    if (data.candidates && data.candidates.length > 1) {
        actionsHtml = data.candidates.map((c) => {
            let label = c[`display_${lang}`] ?? c.display_en ?? c.operation;
            if (!label && c.operation === 'differentiate' && c.args?.var) {
                label = _t('ui.diff_var_label', {var: c.args.var});
            }
            return `<button class="confirm-btn candidate"
                data-op="${escapeHtml(c.operation)}"
                data-args='${escapeJsonAttr(c.args ?? {})}'
                data-candidates='${escapeHtml(candidatesJson)}'
                ${userTextAttr}
                >${escapeHtml(label)}</button>`;
        }).join('') +
            `<button class="cancel-btn">${_t('ui.cancel')}</button>`;
    } else {
        const pendingDepsAttr = `data-pending-deps='${escapeJsonAttr(data.pending_dependencies ?? [])}'`;
        actionsHtml =
            `<button class="confirm-btn"
                data-op="${escapeHtml(data.operation)}"
                data-args='${escapeJsonAttr(data.args ?? {})}'
                data-candidates='${escapeHtml(candidatesJson)}'
                ${pendingDepsAttr}
                ${userTextAttr}
                >${_t('ui.execute')}</button>
             <button class="cancel-btn">${_t('ui.cancel')}</button>`;
    }
    return `<div>${promptText}</div><div class="confirm-actions">${actionsHtml}</div>`;
}
