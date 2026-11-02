// ui/notebook_handoff.js — SCALAR-HANDOFF front-end (docs/34 §5).
//
// The button is the same act as the NL command `E5 を Maxima セッションに渡して`,
// with one difference: a button cannot be read the way a typed sentence can, so
// it PREVIEWS first.  What the preview shows is the carried declarations —
// that list is what decides whether the target session can actually continue
// the calculation (measured: an uncarried `omega_D` turned the continuation's
// answer into `asinh(omega(D)/Delta)` with no error anywhere).
//
// Visible only in a Julia session: the crossing exists because operator algebra
// and scalar algebra live in different backends, so Maxima → Maxima is not a
// thing anyone needs.

import { escapeHtml, prepareLaTeX } from './notebook_utils.js';

export function initHandoff({ getSelectedExprId, getSelectedExprLatex }) {
    const btn      = document.getElementById('handoff-btn');
    const modal    = document.getElementById('handoff-modal');
    const exprBox  = document.getElementById('handoff-modal-expr');
    const bodyBox  = document.getElementById('handoff-modal-body');
    const closeBtn = document.getElementById('handoff-modal-close');
    const cancelBtn = document.getElementById('handoff-cancel-btn');
    const confirmBtn = document.getElementById('handoff-confirm-btn');
    if (!btn || !modal) return { updateHandoffButton: () => {} };

    let pending = null;      // { expr_id, target_backend }

    function updateHandoffButton() {
        const isJulia = (window._sessionPrimaryBackend === 'julia_full');
        btn.style.display = (isJulia && getSelectedExprId()) ? '' : 'none';
    }

    function close() { modal.style.display = 'none'; pending = null; }

    async function openPreview() {
        const exprId = getSelectedExprId();
        if (!exprId) return;
        pending = { expr_id: exprId, target_backend: 'maxima' };
        exprBox.innerHTML = '';
        bodyBox.innerHTML = `<div>${escapeHtml(_t('ui.handoff_loading'))}</div>`;
        modal.style.display = 'flex';

        const latex = getSelectedExprLatex();
        if (latex && window.katex) {
            try {
                katex.render(prepareLaTeX(latex), exprBox, { throwOnError: false });
            } catch (e) { exprBox.textContent = latex; }
        } else if (latex) {
            exprBox.textContent = latex;
        }

        let res;
        try {
            res = await (await fetch('/api/notebook/handoff/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pending),
            })).json();
        } catch (e) {
            bodyBox.innerHTML = `<div class="handoff-error">${escapeHtml(String(e))}</div>`;
            return;
        }

        if (!res.ok) {
            // A refusal is the useful answer here — say it and offer no button.
            bodyBox.innerHTML = `<div class="handoff-error">${escapeHtml(res.error || '')}</div>`;
            confirmBtn.style.display = 'none';
            return;
        }
        confirmBtn.style.display = '';

        const decls = res.declarations || [];
        const rows = decls.map(d => {
            const bits = [d.domain, d.sign].filter(Boolean).join(', ');
            return `<li><code>${escapeHtml(d.name)}</code>${bits ? ' — ' + escapeHtml(bits) : ''}</li>`;
        }).join('');
        let html = `<div>${escapeHtml(_t('ui.handoff_carries').replace('{count}', decls.length))}</div>`;
        html += rows ? `<ul style="margin:6px 0 0 1.2em;">${rows}</ul>` : '';
        if ((res.undeclared || []).length) {
            html += `<div style="margin-top:8px; color:#8a6d3b;">`
                 +  escapeHtml(_t('ui.handoff_undeclared').replace(
                        '{names}', (res.undeclared || []).join(', ')))
                 +  `</div>`;
        }
        bodyBox.innerHTML = html;
    }

    async function confirm() {
        if (!pending) return;
        bodyBox.innerHTML = `<div>${escapeHtml(_t('ui.handoff_creating'))}</div>`;
        confirmBtn.disabled = true;
        let res;
        try {
            res = await (await fetch('/api/notebook/handoff/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pending),
            })).json();
        } catch (e) {
            bodyBox.innerHTML = `<div class="handoff-error">${escapeHtml(String(e))}</div>`;
            confirmBtn.disabled = false;
            return;
        }
        confirmBtn.disabled = false;
        if (!res.ok) {
            bodyBox.innerHTML = `<div class="handoff-error">${escapeHtml(res.error || '')}</div>`;
            return;
        }
        // Say what was verified — and, just as plainly, what was not.
        const cls = (res.verification && res.verification.status === 'differed')
            ? 'handoff-error' : '';
        bodyBox.innerHTML =
            `<div>${escapeHtml(res.message || '')}</div>` +
            `<div class="${cls}" style="margin-top:6px;">${escapeHtml(res.note || '')}</div>` +
            `<div style="margin-top:12px;">` +
            `<button id="handoff-open-btn">${escapeHtml(_t('ui.handoff_open'))}</button>` +
            `</div>`;
        confirmBtn.style.display = 'none';
        const openBtn = document.getElementById('handoff-open-btn');
        if (openBtn) {
            // Opening is a separate, explicit act: the source session is mid
            // calculation, and folding it away is the last thing the researcher
            // wants by surprise.
            openBtn.addEventListener('click', () => {
                window.location.href = `/?load=${encodeURIComponent(res.filename)}`;
            });
        }
    }

    btn.addEventListener('click', openPreview);
    closeBtn?.addEventListener('click', close);
    cancelBtn?.addEventListener('click', close);
    confirmBtn?.addEventListener('click', confirm);
    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });

    return { updateHandoffButton };
}
