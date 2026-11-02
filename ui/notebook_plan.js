// ui/notebook_plan.js — C-Ext-5 plan-result rendering: a compact per-step
// status checklist for a completed/stopped plan.
//
// Both plan-execution paths already return the full final plan_step_results
// synchronously (the Executor runs to completion or stop before the HTTP
// response is built — confirmed by reading ui/routes/operations.py and
// notebook/engine_plan_mixin.py), so this is a pure render of data already in
// hand — no polling loop. Each step's own expression already appears as an
// ordinary CAS timeline entry (cas_operation Step) on reload; this card is
// an additional compact summary, not a duplicate of those.

import { escapeHtml } from './notebook_utils.js';
import { renderConfirmationHtml } from './notebook_timeline.js';

const _STATUS_ICON = { success: '✓', failed: '✗', skipped: '⊘' };

export function initPlan({ appendEntry, triggerAutosave }) {
    function renderPlanResultHtml(data) {
        const summary = data.message ?? data.operation_label ?? '';
        const results = data.plan_step_results ?? [];
        const rows = results.map(r => {
            const icon = _STATUS_ICON[r.status] ?? '?';
            const cls  = `plan-step plan-step-${escapeHtml(r.status ?? '')}`;
            const detail = r.status === 'failed'
                ? escapeHtml(r.error || '')
                : escapeHtml(r.value || '');
            return `<div class="${cls}">`
                 + `<span class="plan-step-icon">${icon}</span> `
                 + `<span class="plan-step-op">${escapeHtml(r.operation ?? '')}</span>`
                 + (detail ? ` <span class="plan-step-detail">${detail}</span>` : '')
                 + `</div>`;
        }).join('');
        // D-5 (open_items.md): a stopped/failed plan gets a retry button —
        // retry_plan (B-3) existed engine-only until now; this is its first UI.
        // Failure→replan loop (2026-07-18) sits alongside it: retry re-runs
        // the SAME plan after the researcher fixes the root cause; replan
        // asks the Adviser to propose a DIFFERENT approach instead.
        const retryable = data.status === 'failed' && data.plan_id;
        const retryBtn = retryable
            ? `<button class="plan-retry-btn" data-plan-id="${escapeHtml(data.plan_id)}">${_t('ui.plan_retry_btn')}</button>`
            : '';
        const replanBtn = retryable
            ? `<button class="plan-replan-btn" data-plan-id="${escapeHtml(data.plan_id)}">${_t('ui.plan_replan_btn')}</button>`
            : '';
        return `<div class="plan-summary">${escapeHtml(summary)}</div>`
             + `<div class="plan-steps">${rows}</div>`
             + retryBtn + replanBtn;
    }

    // Attach the retry-click handler to a freshly-inserted 'plan' timeline
    // entry (mirrors ui/notebook_branch.js::attachLiveForkButton — wired
    // fresh per entry from appendEntry, not a delegated listener).
    function attachRetryButton(entry) {
        const btn = entry.querySelector('.plan-retry-btn');
        if (!btn) return;
        btn.addEventListener('click', async () => {
            const planId = btn.dataset.planId;
            btn.disabled = true;
            try {
                const res = await fetch(`/api/notebook/plan/retry/${encodeURIComponent(planId)}`,
                    { method: 'POST' });
                const data = await res.json();
                if (data.ok === false && data.detail) {
                    appendEntry('supervisor', `❌ ${escapeHtml(data.detail)}`);
                    return;
                }
                appendEntry('plan', renderPlanResultHtml(data), { stepId: data.step_id });
                triggerAutosave?.();
            } catch (err) {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
            } finally {
                btn.disabled = false;
            }
        });
    }

    // Failure→replan loop (open_items C-Ext-5, 2026-07-18): asks the
    // Adviser for a DIFFERENT approach, informed by what failed.  The
    // response is the SAME needs_confirmation shape a fresh 「解法の候補」
    // ask produces, so it's rendered via the shared renderConfirmationHtml
    // and its confirm/cancel buttons are picked up by the existing
    // delegated .confirm-btn/.cancel-btn listener in notebook_main.js — no
    // new click wiring needed for the resulting candidates card itself,
    // only for this button (mirrors attachRetryButton above).
    function attachReplanButton(entry) {
        const btn = entry.querySelector('.plan-replan-btn');
        if (!btn) return;
        btn.addEventListener('click', async () => {
            const planId = btn.dataset.planId;
            btn.disabled = true;
            try {
                const res = await fetch(`/api/notebook/plan/replan/${encodeURIComponent(planId)}`,
                    { method: 'POST' });
                const data = await res.json();
                if (data.ok === false) {
                    appendEntry('supervisor', `❌ ${escapeHtml(data.detail ?? _t('ui.unknown_error'))}`);
                    return;
                }
                appendEntry('supervisor', renderConfirmationHtml(data));
            } catch (err) {
                appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
            } finally {
                btn.disabled = false;
            }
        });
    }

    // ── C-Ext-5 B-4 (read-only): branch comparison for plan_execution steps.
    // Adapted from _renderBranchComparison in ui/notebook_numeric.js — same
    // table shape (columns = branches, rows = groups), but cells show
    // plan_id/status/summary only (plans have no figure artifacts).
    const compareBtn   = document.getElementById('plan-compare-btn');
    const compareModal = document.getElementById('plan-compare-modal');
    const compareClose = document.getElementById('plan-compare-close');
    const compareEmpty = document.getElementById('plan-compare-empty');
    const compareBody  = document.getElementById('plan-compare-body');

    function renderPlanBranchComparisonHtml(cmp) {
        compareBody.textContent = '';
        const groups = (cmp && cmp.groups || []).filter(g => g.comparable);
        if (!groups.length) {
            compareEmpty.style.display = 'block';
            return;
        }
        compareEmpty.style.display = 'none';
        const branchIds = [...new Set(groups.flatMap(g => Object.keys(g.branches)))].sort();
        const table = document.createElement('table');
        table.style.cssText = 'border-collapse:collapse; font-size:.82em; min-width:100%;';
        const head = table.insertRow();
        const corner = head.insertCell();
        corner.textContent = _t('ui.plan_branch_compare_group');
        branchIds.forEach(b => { const c = head.insertCell(); c.textContent = b; });
        [...head.cells].forEach(c => {
            c.style.cssText = 'border:1px solid #e2e8f0; padding:6px 10px; background:#f8fafc; font-weight:bold; text-align:left;';
        });
        groups.forEach(g => {
            const row = table.insertRow();
            const label = row.insertCell();
            label.textContent = `${g.label}${g.target ? ' · ' + g.target : ''}`;
            branchIds.forEach(b => {
                const cell = row.insertCell();
                (g.branches[b] || []).forEach(entry => {
                    const box = document.createElement('div');
                    box.style.cssText = 'margin-bottom:6px;';
                    const line = document.createElement('div');
                    line.textContent = `${entry.plan_id} · ${entry.status}`;
                    line.style.fontWeight = 'bold';
                    box.appendChild(line);
                    if (entry.summary) {
                        const s = document.createElement('div');
                        s.textContent = entry.summary;
                        s.style.cssText = 'color:#475569;';
                        box.appendChild(s);
                    }
                    cell.appendChild(box);
                });
            });
            [...row.cells].forEach(c => {
                c.style.cssText += 'border:1px solid #e2e8f0; padding:6px 10px; vertical-align:top;';
            });
        });
        compareBody.appendChild(table);
    }

    async function openPlanCompareModal() {
        compareModal.style.display = 'flex';
        compareEmpty.style.display = 'none';
        compareBody.textContent = '…';
        try {
            const r = await fetch('/api/notebook/plan/branch_comparison').then(res => res.json());
            renderPlanBranchComparisonHtml(r.comparison);
        } catch (err) {
            compareBody.textContent = String(err);
        }
    }

    if (compareBtn) compareBtn.addEventListener('click', openPlanCompareModal);
    if (compareClose) compareClose.addEventListener('click',
        () => { compareModal.style.display = 'none'; });
    if (compareModal) compareModal.addEventListener('click',
        (e) => { if (e.target === compareModal) compareModal.style.display = 'none'; });

    return {
        renderPlanResultHtml, renderPlanBranchComparisonHtml,
        attachRetryButton, attachReplanButton,
    };
}
