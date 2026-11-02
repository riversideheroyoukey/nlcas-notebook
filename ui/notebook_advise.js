// ui/notebook_advise.js — LLM adviser modal (R6, docs/27).
//
// Extracted verbatim from notebook_main.js's "Advise modal" section.  The
// adviser-status and last-advise-text are used only by this family, so they
// become module-internal state.  initAdvise() wires the button + modal and
// returns { loadAdviserList } so the main module can populate the dropdown at
// startup.
//
// Dependencies come in via DI: getSelectedExprId / getSelectedExprLatex read
// the shared selection state that lives in notebook_main.js; appendEntry is the
// timeline DOM/state hub (kept in main).  `_t` and `katex` resolve as globals.

import { escapeHtml, prepareLaTeX } from './notebook_utils.js';
import { normalizeMathDelimiters, renderNoteHtml } from './notebook_timeline.js';

export function initAdvise({ getSelectedExprId, getSelectedExprLatex, appendEntry }) {
    const adviseBtn           = document.getElementById('advise-btn');
    const adviseModal          = document.getElementById('advise-modal');
    const adviseModalTitle     = document.getElementById('advise-modal-title');
    const adviseModalExpr      = document.getElementById('advise-modal-expr');
    const adviseModalAdviser   = document.getElementById('advise-modal-adviser');
    const adviseModalQuestion  = document.getElementById('advise-modal-question');
    const adviseModalSubmit    = document.getElementById('advise-modal-submit');
    const adviseModalResult    = document.getElementById('advise-modal-result');
    const adviseModalClose     = document.getElementById('advise-modal-close');
    const adviseSaveArea       = document.getElementById('advise-save-area');
    const adviseSaveBtn        = document.getElementById('advise-save-btn');

    let adviserStatus  = 'checking';   // 'checking' | 'available' | 'unavailable'
    let lastAdviseText = '';           // raw LLM text of the last successful advise

    // Update advise button appearance based on server reachability
    function setAdviserStatus(status) {
        adviserStatus = status;
        adviseBtn.classList.remove('op-btn-unavailable', 'op-btn-checking');
        if (status === 'checking') {
            adviseBtn.textContent = _t('ui.advising');
            adviseBtn.disabled    = true;
            adviseBtn.classList.add('op-btn-checking');
        } else if (status === 'available') {
            adviseBtn.textContent = _t('ui.advise_available');
            adviseBtn.disabled    = false;
        } else {
            adviseBtn.textContent = _t('ui.advise_unavailable');
            adviseBtn.disabled    = false;
            adviseBtn.classList.add('op-btn-unavailable');
        }
    }

    // Fetch all adviser endpoints, populate dropdown, update button state.
    async function loadAdviserList() {
        setAdviserStatus('checking');
        try {
            const res  = await fetch('/api/notebook/adviser-list');
            const data = await res.json();
            const list = data.advisers ?? [];

            adviseModalAdviser.innerHTML = '';
            let anyAvailable = false;
            for (const adv of list) {
                const opt    = document.createElement('option');
                opt.value    = adv.name;
                const dot    = adv.available ? '●' : '○';
                const scope  = adv.scope === 'cloud' ? ' ☁' : '';
                opt.textContent = `${dot} ${adv.model || adv.name}${scope}`;
                if (!adv.available) opt.style.color = '#aaa';
                adviseModalAdviser.appendChild(opt);
                if (adv.available) anyAvailable = true;
            }

            if (list.length === 0) {
                const opt = document.createElement('option');
                opt.textContent = _t('ui.no_llm');
                opt.value = '';
                adviseModalAdviser.appendChild(opt);
            }

            setAdviserStatus(anyAvailable ? 'available' : 'unavailable');
            return anyAvailable;
        } catch (_) {
            setAdviserStatus('unavailable');
            return false;
        }
    }

    // Render LLM commentary: inline $...$ and display $$...$$ via KaTeX.
    function renderMathInText(text) {
        text = normalizeMathDelimiters(text);
        const segments = text.split(/(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)/);
        return segments.map(seg => {
            if (seg.startsWith('$$') && seg.endsWith('$$') && seg.length > 4) {
                try { return katex.renderToString(seg.slice(2, -2), { displayMode: true,  throwOnError: false }); }
                catch (_) { return escapeHtml(seg); }
            }
            if (seg.startsWith('$') && seg.endsWith('$') && seg.length > 2) {
                try { return katex.renderToString(seg.slice(1, -1), { displayMode: false, throwOnError: false }); }
                catch (_) { return escapeHtml(seg); }
            }
            return escapeHtml(seg).replace(/\n/g, '<br>');
        }).join('');
    }

    function openAdviseModal() {
        const selectedExprId    = getSelectedExprId();
        const selectedExprLatex = getSelectedExprLatex();
        adviseModalTitle.textContent    = _t('ui.advise_title_expr', {expr_id: selectedExprId});
        adviseModalQuestion.value       = '';
        adviseModalResult.style.display = 'none';
        adviseModalResult.innerHTML     = '';
        adviseSaveArea.style.display    = 'none';
        adviseSaveBtn.textContent       = _t('ui.save_as_memo_btn');
        adviseSaveBtn.disabled          = false;
        lastAdviseText                  = '';
        adviseModalExpr.innerHTML       = '';
        if (selectedExprLatex) {
            try { katex.render(prepareLaTeX(selectedExprLatex), adviseModalExpr, { displayMode: true, throwOnError: false }); }
            catch (_) { adviseModalExpr.textContent = selectedExprLatex; }
        } else {
            adviseModalExpr.textContent = selectedExprId;
        }
        adviseModal.style.display = 'flex';
        adviseModalQuestion.focus();
    }

    adviseBtn.addEventListener('click', async () => {
        if (!getSelectedExprId()) return;
        if (adviserStatus === 'unavailable') {
            const ok = await loadAdviserList();
            if (!ok) return;
        }
        openAdviseModal();
    });

    adviseModalClose.addEventListener('click', () => {
        adviseModal.style.display = 'none';
    });

    adviseModal.addEventListener('click', e => {
        if (e.target === adviseModal) adviseModal.style.display = 'none';
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && adviseModal.style.display === 'flex')
            adviseModal.style.display = 'none';
    });

    adviseModalQuestion.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.isComposing) adviseModalSubmit.click();
    });

    adviseModalSubmit.addEventListener('click', async () => {
        const selectedExprId = getSelectedExprId();
        if (!selectedExprId) return;
        const question = adviseModalQuestion.value.trim();

        adviseModalSubmit.disabled      = true;
        adviseModalSubmit.textContent   = _t('ui.advising_progress');
        adviseModalResult.style.display = 'block';
        adviseModalResult.innerHTML     = `<span style="color:#888;">${_t('ui.advising_spinner')}</span>`;
        adviseSaveArea.style.display    = 'none';
        lastAdviseText                  = '';

        try {
            const res  = await fetch('/api/notebook/advise', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ expr_id: selectedExprId, question,
                                         adviser_name: adviseModalAdviser.value || null }),
            });
            const data = await res.json();
            if (data.ok) {
                const normalizedText         = normalizeMathDelimiters(data.text);
                adviseModalResult.innerHTML  = renderMathInText(normalizedText);
                lastAdviseText               = normalizedText;
                adviseSaveArea.style.display = 'block';
                adviseSaveBtn.textContent    = _t('ui.save_as_memo_btn');
                adviseSaveBtn.disabled       = false;
            } else {
                adviseModalResult.innerHTML =
                    `<span style="color:#dc3545;">❌ ${escapeHtml(data.error ?? _t('ui.advise_error'))}</span>`;
            }
        } catch (err) {
            adviseModalResult.innerHTML =
                `<span style="color:#dc3545;">❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}</span>`;
        } finally {
            adviseModalSubmit.disabled    = false;
            adviseModalSubmit.textContent = _t('ui.advise_submit_done');
        }
    });

    adviseSaveBtn.addEventListener('click', async () => {
        const selectedExprId = getSelectedExprId();
        if (!lastAdviseText || !selectedExprId) return;
        const noteText = `${_t('ui.advise_note_prefix', {expr_id: selectedExprId})}\n${lastAdviseText}`;
        adviseSaveBtn.disabled   = true;
        adviseSaveBtn.textContent = _t('ui.memo_saving');
        try {
            const res  = await fetch('/api/notebook/note', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text: noteText }),
            });
            const data = await res.json();
            if (data.ok ?? res.ok) {
                appendEntry('note', renderNoteHtml(noteText), { stepId: data.step_id });
                adviseSaveBtn.textContent = _t('ui.saved_done');
            } else {
                adviseSaveBtn.textContent = _t('ui.memo_save_failed');
                adviseSaveBtn.disabled    = false;
            }
        } catch (_) {
            adviseSaveBtn.textContent = _t('ui.memo_save_error');
            adviseSaveBtn.disabled    = false;
        }
    });

    return { loadAdviserList };
}
