// ui/notebook_branch.js — Live branch (in-session fork) UI: a branch
// indicator + switcher in the header, and a per-timeline-entry "fork here"
// button (🌿) distinct from the existing file-snapshot fork (💾) button.
//
// Unlike /fork (file-snapshot save as a new file, wired in notebook_main.js's
// own fork-modal block), this creates a SECOND branch that coexists with
// others in the SAME loaded session — required for numeric_branch_comparison()
// to ever have two branches to compare. See docs/developer_manuals/02_history_spec.md §11.

export function initBranch({ appendEntry, loadInitialSession, resetSelection }) {
    const timeline = document.getElementById('timeline');

    const branchLabel    = document.getElementById('active-branch-label');
    const branchSwitcher = document.getElementById('branch-switcher');

    const liveBranchModal        = document.getElementById('live-branch-modal');
    const liveBranchModalLabel   = document.getElementById('live-branch-modal-label');
    const liveBranchModalOk      = document.getElementById('live-branch-modal-ok');
    const liveBranchModalCancel  = document.getElementById('live-branch-modal-cancel');
    const liveBranchModalCancel2 = document.getElementById('live-branch-modal-cancel2');

    const closeLiveBranchModal = () => { liveBranchModal.style.display = 'none'; };
    liveBranchModalCancel?.addEventListener('click', closeLiveBranchModal);
    liveBranchModalCancel2?.addEventListener('click', closeLiveBranchModal);

    async function _reloadTimeline() {
        timeline.innerHTML = '';
        resetSelection();
        await loadInitialSession();
    }

    let _liveForkStepId = null;
    function liveForkAtStep(stepId) {
        _liveForkStepId = stepId;
        liveBranchModalLabel.value = '';
        liveBranchModal.style.display = 'flex';
        liveBranchModalLabel.focus();
    }

    liveBranchModalOk?.addEventListener('click', async () => {
        const stepId = _liveForkStepId;
        const value  = liveBranchModalLabel.value.trim();
        closeLiveBranchModal();
        try {
            const res  = await fetch('/api/notebook/branch/fork', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ step_id: stepId, label: value }),
            });
            const data = await res.json();
            if (data.ok) {
                await _reloadTimeline();
                appendEntry('system', _t('ui.branch_forked_toast', {branch_id: data.branch_id}));
            }
        } catch (_) {}
    });

    // Attach a live-fork button to a timeline entry that has a step_id.
    function attachLiveForkButton(entry) {
        if (!entry.dataset.stepId) return;
        const btn = document.createElement('button');
        btn.className   = 'live-branch-fork-btn';
        btn.title       = _t('ui.branch_fork_step_title');
        btn.textContent = '🌿';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            liveForkAtStep(entry.dataset.stepId);
        });
        const removeBtn = entry.querySelector('.cas-remove-btn');
        if (removeBtn) removeBtn.before(btn);
        else {
            const idBadge = entry.querySelector('.expr-id');
            if (idBadge) idBadge.after(btn);
            else entry.appendChild(btn);
        }
    }

    // Populate the branch indicator + switcher from GET /session's response.
    function refreshBranchList(sessionData) {
        const branches = sessionData.branches || {};
        const active   = sessionData.active_branch || 'main';
        if (branchLabel) {
            branchLabel.textContent = `${_t('ui.active_branch_prefix')}: ${active}`;
        }
        if (branchSwitcher) {
            branchSwitcher.innerHTML = '';
            Object.keys(branches).forEach(bid => {
                const opt = document.createElement('option');
                opt.value = bid;
                opt.textContent = branches[bid].label ? `${bid} (${branches[bid].label})` : bid;
                if (bid === active) opt.selected = true;
                branchSwitcher.appendChild(opt);
            });
        }
    }

    branchSwitcher?.addEventListener('change', async () => {
        const branchId = branchSwitcher.value;
        try {
            const res  = await fetch('/api/notebook/branch/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ branch_id: branchId }),
            });
            const data = await res.json();
            if (data.ok) await _reloadTimeline();
        } catch (_) {}
    });

    return { attachLiveForkButton, refreshBranchList };
}
