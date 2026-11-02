// ui/notebook_persistence.js — session save/load/export/shutdown/autosave
// (R6, docs/27).  Extracted verbatim from notebook_main.js's "Save dialog" /
// "LaTeX export" / load-modal / "Shutdown" / "Autosave" sections.
//
// Dependencies come in via DI: appendEntry is the timeline DOM/state hub
// (kept in main); loadInitialSession renders a freshly-loaded session into
// the timeline (kept in main, next to the appendEntry-based renderers it
// calls); resetSelection clears main's selectedExprId/opButtons after a load.
// `_t`, `confirm`, `alert`, `prompt`, `window._reloadI18n` resolve as globals.

import { escapeHtml } from './notebook_utils.js';

export function initPersistence({ appendEntry, loadInitialSession, resetSelection }) {
    const timeline   = document.getElementById('timeline');
    const opButtons  = document.getElementById('op-buttons');

    // ── Save dialog ───────────────────────────────────────────────────────

    const saveModal          = document.getElementById('save-modal');
    const saveModalOverwrite = document.getElementById('save-modal-overwrite');
    const saveModalDivider   = document.getElementById('save-modal-divider');
    const saveModalNew       = document.getElementById('save-modal-new');
    const saveModalFilename  = document.getElementById('save-modal-filename');
    const saveModalSkip      = document.getElementById('save-modal-skip');
    const saveModalCancel    = document.getElementById('save-modal-cancel');
    const saveModalPathBar   = document.getElementById('save-modal-path-bar');
    const saveModalDirList   = document.getElementById('save-modal-dir-list');
    const saveModalMkdir     = document.getElementById('save-modal-mkdir');

    let _saveModalCallback = null;
    let _saveBrowsePath    = '';   // relative to sessions/

    function _renderSaveDirList(data) {
        _saveBrowsePath = data.path ?? '';
        const display = 'sessions/' + (_saveBrowsePath ? _saveBrowsePath + '/' : '');
        if (saveModalPathBar) saveModalPathBar.textContent = display;
        if (!saveModalDirList) return;

        const rows = [];
        if (data.can_go_up) {
            rows.push(`<div class="save-dir-row save-dir-up" data-path="${escapeHtml(data.parent ?? '')}"
                style="padding:5px 10px; cursor:pointer; color:#888; font-size:.82em; border-bottom:1px solid #f0f0f0;">
                ↑ ..</div>`);
        }
        for (const dir of (data.dirs ?? [])) {
            const full = _saveBrowsePath ? `${_saveBrowsePath}/${dir}` : dir;
            rows.push(`<div class="save-dir-row" data-path="${escapeHtml(full)}"
                style="padding:5px 10px; cursor:pointer; font-size:.85em; border-bottom:1px solid #f8f8f8;">
                📁 ${escapeHtml(dir)}</div>`);
        }
        for (const file of (data.files ?? [])) {
            const stem = file.replace(/\.json$/, '');
            rows.push(`<div class="save-file-row" data-filename="${escapeHtml(file)}"
                style="padding:5px 10px; cursor:pointer; font-size:.85em; border-bottom:1px solid #f8f8f8; color:#555;">
                📄 ${escapeHtml(stem)}</div>`);
        }
        if (!rows.length) {
            saveModalDirList.innerHTML = `<div style="padding:7px 10px; color:#bbb; font-size:.8em;"
                data-i18n="ui.save_dir_empty">（ファイルなし）</div>`;
        } else {
            saveModalDirList.innerHTML = rows.join('');
            saveModalDirList.querySelectorAll('.save-dir-row').forEach(el => {
                el.addEventListener('mouseenter', () => el.style.background = '#f5f7fa');
                el.addEventListener('mouseleave', () => el.style.background = '');
                el.addEventListener('click', async () => {
                    const qs = el.dataset.path !== undefined ? `?path=${encodeURIComponent(el.dataset.path)}` : '';
                    const d = await fetch(`/api/notebook/browse-dirs${qs}`).then(r => r.json()).catch(() => ({}));
                    _renderSaveDirList(d);
                });
            });
            saveModalDirList.querySelectorAll('.save-file-row').forEach(el => {
                el.addEventListener('mouseenter', () => el.style.background = '#f0f4ff');
                el.addEventListener('mouseleave', () => el.style.background = '');
                el.addEventListener('click', () => {
                    const stem = el.dataset.filename.replace(/\.json$/, '');
                    if (saveModalFilename) {
                        saveModalFilename.value = stem;
                        saveModalFilename.focus();
                    }
                });
            });
        }
    }

    async function openSaveModal(onDone) {
        _saveModalCallback = onDone ?? null;
        _saveBrowsePath = '';
        const [statusRes, dirRes] = await Promise.allSettled([
            fetch('/api/notebook/save-status').then(r => r.json()),
            fetch('/api/notebook/browse-dirs').then(r => r.json()),
        ]);
        const cur = statusRes.status === 'fulfilled' ? statusRes.value.current_filename : null;
        if (cur) {
            saveModalOverwrite.style.display = '';
            saveModalOverwrite.textContent = _t('ui.overwrite_btn', {file: cur});
            if (saveModalDivider) saveModalDivider.style.display = 'flex';
            // Pre-navigate dir browser to the folder of the current file
            const curDir = cur.includes('/') ? cur.substring(0, cur.lastIndexOf('/')) : '';
            const initPath = curDir || '';
            const qs = initPath ? `?path=${encodeURIComponent(initPath)}` : '';
            const d = initPath
                ? await fetch(`/api/notebook/browse-dirs${qs}`).then(r => r.json()).catch(() => dirRes.value ?? {})
                : (dirRes.status === 'fulfilled' ? dirRes.value : {});
            _renderSaveDirList(d);
        } else {
            saveModalOverwrite.style.display = 'none';
            if (saveModalDivider) saveModalDivider.style.display = 'none';
            if (dirRes.status === 'fulfilled') _renderSaveDirList(dirRes.value);
        }
        saveModalFilename.value = '';
        saveModal.style.display = 'flex';
        saveModalFilename.focus();
    }

    async function doSave(overwrite) {
        saveModal.style.display = 'none';
        const userFilename = saveModalFilename.value.trim();
        const btn = document.getElementById('save-btn');
        if (btn) { btn.disabled = true; btn.textContent = _t('ui.saving'); }
        try {
            const res  = await fetch('/api/notebook/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    overwrite,
                    filename:  overwrite ? '' : userFilename,
                    directory: overwrite ? '' : _saveBrowsePath,
                }),
            });
            const data = await res.json();
            if (data.ok) {
                if (btn) {
                    btn.textContent = _t('ui.saved_done');
                    setTimeout(() => { btn.textContent = _t('ui.save_btn'); btn.disabled = false; }, 2000);
                }
                if (_saveModalCallback) _saveModalCallback(true, data.filename);
            } else {
                alert(_t('ui.save_failed') + ': ' + (data.detail || _t('ui.unknown_error')));
                if (btn) { btn.textContent = _t('ui.save_btn'); btn.disabled = false; }
                if (_saveModalCallback) _saveModalCallback(false);
            }
        } catch (e) {
            alert(_t('ui.network_error') + ': ' + e);
            if (btn) { btn.textContent = _t('ui.save_btn'); btn.disabled = false; }
            if (_saveModalCallback) _saveModalCallback(false);
        }
    }

    saveModalOverwrite?.addEventListener('click', () => doSave(true));
    saveModalNew?.addEventListener('click',       () => doSave(false));
    saveModalFilename?.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSave(false); });
    saveModalSkip?.addEventListener('click', () => {
        saveModal.style.display = 'none';
        if (_saveModalCallback) _saveModalCallback(false);
    });
    saveModalCancel?.addEventListener('click', () => {
        saveModal.style.display = 'none';
        _saveModalCallback = null;
    });
    saveModalMkdir?.addEventListener('click', async () => {
        const name = prompt(_t('ui.save_new_folder_prompt') || 'フォルダ名:');
        if (!name?.trim()) return;
        const newPath = _saveBrowsePath ? `${_saveBrowsePath}/${name.trim()}` : name.trim();
        await fetch('/api/notebook/mkdir', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: newPath}),
        }).catch(() => {});
        const qs = _saveBrowsePath ? `?path=${encodeURIComponent(_saveBrowsePath)}` : '';
        const d = await fetch(`/api/notebook/browse-dirs${qs}`).then(r => r.json()).catch(() => ({}));
        _renderSaveDirList(d);
    });

    document.getElementById('save-btn')?.addEventListener('click', () => openSaveModal(null));

    // ── LaTeX export (standalone header button) ───────────────────────────

    async function doExportLatex() {
        try {
            const res = await fetch('/api/notebook/export/latex');
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            if (data.ok) {
                appendEntry('system', _t('ui.latex_exported', {path: data.path}));
            } else {
                appendEntry('supervisor', `❌ ${escapeHtml(data.detail ?? _t('ui.unknown_error'))}`);
            }
        } catch (err) {
            appendEntry('supervisor', `❌ ${_t('ui.network_error')}: ${escapeHtml(String(err))}`);
        }
    }
    document.getElementById('export-latex-btn')?.addEventListener('click', doExportLatex);

    const loadModal      = document.getElementById('load-modal');
    const sessionList    = document.getElementById('session-list');
    const loadModalClose = document.getElementById('load-modal-close');
    const loadPathBar    = document.getElementById('load-modal-path-bar');
    const loadDirList    = document.getElementById('load-modal-dir-list');

    let _loadBrowsePath = '';

    function _renderLoadDirList(data) {
        _loadBrowsePath = data.path ?? '';
        const display = 'sessions/' + (_loadBrowsePath ? _loadBrowsePath + '/' : '');
        if (loadPathBar) loadPathBar.textContent = display;
        if (!loadDirList) return;
        const rows = [];
        if (data.can_go_up) {
            rows.push(`<div class="load-dir-row" data-path="${escapeHtml(data.parent ?? '')}"
                style="padding:4px 10px; cursor:pointer; color:#888; font-size:.82em; border-bottom:1px solid #f0f0f0;">
                ↑ ..</div>`);
        }
        for (const dir of (data.dirs ?? [])) {
            const full = _loadBrowsePath ? `${_loadBrowsePath}/${dir}` : dir;
            const isAutosave = dir === 'autosave';
            const icon = isAutosave ? '🔄' : '📁';
            const style = isAutosave ? 'color:#b06000;' : '';
            rows.push(`<div class="load-dir-row" data-path="${escapeHtml(full)}"
                style="padding:4px 10px; cursor:pointer; font-size:.85em; border-bottom:1px solid #f8f8f8; ${style}">
                ${icon} ${escapeHtml(dir)}</div>`);
        }
        if (!rows.length) {
            loadDirList.innerHTML = `<div style="padding:5px 10px; color:#bbb; font-size:.8em;"
                data-i18n="ui.save_dir_empty">（サブフォルダなし）</div>`;
        } else {
            loadDirList.innerHTML = rows.join('');
            loadDirList.querySelectorAll('.load-dir-row').forEach(el => {
                el.addEventListener('mouseenter', () => el.style.background = '#f5f7fa');
                el.addEventListener('mouseleave', () => el.style.background = '');
                el.addEventListener('click', () => _loadNavigate(el.dataset.path ?? ''));
            });
        }
    }

    async function _loadNavigate(path) {
        sessionList.innerHTML = `<div style="color:#888;">${_t('ui.session_list_loading')}</div>`;
        const qs = path ? `?path=${encodeURIComponent(path)}` : '';
        const [dirRes, sessRes] = await Promise.allSettled([
            fetch(`/api/notebook/browse-dirs${qs}`).then(r => r.json()),
            fetch(`/api/notebook/sessions${qs}`).then(r => r.json()),
        ]);
        if (dirRes.status === 'fulfilled') _renderLoadDirList(dirRes.value);
        _renderSessionList(sessRes.status === 'fulfilled' ? sessRes.value.sessions ?? [] : []);
    }

    function _renderSessionList(sessions) {
        if (!sessions.length) {
            sessionList.innerHTML = `<div style="color:#888; font-size:.85em; padding:6px 2px;">${_t('ui.session_list_empty')}</div>`;
            return;
        }
        sessionList.innerHTML = '';
        for (const s of sessions) {
            const item = document.createElement('div');
            item.style.cssText = 'display:flex; justify-content:space-between; align-items:center; gap:12px; padding:8px 12px; border:1px solid #ddd; border-radius:6px;';
            const infoText = _t('ui.session_info', {date: s.modified_at, exprs: s.expr_count, steps: s.step_count});
            const basePart = s.filename.includes('/') ? s.filename.substring(s.filename.lastIndexOf('/') + 1) : s.filename;
            item.innerHTML =
                `<div style="min-width:0; overflow:hidden;">
                    <div style="font-weight:600; font-family:monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(basePart)}</div>
                    <div style="font-size:.85em; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(s.title)}</div>
                    <div style="font-size:.78em; color:#888; margin-top:2px;">${escapeHtml(infoText)}</div>
                </div>
                <button style="padding:4px 12px; flex-shrink:0;">${_t('ui.load_item_btn')}</button>`;
            item.querySelector('button').addEventListener('click', () => doLoad(s.filename));
            sessionList.appendChild(item);
        }
    }

    document.getElementById('load-btn')?.addEventListener('click', async () => {
        _loadBrowsePath = '';
        sessionList.innerHTML = `<div style="color:#888;">${_t('ui.session_list_loading')}</div>`;
        loadModal.style.display = 'flex';
        await _loadNavigate('');
    });

    loadModalClose?.addEventListener('click', () => { loadModal.style.display = 'none'; });
    loadModal?.addEventListener('click', (e) => { if (e.target === loadModal) loadModal.style.display = 'none'; });

    async function doLoad(filename) {
        if (!confirm(_t('ui.confirm_load', {filename}))) return;
        loadModal.style.display = 'none';
        appendEntry('system', _t('ui.loading_session'));
        try {
            const res  = await fetch('/api/notebook/load', { method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.detail || _t('ui.unknown_error'));
            if (window._reloadI18n) await window._reloadI18n();

            // Rebuild timeline from loaded session
            timeline.innerHTML = '';
            resetSelection();
            await loadInitialSession();
            const count = Object.keys(data.expressions).length;
            appendEntry('system', _t('ui.load_complete', {title: data.title, count}));
        } catch (e) {
            appendEntry('system', _t('ui.load_failed', {error: String(e)}));
        }
    }

    // ── Shutdown ──────────────────────────────────────────────────────────

    document.getElementById('shutdown-btn')?.addEventListener('click', () => {
        // Show save dialog first; after save (or skip), proceed to exit
        openSaveModal(async (_saved) => {
            await fetch('/api/shutdown', { method: 'POST' }).catch(() => {});
            window.close();
        });
        // Replace skip label with "exit without saving" for exit context
        saveModalSkip.textContent = _t('ui.skip_save_exit');
        // Restore label when modal is dismissed without exit
        const restoreLabel = () => { saveModalSkip.textContent = _t('ui.skip_save_continue'); };
        saveModalCancel.addEventListener('click', restoreLabel, { once: true });
    });

    // ── Autosave ──────────────────────────────────────────────────────────

    async function triggerAutosave() {
        try {
            const res  = await fetch('/api/notebook/autosave', { method: 'POST' });
            const data = await res.json();
            if (data.ok && !data.skipped) {
                const ind = document.getElementById('autosave-indicator');
                if (ind) {
                    ind.textContent = _t('ui.autosave_done');
                    ind.style.opacity = '1';
                    setTimeout(() => { ind.style.opacity = '0'; }, 2000);
                }
            }
        } catch (_) {}
    }

    // Periodic autosave (every 2 min)
    setInterval(triggerAutosave, 2 * 60 * 1000);

    return { triggerAutosave, openSaveModal };
}
