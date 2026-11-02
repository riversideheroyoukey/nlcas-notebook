/* Shared front-end i18n bootstrap.
 *
 * Defines window._t(key, vars) and applies [data-i18n] / [data-i18n-ph] from
 * i18n/ui_messages.json, fetched through /api/notebook/ui-lang.
 *
 * Both pages load this — notebook.html and the handwriting page (index.html,
 * which runs standalone AND inside the notebook's iframe).  It lives in one
 * file because the handwriting page had no i18n at all: every label there was
 * hardcoded, mostly in English with a Japanese `✓ 挿入` among them, so an
 * English session showed one Japanese button.  Copying the loader into that
 * page would have fixed today's symptom and left two loaders to drift.
 *
 * A CLASSIC script, deliberately, and loaded before the page's module: classic
 * scripts run in source order and modules are deferred, so `window._t` exists
 * by the time any module body runs.  Loading this as a module would reverse
 * that.
 *
 * Translations arrive asynchronously; the markup therefore carries the Japanese
 * text as its literal content, which is what shows for the instant before the
 * fetch returns (and if the fetch fails).  Keep the literal and the `ja` entry
 * in ui_messages.json in step.
 */
(function () {
    let _msgs = {};
    let _lang = 'ja';
    window._lang = 'ja';

    window._t = function (key, vars) {
        const entry = _msgs[key];
        if (!entry) return key;
        const tpl = entry[_lang] || entry['en'] || key;
        if (!vars) return tpl;
        return tpl.replace(/\{(\w+)\}/g, function (_, k) {
            return vars[k] !== undefined ? vars[k] : '';
        });
    };

    function applyI18n() {
        document.documentElement.lang = _lang;
        document.querySelectorAll('[data-i18n]').forEach(function (el) {
            el.textContent = window._t(el.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-ph]').forEach(function (el) {
            const val = window._t(el.dataset.i18nPh);
            el.placeholder = val;
            if (el.dataset.placeholder !== undefined) {
                el.dataset.placeholder = val;
            }
        });
        // Tooltips.  Markup had no way to key one, so the four written directly
        // in the handwriting page's HTML were fixed text whatever the language
        // — three Japanese, one English.  (The tooltips built in script were
        // always fine; those go through _t() at render time.)
        document.querySelectorAll('[data-i18n-title]').forEach(function (el) {
            el.title = window._t(el.dataset.i18nTitle);
        });
        // Labels a page sets from script (a toggle's caption, a status line)
        // cannot be re-read from the DOM, so each page re-renders its own.
        if (typeof window._onI18nApplied === 'function') window._onI18nApplied();
    }

    function _fetchAndApply() {
        return fetch('/api/notebook/ui-lang')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                _msgs = data.messages || {};
                _lang = data.lang || 'ja';
                window._lang = _lang;
                applyI18n();
            })
            .catch(function () {});
    }

    window._reloadI18n = _fetchAndApply;

    _fetchAndApply();
})();
