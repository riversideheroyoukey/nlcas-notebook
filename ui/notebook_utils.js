// ui/notebook_utils.js — pure presentation helpers shared across the notebook
// front-end modules (R6, docs/27).  Extracted verbatim from notebook_main.js's
// former "Utilities" section; no shared state, only the global `katex` (loaded
// via <script> before the module graph runs).

// Convert full-width ASCII variants and math-only Unicode symbols to half-width.
// Mirrors _fullwidth_to_halfwidth() in notebook_bridge.py.
export function fullwidthToHalfwidth(text) {
    let out = '';
    for (const ch of text) {
        const cp = ch.codePointAt(0);
        if (cp >= 0xFF01 && cp <= 0xFF5E) {
            out += String.fromCodePoint(cp - 0xFEE0);
        } else if (cp === 0x3000) {
            out += ' ';
        } else if (cp === 0x2212) {   // MINUS SIGN −
            out += '-';
        } else {
            out += ch;
        }
    }
    // ー/–/— → '-' only when flanked by math characters
    return out.replace(/(?<=[0-9A-Za-z.)^*/])[–—ー](?=[0-9A-Za-z.(])/g, '-');
}

export function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// For embedding JSON in single-quoted HTML attributes: must also escape apostrophes.
export function escapeJsonAttr(obj) {
    return JSON.stringify(obj)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/'/g, '&#39;');
}

export function stripLatexDelimiters(s) {
    return s
        .replace(/^\$\$([\s\S]*?)\$\$$/, '$1')
        .replace(/^\\\[([\s\S]*?)\\\]$/,  '$1')
        .replace(/^\\\(([\s\S]*?)\\\)$/,  '$1')
        .replace(/^\$([^$\n]+)\$$/,        '$1')
        .replace(/^```[a-z]*\n?([\s\S]*?)\n?```$/, '$1')
        .trim();
}

// Minimal pre-processing before passing stored LaTeX to KaTeX.
//
// ARCHITECTURE NOTE: All semantic math transformations (function names,
// multiplication symbols, ^(...) superscripts, \rm, etc.) are the
// responsibility of the Python backend layer (BackendHandler.normalize_display_latex
// for user input; BackendHandler.postprocess_result_latex for CAS output).
// JS must NOT do semantic math work — that would violate the Backend
// Encapsulation Rule and break backend-agnostic rendering.
//
// The transforms below are LEGACY COMPATIBILITY only: they handle expressions
// stored in sessions before the Python normalization layer was in place.
// They are intentionally kept narrow so that properly-normalised LaTeX
// (stored by modern code) passes through unchanged.
export function prepareLaTeX(s) {
    // Legacy: strip surrounding math-mode delimiters left in old stored data
    s = s.replace(/^\s*\$\$\s*([\s\S]*?)\s*\$\$\s*$/, '$1');
    s = s.replace(/^\s*\$\s*([\s\S]*?)\s*\$\s*$/, '$1');
    s = s.replace(/^\s*\\\[\s*([\s\S]*?)\s*\\\]\s*$/, '$1');

    // Legacy: ^(expr) → ^{expr}  (Maxima canonical stored before normalization)
    let out = '';
    const stack = [];
    for (let i = 0; i < s.length; i++) {
        if (s[i] === '^' && s[i + 1] === '(') {
            out += '^{'; stack.push('exp'); i++;
        } else if (s[i] === '(') {
            out += '('; stack.push('paren');
        } else if (s[i] === ')') {
            out += (stack.length && stack[stack.length - 1] === 'exp') ? '}' : ')';
            stack.pop();
        } else { out += s[i]; }
    }

    // Legacy: protect ^* before blanket * removal
    out = out.replace(/\^\{\s*\*\s*\}/g, '^{\\ast}');
    out = out.replace(/\^\*/g, '^{\\ast}');
    // Legacy: bare * → space for OLD STORED Maxima canonical expressions (a*b style).
    // NOTE: This intentionally does NOT run on new data because normalize_display_latex
    // (Python) now preserves * as-is — * means convolution in signal processing,
    // Hodge star in differential geometry, etc.  New stored LaTeX keeps * verbatim.
    out = out.replace(/(?<!\\)\*/g, ' ');

    // Legacy: bare function names → \commands (old stored un-prefixed names)
    out = out.replace(
        /(?<!\\)\b(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|log|ln|exp|det|lim)\b(?=\s*[({a-zA-Z%])/g,
        '\\$1'
    );
    // Legacy: sqrt(x) / sqrt{x} → \sqrt{x}
    // (?<!\\) guards against an already-escaped \sqrt being turned into
    // \\sqrt (which KaTeX renders as a line break + literal "sqrt").
    // NOTE: mirrored on the backend in
    // notebook/runtime/backend_handler.py normalize_display_latex (step 7).
    // Keep both copies in sync when changing this rule.
    out = out.replace(/(?<!\\)\bsqrt\{([^{}]*)\}/g, '\\sqrt{$1}');
    out = out.replace(/(?<!\\)\bsqrt\(([^()]*)\)/g, '\\sqrt{$1}');
    // Legacy: \mbox → \text (old stored Maxima output before postprocess fixed it)
    out = out.replace(/\\mbox\{([^{}]*)\}/g, '\\text{$1}');

    return out;
}

export function renderKatex(latex, displayMode = false) {
    const el = document.createElement(displayMode ? 'div' : 'span');
    el.className = 'cas-math';
    try {
        katex.render(prepareLaTeX(latex), el, { displayMode, throwOnError: false });
    } catch (_) {
        el.textContent = latex;
    }
    return el;
}
