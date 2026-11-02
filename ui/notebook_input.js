// ui/notebook_input.js — the input surface (R6, docs/27): input-box
// serialization, math-token insertion, and the handwriting modal.
//
// Extracted from notebook_main.js.  savedRange (cursor position across the modal)
// is module-internal now; the DOM refs are looked up here.  initInput(deps)
// self-wires the modal buttons + the iframe math-insert message, and returns the
// serialization helpers the rest of the app calls.  fullwidthToHalfwidth /
// stripLatexDelimiters come from notebook_utils.js; _t is unused here.

import { fullwidthToHalfwidth, stripLatexDelimiters } from './notebook_utils.js';

export function initInput({ sendDomainsToIframe }) {
    const inputBox    = document.getElementById('input-box');
    const hwModal     = document.getElementById('hw-modal');
    const hwBtn       = document.getElementById('hw-btn');
    const hwCancelBtn = document.getElementById('hw-cancel-btn');
    const hwOverlay   = document.getElementById('hw-modal-overlay');
    let savedRange = null;

    // ── Input serialization ───────────────────────────────────────────────

    // Normalize full-width characters in inputBox text nodes in place.
    // Skips math-token spans. Preserves cursor position (conversion is 1:1).
    function normalizeInputBoxText() {
        const sel = window.getSelection();
        for (const node of inputBox.childNodes) {
            if (node.nodeType !== Node.TEXT_NODE) continue;
            const normalized = fullwidthToHalfwidth(node.textContent);
            if (normalized === node.textContent) continue;
            let savedOffset = -1;
            if (sel?.rangeCount > 0 && sel.getRangeAt(0).startContainer === node) {
                savedOffset = sel.getRangeAt(0).startOffset;
            }
            node.textContent = normalized;
            if (savedOffset >= 0) {
                const range = document.createRange();
                range.setStart(node, Math.min(savedOffset, node.length));
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
            }
        }
    }

    function serializeInput() {
        const tokens = [];
        for (const node of inputBox.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                const value = fullwidthToHalfwidth(node.textContent);
                if (value) tokens.push({ type: 'text', value });
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                if (node.classList.contains('math-token')) {
                    tokens.push({
                        type: 'math',
                        latex:     node.dataset.latex     ?? '',
                        canonical: node.dataset.canonical ?? '',
                    });
                } else if (node.tagName === 'BR') {
                    tokens.push({ type: 'text', value: ' ' });
                } else {
                    // Nested elements (e.g. paste-inserted spans): flatten to text
                    const text = node.textContent;
                    if (text) tokens.push({ type: 'text', value: text });
                }
            }
        }
        // Merge adjacent text tokens
        return tokens.reduce((acc, tok) => {
            if (tok.type === 'text' && acc.length > 0 &&
                acc[acc.length - 1].type === 'text') {
                acc[acc.length - 1].value += tok.value;
            } else {
                acc.push(tok);
            }
            return acc;
        }, []);
    }

    function clearInput() {
        inputBox.innerHTML = '';
    }

    function isInputEmpty() {
        return serializeInput().every(
            t => t.type === 'text' && !t.value.trim()
        ) || serializeInput().length === 0;
    }

    // ── Math token insertion ──────────────────────────────────────────────

    function insertMathToken(latex, canonical) {
        // Insert as plain LaTeX text so the input box behaves the same as when
        // the user types LaTeX notation directly.  Rendering happens in the
        // timeline after the expression is submitted and processed — not in the
        // input box itself.  (Keeping canonical unused here; it travels with the
        // token once the user submits via serializeInput.)
        const clean = stripLatexDelimiters(latex);
        const node = document.createTextNode(clean + ' ');

        inputBox.focus();
        if (savedRange) {
            // Restore saved selection inside inputBox
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(savedRange);
            savedRange = null;
        }

        const sel = window.getSelection();
        if (sel && sel.rangeCount > 0) {
            const range = sel.getRangeAt(0);
            // Make sure the range is inside inputBox
            if (inputBox.contains(range.commonAncestorContainer) ||
                range.commonAncestorContainer === inputBox) {
                range.deleteContents();
                range.insertNode(node);
                range.setStartAfter(node);
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
            } else {
                inputBox.appendChild(node);
            }
        } else {
            inputBox.appendChild(node);
        }
    }

    // ── Modal ─────────────────────────────────────────────────────────────

    function openModal() {
        const sel = window.getSelection();
        savedRange = (sel && sel.rangeCount > 0)
            ? sel.getRangeAt(0).cloneRange()
            : null;
        hwModal.style.display = 'flex';
        sendDomainsToIframe();
    }

    function closeModal() {
        hwModal.style.display = 'none';
        inputBox.focus();
    }

    hwBtn.addEventListener('click', openModal);
    hwCancelBtn.addEventListener('click', closeModal);
    hwOverlay.addEventListener('click', closeModal);

    // Receive math-insert message from the iframe
    window.addEventListener('message', (e) => {
        if (e.data?.type === 'math-insert') {
            insertMathToken(e.data.latex ?? '', e.data.canonical ?? '');
            closeModal();
        }
    });

    return { serializeInput, clearInput, isInputEmpty, normalizeInputBoxText };
}
