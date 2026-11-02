// ui/main.js
import { importMathToCanvas } from './canvas_utils.js';

// window._t comes from ui_i18n.js (a classic script, so it is already defined
// when this module body runs).  The fallback keeps the page working if that
// file is ever not loaded — a key on screen, not a crash.
const _t = (k, v) => (window._t ? window._t(k, v) : k);

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('draw-canvas');
    const engineSelect = document.getElementById('engine-select');
    const refinerSelect = document.getElementById('refiner-select');
    const recognizeBtn = document.getElementById('recognize-btn');
    const modeBtn = document.getElementById('mode-btn');
    const clearBtn = document.getElementById('clear-btn');
    const editBtn = document.getElementById('edit-on-canvas-btn');
    const zoomInput = document.getElementById('math-zoom');
    const resultArea = document.getElementById('result-area-content');
    const mathDiv = document.getElementById('math');
    
    // 第3の領域用
    const latexInput = document.getElementById('latex-input');
    const typesetBtn = document.getElementById('typeset-btn');
    const addToNotebookBtn = document.getElementById('add-to-notebook-btn');
    const notebookPanel = document.getElementById('notebook-panel');
    const notebookExpressions = document.getElementById('notebook-expressions');
    const notebookClearBtn = document.getElementById('notebook-clear-btn');

    if (!canvas) return;
    // desynchronized: true bypasses VSync compositing, eliminating ~16ms of input-display lag
    const ctx = canvas.getContext('2d', { desynchronized: true });
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // ── Domain hints received from parent notebook ───────────────────────
    let _domains = [];
    window.addEventListener('message', (e) => {
        if (e.data?.type === 'set_domains') {
            _domains = e.data.domains ?? [];
        }
    });

    // ── Modal-context detection ──────────────────────────────────────────
    // When loaded inside the notebook's handwriting modal, show Insert and
    // hide Exit so the user returns to the notebook rather than closing the tab.
    const isInModal = window !== window.top;
    if (isInModal) {
        const insertBtn       = document.getElementById('insert-btn');
        const shutdownBtn     = document.getElementById('shutdown-btn');
        const addToNotebookBtn = document.getElementById('add-to-notebook-btn');
        // '' (not a display type) — the stylesheet decides how a toolbar button
        // lays out; naming one here would silently opt this button out of the
        // row's equal-height rule.
        if (insertBtn)        insertBtn.style.display        = '';
        if (shutdownBtn)      shutdownBtn.style.display      = 'none';
        if (addToNotebookBtn) addToNotebookBtn.style.display = 'none';
    }

    // Insert button: send current LaTeX (and canonical, if available) to parent
    document.getElementById('insert-btn')?.addEventListener('click', () => {
        const latex = latexInput?.value?.trim() ?? '';
        if (!latex) return;
        const payload = {
            type:      'math-insert',
            latex:     latex,
            canonical: '',   // server resolves via latex_to_canonical()
        };
        if (isInModal) {
            window.parent.postMessage(payload, '*');
        } else {
            // Standalone fallback: copy to clipboard
            navigator.clipboard?.writeText(latex).catch(() => {});
        }
    });

    // ── Template palette (open_items I-1) ─────────────────────────────────
    // A lightweight, dependency-free template-input system integrated into the
    // handwriting screen's LaTeX Editor.  Palette buttons insert LaTeX
    // skeletons into #latex-input; empty {} are placeholders the caret jumps
    // into, and Tab cycles to the next one.  Preview reuses the existing
    // Typeset path (KaTeX), and ✓挿入 / + Notebook already ship the textarea's
    // LaTeX downstream — so this adds a third authoring mode with no change to
    // the merge point or the 3-layer compiler.  Template LaTeX is authored by
    // us (math_templates.json), so there are no output quirks to normalise.
    const tmplPalette = document.getElementById('tmpl-palette');

    function tmplTypeset() {
        if (typesetBtn) typesetBtn.click();
    }

    function tmplInsertSkeleton(latex) {
        if (!latexInput) return;
        const start = latexInput.selectionStart ?? latexInput.value.length;
        const end   = latexInput.selectionEnd   ?? start;
        const v     = latexInput.value;
        latexInput.value = v.slice(0, start) + latex + v.slice(end);
        const localGap = latex.indexOf('{}');
        const pos = (localGap >= 0) ? start + localGap + 1 : start + latex.length;
        latexInput.focus();
        latexInput.setSelectionRange(pos, pos);
        tmplTypeset();
    }

    // Move the caret into the next empty "{}" at or after `from`.
    function tmplFocusNextPlaceholder(from) {
        if (!latexInput) return;
        const v = latexInput.value;
        const idx = v.indexOf('{}', from);
        latexInput.focus();
        if (idx >= 0) latexInput.setSelectionRange(idx + 1, idx + 1);
        else          latexInput.setSelectionRange(v.length, v.length);
    }

    if (latexInput) {
        latexInput.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                tmplFocusNextPlaceholder(latexInput.selectionEnd ?? 0);
            }
        });
    }

    function tmplBuildPalette(groups) {
        if (!tmplPalette) return;
        tmplPalette.innerHTML = '';
        groups.forEach((g, gi) => {
            if (gi > 0) {
                const sep = document.createElement('span');
                sep.className = 'tmpl-sep';
                tmplPalette.appendChild(sep);
            }
            g.items.forEach((it) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'tmpl-key';
                b.textContent = it.label;
                b.title = it.latex;          // technical hint, not translatable copy
                b.addEventListener('click', () => tmplInsertSkeleton(it.latex));
                tmplPalette.appendChild(b);
            });
        });
    }

    (async function tmplLoadPalette() {
        if (!tmplPalette) return;
        try {
            const r = await fetch('math_templates.json');
            const data = await r.json();
            tmplBuildPalette(data.groups || []);
        } catch (_) { /* palette is optional; ignore load failure */ }
        // Append the display-zoom buttons to the end of the palette row, after a
        // divider, so they line up with the Greek/operator keys.  Moving the
        // existing element keeps its IDs and event listeners intact.
        const zoomGroup = document.getElementById('math-display-zoom');
        if (zoomGroup && tmplPalette) {
            const sep = document.createElement('span');
            sep.className = 'tmpl-sep';
            tmplPalette.appendChild(sep);
            tmplPalette.appendChild(zoomGroup);
            zoomGroup.style.marginTop = '0';
        }
    })();

    // ── LaTeX Result display zoom ─────────────────────────────────────────
    // The rendered result can be taller/wider than its panel (matrices, tall
    // fractions).  #result-area now scrolls and is user-resizable; these
    // buttons additionally scale the #math font so the user can fit or magnify
    // the result without re-running OCR.
    let _mathScale = 1;
    const MATH_SCALE_MIN = 0.5, MATH_SCALE_MAX = 4.0, MATH_SCALE_STEP = 0.25;
    const mathZoomResetBtn = document.getElementById('math-zoom-reset');
    function applyMathScale() {
        if (mathDiv) mathDiv.style.setProperty('--math-scale', String(_mathScale));
        if (mathZoomResetBtn) mathZoomResetBtn.textContent = Math.round(_mathScale * 100) + '%';
    }
    function bumpMathScale(delta) {
        _mathScale = Math.min(MATH_SCALE_MAX,
                              Math.max(MATH_SCALE_MIN,
                                       Math.round((_mathScale + delta) / MATH_SCALE_STEP) * MATH_SCALE_STEP));
        applyMathScale();
    }
    document.getElementById('math-zoom-in') ?.addEventListener('click', () => bumpMathScale( MATH_SCALE_STEP));
    document.getElementById('math-zoom-out')?.addEventListener('click', () => bumpMathScale(-MATH_SCALE_STEP));
    mathZoomResetBtn?.addEventListener('click', () => { _mathScale = 1; applyMathScale(); });
    applyMathScale();

    function stripLatexDelimiters(s) {
        return s
            .replace(/^\$\$([\s\S]*?)\$\$$/, '$1')
            .replace(/^\\\[([\s\S]*?)\\\]$/, '$1')
            .replace(/^\\\(([\s\S]*?)\\\)$/, '$1')
            .replace(/^\$([^$\n]+)\$$/, '$1')
            .replace(/^```[a-z]*\n?([\s\S]*?)\n?```$/, '$1')
            .trim();
    }

    const eraserCursor = document.createElement('div');
    eraserCursor.id = 'eraser-cursor';
    document.body.appendChild(eraserCursor);

    let isDrawing = false;
    let drawingPointerId = null;   // track which pointer started the stroke
    let isEraser = false;
    let lastX = 0;
    let lastY = 0;
    const penWidth = 4;
    const eraserWidth = 20;

    // Prevent browser scroll/zoom on the canvas (critical for stylus input)
    canvas.style.touchAction = 'none';

    // Refreshed once per stroke (on pointerdown) so coalesced events share the same rect.
    let canvasRect = canvas.getBoundingClientRect();

    function clearCanvas() {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function getCoords(e) {
        return {
            x: (e.clientX - canvasRect.left) * (canvas.width  / canvasRect.width),
            y: (e.clientY - canvasRect.top)  * (canvas.height / canvasRect.height)
        };
    }

    function drawSegment(fromX, fromY, toX, toY) {
        ctx.beginPath();
        ctx.globalCompositeOperation = 'source-over';
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        if (isEraser) {
            ctx.strokeStyle = "white";
            ctx.lineWidth = eraserWidth;
        } else {
            ctx.strokeStyle = "#000";
            ctx.lineWidth = penWidth;
        }
        ctx.stroke();
    }

    function updateMode(forceEraser = null) {
        isEraser = (forceEraser !== null) ? forceEraser : !isEraser;
        if (modeBtn) modeBtn.innerText = _t(isEraser ? 'ui.hwpage_pen'
                                                     : 'ui.hwpage_eraser');
        canvas.style.cursor = isEraser ? 'none' : 'crosshair';
        eraserCursor.style.display = isEraser ? 'block' : 'none';
    }

    modeBtn?.addEventListener('click', () => updateMode());
    // The mode button's caption depends on STATE, so [data-i18n] alone would be
    // wrong: translations arrive asynchronously, and if they land after the user
    // has switched to the eraser, applyI18n() would relabel it back to "Eraser"
    // while the eraser is active.  Re-render it from the state instead.
    window._onI18nApplied = () => updateMode(isEraser);
    canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); }, false);

    // Eraser cursor overlay
    window.addEventListener('pointermove', (e) => {
        if (isEraser) {
            eraserCursor.style.width  = `${eraserWidth}px`;
            eraserCursor.style.height = `${eraserWidth}px`;
            eraserCursor.style.left   = `${e.clientX - eraserWidth / 2}px`;
            eraserCursor.style.top    = `${e.clientY - eraserWidth / 2}px`;
            eraserCursor.style.display = 'block';
        } else {
            eraserCursor.style.display = 'none';
        }
    });

    canvas.addEventListener('pointerdown', (e) => {
        if (e.button === 2) { e.preventDefault(); updateMode(); return; }
        if (drawingPointerId !== null) {
            // Pen always preempts an ongoing touch/mouse stroke (palm-before-pen case).
            // For a second touch or mouse while already drawing: ignore.
            if (e.pointerType !== 'pen') return;
            isDrawing = false;
            drawingPointerId = null;
        }
        drawingPointerId = e.pointerId;
        canvasRect = canvas.getBoundingClientRect(); // refresh rect at stroke start
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {} // keep events even if pointer leaves canvas
        isDrawing = true;
        const { x, y } = getCoords(e);
        [lastX, lastY] = [x, y];
        ctx.beginPath();
        ctx.globalCompositeOperation = 'source-over';
        if (isEraser) {
            ctx.fillStyle = "white";
            ctx.arc(x, y, eraserWidth / 2, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.fillStyle = "#000";
            ctx.arc(x, y, penWidth / 2, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    canvas.addEventListener('pointermove', (e) => {
        if (!isDrawing || e.pointerId !== drawingPointerId) return;
        e.preventDefault();
        // getCoalescedEvents() recovers sub-frame samples from high-rate digitizers
        const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
        for (const evt of events) {
            const { x, y } = getCoords(evt);
            drawSegment(lastX, lastY, x, y);
            [lastX, lastY] = [x, y];
        }
    });

    const _stopDrawing = (e) => {
        if (e.pointerId === drawingPointerId) { isDrawing = false; drawingPointerId = null; }
    };
    canvas.addEventListener('pointerup',     _stopDrawing);
    canvas.addEventListener('pointercancel', _stopDrawing);
    window.addEventListener('pointerup',     _stopDrawing);

    // 清書ボタンの処理
    if (typesetBtn) {
        typesetBtn.addEventListener('click', () => {
            const raw = latexInput.value;
            const currentLatex = stripLatexDelimiters(raw);
            if (currentLatex !== raw) latexInput.value = currentLatex;

            if (mathDiv && currentLatex) {
                try {
                    katex.render(currentLatex, mathDiv, {
                        displayMode: true,
                        throwOnError: false
                    });
                } catch (err) {
                    console.error("KaTeX rendering error:", err);
                }
            }
        });
    } else {
        console.error("Typeset button NOT found in DOM.");
    }

    recognizeBtn?.addEventListener('click', async () => {
        // --- 1. 思考中（実行中）の状態にする ---
        const originalColor = recognizeBtn.style.background; // 元の色を保存
        const originalText = recognizeBtn.innerText;
    
        recognizeBtn.disabled = true; // ボタンを押せないようにする
        recognizeBtn.style.background = "#f39c12"; // 思考中はオレンジ色（またはお好みの色）に
        recognizeBtn.innerText = "Processing..."; // テキストも変更        
        
        if (mathDiv) mathDiv.innerHTML = ''; 
        if (resultArea) resultArea.textContent = _t('ui.hwpage_processing');
        if (editBtn) editBtn.style.display = 'none';
        const zoomLabel = document.getElementById('zoom-label');
        if (zoomLabel) zoomLabel.style.display = 'none';

        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;

        const dataUrl = canvas.toDataURL('image/png');
        const imageData = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
        
        try {
            const res = await fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData, engine_id: engineSelect.value, refiner_id: refinerSelect.value, domains: _domains })
            });
            const { task_id } = await res.json();
            
            const poll = setInterval(async () => {
                try {
                    const r = await fetch(`/api/vlm_result/${task_id}`);
                    const d = await r.json();

                    if (d.status === 'completed' || d.status === 'failed') {
                        // 処理が完了（成功または失敗）したのでタイマーを止める
                        clearInterval(poll);

                        // --- 1. ボタンの状態を元に戻す ---
                        recognizeBtn.disabled = false;
                        recognizeBtn.style.background = ""; // style.cssの元の色（緑）に戻る
                        recognizeBtn.innerText = originalText;

                        if (d.status === 'completed') {
                            const recognizedLatex = d.result.recognized_latex;

                            // Strip delimiters that VLMs sometimes wrap around LaTeX
                            const cleanLatex = stripLatexDelimiters(recognizedLatex);

                            // --- 2. エディタ（第3の領域）に流し込む ---
                            if (latexInput) {
                                latexInput.value = cleanLatex;
                            }

                            // --- 3. 清書表示（第2の領域）を更新 ---
                            if (mathDiv) {
                                katex.render(cleanLatex, mathDiv, {
                                    displayMode: true,
                                    throwOnError: false
                                });
                            }

                            // --- 4. 編集用ボタン類の表示 ---
                            if (editBtn) {
                                // '' so .panel-action's own display applies
                                editBtn.style.display = '';
                                if (zoomLabel) zoomLabel.style.display = 'flex';
                            
                                // Edit on Canvasボタンのクリックイベントを設定
                                editBtn.onclick = async () => {
                                    await importMathToCanvas(ctx, canvas, mathDiv, zoomInput, clearCanvas, updateMode);
                                };
                            }
                        
                            if (resultArea) resultArea.innerHTML = ''; // 思考中メッセージを消す

                        } else {
                            // 失敗時の処理
                            if (resultArea) resultArea.textContent =
                                _t('ui.hwpage_error', { msg: d.error });
                        }
                    }
                } catch (pollErr) {
                    // ポーリング自体の通信エラーなど
                    clearInterval(poll);
                    recognizeBtn.disabled = false;
                    recognizeBtn.style.background = "";
                    recognizeBtn.innerText = originalText;
                    console.error("Polling error:", pollErr);
                }
            }, 1000); // 1秒ごとに確認

        } catch (e) {
            // 初回の送信エラー時
            recognizeBtn.disabled = false;
            recognizeBtn.style.background = "";
            recognizeBtn.innerText = originalText;
            if (resultArea) resultArea.textContent = _t('ui.hwpage_conn_error');
            console.error("Recognition start error:", e);
        }
    }); // recognizeBtn.onclick の締め

    // ---------------------------------------------------------------------------
    // Notebook integration
    // ---------------------------------------------------------------------------

    async function refreshNotebook() {
        try {
            const res = await fetch('/api/notebook/session');
            const data = await res.json();
            const exprs = data.expressions || {};
            const ids = Object.keys(exprs);
            if (ids.length === 0) {
                notebookPanel.style.display = 'none';
                return;
            }
            notebookPanel.style.display = 'block';
            notebookExpressions.innerHTML = ids.map(id => {
                const e = exprs[id];
                const latexDisplay = e.latex
                    ? `<span class="nb-latex" data-latex="${escapeAttr(e.latex)}"></span>`
                    : '';
                return `<div class="nb-expr"><span class="nb-id">${id}</span> ${latexDisplay}</div>`;
            }).join('');
            // Render KaTeX for each expression
            notebookExpressions.querySelectorAll('.nb-latex').forEach(el => {
                const raw = el.getAttribute('data-latex');
                if (raw) {
                    try { katex.render(raw, el, { displayMode: false, throwOnError: false }); }
                    catch (_) { el.textContent = raw; }
                }
            });
        } catch (_) {}
    }

    function escapeHtml(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function escapeAttr(s) {
        return String(s).replace(/"/g,'&quot;');
    }

    addToNotebookBtn?.addEventListener('click', async () => {
        const latex = latexInput?.value?.trim();
        if (!latex) { alert('LaTeX エディタが空です。'); return; }

        addToNotebookBtn.disabled = true;
        addToNotebookBtn.textContent = _t('ui.hwpage_registering');
        try {
            const res = await fetch('/api/notebook/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ latex }),
            });
            const data = await res.json();
            if (data.ok) {
                await refreshNotebook();
            } else {
                alert('登録失敗: ' + (data.detail || data.error || '不明なエラー'));
            }
        } catch (e) {
            alert('通信エラー: ' + e);
        } finally {
            addToNotebookBtn.disabled = false;
            addToNotebookBtn.textContent = _t('ui.hwpage_add_notebook');
        }
    });

    notebookClearBtn?.addEventListener('click', async () => {
        if (!confirm('セッションを初期化しますか？\n（登録した式と履歴がすべて削除されます）')) return;
        try {
            await fetch('/api/notebook/session', { method: 'DELETE' });
            await refreshNotebook();
        } catch (e) { alert('エラー: ' + e); }
    });

    // Initial load
    refreshNotebook();

    clearBtn?.addEventListener('click', () => {
        // 上：キャンバス
        clearCanvas();
        // 中：LaTeX エディタ
        if (latexInput) latexInput.value = '';
        // 下：KaTeX レンダリング結果
        if (mathDiv) mathDiv.innerHTML = '';
        if (resultArea) resultArea.innerHTML = '';
        if (editBtn) editBtn.style.display = 'none';
        const zoomLabel = document.getElementById('zoom-label');
        if (zoomLabel) zoomLabel.style.display = 'none';
    });

    document.getElementById('shutdown-btn')?.addEventListener('click', async () => {
        if (confirm('Exit system?')) {
            await fetch('/api/shutdown', {method:'POST'});
            window.close();
        }
    });

    async function loadModels() {
        const res = await fetch('/api/models');
        const data = await res.json();
        engineSelect.innerHTML = (data.default_engine === 'none' ? `<option value="none" selected>${_t('ui.hwpage_select_engine')}</option>` : '') + 
            data.engines.map(id => `<option value="${id}" ${id === data.default_engine ? 'selected' : ''}>${id}</option>`).join('');
        refinerSelect.innerHTML = data.refiners.map(id => 
            `<option value="${id}" ${id === data.default_refiner ? 'selected' : ''}>${id}</option>`
        ).join('');
    }

    clearCanvas();
    loadModels();
});