// ui/canvas_utils.js

export async function importMathToCanvas(ctx, canvas, mathDiv, zoomInput, clearCanvas, updateMode) {
    if (!mathDiv) return;
    const target = mathDiv.querySelector('.katex-html') || mathDiv;

    try {
        const tempCanvas = await window.html2canvas(target, {
            backgroundColor: null,
            scale: 3,
            useCORS: true
        });

        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // --- 文字が存在する「左端」をピクセル単位で探査 ---
        let firstVisibleX = tempCanvas.width;
        for (let y = 0; y < tempCanvas.height; y++) {
            for (let x = 0; x < tempCanvas.width; x++) {
                const alpha = data[(y * tempCanvas.width + x) * 4 + 3];
                if (alpha > 0) { // 透明でないピクセルを発見
                    if (x < firstVisibleX) firstVisibleX = x;
                }
            }
        }
        // もし文字が見つからなければ0にする
        if (firstVisibleX === tempCanvas.width) firstVisibleX = 0;

        clearCanvas();

        const imgW = tempCanvas.width;
        const imgH = tempCanvas.height;
        const aspect = imgW / imgH;
        const userZoom = zoomInput ? parseFloat(zoomInput.value) : 10.0;

        // 拡大係数
        const baseScale = 800;
        let drawWidth = baseScale * userZoom;
        let drawHeight = drawWidth / aspect;

        // --- 縦方向: 上下マージンを確保してキャンバス内に収める ---
        const marginV = 120;  // 上下それぞれの余白(px)
        const marginL = 160;  // 左余白(px)
        const maxH = canvas.height - 2 * marginV;
        if (drawHeight > maxH) {
            const scale = maxH / drawHeight;
            drawWidth *= scale;
            drawHeight = maxH;
        }

        // --- 描画位置の計算 ---
        // tempCanvas内での「文字までの空白の割合」を計算し、描画時のズレを相殺する
        const cropRatioX = firstVisibleX / imgW;
        const shiftX = drawWidth * cropRatioX;

        const offsetX = marginL - shiftX;  // 左端 marginL px に数式の先頭を揃える
        const offsetY = Math.round((canvas.height - drawHeight) / 2);  // 上下中央

        ctx.globalCompositeOperation = 'source-over';
        ctx.imageSmoothingEnabled = true;
        ctx.globalAlpha = 0.7; 

        // 描画実行
        ctx.drawImage(tempCanvas, offsetX, offsetY, drawWidth, drawHeight);
        
        ctx.globalAlpha = 1.0;
        updateMode(false);
        ctx.getImageData(0, 0, 1, 1);
        
        console.log(`FirstVisibleX: ${firstVisibleX}, shiftX: ${shiftX.toFixed(1)}, finalOffsetX: ${offsetX.toFixed(1)}`);
    } catch (err) {
        console.error("html2canvas error:", err);
    }
}