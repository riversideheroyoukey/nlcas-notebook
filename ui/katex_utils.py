# ui/katex_utils.py
# Version: v1.1.5
# Description: Fixed utility for managing KaTeX local assets.

import os
import requests
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
LIB_DIR = STATIC_DIR / "lib" / "katex"
FONTS_DIR = LIB_DIR / "fonts"

def ensure_katex_assets():
    """KaTeXのJS/CSSおよびフォントがローカルに存在することを確認し、なければダウンロードする"""
    if not FONTS_DIR.exists():
        FONTS_DIR.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/"
    
    # メイン資産
    assets = {
        "katex.min.js": base_url + "katex.min.js",
        "katex.min.css": base_url + "katex.min.css"
    }
    
    # 最小限必要なフォント
    fonts = [
        "KaTeX_Main-Regular.woff2",
        "KaTeX_Math-Italic.woff2",
        "KaTeX_Size1-Regular.woff2"
    ]

    for f, url in assets.items():
        p = LIB_DIR / f
        if not p.exists():
            print(f"📥 Downloading {f}...")
            with open(p, "wb") as out:
                out.write(requests.get(url).content)
                
    for f in fonts:
        p = FONTS_DIR / f
        if not p.exists():
            print(f"📥 Downloading font: {f}...")
            with open(p, "wb") as out:
                out.write(requests.get(base_url + "fonts/" + f).content)
                
    print("✅ KaTeX assets are ready locally.")