# Third-party components

This file describes third-party components independently of the license for
NLCAS Notebook's original code. The latter has not yet been selected.

| Component | Distribution / notice |
|---|---|
| html2canvas 1.4.1 | Bundled at `ui/static/lib/html2canvas.min.js`; MIT notice retained in that file and [LICENSES/html2canvas-MIT.txt](LICENSES/html2canvas-MIT.txt). [Upstream license](https://github.com/niklasvh/html2canvas/blob/v1.4.1/LICENSE) |
| KaTeX | Downloaded at application startup by `ui/katex_utils.py` (currently 0.16.9). The public snapshot omits the developer's downloaded KaTeX cache. [Upstream license](https://github.com/KaTeX/KaTeX/blob/v0.16.9/LICENSE) |
| Maxima | Installed separately; not bundled. [Project](https://maxima.sourceforge.io/) |
| Julia and Julia packages | Installed separately from `Project.toml` and `Manifest.toml`; not bundled. Each dependency retains its own license |
| Python packages | Installed separately from `requirements.txt` and `requirements-numeric.txt`; not bundled. Each dependency retains its own license |
| LLM / OCR model weights | Not bundled. Configure your own endpoints and obtain models under their respective terms |

The official upstream manuals retained in the local development workspace
under `docs/misc/` are excluded from the public snapshot. The dependency
lists above identify separate installation requirements, rather than
relicensing those projects under this project's eventual license.
