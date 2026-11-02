r"""公開実例が**実際に組版できる**か（TEX-EXPORT-SCAN (a)）。

静的な走査（`notebook/runtime/tex_sanity.py`）は「読めば分かる」壊れ方を捕まえるが、
**TeX に通るか**は TeX にしか答えられない。起票は「TeX 環境が入ったら」着手条件つきで
待っていた——2026-08-22 に入っていることが分かり、走らせたら**96本中3本が組版に
失敗していた**:

    scenario_greens_function_from_hamiltonian_{jp,en}.tex
        l.41 `$\hat$` —— 引数を取るコマンドが引数から切り離され、単体では不正な TeX
        （利用者が実際に踏んで報告した。エクスポータの断片規則が空白区切りの引数を
        知らなかった）
    scenario_integration_variable_notation_en.tex
        l.65 `$sin(θ)^2$` —— Unicode のギリシャ文字が数式の中に生のまま
        （`Not two-byte family`。変換表はエクスポータにあったのに、数式にする枝が
        使っていなかった）

**「コンパイルは通るが誤り」は別問題**として残る（起票の (b)＝JSON↔TeX 照合）。
ただし通らないものは、通らないと言えるようになった。

**既定で走る。** 実例の再生成チェック（`--run-scenarios`）と違って CAS を使わない
ので、96本で40秒足らずで終わる——そして opt-in にしておいたことが、上の3本が
誰にも気づかれずに残っていた理由でもある。`uplatex` が無い環境ではスキップする。
"""
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = sorted((REPO / "domains").glob("*/examples/*.tex"))

pytestmark = pytest.mark.skipif(
    shutil.which("uplatex", path="/Library/TeX/texbin:/usr/local/texlive/bin"
                 ) is None and shutil.which("uplatex") is None,
    reason="uplatex が無い環境ではこの検査はできない",
)


def _uplatex() -> str:
    return (shutil.which("uplatex")
            or shutil.which("uplatex", path="/Library/TeX/texbin")
            or shutil.which("uplatex", path="/usr/local/texlive/bin"))


def test_there_are_examples_to_check():
    """実例が0本なら、この検査は何も言っていない（緑が空振りになる）。"""
    assert len(EXAMPLES) > 50, len(EXAMPLES)


@pytest.mark.parametrize("tex", EXAMPLES, ids=lambda p: p.stem)
def test_every_published_example_compiles(tex, tmp_path):
    work = tmp_path / tex.stem
    work.mkdir()
    shutil.copy(tex, work / tex.name)
    # 図（TikZ）を使う実例があるので、同じディレクトリの添付も持っていく。
    for extra in tex.parent.glob(f"{tex.stem}*.svg"):
        shutil.copy(extra, work / extra.name)
    proc = subprocess.run(
        [_uplatex(), "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=work, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        log = (work / f"{tex.stem}.log")
        errors = [ln for ln in (log.read_text(errors="replace").splitlines()
                                if log.exists() else [])
                  if ln.startswith("! ") or ln.startswith("l.")][:6]
        pytest.fail(f"{tex.name} が組版できない:\n  " + "\n  ".join(errors))
