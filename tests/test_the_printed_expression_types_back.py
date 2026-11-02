r"""NL-GRAMMAR ① 五十六問目「系が印刷した式を打ち直すと同じ式になるか」（2026-09-25）。

綴り空間の3つの読みは尽きた（未打鍵 0／言語中立の非対称1件は語順が理由で決着／
名前の違う鏡2組は測って白）。そこで**4つ目の供給源**——「系が印刷した綴りを打ち
直す」——を機械化した: **公開実例の `latex` を入力として読み直し、保存されている
`canonical` と比べる**。R3（`tests/test_user_cas_invariants.py`）は同じ問いに
答えているが **curated 16件**で、こちらは Maxima セッションの **805式**。

実測 74/805 が食い違い、大半は既知・設計どおり（`'integrate` の引用符、Laplace の
定義カード、tex1 の `\pmatrix` ラッパ＝R3 が既知の穴として固定済み）。
**新しい欠陥は2つ、どちらも公開実例の中**:

| 印刷した行（公開 .tex / .json） | 打ち直すと |
|---|---|
| `… = \cos(b-a) \dot{b} m p q + \dot{a} \left(m+M\right) p^2` | `a_dot(m+M)` ——**関数呼び出し** |
| `\frac{m \left(2 r^2 \left(\frac{d}{d t}\theta\right)…\right)}{2}` | `d * (d*t)^-1 * theta` ——**∂ が約分** |

**① 空白付きの括弧は積である——規則は在って、片方の枝しか訊いていなかった。**
`f (x)` → `f * x`、`y (m+M)` → `y * (m+M)` は昔から正しい（`spaced` をパーサが
記録し、解決器が文脈で決める）。**下付きを畳む枝**は訊いておらず、同じ空白が
`a_dot` のあとでは呼び出しになった。二重橋の Lagrangian は
`∂Lag/∂ȧ = cos(b-a)ḃmpq + ȧ(m+M)p²` と印刷されており、研究者がその行を写すと
`a_dot` に `(m+M)` を**適用**した別物になる。

**② 導関数でない分数は、その中を見ていなかった。** 走査は `\frac{` を見つけて
読めなければ**分母の終わりまで丸ごと飛ばす**ので、外側の分数の中の
`\frac{d}{dt}\theta` を一度も見ない。これは BP-9 が防ぐために在る当の失敗
（「`d` と `\partial` が普通の記号に落ちて ∂φ/∂x が φ/x に約分される」）で、
公開されている中心力の例3式（日英）がそれだった。

**残す1件は設計の問いなので別起票**（DOT-DISPLAY-IS-AN-ATOM）: `'diff(r,t,1)` が
`\dot{r}` と印刷され、打ち直すと**独立な原子** `r_dot` になる。解析力学では
`q` と `q̇` は独立原子だと決めてあるので、これは欠陥ではなく**どちらの規約で
読むか**の問いである。

> **その起票は 2026-09-25 に解決した**（CLAUDE.md「宣言が答えを変えてはならない」）。
> **文字列としての差は残る**——`canonical` は `r_dot`、印刷の出所は `'diff(r,t,1)` —— が、
> **CAS の対象としては同じものに着地する**: `_prepare_expr_lookup` の表が時間依存量の
> `r_dot` を `'diff(r,t,1)` へ写すので、打ち直した式を使うと `m*('diff(r,t,1))^2` になる
> （実測）。判別子は最初から**宣言**であり、従っていなかったのは変数を綴る1つの関数
> だけだった——そこには**生きた silent-wrong**（`∂/∂q̇` が `0` を `completed`）が在った。
> だからこの6件は、この測定器にとっては**既知で等価な差**であって塞がりではない。

**数え直した（2026-09-25）——「ほとんどは既知」は成り立たなかった。** 上の2件を
直したあとの食い違いは **68 / 805**。族に分けると（**等価と分かっている差は最後に
当てる**——先に当てると生きた欠陥が裏に隠れる）:

| 族 | 件 | 状態 |
|---|---|---|
| tex1 の波括弧つき積分 | 10 | **未記録** → TEX1-PRINTS-WHAT-IT-CANNOT-READ |
| `\ifx\endpmatrix` マクロ包み | 10 | 同上（R3 が固定しているのは素の `\pmatrix`） |
| Laplace の定義カード | 8 | 既知 |
| 集合の波括弧 `\{3,1\}` → **反交換子** | 8 | **未記録**・silent wrong |
| **併置×冪** `a (b)^2` → `(a*b)^2` | 6 | **未記録**・silent wrong → JUXT-POW-PRECEDENCE |
| 表示が式番号（`E5 + E7`） | 4 | 数式でない（測定器の対象外） |
| 天井関数 `\lceil x \rceil` → **積** | 4 | **未記録**・silent wrong |
| Σ のプレースホルダ範囲 | 4 | 既知 |
| Itô の未簡約形・符号の正規化 | 6 | **CAS 差分 0**（意味は同じ） |
| ドット表示（DOT-DISPLAY） | 2 | 既知・等価 |
| 非可換冪 `^^` | 2 | 既知（表示復元の設計） |
| plain TeX の `\over` | 2 | **未記録** |
| 連鎖等式の表示 | 2 | 測定器の対象外（`strip_lhs` は先頭の `=` だけ外す） |

**JSON の `latex` が公開 `.tex` と同じ文字列であることは確認した**（`.tex` を grep）
——測定器が読んでいるのは、研究者が画面と論文で見る綴りそのものである。
"""
import glob
import json

import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.shared_parser import UnknownNotationError, strip_lhs

_DISPLAY_ONLY = ("lattice_descriptor", "integral_descriptor", "pair_representation",
                 "extremum_descriptor", "angular_descriptor")


@pytest.fixture(scope="module")
def handler():
    return NotebookEngine(title="printed")._handler


class TestASpaceBeforeTheParenthesisIsMultiplication:
    r"""`f (x)` の規則は**名前の形によらない**。"""

    @pytest.mark.parametrize("written,code", [
        # 素の名前——昔から正しい
        ("f (x)",        "f * x"),
        ("y (m+M)",      "y * (m + M)"),
        ("f(x)",         "f(x)"),
        # 装飾名——ここが呼び出しになっていた
        ("a_dot (m+M)",  "a_dot * (m + M)"),
        ("a_dot(m+M)",   "a_dot(m + M)"),
        ("q_ddot (t)",   "q_ddot * t"),
        ("H_hat (x)",    "H_hat * x"),
        # 添字——空白のときは**括弧が無い場合と同じ読み**（`a_i` は `a(i)`）
        ("a_i (k)",      "a(i) * k"),
        ("a_i(k)",       "a_i(k)"),
        ("A_{i,j} (x)",  "A(i, j) * x"),
        ("A_{i,j}(x)",   "A[i, j](x)"),
        # 既知の関数は空白があっても呼び出し
        ("sin (x)",      "sin(x)"),
    ])
    def test_lowering(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written

    @pytest.mark.parametrize("written,code", [
        ("f (x, y)",       "f(x, y)"),
        ("a_dot (x, y)",   "a_dot(x, y)"),
        ("A_{i,j} (x, y)", "A[i, j](x, y)"),
    ])
    def test_a_comma_list_is_never_a_product(self, handler, written, code):
        """**引数が2つ以上なら積ではありえない**——落とすと `y` が消える。

        変異試験で「引数は1つ」を外しても当時のどのセルも落ちなかった。測ると
        `f (x, y)` が `f * x` になり、研究者が書いた `y` が**黙って消える**。
        素の名前の枝には最初から在った条件で、下付きの枝に写したときも要る。
        """
        assert handler.to_backend_code(written).code == code, written

    def test_the_bare_name_and_the_spaced_head_agree(self, handler):
        """組み直した頭は「括弧が無ければ何だったか」と一致すること。"""
        for bare, spaced in (("a_dot", "a_dot (x)"), ("a_i", "a_i (x)"),
                             ("q_ddot", "q_ddot (x)")):
            head = handler.to_backend_code(spaced).code.split(" * ")[0]
            assert head == handler.to_backend_code(bare).code, (bare, spaced)


class TestADerivativeInsideAnotherFractionIsRead:
    r"""外側の分数が導関数でなくても、その中の導関数は読む。"""

    @pytest.mark.parametrize("written,code", [
        (r"\frac{d}{d t} \theta",              "'diff(theta,t,1)"),
        (r"\frac{\left(\frac{d}{d t} \theta\right)}{2}",
         "(('diff(theta,t,1)))/(2)"),
        (r"\frac{m \left(2 r^2 \left(\frac{d}{d t} \theta\right)\right)}{2}",
         "((m*(2*r^2*('diff(theta,t,1)))))/(2)"),
        # 導関数でない分数はそのまま
        (r"\frac{a}{b}", "((a)/(b))"),
    ])
    def test_nested(self, handler, written, code):
        got = handler.latex_to_canonical(written)
        assert got.replace("(", "").replace(")", "") == code.replace("(", "").replace(")", ""), got

    def test_the_published_central_force_line(self, handler):
        r"""公開実例の E5 ——この行がそのまま `d`/`t` を普通の記号にしていた。"""
        published = (r"\frac{m \left(2 r^2 \left(\frac{d}{d t} \theta\right) "
                     r"\left(\frac{d^2}{d t^2} \theta\right)+2 r \dot{r} "
                     r"\left(\frac{d}{d t} \theta\right)^2+2 \dot{r} \ddot{r}\right)}{2}"
                     r"+\frac{k \dot{r}}{r^2}")
        back = handler.latex_to_canonical(published)
        assert "'diff(theta,t,1)" in back and "'diff(theta,t,2)" in back, back
        assert "(d)/(d*t)" not in back, back


class TestThePublishedDisplaysStillReadBack:
    r"""**測定器そのもの**——公開実例の `latex` を打ち直す。

    ここは R3（curated 16件）が見ていない範囲で、この問いはそこから出た。
    数を固定するのではなく、**この2つの形が公開物に残っていないこと**を固定する:
    直したあとに同じ形が現れたら、それは退行である。
    """

    def _rows(self):
        for f in sorted(glob.glob("domains/*/examples/*.json")):
            j = json.load(open(f, encoding="utf-8"))
            if j["session"].get("primary_backend") != "maxima":
                continue
            for x in j["expressions"].values():
                lat, can = (x.get("latex") or "").strip(), (x.get("canonical") or "").strip()
                if not lat or not can or any(x.get(k) for k in _DISPLAY_ONLY):
                    continue
                yield f, x["display_id"], lat

    def test_no_published_display_cancels_a_derivative(self, handler):
        bad = []
        for f, eid, lat in self._rows():
            try:
                back = handler.latex_to_canonical(strip_lhs(lat) or lat)
            except (UnknownNotationError, ValueError, AttributeError, TypeError):
                continue
            if "(d)/(d*t)" in back.replace(" ", "") or "(d)/(d" in back.replace(" ", ""):
                bad.append((f.split("/")[-1], eid))
        assert not bad, f"a published display still cancels its derivative: {bad}"

    def test_no_published_display_turns_a_spaced_paren_into_a_call(self, handler):
        bad = []
        for f, eid, lat in self._rows():
            try:
                back = handler.to_backend_code(
                    handler.latex_to_canonical(strip_lhs(lat) or lat)).code
            except (UnknownNotationError, ValueError, AttributeError, TypeError):
                continue
            for name in ("a_dot(", "b_dot(", "r_dot(", "q_dot(", "r_ddot("):
                if name in back.replace(" ", ""):
                    bad.append((f.split("/")[-1], eid, name))
        assert not bad, f"a published display reads back as a CALL: {bad}"
