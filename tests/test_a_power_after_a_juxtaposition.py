r"""JUXT-POW-PRECEDENCE（2026-09-25）——併置のあとの冪が、併置ぜんぶに掛かっていた。

    a (b)^2        →  (a * b)^2      ✘
    a*(b)^2        →  a * b^2        ✔
    a b^2          →  a * b^2        ✔
    a^2 (b)        →  a^(2 * b)      ✘   ← 指数が後ろの因子を飲み込む

**`*` を書けば正しく、括弧を書かなければ正しく、この系自身の乗算である併置のときだけ
壊れていた。** 公開実例の中心力のエネルギー（日英の E5/E6）とニュースベンダーの
費用関数（E1）がこれで、CAS 差分は 0 でない——**研究者が書いた式と別の量**が
`completed` で返っていた。

**根は2つ、どちらも「空白＋括弧で作られた FunctionCall が `^` の相手として原子扱い
される」こと。** 直す場所が違うのは、決められる層が違うから:

- **底の側は意味の問い**——`sin (x)^2` は呼び出しなので `sin(x)^2` が正しく、
  `a (b)^2` は積なので `a*b^2` が正しい。「呼び出しか積か」を知っているのは
  リゾルバだけ（`_spaced_paren_is_a_product`）なので、そこで組み替える。
- **指数の側は文法の問い**——空白は「指数はここで終わり」を言っている。研究者が
  積を指数にしたければ波括弧を書く（`a^{2 (b)}` は変わらない）。パーサが答える。

**連鎖は②があって初めて直る。** `a (b)^2 (c)^3` は①だけだと `a*b^(2*c^3)` にしか
ならない——指数の結合が既に間違って焼かれているため。

**組み替えの結果は `Juxtaposition`**（`Mul` ではない）——パーサが空白の積に対して
作る当の節点なので、可換かどうかの判断は `_flatten_juxtaposition` の1箇所に残る。
ただし**実測では差が出ない**（宣言された行列3つ・Julia の演算子・11綴りで 0 変化）
ので、これは guard ではなく「答えを1箇所に置く」という理由での選択である。

**2つの形を1つのヘルパが割る**（`_spaced_product_split`）: 括弧は素の名前にも
（`a (b)`）下付きの名前にも（`c_o (Q-a)`、パーサは**下付きのオペランド**に付ける）
付く。既に在った適用下付きの枝もこれを読むので、答えは1箇所にある。
"""
import pytest

from notebook.engine import NotebookEngine


@pytest.fixture(scope="module")
def handler():
    return NotebookEngine(title="juxtpow")._handler


class TestTheExponentBelongsToTheFactor:
    r"""空白＋括弧は乗算なので、冪はその**因子**だけに掛かる。"""

    @pytest.mark.parametrize("written,code", [
        ("a (b)^2",       "a * b^2"),
        ("a (b+c)^2",     "a * (b + c)^2"),
        ("a (b)^2 + c",   "a * b^2 + c"),
        ("2 (b)^2",       "2 * b^2"),
        ("a (b)^n",       "a * b^n"),
        ("(a) (b)^2",     "a * b^2"),
        ("a b (c)^2",     "a * b * c^2"),
        ("a (b) (c)^2",   "a * b * c^2"),
        # 下付きの頭——括弧は下付きのオペランドに付く（公開実例の綴り）
        ("c_o (Q-a)^2",   "c(o) * (Q - a)^2"),
        ("r_dot (x)^2",   "r_dot * x^2"),
        ("a_i (k)^2",     "a(i) * k^2"),
    ])
    def test_base_side(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written

    @pytest.mark.parametrize("written", [
        "a (b)^2", "a (b+c)^2", "2 (b)^2", "a (b)^n", "c_o (Q-a)^2", "a (b)^{-1}",
    ])
    def test_the_space_agrees_with_the_star(self, handler, written):
        """**不変条件はこれ**——`a (b)^2` と `a*(b)^2` は同じ式である。"""
        starred = written.replace(" (", "*(", 1)
        assert handler.to_backend_code(written).code == \
               handler.to_backend_code(starred).code, (written, starred)


class TestTheExponentDoesNotSwallowTheNextFactor:
    r"""空白は「指数はここで終わり」を言っている。"""

    @pytest.mark.parametrize("written,code", [
        ("a^2 (b)",       "a^2 * b"),
        ("a^n (b)",       "a^n * b"),
        ("a^2 (b)^3",     "a^2 * b^3"),
        ("a (b)^2 (c)^3", "a * b^2 * c^3"),
    ])
    def test_script_side(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written

    def test_braces_say_the_product_is_the_exponent(self, handler):
        """波括弧を書けば積が指数になる——そこは変わらない。"""
        assert handler.to_backend_code("a^{2 (b)}").code == \
               handler.to_backend_code("a^{2 b}").code

    @pytest.mark.parametrize("written,code", [
        # **空白が無ければ抑制しない**——`a^{f(x)}` の glued な綴り。変異で名指された:
        # `peek_ws_before()` を外すと、この2つが `a^f * x` になる。
        ("a^f(x)",   "a^f(x)"),
        ("a^sin(x)", "a^sin(x)"),
    ])
    def test_a_glued_parenthesis_is_still_the_exponent(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written

    @pytest.mark.parametrize("written,code", [
        # **入れ子はまっさらから読む**——括弧の中は指数の一部で、そこでは
        # 空白＋括弧が呼び出しでありうる。変異で名指された: `parse_expr` が
        # フラグを落とさないと `a^(sin (x))` が `a^(sin * x)` になる。
        ("a^(sin (x))", "a^sin(x)"),
        ("a^(2 (b))",   "a^(2 * b)"),
        ("a^(2 b)",     "a^(2 * b)"),
    ])
    def test_a_nested_group_is_parsed_afresh(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written

    def test_the_exponent_keeps_its_own_parentheses(self, handler):
        r"""`e^{-i\omega(t-t')}` ——指数の中の括弧は指数の一部。"""
        code = handler.to_backend_code(
            handler.latex_to_canonical(r"e^{-i \omega (t-t')}")).code
        assert code.startswith("%e^("), code
        assert "omega" in code and "*" in code, code


class TestAGenuineCallIsUntouched:
    r"""呼び出しと積の区別は動かさない——決めているのは1つの述語である。"""

    @pytest.mark.parametrize("written,code", [
        ("sin (x)^2",   "sin(x)^2"),
        ("sin(x)^2",    "sin(x)^2"),
        ("f(x)^2",      "f(x)^2"),
        ("f (x)",       "f * x"),
        ("a (b)",       "a * b"),
        ("a*(b)^2",     "a * b^2"),
        ("a b^2",       "a * b^2"),
        ("(a b)^2",     "(a * b)^2"),
        ("2 x^2",       "2 * x^2"),
        ("a^2 b",       "a^2 * b"),
        # 引数が2つ以上なら積ではありえない（前回の変異が名指した条件）
        ("f (x, y)^2",  "f(x, y)^2"),
    ])
    def test_unchanged(self, handler, written, code):
        assert handler.to_backend_code(written).code == code, written


class TestThePublishedDisplaysReadBackAsTheSameQuantity:
    r"""公開実例の綴りそのもの——CAS に差を訊く（R3 と同じ読み方）。"""

    @pytest.mark.parametrize("written,code", [
        # 中心力のエネルギー（E5/E6、日英）
        (r"2 r \dot{r} \left(\frac{d}{d t} \theta\right)^2",
         "2 * r * r_dot * 'diff(theta,t,1)^2"),
        # ニュースベンダーの費用関数（E1、日英）
        ("c_o (Q - a)^2 / (2 (b - a))",
         "c(o) * (Q - a)^2 * (2 * (b - a))^-1"),
    ])
    def test_published(self, handler, written, code):
        got = handler.to_backend_code(handler.latex_to_canonical(written)).code
        assert got == code, got
