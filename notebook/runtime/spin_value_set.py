r"""スピン射影 $m$ の**値集合**——`\sum_\sigma` が何を走るのか。

`\sum_\sigma` は範囲が書かれていないので、QuantumAlgebra の形式和にそのまま乗る。
それは書かれたものと一致するが、**QA は形式和どうしの交換子を畳まない**: 実測
2026-08-22、`[\sum_\sigma c^†_{1\sigma}c_{2\sigma}, \sum_\sigma \hat n_{1\sigma}]`
は正しい項に加えて4演算子の項を2つ残す——それらはダミーの入れ替えで相殺するが、
QA は項をまたぐダミーの正準化をしない（`[A,A]=0` は正しく出るので、欠陥ではなく
限界）。スピン値を明示して書けば 0 になる。

つまりここでも「受け取れないなら展開して渡す」が効く。**足りないのは値集合**で、
宣言が記録していたのは `statistics=fermi` だけだった。

**表現（研究者の提案、2026-08-22）:**

- フェルミオンは半整数スピンなので、射影を**符号付きの分子**で列挙する
  （電子 = $\pm 1/2$ → `[-1, 1]`）。分子で持つのは半整数を float にしないため
  ——F-9 の `Surd` と同じ理由で、`1/2` を浮動小数にした瞬間に厳密性が消える。
- ボソンは整数スピンなので、射影は**整数の範囲**（$-S..S$）。

**統計は分子の偶奇を決めるが、大きさは決めない**（スピン 3/2 のフェルミオンは
実在する——半導体の $j=3/2$ 価電子帯）。だから $2S$ は粒子ごとのデータで、
`models/particles.json` の `two_s` に置いてある。梯子から外れる粒子
（質量ゼロ ⇒ ヘリシティ $\pm S$ のみ。光子に $m=0$ は無い）は
`projection_numerators` で明示する。

**添字の綴りは分子ではない。** 実測、QA は添字を区切りなしで連結するので
`c(1,1)` は `c(11)` と印字され——サイト 11 と同じ文字列になる。矢印なら
`c(1↑)` で一意。だから分子は値集合の表現にとどめ、QA の添字には
1文字の綴り（spin 1/2 なら矢印）を使う。
"""
import json
from pathlib import Path
from typing import List, Optional

_PARTICLES_CACHE: "Optional[List[dict]]" = None


def load_particles() -> "List[dict]":
    """`models/particles.json` の粒子表。**バックエンド中立なデータ**である。

    読み込みがここにあるのは、この表が CAS の知識ではなく物理の知識だからで、
    Julia ハンドラが持っていると共有層（`engine.py`）が粒子を知るために
    バックエンドを import することになる——それは Core Principle #10 が禁じている
    向きで、`tests/test_code_hygiene.py` の CAS-import ratchet が実際に捕まえた。
    """
    global _PARTICLES_CACHE
    if _PARTICLES_CACHE is None:
        path = Path(__file__).resolve().parent.parent.parent / "models" / "particles.json"
        with path.open(encoding="utf-8") as f:
            _PARTICLES_CACHE = json.load(f)["particles"]
    return _PARTICLES_CACHE


def projections_for_particle(particle_key: str) -> "Optional[List[int]]":
    """粒子キーから射影の値集合（分子）。不明なキーでは None——推測しない。"""
    if not particle_key:
        return None
    for p in load_particles():
        if p.get("key") == particle_key:
            return projection_numerators(p)
    return None

#: 射影の**分子**（÷2）→ QA の添字に置ける1文字の綴り。
#:
#: **1文字であることは QA の要求**（`QuIndex` は1 unicode 文字）で、**数字が
#: 使えないのは印字の衝突**である——実測 2026-09-12、`c(1, QuIndex("0"))` は
#: `c(10)` と印字され、サイト 10 と同じ文字列になる（`"1"` も同様）。
#: 1文字なら何でも置けて区別も付く（`⇑ ⇓ ⁺ ⁰ ⁻ ▲ ▼ α β` を実測）ので、
#: 選ぶ基準は**物理の慣用**である（研究者の選択、2026-09-12）:
#:
#: - 半整数（$m=\pm 1/2$）は矢印。
#: - 整数（$m=+1,0,-1$）は上付き符号——$\pi^+\pi^0\pi^-$ と同じ書き方で、
#:   ヘリシティ $\pm1$ の光子・グルーオンは `⁺ ⁻` の2値になる。
#:
#: **表に無い分子があれば、その値集合は展開しない。** グラビトン（$m=\pm2$、
#: 分子 $\pm4$）に1文字の慣用綴りは無いので形式和のまま——推測しない。
#: 3/2 の慣用（重い/軽い正孔の `⇑ ↑ ↓ ⇓`）を入れていないのは、**粒子表に
#: $2S=3$ の粒子が1つも無い**からで、通らない綴りは文法に入れない。
PROJECTION_LABEL: dict[int, str] = {
    -1: "↓", 1: "↑",            # 半整数: m = ∓1/2
    -2: "⁻", 0: "⁰", 2: "⁺",     # 整数:   m = -1, 0, +1
}

#: 旧名。半整数だけを指していたので、読む側が「半整数の表」と誤解しないよう
#: 別名で残す（削除するのは呼び出し側を数えてから）。
HALF_INTEGER_LABEL: dict[int, str] = {-1: "↓", 1: "↑"}

#: 射影ラベル → **画面に出す LaTeX**。ラベルそのもの（`⁺`）は生の unicode 上付き
#: なので、`$…$` の中に置くと uplatex が組めない——ギリシャ字と同じ罠
#: （CLAUDE.md「公開 .tex は COMPILE する」）。矢印は命令、整数の射影は素の符号で、
#: どちらも物理の書き方である（$a_{1+}$）。**ここが唯一の表**で、バックエンドの
#: 表示側はこれを読む。
PROJECTION_LATEX: dict[str, str] = {
    "↑": r"\uparrow", "↓": r"\downarrow",
    "⁺": "+", "⁰": "0", "⁻": "-",
}


def ladder_numerators(two_s: int) -> "List[int]":
    r"""$-S..S$ の射影を**分子**（÷2）で。``two_s`` は $2S$。

    $2S=1$ → ``[-1, 1]``（電子）、$2S=2$ → ``[-2, 0, 2]``（$m=-1,0,1$）、
    $2S=0$ → ``[0]``。分子で返すので半整数が float にならない。
    """
    if two_s < 0:
        raise ValueError(f"two_s must be non-negative, got {two_s}")
    return list(range(-two_s, two_s + 1, 2))


def projection_numerators(particle: dict) -> "Optional[List[int]]":
    """粒子データから射影の値集合（分子）を読む。決まらなければ None。

    `projection_numerators` が明示されていればそれ（梯子から外れる粒子）、
    無ければ `two_s` からの梯子。`two_s` も無ければ None——**推測しない**。
    """
    explicit = particle.get("projection_numerators")
    if explicit is not None:
        return list(explicit)
    two_s = particle.get("two_s")
    if two_s is None:
        return None
    return ladder_numerators(int(two_s))


def index_labels(numerators: "List[int]") -> "Optional[List[str]]":
    r"""射影の分子を、QA の添字に置ける1文字の綴りに。決まらなければ None。

    **全部の分子に綴りがあるときだけ**返す（`PROJECTION_LABEL`）。1つでも
    表に無ければ None で、呼び手は展開せず形式和のまま渡す——綴りを推測しない。

    **値が1つの集合は展開しない。** スピン 0 の粒子（フォノン・励起子…）の
    値集合は `[0]` で、`\sum_\sigma` を1項に畳むと `a(1)` と `a(1⁰)` という
    **別の QA 対象**が同じセッションに並ぶ。畳む利得（QA が項をまたぐダミーを
    正準化しない、という限界の回避）はそもそも項が1つなら無い。
    """
    if len(numerators) < 2:
        return None
    labels = [PROJECTION_LABEL.get(n) for n in numerators]
    if any(lbl is None for lbl in labels):
        return None
    return [lbl for lbl in labels if lbl is not None]


def format_numerator(n: int) -> str:
    r"""分子を人が読む形に: ``-1`` → ``-1/2``、``2`` → ``1``、``0`` → ``0``。"""
    if n % 2 == 0:
        return str(n // 2)
    return f"{n}/2"


def expand_spin_index_sum(node, spin_projections: dict) -> "Optional[object]":
    r"""``\sum_\sigma f(\sigma)`` を宣言された値集合の項に展開する。展開できなければ None。

    展開する条件（どれも「書かれていること」と「宣言されていること」だけで決まる）:

    - 束縛添字が**ちょうど1つ**で、範囲が**書かれていない**（範囲つきは
      `finite_sum_expansion` の領分）。
    - その添字の名前が**スピン変数の綴り**である（`\sigma` 等——`SPIN_LABEL_CANON`
      に載っている語）。素の `s` は展開しない: 1文字にスピンの意味を読み込むのは、
      この系が他所でしない推測である（矢印＝値／σ＝変数の区別と同じ線）。
    - 本体に出てくる演算子のうち、値集合が**宣言されているもの**が値集合について
      一致し、その集合に1文字の綴りがある（spin 1/2 なら矢印）。
      食い違えば展開しない——多species を推測で潰さない。

    展開が要る理由は QA の限界である: 形式和どうしの交換子で、ダミー入れ替えで
    相殺する項を QA は畳まない（モジュール冒頭の実測）。値を入れれば畳む。
    """
    from notebook.models.ir import Add, IndexSum, IndexSymbol
    from notebook.runtime.spin_label_canon import canon_spin_label

    if not isinstance(node, IndexSum) or node.op != "sum":
        return None
    if len(node.dummy_indices) != 1 or node.lower is not None or node.upper is not None:
        return None
    name = node.dummy_indices[0].name
    if canon_spin_label(name) is None:          # `\sigma` は載っている、素の `s` は載っていない
        return None
    if not spin_projections:
        return None

    bases = {b for b in spin_projections if _mentions_base(node.body, b)}
    sets = {tuple(spin_projections[b]) for b in bases}
    if len(sets) != 1:
        return None                              # 未宣言、または species が食い違う
    labels = index_labels(list(next(iter(sets))))
    if not labels:
        return None                              # 1文字の綴りが無い（整数スピン・3/2 …）

    nums = list(next(iter(sets)))
    terms = [_substitute_projection(node.body, name, lbl, n)
             for lbl, n in zip(labels, nums)]
    return terms[0] if len(terms) == 1 else Add(args=terms)


def _substitute_projection(node, name: str, label: str, numerator: int):
    r"""``\sigma`` を、**位置に応じて**綴りか値かに置き換える。

    添字の位置（`IndexSymbol`）には1文字の綴り（矢印）、それ以外——$\hat S^z$ を
    $\sum_{i\sigma}\sigma\,\hat n_{i\sigma}$ と書いたときの係数の $\sigma$ ——には
    **値**が入る。IR が既にこの2つを別のノードで持っているので、判定に推測はいらない
    （`Symbol("sigma")` が係数、`IndexSymbol("sigma")` が添字）。

    値は分子から**厳密**に組む: 奇数なら `numerator * (1//2)`、偶数なら整数。
    ここで `0.5` を作ると F-9 が禁じている float がスピン代数に入り込む。

    これが無いと、係数の位置にも矢印が入って Julia が生の `ParseError:` を返して
    いた（実測 2026-08-22）——研究者から見れば、書いた式が理由なく通らない。
    """
    from notebook.models.ir import IndexSymbol, Mul, Number, Pow, Symbol

    if isinstance(node, IndexSymbol):
        return IndexSymbol(name=label) if node.name == name else node
    if isinstance(node, Symbol):
        bare = node.name[1:] if node.name.startswith(":") else node.name
        if bare != name:
            return node
        if numerator % 2 == 0:
            return Number(value=numerator // 2)
        return Mul(args=[Number(value=numerator),
                         Pow(base=Number(value=2), exponent=Number(value=-1))])

    fields = getattr(type(node), "model_fields", None)
    if fields is None:
        return node
    updates = {}
    for key in fields:
        val = getattr(node, key, None)
        if isinstance(val, list):
            new_list = [_substitute_projection(v, name, label, numerator)
                        if hasattr(type(v), "model_fields") else v for v in val]
            if new_list != val:
                updates[key] = new_list
        elif hasattr(type(val), "model_fields"):
            new_val = _substitute_projection(val, name, label, numerator)
            if new_val is not val:
                updates[key] = new_val
    return node.model_copy(update=updates) if updates else node


def _mentions_base(node, base: str) -> bool:
    """*node* の中に名前 *base* の記号が出てくるか（素朴な再帰走査）。"""
    from notebook.models.ir import Symbol
    if isinstance(node, Symbol):
        n = node.name
        return (n[1:] if n.startswith(":") else n) == base
    fields = getattr(type(node), "model_fields", None)
    if fields is None:
        return False
    for key in fields:
        val = getattr(node, key, None)
        if isinstance(val, list):
            if any(_mentions_base(v, base) for v in val):
                return True
        elif _mentions_base(val, base):
            return True
    return False


# ---------------------------------------------------------------------------
# 核種（NMR/NQR）
# ---------------------------------------------------------------------------
#
# **核種は粒子表とは別の軸**なので、データも別ファイル（`models/nuclides.json`）。
# 表の同一性が「粒子の種類」（電子・陽子・光子）であるのに対し、こちらは同位体で
# ある: $^{13}$C のスピン 1/2 は「炭素」からは出ない（$^{12}$C は 0、$^{14}$C も 0）。
# 統計は質量数の偶奇から厳密に決まるので保存しない。

_NUCLIDES_CACHE: "Optional[List[dict]]" = None


def load_nuclides() -> "List[dict]":
    """`models/nuclides.json` の核種表（バックエンド中立なデータ）。"""
    global _NUCLIDES_CACHE
    if _NUCLIDES_CACHE is None:
        path = Path(__file__).resolve().parent.parent.parent / "models" / "nuclides.json"
        with path.open(encoding="utf-8") as f:
            _NUCLIDES_CACHE = json.load(f)["nuclides"]
    return _NUCLIDES_CACHE


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def nuclide_spellings(nuclide: dict) -> "List[str]":
    r"""1つの核種の書き方を機械的に作る: ``13C`` ``^{13}C`` ``^13C`` ``C-13`` ``¹³C``。

    データに列挙しないのは、綴りが (A, 元素記号) から決まるからで、列挙すると
    片方だけ足し忘れる。
    """
    a, sym = nuclide["mass_number"], nuclide["element"]
    return [f"^{{{a}}}{sym}", f"^{a}{sym}", f"{str(a).translate(_SUPERSCRIPT_DIGITS)}{sym}",
            f"{sym}-{a}", f"{a}{sym}"]


def find_nuclide(text: str) -> "Optional[dict]":
    r"""*text* に書かれている核種。無ければ None。

    **長い綴りから照合し、境界を見る。** `13C` は `113Cd` の部分文字列なので、
    素朴な `in` では $^{113}$Cd を $^{13}$C と読み違える（実測で確かめた罠）。
    """
    if not text:
        return None
    best: "tuple[int, Optional[dict]]" = (0, None)
    for nuc in load_nuclides():
        for spelling in nuclide_spellings(nuc):
            i = text.find(spelling)
            while i >= 0:
                before = text[i - 1] if i > 0 else ""
                after = text[i + len(spelling):i + len(spelling) + 1]
                if not before.isdigit() and not (after.isalnum() and after.isascii()):
                    if len(spelling) > best[0]:
                        best = (len(spelling), nuc)
                    break
                i = text.find(spelling, i + 1)
    return best[1]


def projections_for_nuclide(text: str) -> "Optional[List[int]]":
    """*text* に書かれた核種の射影の値集合（分子）。核種が無ければ None。"""
    nuc = find_nuclide(text)
    return None if nuc is None else ladder_numerators(int(nuc["two_s"]))
