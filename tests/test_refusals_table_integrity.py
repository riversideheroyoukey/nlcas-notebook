r"""`docs/refusals.md` の表そのものを検査する。

この一覧は研究者が「何が通らないか」を見る唯一の場所なので、**表として壊れたら
中身が正しくても届かない**。2026-08-18 の監査で、前日に自分が追加した縦棒の3行が
セルを分裂させていた——マークダウンの表ではバッククォートの中でも `|` が区切りに
なるため、`\|` と書く必要がある。行の内容は正しかったのに、表としては読めなかった。

もう1つ固定するのは 🟠/🟡 に起票 ID があること。ただし**この検査が見るのは ID の
有無だけで、その行が本当にその分類かは見ない**——同じ日にその取り違えで、そもそも
🟠 でない項目を起票して取り下げている（open_items INTEGRAL-VAR-INDEXED）。分類の
妥当性は人が実測して決めるものとして残る。
"""
import pathlib
import re

import pytest

DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "refusals.md"

#: 同じ検査を `open_items.md` にも掛ける。**2026-08-18 に3度、当方が同じ壊し方を
#: した**——アーカイブ行を書くとき本文に改行を埋め込み、マークダウンの表としては
#: 行が割れて続き段落が表の外に落ちていた（内容は残るが表としては読めない）。
#: refusals.md にだけ検査を付けて open_items.md に付けなかったのが、繰り返した理由。
OPEN_ITEMS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "open_items.md"

#: マークダウンの表がセルを分ける規則——`\` で逃がされていない `|` のみ。
_SPLIT = re.compile(r"(?<!\\)\|")


def _cells(line: str) -> "list[str]":
    return [c.strip() for c in _SPLIT.split(line.strip().strip("|"))]


def _tables() -> "list[tuple[int, int, str]]":
    """(行番号, 期待列数, 行) をデータ行ぶんだけ返す。

    この文書は3列・4列・5列の表を併せ持つので、**表ごとに**区切り行の列数を
    期待値とする（全体を1つの数で見ると誤検出になる——最初にそう書いて14件の
    偽陽性を出した）。
    """
    out, expect = [], None
    for n, line in enumerate(DOC.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip().startswith("|"):
            expect = None
            continue
        cells = _cells(line)
        if set("".join(cells)) <= set("-: "):
            expect = len(cells)
            continue
        if expect is None:
            continue
        out.append((n, expect, line))
    return out


def test_the_document_has_tables():
    rows = _tables()
    assert len(rows) > 40, f"only {len(rows)} data rows found"


def test_every_row_has_its_table_s_column_count():
    """壊れた行は、内容が正しくても読者に届かない。"""
    bad = [f"L{n}: {len(_cells(line))} cells != {expect} — {_cells(line)[0][:44]}"
           for n, expect, line in _tables() if len(_cells(line)) != expect]
    assert not bad, "行がセルに分裂している（`|` は `\\|` と書く）:\n  " + "\n  ".join(bad)


def test_a_literal_pipe_is_escaped():
    """バッククォートの中でも `|` は区切りになる——ここが 2026-08-18 の実害。"""
    bad = []
    for n, _expect, line in _tables():
        for m in re.finditer(r"`[^`]*`", line):
            if re.search(r"(?<!\\)\|", m.group(0)):
                bad.append(f"L{n}: {m.group(0)[:40]}")
    assert not bad, "コードスパン内の素の `|`:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("kind", ["🟠", "🟡"])
def test_interim_and_undecided_rows_name_a_ticket(kind):
    """規則2（と 2026-08-17 に広げた 🟠 版）。

    **この検査は ID の有無しか見ない。** その行が本当に 🟠 かは人が実測して決める
    ——同じ日に取り違えて、🟠 でない項目を起票し取り下げている。
    """
    bad = []
    for n, expect, line in _tables():
        cells = _cells(line)
        if expect != 5 or kind not in cells[2]:
            continue
        if "解決" in cells[2]:                     # 解決済みの記録行は起票不要
            continue
        if not re.search(r"起票\s*[A-Z][A-Z0-9\-]+|\bBP-\d\b", cells[-1]):
            bad.append(f"L{n}: {cells[0][:48]}")
    assert not bad, f"{kind} なのに起票 ID が無い行:\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------------------
# open_items.md —— 表の行が1行で閉じているか
# ---------------------------------------------------------------------------

def _open_items_rows() -> "list[tuple[int, str]]":
    """`| ` で始まる行だけ。字下げされた入れ子の表は対象外（本文の一部）。"""
    return [(n, line) for n, line in
            enumerate(OPEN_ITEMS.read_text(encoding="utf-8").split("\n"), 1)
            if line.startswith("| ")]


def test_open_items_has_tables():
    assert len(_open_items_rows()) > 60


def test_every_open_items_row_closes_on_one_line():
    r"""アーカイブ行の本文に改行を入れると、行が割れて続きが表の外に落ちる。

    2026-08-18 に当方が3度やった（INTEGRAL-VAR-INDEXED ほか5行 → 接合、
    EN-MULTI-OPSUBST ほか2行 → 接合）。内容は残るが**表としては読めない**ので、
    公開物としては壊れている。長い行になるが、1行で閉じること。
    """
    bad = [f"L{n}: {line[:56]}" for n, line in _open_items_rows()
           if not line.rstrip().endswith("|")]
    assert not bad, (
        "表の行が1行で閉じていない（本文に改行を入れない）:\n  " + "\n  ".join(bad))


def test_no_paragraph_is_stranded_after_a_table_row():
    """割れた行の続きは、表の外の裸の段落として残る——その形も直接見る。"""
    lines = OPEN_ITEMS.read_text(encoding="utf-8").split("\n")
    bad = []
    for n, line in enumerate(lines[1:], 2):
        prev = lines[n - 2]
        if (prev.startswith("| ") and prev.rstrip().endswith("|")
                and line.startswith("**") and line.rstrip().endswith("|")):
            bad.append(f"L{n}: {line[:56]}")
    assert not bad, "表から落ちた段落:\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------------------
# 種別の綴り —— 凡例に無いものが現れない
# ---------------------------------------------------------------------------

def _legend_kinds() -> "set[str]":
    """凡例表が定義している種別の記号（🟡 / 🟠 / 🔵 / —）。"""
    kinds = set()
    for _n, _e, line in _tables():
        cells = _cells(line)
        if len(cells) != 3:
            continue
        marker = cells[0].replace("*", "").strip()
        # 凡例行は「記号 + 名前」（`🟡 暫定`）か記号のみ（`—`）。他の3列表（F 表）と
        # 取り違えないよう、意味の列が種別の説明であることも見る。
        if marker and (marker.split()[0] in ("🟡", "🟠", "🔵", "—")
                       and ("拒否" in cells[1] or "読み方" in cells[1]
                            or "primitive" in cells[1] or "書き方" in cells[1])):
            kinds.add(marker.split()[0])
    return kinds


def test_the_legend_defines_its_kinds():
    r"""2026-08-19 に「—」を落とした——**この文書は「通らないもの」だけを載せる**。

    解決済みの記法を「—」種別でここに残していたが、目的（`\int d^3k` が通らない、を
    1箇所で）に反するうえ、測ると **user_guide.md の欠落をこの文書で埋め合わせて**
    いた（`\sin^2 x`・`\sin^{-1} x`・`\max_{i=1}^{3}` は user_guide に1件も無かった）。
    通る書き方は user_guide へ移し、断りの行が広すぎる限界を伝えないよう、対になる
    「通る形」は各行の理由・回避策に書く。
    """
    assert _legend_kinds() == {"🟡", "🟠", "🔵"}, _legend_kinds()


def test_every_kind_cell_uses_a_defined_symbol():
    r"""**2026-08-18 にここで穴が開いた。** 拒否を解いて「計算できる」行に変えていく
    過程で「—」という4つ目の種別を導入したが、**凡例に足していなかった**——文書自身の
    更新規則が「種別を必ず付ける」と言っている当のものである。綴りも `—` と
    `— 解決（元 🟡）` の2通りに割れていた。

    種別は読者が「もう考えなくてよいか」を判断する唯一の手がかりなので、
    **凡例に無い記号が現れないこと**を機械的に固定する。
    """
    legend = _legend_kinds()
    bad = []
    for n, expect, line in _tables():
        cells = _cells(line)
        if expect != 5 or cells[0] in ("記法", "状況", "記法・操作"):
            continue
        symbol = cells[2].replace("*", "").strip()[:1]
        if symbol and symbol not in legend:
            bad.append(f"L{n}: 種別 {cells[2]!r} — {cells[0][:40]}")
    assert not bad, "凡例に無い種別:\n  " + "\n  ".join(bad)
