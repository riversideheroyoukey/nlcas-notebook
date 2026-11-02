"""`docs/open_items.md` の索引と本文が食い違っていないかを機械的に検査する。

**なぜ必要か（2026-08-12 の突き合わせで実測した3件）。** この台帳は「優先度一覧の
表」と「本文の節」の二重管理になっており、片方だけを編集したときに検出する仕組みが
無かった。実際に壊れていたのは全て**編集時の巻き添え**で、書き足しの誤りは1件も
無い:

1. **課題ごと消えた。** `B-2（残）` の節が、隣接する `B-10` の節を削除するついでに
   消えていた（2026-08-08、`d74b7f9`）。先送り表の行だけが残ってリンク切れになり、
   課題の内容が4日間ドキュメントから失われていた。
2. **成果が消えた。** `ONE-QUESTION-MANY-ANSWERS` の状態欄が13コミット古く、本文には
   3期ぶんの実施記録があるのに「未着手・設計段階」と書かれていた。
3. **項目が索引から見えなかった。** `J-3` は本文に節があり規模小・優先度低の実施可能な
   項目なのに、優先度一覧のどの表にも載っていなかった。

この検査が catch できるのは 1 と 3（構造の食い違い）で、**2 は catch できない**——
状態欄の記述が古いことは、文章を読まないと分からない。そこは人間の突き合わせが要る。

**変異試験で確認済み（2026-08-12）。** 検査を書いただけでは「効いている」ことに
ならないので、壊して落ちるかを1件ずつ測った:

| 変異 | 結果 |
|---|---|
| `B-2（残）` の節を消す（**実際に起きた壊れ方**） | ✓ 落ちる |
| `J-3` の索引行を消す（**実際に起きた壊れ方**） | ✓ 落ちる |
| 参照を「scratchpad の \\`split_audit.py\\`」に戻す（**同上**） | ✓ 落ちる |
| 参照ファイルパスを存在しない名前に変える | ✓ 落ちる |
| アンカーの綴りを崩す（`#bp-5-julia` → `#bp-5-Julia`） | ✓ 落ちる |
| 引き継ぎメモを README の一覧から外す | ✓ 落ちる |
| 見出しだけ改名して索引を放置 | ✓ 落ちる（2件） |
| 対照: 無害なコメントを足す | ✓ 落ちない（偽陽性なし） |

最初の実行では2つの変異が空振りしていた（BSD `sed` の空正規表現再利用が効かず、
ファイルが変わっていなかった）。**変異試験は「変異が適用されたこと」まで確かめないと、
検査が通ったのか変異が無かったのかを区別できない。**

CAS を起動しないので `-m "not slow"` の高速サブセットに入る（全体 0.05 秒）。
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "open_items.md"
HANDOFF_DIR = REPO / "docs" / "handoffs"

# 索引に載っていなくてよい本文の節と、その理由。
# 増やすときは理由を書くこと（「載せ忘れ」を「例外」で隠さないため）。
INDEX_EXEMPT = {
    "J-1": "「既知の制限・設計上の決定事項」であって着手対象ではない",
    "BP-3": "F-3 item J への参照のみで、独立した課題ではない",
}

# 実在しなくてよい参照パスと、その理由。
PATH_EXEMPT = {
    "tests/test_backend_unification.py": "C-Ext-4 Phase 5 の予定ファイル（未実装と明記）",
}

# `supervisor_hint.txt` のように、特定の場所を指さず一般名として書かれるもの。
GENERIC_BASENAMES = {
    "supervisor_hint.txt", "ocr_hint.txt", "meta.json",
    "categories.json", "endpoint.json",
}

# 参照パスを探すルート（台帳はリポジトリ相対でもファイル名だけでも書かれている）
SEARCH_ROOTS = [
    "", "docs", "docs/developer_manuals", "docs/handoffs", "docs/scenarios",
    "docs/archive", "tests", "notebook", "notebook/runtime", "notebook/models",
    "notebook/numeric", "ui", "ui/routes", "scripts", "scripts/audit",
    "i18n", "domains", "backends",
]


def _slug(heading: str) -> str:
    """見出し → アンカー。

    GitHub の変換を厳密に再現するものではない。**必要なのは自己整合性**——
    同じ関数で見出しとリンクを突き合わせれば、手書きのアンカーが見出しから
    ずれたこと（実際に `MIXED-SCRIPT-NESTING` で下線が落ちていた）は捕まる。
    GitHub 上で本当に飛べるかは、この検査の保証範囲ではない。
    """
    h = re.sub(r"[`*\[\]]", "", heading.strip()).lower()
    return "".join(
        c if (c.isalnum() or c in "-_") else "-" if c == " " else ""
        for c in h
    )


@pytest.fixture(scope="module")
def ledger():
    text = LEDGER.read_text(encoding="utf-8")
    lines = text.splitlines()
    archive_at = next(
        (i for i, l in enumerate(lines) if l.startswith("## 実装済み項目")),
        len(lines))
    index_end = next(
        (i for i, l in enumerate(lines) if l.startswith("## 実践ループ")),
        archive_at)
    return {
        "text": text,
        "lines": lines,
        "index": "\n".join(lines[:index_end]),
        "archive_at": archive_at,
        # 本文（アーカイブより手前）の節: (行番号, 見出し)
        "sections": [
            (i + 1, l[4:].strip())
            for i, l in enumerate(lines[:archive_at])
            if l.startswith("### ") and not l.startswith("### ▶")
        ],
        "headings": {
            _slug(m.group(1)) for m in re.finditer(r"^#{2,4}\s+(.+)$", text, re.M)
        },
    }


class TestAnchors:
    def test_every_internal_link_resolves(self, ledger):
        """`](#...)` の全てに対応する見出しがあること。

        リンク切れは「節が消えた」の症状として出る——`B-2（残）` はこれで
        検出できたはずだった。
        """
        bad = []
        for m in re.finditer(r"\]\(#([^)]+)\)", ledger["text"]):
            if m.group(1) not in ledger["headings"]:
                line = ledger["text"][: m.start()].count("\n") + 1
                bad.append(f"L{line}: #{m.group(1)}")
        assert not bad, (
            "対応する見出しが無い内部リンク:\n  " + "\n  ".join(bad)
            + "\n\n節が消えたか、アンカーの綴りが見出しからずれている。"
        )


class TestIndexCoverage:
    def test_every_open_item_appears_in_the_priority_index(self, ledger):
        """本文の節は、優先度一覧のどれかの表に載っていること。

        載っていない項目は一覧から見えず、実質存在しないのと同じになる
        （`J-3` が実際にそうなっていた）。
        """
        missing = []
        for line_no, heading in ledger["sections"]:
            ident = re.split(r"[.．]", heading)[0]
            ident = re.sub(r"（[^）]*）", "", ident).strip()
            if not ident or ident in INDEX_EXEMPT:
                continue
            if ident not in ledger["index"]:
                missing.append(f"L{line_no}: {ident} — {heading[:50]}")
        assert not missing, (
            "本文に節があるが優先度一覧に載っていない項目:\n  "
            + "\n  ".join(missing)
            + "\n\n表に行を足すか、意図的なら INDEX_EXEMPT に理由つきで登録する。"
        )

    def test_index_exemptions_still_exist(self, ledger):
        """INDEX_EXEMPT が実在の節を指していること（例外の化石化を防ぐ）。"""
        idents = {
            re.sub(r"（[^）]*）", "", re.split(r"[.．]", h)[0]).strip()
            for _, h in ledger["sections"]
        }
        stale = sorted(set(INDEX_EXEMPT) - idents)
        assert not stale, (
            f"INDEX_EXEMPT が存在しない節を指している: {stale}\n"
            "節が消えたか改名された。例外を消すこと。"
        )


class TestReferences:
    def test_no_reference_to_a_temporary_directory(self, ledger):
        """一時ディレクトリを指す参照は、書いた翌日に無効になる。

        実例: 残作業が計測プラグインを「scratchpad の `split_audit.py`」と
        書いていたが、実物は `scripts/audit/` に入っていた。

        **単語の言及と参照を区別する。** 「scratchpad にあると書いていたが」のような
        経緯の説明は参照ではない。欠陥の形は「一時ディレクトリの名前と、
        バッククォートのファイル名が同じ行に並ぶ」——`scratchpad の
        \\`split_audit.py\\`` がまさにそれで、スラッシュは付いていなかったので
        パスらしさでは判別できない。リテラルなパス（`/tmp/...`）は単独で拾う。
        """
        tmp_word = re.compile(r"scratchpad|一時ディレクトリ")
        tmp_path = re.compile(r"/private/tmp/|(?<![\w/])/tmp/")
        backticked_file = re.compile(r"`[^`]*\.[A-Za-z0-9]{1,4}`")
        bad = []
        for i, line in enumerate(ledger["lines"], 1):
            if tmp_path.search(line):
                bad.append(f"L{i}: {line.strip()[:90]}")
            elif tmp_word.search(line) and backticked_file.search(line):
                bad.append(f"L{i}: {line.strip()[:90]}")
        assert not bad, (
            "一時ディレクトリへの参照:\n  " + "\n  ".join(bad)
            + "\n\nリポジトリ内の実在パスに置き換えること。"
        )

    def test_every_referenced_path_exists(self, ledger):
        """バッククォートで囲まれたファイル参照が実在すること。"""
        pat = re.compile(
            r"`([A-Za-z0-9_./-]+\.(?:py|json|md|jl|txt|ps1|html|js|toml|ini))`")
        missing = []
        for i, line in enumerate(ledger["lines"], 1):
            for m in pat.finditer(line):
                p = m.group(1)
                if p in PATH_EXEMPT or Path(p).name in GENERIC_BASENAMES:
                    continue
                if p.startswith("_") or p.count("/") == 0 and p.startswith("scenario_"):
                    continue          # `_jp.md` 等の接尾辞パターン
                if any((REPO / r / p).exists() for r in SEARCH_ROOTS):
                    continue
                missing.append(f"L{i}: {p}")
        assert not missing, (
            "実在しないファイル参照:\n  " + "\n  ".join(missing)
            + "\n\n移動・改名されたか、未実装の予定ファイル（なら PATH_EXEMPT に"
              "理由つきで登録する）。"
        )

    def test_path_exemptions_are_still_absent(self):
        """PATH_EXEMPT のファイルが出来ていたら、例外を外すべき合図。"""
        landed = [p for p in PATH_EXEMPT if (REPO / p).exists()]
        assert not landed, (
            f"PATH_EXEMPT のファイルが実在するようになった: {landed}\n"
            "例外を外し、該当項目の状態も更新すること。"
        )


class TestHandoffIndex:
    """`docs/handoffs/README.md` の一覧と実ファイルの対応。

    引き継ぎメモは「状態」を一覧で持つので、同じ二重管理の問題を抱える——
    `one_question_many_answers_handoff.md` の状態が13コミット古かったのは
    ここと文書ヘッダの両方だった。
    """

    def test_every_handoff_doc_is_listed(self):
        readme = (HANDOFF_DIR / "README.md").read_text(encoding="utf-8")
        docs = sorted(
            p.name for p in HANDOFF_DIR.glob("*.md") if p.name != "README.md")
        missing = [d for d in docs if d not in readme]
        assert not missing, (
            f"README の一覧に無い引き継ぎメモ: {missing}\n"
            "「いま置いてあるもの」の表に行を足すこと（役目を終えたなら archive/ へ）。"
        )

    def test_every_listed_handoff_exists(self):
        readme = (HANDOFF_DIR / "README.md").read_text(encoding="utf-8")
        listed = set(re.findall(r"\(([a-z0-9_]+_handoff\.md)\)", readme))
        gone = sorted(n for n in listed if not (HANDOFF_DIR / n).exists())
        assert not gone, (
            f"README が指す引き継ぎメモが存在しない: {gone}\n"
            "archive/ へ移したなら、現役の一覧からは行を消すこと。"
        )


class TestSolvedItemsAreArchived:
    """解決した項目が本文と索引に残っていないこと。

    **なぜ必要か（2026-09-13 に測った）。** この文書は冒頭で自分の規則を述べている
    ——「解決済み項目は本文から除き、末尾の『実装済み項目（参照用アーカイブ）』に
    1行で記録する」。**その規則は 2026-08-17 を最後に適用されていなかった**:
    アーカイブ表の最新行が 2026-08-16 で、以後ひと月ぶんの解決済み項目は
    (a) 本文に節のまま（15節・816行＝本文の15%）か (b) 索引の行だけ、という形で
    溜まっていた。**中優先度の表28行のうち18行が解決済み**——「次に何をするか」を
    読む表の3分の2が、済んだ仕事だった。

    上の `TestIndexCoverage` の docstring が「状態欄の記述が古いことは文章を
    読まないと分からない」と書いているとおり、**記述の鮮度そのものは機械では
    見えない**。だがこの2つは形で見える:

    * 見出しが「—— 解決」で終わっていれば、その節は丸ごと済んでいる
      （「本体は解決、残は…」のような部分解決は末尾が違うので当たらない）;
    * 優先度表の最後の欄（規模／着手条件）が `済`・`完了` なら、その行は
      仕事の大きさではなく**状態**を書いている＝アーカイブへ移す時期を過ぎている。

    見えない残りの半分（本文の記述が古い）は人が読むしかない。**この検査は
    「全部見た」とは言わない**——言えるのは「この2つの形は残っていない」だけ。
    """

    def test_no_body_section_reads_as_fully_solved(self, ledger):
        FINISHED = ("解決", "完了", "根治", "済")
        bad = [
            f"L{n}: {h}" for n, h in ledger["sections"]
            if h.rstrip().rstrip("。.").endswith(FINISHED)
        ]
        assert not bad, (
            "本文に丸ごと解決済みの節が残っている:\n  " + "\n  ".join(bad)
            + "\n\n文書冒頭の規則どおり、末尾のアーカイブ表へ1行で移すこと"
            "（詳細は各設計文書・git 履歴・実例が持つ）。"
        )

    def test_no_priority_row_reads_as_finished(self, ledger):
        bad = []
        for i, line in enumerate(ledger["index"].splitlines(), start=1):
            if not line.startswith("| ["):
                continue
            last = re.split(r"(?<!\\)\|", line)[-2].strip()
            if last.strip("*") in ("済", "完了"):
                bad.append(f"L{i}: {line[:60]}… → 「{last}」")
        assert not bad, (
            "優先度一覧に、最後の欄が状態（済／完了）になっている行がある:\n  "
            + "\n  ".join(bad)
            + "\n\nその欄は規模／着手条件を書くところで、済んだ項目は"
            "アーカイブ表へ移す。"
        )
