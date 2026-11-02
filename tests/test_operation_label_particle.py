"""自分の目的語を取る display は、目的語ではなく**所有者**を取る。

タイムラインのカードのラベルは ``engine._visible_action`` が組み立て、最後に
``engine.action_on_expr`` で式参照を付ける。その日本語は ``{expr_ref} を{display}``
で、``展開する`` のような display を想定して書かれていた——ところが**103操作のうち
70**の日本語 display は ``方程式を解く``・``固有値を求める``・``実部を取り出す`` と
自分の ``を`` を既に持っており、結果は ``E5 を方程式を解く``、目的語が2つで動詞が1つ
という非文になる。

**実測 2026-08-27: そのラベルは公開実例に32件、既に出ていた。** それでも誰にも
指摘されなかったのは、`solve` のように**ラベルが画面に一度も届いていなかった**
操作が多かったからでもある（同日修正）。届くようにする以上、文として読めなければ
ならない。

英語にこの区別は要らない（``{display} [{expr_ref}]`` は目的語を取らない）ので、
2つの型の英語は同一である。
"""
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb

_SCHEMA = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent
     / "schemas" / "common_operations.schema.json").read_text(encoding="utf-8"))
_OPS = _SCHEMA["operations"] if isinstance(_SCHEMA, dict) and "operations" in _SCHEMA else _SCHEMA
_OPS = _OPS if isinstance(_OPS, list) else list(_OPS.values())

#: (operation, its Japanese display) for every operation whose display already
#: governs an object.  Read from the schema, not listed here: a new operation
#: with a `を` in its display joins this test by existing.
_WITH_WO = sorted(
    {(o.get("name") or o.get("operation") or "?", o["display"]["ja"])
     for o in _OPS
     if isinstance(o.get("display"), dict) and o["display"].get("ja")
     and "を" in o["display"]["ja"]})


@pytest.fixture(scope="module")
def engine():
    nb.start_engine(title="label_particle", mode="standard", language="ja",
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    # An EQUATION: the both-sides family drops its "両辺" prefix for anything
    # else (`_BOTH_SIDES_OPS`'s short form), and that branch is right — so the
    # long forms can only be exercised on an equation.
    eng.register_expression("a*x + b = 0", latex="a x + b = 0")
    yield eng
    nb.stop_engine()


class TestTheJapaneseLabelIsGrammatical:
    def test_the_schema_really_has_such_displays(self):
        """A guard: if the schema stopped spelling displays this way the
        parametrised test below would silently cover nothing."""
        assert len(_WITH_WO) >= 50, f"only {len(_WITH_WO)} — check the schema shape"

    @pytest.mark.parametrize("name,display", _WITH_WO)
    def test_an_object_taking_display_gets_the_possessor(self, engine, name, display):
        label = engine._visible_action(name, {"expr": "E1"})
        if not label.startswith("E1 "):
            pytest.skip(f"{name} builds its own sentence")
        assert not re.match(r"^E1 を.*を", label), label

    @pytest.mark.parametrize("name,expected", [
        ("expand",           "E1 を展開する"),
        ("to_polar",         "E1 を極形式に変換する"),
        ("normal_order",     "E1 を正規順序にする"),
    ])
    def test_a_display_without_one_still_takes_the_object(self, engine, name, expected):
        """The rule must not fire everywhere.  `極形式に変換する` and
        `正規順序にする` carry a case particle but mark a GOAL, not something
        the expression has, so they keep を."""
        assert engine._visible_action(name, {"expr": "E1"}) == expected

    @pytest.mark.parametrize("name,expected", [
        ("solve_equation", "E1 の方程式を解く"),
        ("eigenvalues",    "E1 の固有値を求める"),
        ("real_part",      "E1 の実部を取り出す"),
        # 両辺 is possessed by the expression too, though the particle marking
        # it is に / から rather than を (measured 2026-09-05: of the eight
        # displays with a case particle and no を, only these three want の —
        # `極形式に変換する` marks a goal and `数値的に解く` a manner).
        ("multiply_both_sides",      "E1 の両辺に乗じる"),
        ("add_to_both_sides",        "E1 の両辺に加える"),
        ("subtract_from_both_sides", "E1 の両辺から引く"),
        ("divide_both_sides",        "E1 の両辺を割る"),
    ])
    def test_the_spellings_that_were_published_wrong(self, engine, name, expected):
        assert engine._visible_action(name, {"expr": "E1"}) == expected


class TestThePublishedExamplesAreClean:
    def test_no_worked_example_carries_the_double_object(self):
        """32 of them did, until 2026-08-27.

        The check is EXACT — `{eid} を` followed by a schema display that itself
        governs an object — not "two を in one label".  Counting particles
        flagged ``E14 を maxima セッションへ渡す準備をしました``, a sentence a
        handler writes itself and which is perfectly grammatical: its two を
        belong to different verbs.  A crude check that fails on correct output
        would be removed by the next reader, and rightly.
        """
        displays = {d for _n, d in _WITH_WO}
        root = pathlib.Path(__file__).resolve().parent.parent / "domains"
        bad = []
        for f in sorted(root.glob("*/examples/*_jp.json")):
            for st in (json.loads(f.read_text(encoding="utf-8")).get("steps") or {}).values():
                va = (st or {}).get("visible_action") or ""
                m = re.match(r"^E\d+ を(.+)$", va)
                if m and m.group(1) in displays:
                    bad.append((f.name, va))
        assert not bad, f"{len(bad)} published labels read as two objects: {bad[:5]}"
