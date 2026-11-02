"""代入の文は2つの規則が読む —— ならば返すものは1つでなければならない。

NL-GRAMMAR ① の四問目。コーパスに残っていた割れ6件のうち**5件が代入**で、
どれも「同じ要求を日英で別の規則が読む」形だった:

    jp `E2 の A を B に置換して`        → subst_replace   (route `substitute`)
    en `In E2, substitute B for A`      → substitute_into (route `substitute_into`)
    jp `E6 の q に p を代入して`         → substitute_into   ← 鏡
    en `substitute p for q in E6`       → subst_replace     ← 鏡

コーパスはこれを「答えは同じで、答えた規則だけが違う」ドリフト検知として記録して
いた。実測するとそうではなかった（2026-08-27）:

* ``substitute_into`` の返す辞書だけ **``operation_label`` が無い**。この欄は
  タイムラインのカードのラベルそのもの（``notebook_timeline.js``）なので、
  同じ要求が**片方の言語でだけラベル無しのカードになる**。担当が言語で逆なので、
  欠けるのは英語のときもあれば日本語のときもある。
* 出ていたラベルのほうも壊れていた。``_visible_action`` の代入の枝が
  ``[{expr_ref}] lhs → rhs`` を組み立て、そのあと ``engine.action_on_expr`` が
  参照を**もう一度**足していた——``E1 を[E1] x → …`` / ``[E1] q → p [E1]``。

**そして最初この検査は返り値の欄しか見ておらず、直したつもりで直っていなかった。**
``operation_label`` を足しても画面は変わらない: ``substitute_into`` は自分の結果を
``registered_expressions`` にも append しており、UI はそれを**先に**描いてから
``data`` を「表示済み」で飛ばす（``notebook_main.js`` 230-247）ので、研究者に届く
カードは append 側——ラベルを持たないほう——だった。だからここは**描かれるカード**を
UI の分岐そのままに再現して見る。欄を見て済ませたのが、この一問での私の誤りである。

この検査が押さえるのは、どちらの規則が読んだかに依らない**性質**である。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


#: (name, setup, jp, en) — the spellings are the corpus's own, reduced to
#: expressions this file can register without a scenario.
PAIRS = [
    ("に置換",   "L = r^2*x/2",   "E1 の x を -2*y/r に置換して",
                                  "In E1, substitute -2*y/r for x"),
    ("に代入",   "H = p*q",       "E1 の q に p を代入して",
                                  "substitute p for q in E1"),
    ("で置き換え", "L = sin(q) + 1", "E1 の sin(q) を q で置き換えて",
                                  "In E1, substitute q for sin(q)"),
]


def _rendered_cards(data):
    """``notebook_main.js`` 230-247 の分岐そのまま。

    `registered_expressions` の各項が先に描かれ、そのどれかが `data.expr_id` と
    一致すれば `data` のカードは描かれない。
    """
    cards = [("registered", e.get("expr_id"), e.get("operation_label"))
             for e in (data.get("registered_expressions") or [])]
    already = any(e.get("expr_id") == data.get("expr_id")
                  for e in (data.get("registered_expressions") or []))
    if not already:
        cards.append(("data", data.get("expr_id"), data.get("operation_label")))
    return cards


def _do(lang, setup, text):
    nb.start_engine(title="subst_label", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    try:
        _run(eng, sup, setup)
        return _run(eng, sup, text)
    finally:
        nb.stop_engine()


class TestBothRulesAnswerTheSameWay:
    @pytest.mark.parametrize("name,setup,jp,en", PAIRS)
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_card_the_researcher_sees_has_a_label(self, name, setup, jp, en, lang):
        """Whichever rule wins, the card that is RENDERED carries the label.

        Not ``r["operation_label"]`` — that field can be present on a response
        whose card never gets drawn from it.
        """
        r = _do(lang, setup, jp if lang == "ja" else en)
        assert r["status"] == "completed", r
        cards = _rendered_cards(r)
        assert cards, r
        for src, eid, label in cards:
            assert label, (
                f"{lang}: the rendered card (from {src}) has no label — the two "
                f"rules return different shapes again")

    @pytest.mark.parametrize("name,setup,jp,en", PAIRS)
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_expression_reference_appears_exactly_once(
            self, name, setup, jp, en, lang):
        """`E1 を[E1] x → …` — the branch built the reference and the generic
        wrapper added it again."""
        r = _do(lang, setup, jp if lang == "ja" else en)
        for _src, _eid, label in _rendered_cards(r):
            assert label.count("E1") == 1, label

    @pytest.mark.parametrize("name,setup,jp,en", PAIRS)
    def test_the_two_languages_reach_the_same_value(self, name, setup, jp, en):
        """The premise the label check rests on: these really are one request."""
        assert _do("ja", setup, jp)["canonical"] == _do("en", setup, en)["canonical"]

    @pytest.mark.parametrize("name,setup,jp,en", PAIRS)
    def test_the_two_languages_show_the_same_replacement(self, name, setup, jp, en):
        """Same `lhs → rhs`, differently framed — the frame is the i18n
        template's business, the arrow is the operation's."""
        ja = _rendered_cards(_do("ja", setup, jp))[0][2]
        en_ = _rendered_cards(_do("en", setup, en))[0][2]
        core = lambda s: s.replace("E1", "").replace("の", "").replace("in", "").strip()
        assert core(ja) == core(en_), (ja, en_)
