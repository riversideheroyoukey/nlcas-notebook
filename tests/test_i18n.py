"""Tests for notebook.runtime.i18n."""
import pytest
from notebook.runtime.i18n import t, reload


@pytest.fixture(autouse=True)
def fresh_cache():
    reload()
    yield
    reload()


def test_known_key_ja():
    result = t("engine.register", "ja", display_id="E1")
    assert result == "E1 を登録する"


def test_known_key_en():
    result = t("engine.register", "en", display_id="E1")
    assert result == "Register E1"


def test_fallback_to_en_for_unknown_lang():
    result = t("engine.register", "fr", display_id="E1")
    assert result == "Register E1"


def test_fallback_to_key_for_unknown_key():
    result = t("nonexistent.key", "ja")
    assert result == "nonexistent.key"


def test_no_kwargs():
    result = t("engine.user_input", "ja")
    assert result == "利用者入力"

    result = t("engine.user_input", "en")
    assert result == "User input"


def test_llm_unknown_fallback():
    result = t("engine.llm_unknown", "ja")
    assert result == "不明"

    result = t("engine.llm_unknown", "en")
    assert result == "unknown"


def test_partial_diff_with_var():
    result = t("engine.partial_diff", "ja", var="x")
    assert result == "x で偏微分する"

    result = t("engine.partial_diff", "en", var="x")
    assert result == "Partial derivative w.r.t. x"


def test_partial_diff_novar():
    result = t("engine.partial_diff_novar", "ja")
    assert result == "偏微分する"


def test_missing_kwarg_does_not_raise():
    # If kwargs are provided but a placeholder is missing from template, return template as-is
    result = t("engine.register", "ja")  # missing display_id
    assert "display_id" in result or result == "{display_id} を登録する"


def test_adviser_system_language_ja():
    result = t("adviser.system_language", "ja")
    assert "日本語" in result


def test_adviser_system_language_en():
    result = t("adviser.system_language", "en")
    assert "English" in result


def test_interpreter_assumed():
    result = t("interpreter.assumed", "ja", condition="x > 0")
    assert result == "✓ 仮定しました: x > 0"

    result = t("interpreter.assumed", "en", condition="x > 0")
    assert result == "✓ Assumed: x > 0"


class TestTheLanguageCodeIsJa:
    r"""言語コードは `ja`。`jp` は**言語コードではない**（2026-09-05）。

    `jp` はリポジトリ内に3か所あるが、いずれも言語コードではない——成果物の
    ファイル名接尾辞（`scenario_x_jp.tex`。`generate_scenario_examples` は
    `language` と `suffix` を別の引数として受ける）、その接尾辞に由来する
    コーパス行の欄名、そして Wigner–Eckart の量子数 j′ である。

    4つ目として `f1_lattice_mention` が名前付きグループを `(?P<jp>…)` と
    綴っていた——全191キーで**言語名のグループはそこだけ**で、`(?P<ja>…)` は
    ゼロだった。`ja` へ揃えたので、これはその固定である。

    **なぜ固定するか:** コーパス行の欄が `jp` であることを知らずに `ja` で読むと
    **日本語の語がすべて「0行」と出て、英語しか候補に上がらない**（`nl_worklist`
    を書き直したときに実際に一度そう読みかけた）。同じ綴りが「言語」と
    「ファイル名」の両方に使われている以上、言語側は1つに固定しておく。
    """

    def _tables(self):
        import json
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent / "i18n"
        return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}

    def test_no_i18n_table_uses_jp_as_a_language_key(self):
        found = []

        def walk(name, node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "jp":
                        found.append(f"{name}:{path}")
                    walk(name, v, f"{path}/{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(name, v, f"{path}[{i}]")

        for name, data in self._tables().items():
            walk(name, data)
        assert found == [], f"言語キーは 'ja'。'jp' が残っている: {found}"

    def test_no_nl_pattern_names_a_group_after_a_language(self):
        import json
        import pathlib
        import re
        path = (pathlib.Path(__file__).resolve().parent.parent
                / "i18n" / "nl_patterns.json")
        data = json.loads(path.read_text())
        bad = []
        for key, value in data.items():
            blob = json.dumps(value, ensure_ascii=False)
            for group in re.findall(r"\?P<([A-Za-z0-9_]+)>", blob):
                if group == "jp":
                    bad.append(key)
        assert bad == [], (
            "正規表現の名前付きグループを言語名 'jp' で綴らない"
            f"（'ja' に揃える）: {bad}"
        )

    def test_the_lattice_mention_group_is_read_under_its_new_name(self):
        """パターンだけ／消費側だけ直すと取り出す名前が変わる、を押さえる。"""
        from ui.interpret_f1 import _find_unrecognized_lattice_name
        assert _find_unrecognized_lattice_name("未知格子で 2x2 のモデル", "ja") == "未知"
        assert _find_unrecognized_lattice_name(
            "body-centered lattice with 2x2 sites", "en") == "body-centered"
