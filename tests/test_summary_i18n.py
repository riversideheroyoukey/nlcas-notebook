"""i18n of the branch-history summary (I18N-SUMMARY, open_items).

_generate_summary_text used to hardcode the Japanese header / assumption / no-step
strings; they are now looked up via t(key, engine.lang).  Covers JA + EN.
"""
from notebook.engine import NotebookEngine
from ui.routes.history import _generate_summary_text


def _summary(lang, *, empty=False):
    with NotebookEngine(title="summary i18n") as e:
        e.session.language = [lang]
        if not empty:
            e.register_expression("a x + b")
            e.add_assumption("x > 0", display="x > 0")
        return _generate_summary_text(e)


def test_summary_ja():
    s = _summary("ja")
    assert "【ブランチ:" in s and "】" in s
    assert "式 1 件" in s
    assert "仮定 1 件" in s
    assert "仮定: x > 0" in s


def test_summary_en():
    s = _summary("en")
    assert "[Branch:" in s and "]" in s
    assert "1 expression(s)" in s
    assert "1 assumption(s)" in s
    assert "Assumption: x > 0" in s
    # no Japanese leakage in the English summary
    assert "件" not in s and "仮定" not in s and "ブランチ" not in s


def test_no_steps_ja():
    assert "（ステップなし）" in _summary("ja", empty=True)


def test_no_steps_en():
    s = _summary("en", empty=True)
    assert "(no steps)" in s
    assert "ステップなし" not in s
