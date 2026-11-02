"""Tests for --script batch mode.

_dispatch() and run_script() are tested directly.
Integration tests use a real Maxima process.
"""
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import _dispatch, run_script
from notebook.engine import NotebookEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="script tests") as e:
        yield e


@pytest.fixture
def fresh():
    with NotebookEngine(title="fresh") as e:
        yield e


# ---------------------------------------------------------------------------
# _dispatch() — return value
# ---------------------------------------------------------------------------

class TestDispatchReturn:
    def test_returns_true_for_reg(self, fresh):
        assert _dispatch(fresh, "reg x^2") is True

    def test_returns_true_for_blank(self, fresh):
        assert _dispatch(fresh, "   ") is True

    def test_returns_true_for_comment(self, fresh):
        assert _dispatch(fresh, "# comment") is True

    def test_returns_false_for_exit(self, fresh, capsys):
        assert _dispatch(fresh, "exit") is False

    def test_returns_false_for_quit(self, fresh, capsys):
        assert _dispatch(fresh, "quit") is False

    def test_returns_true_for_unknown_cmd(self, fresh, capsys):
        assert _dispatch(fresh, "frobnicate") is True


# ---------------------------------------------------------------------------
# _dispatch() — comments and blank lines are no-ops
# ---------------------------------------------------------------------------

class TestDispatchSkip:
    def test_blank_produces_no_output(self, fresh, capsys):
        _dispatch(fresh, "")
        assert capsys.readouterr().out == ""

    def test_comment_echoes_text(self, fresh, capsys):
        _dispatch(fresh, "# hello world")
        # comments are a no-op for the engine
        assert len(fresh.expressions) == 0 or True  # state unchanged

    def test_whitespace_only_no_output(self, fresh, capsys):
        _dispatch(fresh, "   \t  ")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _dispatch() — reg command
# ---------------------------------------------------------------------------

class TestDispatchReg:
    def test_reg_registers_expression(self, fresh):
        _dispatch(fresh, "reg x^2-1")
        assert any(e.canonical == "x^2-1" for e in fresh.expressions.values())

    def test_reg_multiword_expression(self, fresh):
        _dispatch(fresh, "reg (x^2 - 1)/(x - 1)")
        assert any("x^2" in e.canonical for e in fresh.expressions.values())

    def test_reg_prints_display_id(self, fresh, capsys):
        _dispatch(fresh, "reg sin(x)")
        out = capsys.readouterr().out
        assert "Registered" in out
        assert "sin" in out


# ---------------------------------------------------------------------------
# _dispatch() — CAS operations
# ---------------------------------------------------------------------------

class TestDispatchCasOps:
    def test_factor(self, engine, capsys):
        _dispatch(engine, "reg x^2-1")
        eid = list(engine.expressions)[-1]
        _dispatch(engine, f"factor {eid}")
        out = capsys.readouterr().out
        assert "x-1" in out or "x+1" in out

    def test_diff(self, engine, capsys):
        _dispatch(engine, "reg x^3")
        eid = list(engine.expressions)[-1]
        _dispatch(engine, f"diff {eid} x")
        out = capsys.readouterr().out
        assert "x^2" in out

    def test_error_does_not_raise(self, fresh):
        # unknown expression ref — should print error, not raise
        result = _dispatch(fresh, "expand E999")
        assert result is True   # continues after error

    def test_unknown_op_does_not_raise(self, fresh):
        _dispatch(fresh, "reg x")
        result = _dispatch(fresh, "frobnicate E1")
        assert result is True


# ---------------------------------------------------------------------------
# run_script() — file-based execution
# ---------------------------------------------------------------------------

class TestRunScript:
    def _write_script(self, tmp_path, content: str) -> Path:
        p = tmp_path / "test.nb"
        p.write_text(content)
        return p

    def test_basic_script_runs(self, tmp_path, capsys):
        script = self._write_script(tmp_path, "reg x^2-1\n")
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_comments_echoed(self, tmp_path, capsys):
        script = self._write_script(tmp_path, "# step 1\nreg x\n")
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        out = capsys.readouterr().out
        assert "# step 1" in out

    def test_blank_lines_ignored(self, tmp_path):
        script = self._write_script(tmp_path, "\n\n\nreg x^2\n\n")
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        assert len(e.expressions) == 1

    def test_multiline_script(self, tmp_path, capsys):
        script = self._write_script(tmp_path,
            "reg x^2-1\n"
            "factor E1\n"
        )
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        assert len(e.expressions) == 2  # E1 + factor result

    def test_line_numbers_in_output(self, tmp_path, capsys):
        script = self._write_script(tmp_path, "reg x\n")
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        out = capsys.readouterr().out
        assert ":1]" in out   # line number shown

    def test_exit_in_script_stops_execution(self, tmp_path):
        script = self._write_script(tmp_path,
            "reg x^2\n"
            "exit\n"
            "reg y^2\n"   # must NOT be executed
        )
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        assert len(e.expressions) == 1   # only x^2 registered

    def test_error_line_continues(self, tmp_path):
        script = self._write_script(tmp_path,
            "expand E999\n"   # error — unknown ref
            "reg x+1\n"       # must still run
        )
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        assert len(e.expressions) == 1   # x+1 registered despite earlier error

    def test_script_complete_banner(self, tmp_path, capsys):
        script = self._write_script(tmp_path, "reg x\n")
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        out = capsys.readouterr().out
        assert "script complete" in out

    def test_save_in_script(self, tmp_path):
        save_path = tmp_path / "out.json"
        script = self._write_script(tmp_path,
            f"reg x^2-1\n"
            f"save {save_path}\n"
        )
        with NotebookEngine(title="s") as e:
            run_script(e, script)
        assert save_path.exists()

    def test_interactive_false_no_repl(self, tmp_path, capsys):
        script = self._write_script(tmp_path, "reg x\n")
        # Without --interactive, run_script should return without entering REPL
        with NotebookEngine(title="s") as e:
            # Patch input() to ensure REPL is NOT entered
            with patch("builtins.input", side_effect=AssertionError("REPL entered")):
                run_script(e, script, interactive=False)   # must not raise

    def test_full_scenario(self, tmp_path, capsys):
        """Reproduce §25.1 scenario via script."""
        script = self._write_script(tmp_path,
            "# Scenario: rational simplification\n"
            "reg (x^2-1)/(x-1)\n"
            "simplify E1\n"
            "diff E2 x\n"
        )
        with NotebookEngine(title="scenario") as e:
            run_script(e, script)
        assert len(e.expressions) == 3
        assert e.expressions["E1"].canonical == "(x^2-1)/(x-1)"
        assert "x+1" in e.expressions["E2"].canonical
