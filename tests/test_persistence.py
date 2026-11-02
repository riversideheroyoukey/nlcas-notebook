"""Tests for NotebookEngine save/load persistence.

Covers:
  - Metadata, expressions, steps, branches roundtrip.
  - ID counters resume correctly after load.
  - New operations work after load (self-contained CAS commands).
  - replay() after load rebuilds CAS state.
  - save → load → save produces identical JSON (idempotent).
"""
import json
import pytest
from pathlib import Path

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.models.step import StepKind, StepStatus
from ui.notebook_bridge import _save_fork_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def saved_session(tmp_path):
    """Engine with 2 expressions and 1 operation, saved to tmp_path/s.json."""
    path = tmp_path / "s.json"
    with NotebookEngine(title="persist test") as e:
        e1 = e.register_expression("x^2-1")
        e.run_operation("factor", {"expr": e1.display_id})
        e.save(path)
        meta = {
            "expr_count": len(e.expressions),
            "step_count": len(e.steps),
            "expr_counter": e._expr_counter,
            "step_counter": e._step_counter,
            "head": e.active_branch.head_step_id,
        }
    return path, meta


# ---------------------------------------------------------------------------
# save() — file is written and parseable
# ---------------------------------------------------------------------------

class TestSave:
    def test_file_created(self, tmp_path):
        path = tmp_path / "out.json"
        with NotebookEngine() as e:
            e.save(path)
        assert path.exists()

    def test_json_parseable(self, tmp_path):
        path = tmp_path / "out.json"
        with NotebookEngine() as e:
            e.register_expression("x+1")
            e.save(path)
        data = json.loads(path.read_text())
        assert "session" in data
        assert "expressions" in data
        assert "steps" in data
        assert "branches" in data
        assert "expr_counter" in data
        assert "step_counter" in data

    def test_version_field(self, tmp_path):
        path = tmp_path / "out.json"
        with NotebookEngine() as e:
            e.save(path)
        data = json.loads(path.read_text())
        assert data["version"] == "2"

    def test_expressions_serialized(self, tmp_path):
        path = tmp_path / "out.json"
        with NotebookEngine() as e:
            e.register_expression("sin(x)")
            e.save(path)
        data = json.loads(path.read_text())
        assert "E1" in data["expressions"]
        assert data["expressions"]["E1"]["canonical"] == "sin(x)"

    def test_rendered_command_preserved(self, tmp_path):
        path = tmp_path / "out.json"
        with NotebookEngine() as e:
            ex = e.register_expression("x^3")
            e.run_operation("differentiate", {"expr": ex.display_id, "var": "x"})
            e.save(path)
        data = json.loads(path.read_text())
        cas_step = [s for s in data["steps"].values() if s["rendered_command"]]
        assert any("diff(" in s["rendered_command"] for s in cas_step)


# ---------------------------------------------------------------------------
# load() — state is fully restored
# ---------------------------------------------------------------------------

class TestLoad:
    def test_session_title(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            assert e.session.title == "persist test"

    def test_session_mode(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            assert e.session.mode == SessionMode.standard_symbolic

    def test_expression_count(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            assert len(e.expressions) == meta["expr_count"]

    def test_step_count(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            assert len(e.steps) == meta["step_count"]

    def test_branch_head_restored(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            assert e.active_branch.head_step_id == meta["head"]

    def test_expr_counter_restored(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            assert e._expr_counter == meta["expr_counter"]

    def test_step_counter_restored(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            assert e._step_counter == meta["step_counter"]

    def test_canonical_value_preserved(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            assert e.expressions["E1"].canonical == "x^2-1"

    def test_rendered_command_preserved(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            cas_steps = [s for s in e.steps.values() if s.rendered_command]
            assert cas_steps
            assert any("factor(" in s.rendered_command for s in cas_steps)

    def test_step_kind_preserved(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            kinds = {s.kind for s in e.steps.values()}
            assert StepKind.expression_registration in kinds
            assert StepKind.cas_operation in kinds

    def test_active_branch_is_main(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            assert e.active_branch_id == "main"


# ---------------------------------------------------------------------------
# ID continuity after load
# ---------------------------------------------------------------------------

class TestIdContinuity:
    def test_next_expr_id_continues(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            new_expr = e.register_expression("x+1")
            expected = f"E{meta['expr_counter'] + 1}"
            assert new_expr.display_id == expected

    def test_next_step_id_continues(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            before = e._step_counter
            e.register_expression("y^2")
            assert e._step_counter == before + 1

    def test_no_id_collision(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            e.register_expression("z")
            all_ids = list(e.expressions.keys())
            assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Operations after load
# ---------------------------------------------------------------------------

class TestOperationsAfterLoad:
    def test_new_operation_on_loaded_expr(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            _, val = e.run_operation("expand", {"expr": "E1"})
            assert "x^2" in val

    def test_new_expr_then_operation(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            ex = e.register_expression("(x+1)^3")
            _, val = e.run_operation("expand", {"expr": ex.display_id})
            assert "x^3" in val

    def test_step_added_after_load(self, saved_session):
        path, meta = saved_session
        with NotebookEngine.load(path) as e:
            before = len(e.steps)
            e.run_operation("expand", {"expr": "E1"})
            assert len(e.steps) == before + 1


# ---------------------------------------------------------------------------
# Replay after load — CAS state rebuilt from scratch
# ---------------------------------------------------------------------------

class TestReplayAfterLoad:
    def test_replay_ok(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            result = e.replay_current_branch()
        assert result.ok

    def test_replay_restores_expressions(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            result = e.replay_current_branch()
        for eid in ["E1", "E2"]:
            assert eid in result.expr_values

    def test_replay_cas_result_correct(self, saved_session):
        path, _ = saved_session
        with NotebookEngine.load(path) as e:
            result = e.replay_current_branch()
        assert "(x" in result.expr_values.get("E2", "")

    def test_full_roundtrip(self, tmp_path):
        """save → PC-restart simulation (new engine) → load → replay → new operation."""
        path = tmp_path / "roundtrip.json"

        # original session
        with NotebookEngine(title="roundtrip") as e:
            ex = e.register_expression("x^2-4")
            e.run_operation("factor", {"expr": ex.display_id})
            e.save(path)

        # simulate restart: fresh engine loaded from file
        with NotebookEngine.load(path) as e:
            result = e.replay_current_branch()
            assert result.ok
            assert "(x" in result.expr_values.get("E2", "")

            # continue working
            _, val = e.run_operation("expand", {"expr": "E2"})
            assert "x^2" in val


# ---------------------------------------------------------------------------
# Idempotence: save → load → save produces identical JSON
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_double_save_identical(self, tmp_path):
        path1 = tmp_path / "a.json"
        path2 = tmp_path / "b.json"
        with NotebookEngine(title="idem") as e:
            e.register_expression("x^2")
            e.save(path1)

        with NotebookEngine.load(path1) as e:
            e.save(path2)

        assert json.loads(path1.read_text()) == json.loads(path2.read_text())


# ---------------------------------------------------------------------------
# Maxima snapshot save / load via session_io
# ---------------------------------------------------------------------------

class TestSnapshotWithSessionIO:
    """Verify that do_load works with and without a .mac snapshot file."""

    def test_snapshot_saved_alongside_json(self, tmp_path):
        """Saving via engine.save_cas_snapshot() creates a .mac file."""
        json_path = tmp_path / "sess.json"
        mac_path  = tmp_path / "sess.mac"
        with NotebookEngine(title="snap test") as e:
            e.register_expression("x^2 - 1")
            e.save(json_path)
            ok = e.save_cas_snapshot(mac_path)
        assert ok
        assert mac_path.exists()
        assert mac_path.stat().st_size > 0

    def test_load_without_snapshot_falls_back_to_replay(self, tmp_path):
        """If no .mac exists, loading from .json + replay still works."""
        json_path = tmp_path / "no_snap.json"
        mac_path  = tmp_path / "no_snap.mac"

        with NotebookEngine(title="no snap") as e:
            ex = e.register_expression("x^2 - 4")
            e.run_operation("factor", {"expr": ex.display_id})
            e.save(json_path)
        # deliberately do NOT save a .mac

        assert not mac_path.exists()
        # Load JSON only — should succeed via replay
        with NotebookEngine.load(json_path) as e2:
            result = e2.replay_current_branch()
            assert result.ok
            assert len(e2.expressions) == 2

    def test_corrupted_snapshot_does_not_prevent_json_open(self, tmp_path):
        """A broken .mac must not prevent the notebook from opening."""
        json_path = tmp_path / "broken.json"
        mac_path  = tmp_path / "broken.mac"

        with NotebookEngine(title="broken snap") as e:
            ex = e.register_expression("x + 1")
            e.save(json_path)

        # Write a command that always errors (error() returns not-ok via errcatch)
        mac_path.write_text('error("forced snapshot failure")$\n', encoding="utf-8")

        # Loading should still succeed (adapter returns False, code falls back to replay)
        with NotebookEngine.load(json_path) as e2:
            # load_cas_snapshot returns False on bad commands
            ok = e2.load_cas_snapshot(mac_path)
            assert not ok          # snapshot failed
            # notebook data is intact from JSON
            assert len(e2.expressions) == 1
            assert "E1" in e2.expressions

    def test_assume_persists_across_snapshot_cycle(self, tmp_path):
        """assume in session → save snapshot → fresh process → load snapshot → assume present."""
        json_path = tmp_path / "assume_snap.json"
        mac_path  = tmp_path / "assume_snap.mac"

        with NotebookEngine(title="assume snap") as e:
            e.add_assumption("gamma > 0")
            e.save(json_path)
            ok = e.save_cas_snapshot(mac_path)
        assert ok

        with NotebookEngine.load(json_path) as e2:
            assert e2.load_cas_snapshot(mac_path)
            facts = e2._adapter.execute("facts()").value
            assert "gamma" in facts


# ---------------------------------------------------------------------------
# Fork snapshot assumptions correctness
# ---------------------------------------------------------------------------

class TestForkSnapshotAssumptions:
    """_save_fork_snapshot must carry the correct active assumptions at the chosen step."""

    def _run_fork(self, engine, step_id, tmp_path, filename="fork.json"):
        import asyncio
        path = asyncio.run(
            _save_fork_snapshot(engine, step_id, filename)
        )
        # _save_fork_snapshot returns just the filename; find it in sessions/
        from ui.session_io import sessions_dir
        return sessions_dir() / path

    def test_assumption_included_when_added_before_fork_step(self, tmp_path):
        """assume added before fork step → present in forked JSON."""
        import asyncio
        with NotebookEngine(title="fork assume") as e:
            _, asm = e.add_assumption("alpha > 0")
            ex = e.register_expression("x^2")
            fork_step_id = list(e.steps.keys())[-1]
            fname = asyncio.run(
                _save_fork_snapshot(e, fork_step_id, "fork_assume_test")
            )
        from ui.session_io import sessions_dir
        data = json.loads((sessions_dir() / fname).read_text())
        assert any(v["canonical"] == "alpha > 0" for v in data["assumptions"].values())

    def test_assumption_excluded_when_added_after_fork_step(self, tmp_path):
        """assume added after fork step → absent in forked JSON."""
        import asyncio
        with NotebookEngine(title="fork assume after") as e:
            ex = e.register_expression("x^2")
            fork_step_id = list(e.steps.keys())[-1]
            e.add_assumption("beta > 0")   # added AFTER the fork point
            fname = asyncio.run(
                _save_fork_snapshot(e, fork_step_id, "fork_assume_after_test")
            )
        from ui.session_io import sessions_dir
        data = json.loads((sessions_dir() / fname).read_text())
        assert not any(v["canonical"] == "beta > 0" for v in data["assumptions"].values())

    def test_removed_assumption_excluded_from_fork(self, tmp_path):
        """assume → forget → fork: assumption must NOT appear in forked JSON."""
        import asyncio
        with NotebookEngine(title="fork forget") as e:
            e.add_assumption("mu > 0")
            e.remove_assumption_by_canonical("mu > 0")
            ex = e.register_expression("x")
            fork_step_id = list(e.steps.keys())[-1]
            fname = asyncio.run(
                _save_fork_snapshot(e, fork_step_id, "fork_forget_test")
            )
        from ui.session_io import sessions_dir
        data = json.loads((sessions_dir() / fname).read_text())
        assert not any(v["canonical"] == "mu > 0" for v in data["assumptions"].values())

    def test_assumption_removed_after_fork_step_still_present(self, tmp_path):
        """assume before fork step, forget after: assumption must appear in forked JSON."""
        import asyncio
        with NotebookEngine(title="fork remove after") as e:
            e.add_assumption("nu > 0")
            ex = e.register_expression("x")
            fork_step_id = list(e.steps.keys())[-1]
            e.remove_assumption_by_canonical("nu > 0")  # removed AFTER fork point
            fname = asyncio.run(
                _save_fork_snapshot(e, fork_step_id, "fork_remove_after_test")
            )
        from ui.session_io import sessions_dir
        data = json.loads((sessions_dir() / fname).read_text())
        assert any(v["canonical"] == "nu > 0" for v in data["assumptions"].values())


# ---------------------------------------------------------------------------
# R-1 (PR-1/PR-3): schema v2 gate + callable guard for legacy migrations
# ---------------------------------------------------------------------------

class TestSchemaV2NoMigration:
    """v2 files must round-trip without any legacy migration rewriting them."""

    def test_function_call_canonicals_survive_roundtrip(self, tmp_path):
        """f(x^2)/g(a*b)/q(t) must NOT be rewritten to products on load."""
        path = tmp_path / "v2.json"
        e = NotebookEngine(title="v2 roundtrip")
        ids = {}
        for canonical in ("f(x^2)", "g(a*b)", "q(t)"):
            ids[canonical] = e.register_expression(canonical).display_id
        e.save(path)
        loaded = NotebookEngine.load(path)
        for canonical, eid in ids.items():
            assert loaded.expressions[eid].canonical == canonical

    def test_v2_load_save_json_identical(self, tmp_path):
        """save → load → save must produce semantically identical canonicals."""
        p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
        e = NotebookEngine(title="v2 idem")
        e.register_expression("f(x^2)")
        e.register_expression("l*m*(l*omega^2)")
        e.save(p1)
        NotebookEngine.load(p1).save(p2)
        c1 = {k: v["canonical"] for k, v in json.loads(p1.read_text())["expressions"].items()}
        c2 = {k: v["canonical"] for k, v in json.loads(p2.read_text())["expressions"].items()}
        assert c1 == c2


class TestLegacyV1Migration:
    """v1 files still get the historical repairs — but never on declared callables."""

    @staticmethod
    def _save_as_v1(engine, path):
        engine.save(path)
        data = json.loads(path.read_text())
        data["version"] = "1"
        path.write_text(json.dumps(data, ensure_ascii=False))

    def test_v1_wrong_func_call_still_repaired(self, tmp_path):
        """Historical P2 damage m(l*omega^2) is still rewritten to m*(l*omega^2)."""
        path = tmp_path / "v1.json"
        e = NotebookEngine(title="v1 repair")
        eid = e.register_expression("m(l*omega^2)").display_id
        self._save_as_v1(e, path)
        loaded = NotebookEngine.load(path)
        assert loaded.expressions[eid].canonical == "m*(l*omega^2)"

    def test_v1_declared_function_not_rewritten(self, tmp_path):
        """A call whose callee is a declared function must survive a v1 load."""
        path = tmp_path / "v1f.json"
        e = NotebookEngine(title="v1 declared")
        e.register_quantity("f", kind="function")
        eid = e.register_expression("f(x^2)").display_id
        self._save_as_v1(e, path)
        loaded = NotebookEngine.load(path)
        assert loaded.expressions[eid].canonical == "f(x^2)"

    def test_v1_declared_operator_not_rewritten(self, tmp_path):
        path = tmp_path / "v1o.json"
        e = NotebookEngine(title="v1 op")
        e.register_quantity("a", kind="operator")
        eid = e.register_expression("a(k^2)").display_id
        self._save_as_v1(e, path)
        loaded = NotebookEngine.load(path)
        assert loaded.expressions[eid].canonical == "a(k^2)"

    def test_missing_version_key_treated_as_v1(self, tmp_path):
        """Pre-versioning files (no 'version' key) still get the repairs."""
        path = tmp_path / "v0.json"
        e = NotebookEngine(title="v0")
        eid = e.register_expression("m(l*omega^2)").display_id
        e.save(path)
        data = json.loads(path.read_text())
        del data["version"]
        path.write_text(json.dumps(data, ensure_ascii=False))
        loaded = NotebookEngine.load(path)
        assert loaded.expressions[eid].canonical == "m*(l*omega^2)"


# ---------------------------------------------------------------------------
# R-2 (PR-2): SymbolContext extras must survive save/load
# ---------------------------------------------------------------------------

class TestSymbolContextExtrasPersistence:
    def test_canonical_pairs_roundtrip(self, tmp_path):
        path = tmp_path / "pairs.json"
        e = NotebookEngine(title="pairs")
        e.symbol_context.add_canonical_pair("q", "p")
        e.symbol_context.add_canonical_pair("theta", "p_theta")
        e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.canonical_pairs == [
            ("q", "p"), ("theta", "p_theta")
        ]

    def test_spin_display_styles_roundtrip(self, tmp_path):
        path = tmp_path / "spin.json"
        e = NotebookEngine(title="spin")
        e.symbol_context.spin_display_styles["S"] = "superscript"
        e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.spin_display_styles == {"S": "superscript"}

    def test_einstein_summation_roundtrip(self, tmp_path):
        path = tmp_path / "einstein.json"
        e = NotebookEngine(title="einstein")
        e.symbol_context.einstein_summation = True
        e.save(path)
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.einstein_summation is True

    def test_old_file_without_extras_loads_defaults(self, tmp_path):
        """Files saved before v2 (no extra keys) load with safe defaults."""
        path = tmp_path / "old.json"
        e = NotebookEngine(title="old")
        e.save(path)
        data = json.loads(path.read_text())
        for key in ("canonical_pairs", "spin_display_styles", "einstein_summation"):
            data.pop(key, None)
        path.write_text(json.dumps(data, ensure_ascii=False))
        loaded = NotebookEngine.load(path)
        assert loaded.symbol_context.canonical_pairs == []
        assert loaded.symbol_context.spin_display_styles == {}
        assert loaded.symbol_context.einstein_summation is False

    def test_einstein_superscript_resolution_consistent_after_load(self, tmp_path):
        """A^i must resolve to an upper INDEX (not a power) after reload,
        exactly as it did before saving — the flag loss changed the meaning."""
        from notebook.models.ir import IndexedSymbol, Pow
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver

        path = tmp_path / "einstein_res.json"
        e = NotebookEngine(title="einstein res")
        e.symbol_context.einstein_summation = True
        parser, resolver = SharedParser(), MaximaSemanticResolver()
        before = resolver.resolve(parser.parse("A^i"), e.symbol_context)
        assert isinstance(before, IndexedSymbol)
        assert before.indices[0].is_up is True
        e.save(path)
        loaded = NotebookEngine.load(path)
        after = resolver.resolve(parser.parse("A^i"), loaded.symbol_context)
        assert isinstance(after, IndexedSymbol)
        assert after.indices[0].is_up is True
