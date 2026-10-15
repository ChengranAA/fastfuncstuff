import subprocess
from types import SimpleNamespace

from fastfuncstuff.benchmark.stages import glmsingle_matlab as stage


def _ctx(tmp_path, *, force_ref):
    noted = {}
    ctx = SimpleNamespace(
        glmsingle_dir=tmp_path,
        force_ref=force_ref,
        data_dir=tmp_path,
        verbose=False,
        note_items=lambda role, ran, total: noted.update({role: (ran, total)}),
    )
    return ctx, noted


def _no_matlab(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("MATLAB must not be launched")

    monkeypatch.setattr(subprocess, "Popen", boom)


def test_existing_outputs_never_launch_matlab(tmp_path, monkeypatch):
    # -force-ffs reaches every stage; a reference stage must obey -force-ref only.
    for name in stage._EXPECTED_OUTPUTS:
        (tmp_path / name).write_text("")
    _no_matlab(monkeypatch)
    ctx, noted = _ctx(tmp_path, force_ref=False)

    assert stage.run_ref(ctx) == 0.0
    assert noted["ref"] == (0, 1)


def test_force_ref_still_reruns(tmp_path, monkeypatch):
    for name in stage._EXPECTED_OUTPUTS:
        (tmp_path / name).write_text("")
    launched = []

    def fake_run(cmd, log_file, cwd):
        launched.append(cmd)
        return 1.0

    monkeypatch.setattr(stage, "_run_matlab_with_log", fake_run)
    monkeypatch.setattr(stage, "_matlab_script", lambda ctx: tmp_path / "x.m")
    ctx, _ = _ctx(tmp_path, force_ref=True)

    assert stage.run_ref(ctx) == 1.0
    assert "rerun = true" in launched[0]


def test_matlab_output_is_not_read_through_a_pipe(tmp_path):
    # A detached MathWorksServiceHost inherits an stdout pipe and a reader that
    # waits for EOF then waits on the daemon.  A child that backgrounds a sleeper
    # holding stdout must not block the stage.
    log = tmp_path / "m.log"
    elapsed = stage._run_matlab_with_log("echo hi; (sleep 30 &) ; exit 0", log, tmp_path)
    assert elapsed < 10
    assert "hi" in log.read_text()
