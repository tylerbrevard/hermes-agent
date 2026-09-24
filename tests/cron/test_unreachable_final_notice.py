"""A suppressed failure notice must have another run left to deliver a result (#109990).

Contract: ``_save_compose_deliver`` holds the failure notice only while the
unreachable-retry ladder can actually schedule a re-run. On the final allowed run of a
finite repeat-limited recurring job, ``mark_job_run`` retires the record (no retry can
follow), so the ordinary failure notice must go out instead.
"""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.parametrize("repeat", [1, 2, None])
def test_retry_notice_matches_remaining_runs(tmp_cron_home, monkeypatch, repeat):
    from cron import scheduler
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job("daily report", "every 24h", repeat=repeat)
    job["_model_unreachable"] = True
    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *args, **kw: delivered.append(args[1])
    )
    outcome = scheduler._RunDelivery(job, False, "ConnectError: unavailable")
    scheduler._save_compose_deliver(
        outcome,
        scheduler._FireOwnership(job, None),
        "",
        "Connection failed",
        adapters=None,
        loop=None,
        verbose=False,
        execution_token=object(),
    )
    assert mark_job_run(job["id"], False, outcome.error, model_unreachable=True)
    remaining = get_job(job["id"])
    assert remaining is not None
    another_run = remaining["next_run_at"] is not None
    # XOR: exactly one of (notice delivered, another run remains) — never neither.
    assert bool(delivered) is not another_run
    if repeat == 1:
        # The only allowed run failed unreachable: the job is completed with no retry,
        # so its failure notice must have been delivered.
        assert remaining["repeat"]["completed"] == repeat
        assert not another_run
        assert delivered
