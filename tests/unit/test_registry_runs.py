from sfe.models.registry import log_run, read_runs


def test_read_runs_empty(tmp_settings):
    assert read_runs().empty


def test_log_run_writes_jsonl_and_read_runs_flattens(tmp_settings):
    log_run(
        "baseline",
        params={"horizon": "D-1"},
        metrics={"sign_brier": 0.21},
        tags={"go": "True"},
    )
    log_run("backtest", params={"horizon": "D-1"}, metrics={"net_margin_eur_per_mwh": 0.003})

    all_runs = read_runs()
    assert set(all_runs["name"]) == {"baseline", "backtest"}
    assert "param.horizon" in all_runs.columns
    assert "metric.sign_brier" in all_runs.columns
    assert "tag.go" in all_runs.columns

    only_baseline = read_runs("baseline")
    assert len(only_baseline) == 1
    assert only_baseline.iloc[0]["metric.sign_brier"] == 0.21
    # sorted by logged_at descending (ties possible at low clock resolution)
    assert list(all_runs["logged_at"]) == sorted(all_runs["logged_at"], reverse=True)
