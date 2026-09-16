from sfe.models.registry import load_model, load_model_meta, model_exists, save_model


def test_save_model_writes_meta_and_roundtrips(tmp_settings):
    save_model({"a": 1}, "dummy", meta={"horizon": "D-1", "n_rows": 42})
    assert model_exists("dummy")
    assert load_model("dummy") == {"a": 1}
    meta = load_model_meta("dummy")
    assert meta["horizon"] == "D-1"
    assert meta["n_rows"] == 42
    assert "saved_at" in meta


def test_missing_model(tmp_settings):
    assert model_exists("nope") is False
    assert load_model_meta("nope") is None
