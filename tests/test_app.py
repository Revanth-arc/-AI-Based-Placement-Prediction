from __future__ import annotations

import pandas as pd
import pytest

import app as placement_app


class FakePipeline:
    classes_ = [0, 1]

    def predict(self, frame):
        return [1] * len(frame)

    def predict_proba(self, frame):
        return [[0.2, 0.8] for _ in range(len(frame))]


def ready_state():
    _, _, metadata = placement_app.prepare_dataset()
    metadata.update({
        "train_rows": 10,
        "validation_rows": 5,
        "test_rows": 5,
        "split": "Stratified 70/15/15 test split",
        "selection_metric": "Validation F1-score",
        "best_model": "Test Model",
        "best_validation_f1": 0.8,
    })
    metrics = [{
        "name": "Test Model", "status": "ready", "best": True,
        "accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.9,
        "validation_f1": 0.8, "test": {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.9, "confusion_matrix": [[4, 1], [1, 4]]},
    }]
    return {"Test Model": FakePipeline()}, metrics, metadata


@pytest.fixture
def client(monkeypatch):
    models, metrics, metadata = ready_state()
    monkeypatch.setattr(placement_app, "_models", models)
    monkeypatch.setattr(placement_app, "_metrics", metrics)
    monkeypatch.setattr(placement_app, "_metadata", metadata)
    monkeypatch.setattr(placement_app, "_model_errors", [])

    def ensure_trained(force=False):
        if force or placement_app._models is None:
            placement_app._models = models
            placement_app._metrics = metrics
            placement_app._metadata = metadata
            placement_app._model_errors = []

    monkeypatch.setattr(placement_app, "ensure_trained", ensure_trained)
    return placement_app.app.test_client()


def make_prediction_payload(metadata):
    payload = {"model": "Test Model"}
    for feature in metadata["features"]:
        payload[feature["name"]] = feature.get("default", "New York")
    return payload


def test_health_and_frontend(client):
    assert client.get("/api/health").status_code == 200
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI-Based Placement" in response.data
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/favicon.ico").status_code == 200
    script = client.get("/static/script.js")
    assert script.status_code == 200
    for route in (b"/api/health", b"/api/dataset", b"/api/metrics", b"/api/predict", b"/api/retrain", b"/api/dataset/reload"):
        assert route in script.data


def test_dataset_endpoint_returns_schema_preview_and_distribution(client):
    response = client.get("/api/dataset")
    assert response.status_code == 200
    body = response.get_json()
    assert body["rows"] == 12000
    assert body["target"] == "Placement Status"
    assert len(body["columns"]) == 11
    assert len(body["preview"]) == 5
    assert body["target_distribution"]["placement"] == 4610
    assert body["synthetic_demo_data"] is True
    assert "synthetic demo data" in body["data_quality_warning"].lower()
    assert "Student ID" in body["excluded_columns"]
    assert "Student ID" not in [feature["name"] for feature in body["features"]]


def test_metrics_endpoint_and_compatibility_alias(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert response.get_json()["best_model"] == "Test Model"
    assert client.get("/api/summary").status_code == 200
    assert client.get("/api/models").status_code == 200


def test_prediction_accepts_feature_schema_and_returns_probability(client):
    _, _, metadata = ready_state()
    response = client.post("/api/predict", json=make_prediction_payload(metadata))
    assert response.status_code == 200
    result = response.get_json()
    assert result["prediction"] == 1
    assert result["label"] == "Placement"
    assert result["placement_probability"] == 0.8
    assert result["evaluation"]["test_rows"] == 5
    assert result["evaluation"]["accuracy"] == 0.9
    assert "label" in result


def test_prediction_warns_for_values_outside_the_training_range(client):
    _, _, metadata = ready_state()
    payload = make_prediction_payload(metadata)
    numeric = next(feature for feature in metadata["features"] if feature["type"] == "number")
    payload[numeric["name"]] = numeric["maximum"] + 1

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    warnings = response.get_json()["input_warnings"]
    assert any(numeric["name"] in warning and "training range" in warning for warning in warnings)

    payload = make_prediction_payload(metadata)
    category = next(feature for feature in metadata["features"] if feature["type"] == "category")
    payload[category["name"]] = "Unseen training category"
    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    warnings = response.get_json()["input_warnings"]
    assert any(category["name"] in warning and "not observed" in warning for warning in warnings)


def test_prediction_rejects_missing_and_invalid_fields(client):
    _, _, metadata = ready_state()
    payload = make_prediction_payload(metadata)
    payload.pop(metadata["features"][0]["name"])
    assert client.post("/api/predict", json=payload).status_code == 400
    payload = make_prediction_payload(metadata)
    numeric = next(feature for feature in metadata["features"] if feature["type"] == "number")
    payload[numeric["name"]] = "not a number"
    assert client.post("/api/predict", json=payload).status_code == 400
    payload = make_prediction_payload(metadata)
    payload["unexpected"] = 1
    assert client.post("/api/predict", json=payload).status_code == 400
    assert client.post("/api/predict", data="{}", content_type="text/plain").status_code == 400


def test_default_model_set_includes_stronger_algorithms():
    model_names = list(placement_app._default_model_specs().keys())
    assert model_names == [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "K-Nearest Neighbors",
        "Gradient Boosting",
    ]
    assert "K-Nearest Neighbors" in model_names
    assert "Gradient Boosting" in model_names


def test_generated_dataset_reaches_target_accuracy_and_precision(monkeypatch, tmp_path):
    dataset_path = tmp_path / "placement_dataset.csv"
    placement_app._create_default_dataset(dataset_path)
    monkeypatch.setattr(placement_app, "DATA_PATH", dataset_path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")
    X, y, _ = placement_app.prepare_dataset()
    models, metrics, metadata, errors = placement_app.train_models()
    assert models
    assert not errors
    best = max(metrics, key=lambda row: row["validation_f1"])
    assert best["test"]["accuracy"] > 0.90
    assert best["test"]["precision"] > 0.90
    assert metadata["best_model"] in models
    assert X.groupby(y)["Internships"].mean()[1] > X.groupby(y)["Internships"].mean()[0]


def test_student_id_is_excluded_from_prediction_features(monkeypatch, tmp_path):
    path = tmp_path / "student_ids.csv"
    pd.DataFrame({
        "Student ID": [f"STU{i:03d}" for i in range(20)],
        "CGPA": list(range(20)),
        "Placement Status": [0, 1] * 10,
    }).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")

    _, _, metadata = placement_app.prepare_dataset()

    assert "Student ID" in metadata["excluded_columns"]
    assert "Student ID" not in [feature["name"] for feature in metadata["features"]]


def test_retrain_and_reload_endpoints(client):
    assert client.post("/api/retrain", json={}).status_code == 200
    response = client.post("/api/dataset/reload", json={})
    assert response.status_code == 200
    assert response.get_json()["rows"] == 12000


def test_dataset_endpoint_reports_missing_file(client, monkeypatch):
    monkeypatch.setattr(placement_app, "DATA_PATH", placement_app.BASE_DIR / "tests" / "missing.csv")
    response = client.get("/api/dataset")
    assert response.status_code == 400
    assert "not found" in response.get_json()["error"].lower()


def test_duplicate_rows_missing_values_and_yes_no_targets(monkeypatch):
    rows = [{" City ": f"Town {i % 3}", "Placement Status": "YES" if i % 2 else "No", "CGPA": None if i == 3 else float(i)} for i in range(20)]
    rows.append(rows[0].copy())
    path = placement_app.BASE_DIR / "tests" / "placement_case.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")
    try:
        X, y, metadata = placement_app.prepare_dataset()
    finally:
        path.unlink(missing_ok=True)
    assert len(X) == len(y) == 20
    assert metadata["duplicate_rows_removed"] == 1
    assert metadata["missing_values"] == 1
    assert y.sum() == 10
    assert metadata["target_mapping"]["1"].lower() == "yes"


def test_ambiguous_target_requires_configuration(monkeypatch):
    path = placement_app.BASE_DIR / "tests" / "ambiguous_case.csv"
    pd.DataFrame({"Placement": [0, 1], "Placement Outcome": [0, 1], "CGPA": [40, 80]}).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")
    try:
        with pytest.raises(placement_app.DatasetError, match="More than one possible target"):
            placement_app.prepare_dataset()
    finally:
        path.unlink(missing_ok=True)


def test_configured_target_is_normalized(monkeypatch):
    path = placement_app.BASE_DIR / "tests" / "configured_case.csv"
    pd.DataFrame({"City": ["A", "B"] * 10, "Placement Status": ["true", "false"] * 10, "CGPA": list(range(20))}).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "placement_status")
    try:
        X, y, metadata = placement_app.prepare_dataset()
    finally:
        path.unlink(missing_ok=True)
    assert metadata["target"] == "Placement Status"
    assert y.nunique() == 2
    assert metadata["feature_count"] == 2


def test_temporal_split_keeps_each_date_in_one_partition(monkeypatch):
    path = placement_app.BASE_DIR / "tests" / "temporal_case.csv"
    rows = []
    for day in range(20):
        for location in ("North", "South"):
            rows.append({
                "Date": f"2025-01-{day + 1:02d}",
                "Location": location,
                "Humidity": 40 + day,
                "Placement Status": int((day + len(location)) % 2),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")
    try:
        X, y, metadata = placement_app.prepare_dataset()
        train_x, val_x, test_x, train_y, val_y, test_y, description = placement_app._split_dataset(X, y, metadata)
        source_dates = pd.read_csv(path, usecols=["Date"])["Date"]
        partitions = [train_x, val_x, test_x]
        date_sets = [set(pd.to_datetime(source_dates.reindex(part.index)).dt.strftime("%Y-%m-%d")) for part in partitions]
        assert not (date_sets[0] & date_sets[1] or date_sets[0] & date_sets[2] or date_sets[1] & date_sets[2])
        assert max(date_sets[0]) < min(date_sets[1]) < max(date_sets[1]) < min(date_sets[2])
        assert "Chronological date-group split" in description
        assert "Date" not in X.columns
        assert (len(train_y), len(val_y), len(test_y)) == (28, 6, 6)
    finally:
        path.unlink(missing_ok=True)


def test_temporal_split_does_not_fall_back_to_random_when_a_date_range_lacks_a_class(monkeypatch):
    path = placement_app.BASE_DIR / "tests" / "temporal_class_case.csv"
    rows = []
    for day in range(20):
        for location in ("North", "South"):
            rows.append({
                "Date": f"2025-02-{day + 1:02d}",
                "Location": location,
                "Humidity": 40 + day,
                "Placement Status": int(day >= 17),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(placement_app, "DATA_PATH", path)
    monkeypatch.setattr(placement_app, "TARGET_ENV", "")
    try:
        X, y, metadata = placement_app.prepare_dataset()
        with pytest.raises(placement_app.DatasetError, match="will not switch to a random split"):
            placement_app._split_dataset(X, y, metadata)
    finally:
        path.unlink(missing_ok=True)
