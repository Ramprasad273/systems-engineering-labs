"""Unit tests for schema validation and evaluation telemetry metrics."""

from src.utils.metrics import (
    validate_json_schema,
    calculate_severity_metrics,
    calculate_binary_anomaly_metrics,
    compare_stage1_vs_stage2
)


def test_validate_json_schema():
    valid_json = '{"root_cause": "DataNode failure", "severity": "P1_CRITICAL", "affected_component": "DataNode", "mitigation_commands": ["restart"], "confidence": 0.95}'
    is_valid, parsed = validate_json_schema(valid_json)
    assert is_valid
    assert parsed["severity"] == "P1_CRITICAL"

    invalid_json = '{"root_cause": 123}'  # missing fields and bad types
    is_valid, _ = validate_json_schema(invalid_json)
    assert not is_valid


def test_calculate_binary_anomaly_metrics():
    preds = [1, 1, 0, 0, 1]
    targets = [1, 0, 0, 1, 1]
    res = calculate_binary_anomaly_metrics(preds, targets)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["tn"] == 1
    assert res["fn"] == 1
    assert round(res["accuracy"], 2) == 0.60


def test_compare_stage1_vs_stage2():
    s2_metrics = {
        "schema_compliance_rate": 94.5,
        "severity_metrics": {"macro_f1": 0.88},
        "binary_anomaly_metrics": {"accuracy": 0.9610, "precision": 0.9520, "recall": 0.8600, "f1": 0.9036},
        "latency_ms": {"p95": 1150.0}
    }
    comp = compare_stage1_vs_stage2("non_existent.json", s2_metrics)
    assert comp["anomaly_detection_performance"]["f1_score"]["stage2_qlora"] == 0.9036


if __name__ == "__main__":
    test_validate_json_schema()
    test_calculate_binary_anomaly_metrics()
    test_compare_stage1_vs_stage2()
    print("All metrics unit tests passed successfully!")
