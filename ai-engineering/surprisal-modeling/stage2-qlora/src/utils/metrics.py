"""Evaluation Telemetry, Schema Compliance Validator, and Stage 1 vs Stage 2 Comparative Analysis.

Provides strict JSON schema validation, multi-class severity classification F1 computation,
binary anomaly detection metrics identical to Stage 1, and automated comparative benchmarking.
"""

import os
import json
import logging

logger = logging.getLogger("stage2.metrics")

REQUIRED_KEYS = {"root_cause", "severity", "affected_component", "mitigation_commands", "confidence"}
SEVERITY_VALUES = {"P0_EMERGENCY", "P1_CRITICAL", "P2_WARNING", "P3_INFO"}


def validate_json_schema(output: str) -> tuple[bool, dict | None]:
    """Validates whether string output conforms strictly to root-cause diagnosis JSON schema.

    Checks:
        1. Parseable JSON syntax.
        2. Presence of all required keys.
        3. Correct type constraints on fields.
        4. Valid enum member for severity.

    Returns:
        Tuple of (is_valid, parsed_dict).
    """
    if not output:
        return False, None

    # Try extracting JSON if wrapped in markdown fences or stray tokens
    clean_output = output.strip()
    if "```json" in clean_output:
        parts = clean_output.split("```json")
        if len(parts) > 1:
            clean_output = parts[1].split("```")[0].strip()
    elif "```" in clean_output:
        parts = clean_output.split("```")
        if len(parts) > 1:
            clean_output = parts[1].strip()

    try:
        data = json.loads(clean_output)
    except Exception:
        return False, None

    if not isinstance(data, dict):
        return False, None

    if not REQUIRED_KEYS.issubset(data.keys()):
        return False, None

    if not isinstance(data.get("root_cause"), str) or not isinstance(data.get("affected_component"), str):
        return False, None

    if not isinstance(data.get("mitigation_commands"), list):
        return False, None

    if not isinstance(data.get("confidence"), (int, float)):
        return False, None

    if data.get("severity") not in SEVERITY_VALUES:
        return False, None

    return True, data


def calculate_severity_metrics(predictions: list[str], targets: list[str]) -> dict[str, float | dict]:
    """Computes Precision, Recall, and F1 across multi-class severity levels."""
    labels = sorted(list(SEVERITY_VALUES))
    per_class = {}
    
    total_correct = 0
    for sev in labels:
        tp = sum(1 for p, t in zip(predictions, targets) if p == sev and t == sev)
        fp = sum(1 for p, t in zip(predictions, targets) if p == sev and t != sev)
        fn = sum(1 for p, t in zip(predictions, targets) if p != sev and t == sev)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class[sev] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        total_correct += tp

    accuracy = total_correct / len(targets) if targets else 0.0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class
    }


def calculate_binary_anomaly_metrics(predictions: list[int], targets: list[int]) -> dict[str, float | int]:
    """Computes exact binary anomaly detection metrics identical to Stage 1.

    Args:
        predictions: List of binary predictions (1=Anomaly, 0=Normal).
        targets: List of ground-truth binary labels (1=Anomaly, 0=Normal).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, tp, fp, tn, fn.
    """
    tp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 1)

    total = len(targets)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def compare_stage1_vs_stage2(stage1_results_path: str, stage2_metrics: dict) -> dict:
    """Generates comprehensive side-by-side performance comparison of Stage 1 vs Stage 2."""
    stage1_metrics = {}
    if os.path.exists(stage1_results_path):
        try:
            with open(stage1_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stage1_metrics = data.get("test_metrics", {})
        except Exception as e:
            logger.warning(f"Could not load Stage 1 evaluation results: {e}")
    else:
        # Default Stage 1 benchmark reported figures
        stage1_metrics = {
            "accuracy": 0.952767,
            "precision": 0.946331,
            "recall": 0.844043,
            "f1": 0.892265,
            "tp": 14212,
            "fp": 806,
            "tn": 55017,
            "fn": 2626
        }

    s2_bin = stage2_metrics.get("binary_anomaly_metrics", {})

    comparison = {
        "architecture_comparison": {
            "stage1": "GPT-2 Unsupervised Surprisal (124M parameters)",
            "stage2": "Qwen-2.5-3B-Instruct QLoRA Fine-Tuned (3.09B parameters, 21M trainable adapter)"
        },
        "anomaly_detection_performance": {
            "accuracy": {
                "stage1_gpt2": stage1_metrics.get("accuracy", 0.0),
                "stage2_qlora": s2_bin.get("accuracy", 0.0),
                "delta": s2_bin.get("accuracy", 0.0) - stage1_metrics.get("accuracy", 0.0)
            },
            "precision": {
                "stage1_gpt2": stage1_metrics.get("precision", 0.0),
                "stage2_qlora": s2_bin.get("precision", 0.0),
                "delta": s2_bin.get("precision", 0.0) - stage1_metrics.get("precision", 0.0)
            },
            "recall": {
                "stage1_gpt2": stage1_metrics.get("recall", 0.0),
                "stage2_qlora": s2_bin.get("recall", 0.0),
                "delta": s2_bin.get("recall", 0.0) - stage1_metrics.get("recall", 0.0)
            },
            "f1_score": {
                "stage1_gpt2": stage1_metrics.get("f1", 0.0),
                "stage2_qlora": s2_bin.get("f1", 0.0),
                "delta": s2_bin.get("f1", 0.0) - stage1_metrics.get("f1", 0.0)
            }
        },
        "structured_diagnostic_capabilities": {
            "stage1_gpt2": "None (unsupervised sequence surprisal scalar only)",
            "stage2_qlora": {
                "json_schema_compliance_rate": stage2_metrics.get("schema_compliance_rate", 0.0),
                "severity_classification_macro_f1": stage2_metrics.get("severity_metrics", {}).get("macro_f1", 0.0),
                "latency_p95_ms": stage2_metrics.get("latency_ms", {}).get("p95", 0.0)
            }
        },
        "vram_footprint_mb": {
            "stage1_gpt2_fp16": 1616.49,
            "stage2_qlora_nf4_bs2": 5120.0,
            "stage2_fp16_unquantized": 12288.0
        }
    }
    return comparison
