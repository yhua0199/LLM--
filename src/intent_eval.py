import json
from pathlib import Path

from common.paths import results_path

# =========================
# 0. 模型配置（保留）
# =========================
MODEL_TAG = "qwen2.5_7b"
LABELS = ["法律类", "违规类", "闲聊类"]


# =========================
# 1. 路径工具
# =========================
def pred_file_path() -> Path:
    return results_path("intent", f"intent_pred_{MODEL_TAG}.jsonl")


def metrics_file_path() -> Path:
    return results_path("intent", f"metrics_{MODEL_TAG}.json")


# =========================
# 2. 工具函数
# =========================
def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_div(a, b):
    return a / b if b != 0 else 0.0


# =========================
# 3. 主评测逻辑（修改为 Macro 指标）
# =========================
def main():
    pred_path = pred_file_path()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    rows = read_jsonl(pred_path)
    total = len(rows)

    # 统计各类的 TP, FP, FN
    stats = {lab: {"TP": 0, "FP": 0, "FN": 0} for lab in LABELS}
    correct_total = 0

    for r in rows:
        gt = r.get("真实类型")
        pred = r.get("预测类型")

        if gt not in LABELS: continue

        if pred == gt:
            stats[gt]["TP"] += 1
            correct_total += 1
        else:
            # 真实为 gt 但预测成了别的，gt 类的缺失 (FN)
            stats[gt]["FN"] += 1
            # 如果预测结果在预定义的标签内，则该预测类多了一个错误 (FP)
            if pred in LABELS:
                stats[pred]["FP"] += 1

    # 计算分类指标
    per_class_metrics = {}
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    for lab in LABELS:
        tp = stats[lab]["TP"]
        fp = stats[lab]["FP"]
        fn = stats[lab]["FN"]

        p = safe_div(tp, tp + fp)
        r_val = safe_div(tp, tp + fn)
        f1_val = safe_div(2 * p * r_val, p + r_val)

        per_class_metrics[lab] = {"Precision": p, "Recall": r_val, "F1": f1_val}
        macro_precision += p
        macro_recall += r_val
        macro_f1 += f1_val

    # 计算最终均值
    num_classes = len(LABELS)
    accuracy = safe_div(correct_total, total)
    avg_precision = macro_precision / num_classes
    avg_recall = macro_recall / num_classes
    avg_f1 = macro_f1 / num_classes

    # ---------- print ----------
    print("=== Intent Classification Evaluation ===")
    print(f"Model tag       : {MODEL_TAG}")
    print(f"Total samples  : {total}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall   : {avg_recall:.4f}")
    print(f"Macro F1 score : {avg_f1:.4f}")
    print("\nPer-class Metrics (Recall / F1):")
    for lab in LABELS:
        m = per_class_metrics[lab]
        print(f"  {lab}: Recall={m['Recall']:.4f}, F1={m['F1']:.4f}")

    # ---------- save ----------
    out_path = metrics_file_path()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_tag": MODEL_TAG,
            "accuracy": accuracy,
            "macro_precision": avg_precision,
            "macro_recall": avg_recall,
            "macro_f1": avg_f1,
            "per_class": per_class_metrics
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
