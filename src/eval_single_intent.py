# -*- coding: utf-8 -*-
"""
eval_single_intent.py

评估“单一模型（intent+rewrite融合）”的输出结果：
- 只评估意图分类（预测类型 vs 真实类型）
- 输入：results/rewrite/qwen2.5_1.5b_intent_rewrite.jsonl（默认，可在配置区改）
- 输出：results/intent/metrics_qwen2.5_1.5b_single.jsonl（按你要求命名）
- 输出内容：accuracy / macro precision/recall/f1 + per-class 指标

依赖：仅 Python 标准库（不需要 sklearn）
"""

import os
import json
from typing import Dict, List, Tuple

# =========================
# 0) 配置区（全部放前面）
# =========================

# 单模型推理输出（jsonl）路径：你按实际文件名改这一行即可
PRED_PATH = "results/rewrite/qwen2.5_1.5b_intent_rewrite.jsonl"

# 指标输出：按你要求输出到 intent 下，并命名为 metrics_qwen2.5_1.5b_single.jsonl
OUT_DIR = "results/intent"
OUT_FILE = "metrics_qwen2.5_1.5b_single.jsonl"
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

# 写到结果里的 model_tag（可改）
MODEL_TAG = "qwen2.5_1.5b"

# 类别集合（固定顺序，保证 per_class 输出稳定）
CLASSES = ["法律类", "违规类", "闲聊类"]

# 容错映射：模型输出可能出现空格/大小写/别名时做纠正（可扩展）
NORMALIZE_MAP = {
    "法律": "法律类",
    "法务类": "法律类",
    "违规": "违规类",
    "敏感": "违规类",
    "闲聊": "闲聊类",
    "聊天": "闲聊类",
}


# =========================
# 1) 工具函数
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_label(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(" ", "")
    # 统一成：法律类/违规类/闲聊类
    if s in NORMALIZE_MAP:
        s = NORMALIZE_MAP[s]
    return s


def load_pairs(pred_path: str) -> Tuple[List[str], List[str], int]:
    """
    从单模型输出 jsonl 读取：
    - 真实类型：真实类型
    - 预测类型：预测类型

    返回：y_true, y_pred, total_lines
    """
    y_true, y_pred = [], []
    total = 0
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            gt = normalize_label(obj.get("真实类型", ""))
            pr = normalize_label(obj.get("预测类型", ""))
            y_true.append(gt)
            y_pred.append(pr)
    return y_true, y_pred, total


def confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]) -> Dict[str, Dict[str, int]]:
    """
    返回嵌套 dict 形式的混淆矩阵 cm[gt][pred] = count
    """
    cm = {gt: {pr: 0 for pr in classes} for gt in classes}
    for gt, pr in zip(y_true, y_pred):
        if gt not in classes:
            continue
        if pr not in classes:
            # 预测不在集合里，记为“全错”，但不新增类别；这里简单跳过计数到任何 pred
            # 你也可以选择把它算到某个“其他类”，但作业通常不需要
            continue
        cm[gt][pr] += 1
    return cm


def precision_recall_f1_from_cm(cm: Dict[str, Dict[str, int]], classes: List[str]) -> Tuple[Dict[str, Dict[str, float]], float, float, float]:
    """
    由 cm 计算 per-class Precision/Recall/F1，以及 macro 指标
    """
    per_class = {}
    precisions, recalls, f1s = [], [], []

    # 为每个类计算 TP/FP/FN
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[gt][c] for gt in classes if gt != c)
        fn = sum(cm[c][pr] for pr in classes if pr != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(classes) if classes else 0.0
    macro_recall = sum(recalls) / len(classes) if classes else 0.0
    macro_f1 = sum(f1s) / len(classes) if classes else 0.0

    return per_class, macro_precision, macro_recall, macro_f1


def accuracy(y_true: List[str], y_pred: List[str], classes: List[str]) -> float:
    correct = 0
    total = 0
    for gt, pr in zip(y_true, y_pred):
        if gt not in classes:
            continue
        total += 1
        if pr == gt:
            correct += 1
    return correct / total if total > 0 else 0.0


# =========================
# 2) 主流程
# =========================

def main():
    print(f"[INFO] PRED_PATH = {PRED_PATH}")
    print(f"[INFO] OUT_PATH  = {OUT_PATH}")
    print(f"[INFO] MODEL_TAG = {MODEL_TAG}")

    y_true, y_pred, total_lines = load_pairs(PRED_PATH)

    acc = accuracy(y_true, y_pred, CLASSES)
    cm = confusion_matrix(y_true, y_pred, CLASSES)
    per_class, macro_p, macro_r, macro_f1 = precision_recall_f1_from_cm(cm, CLASSES)

    metrics = {
        "model_tag": MODEL_TAG,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class
    }

    safe_mkdir(OUT_DIR)
    # 你要求输出 jsonl：这里写一行即可
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    print(f"[DONE] Read {total_lines} lines.")
    print(f"[DONE] Metrics saved to: {OUT_PATH}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
