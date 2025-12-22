# -*- coding: utf-8 -*-
"""
eval_single_model_intent.py

功能：
- 读取单模型融合输出（jsonl）：每行形如
  {"原始问题":"...","改写问题":"...","预测类型":"...","真实类型":"..."}
- 只评估意图分类：预测类型 vs 真实类型
- 输出指标到 results/intent/metrics_qwen2.5_1.5b_single.json

不依赖 sklearn（纯标准库）
"""

import os
import json
from typing import Dict, List, Tuple

# =========================
# 0) 配置区（全部放前面）
# =========================


# ===== 强制把工作目录切到项目根目录 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

# ===== 单模型预测结果（你红框的那个文件）=====
PRED_PATH = "results/rewrite/qwen2.5_1.5b_intent_rewrite.jsonl"

# ===== 输出到 intent 目录 =====
OUT_DIR = "results/intent"
OUT_FILE = "metrics_qwen2.5_1.5b_single.json"
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

MODEL_TAG = "qwen2.5_1.5b"


# 固定三分类
CLASSES = ["法律类", "违规类", "闲聊类"]

# 预测标签归一化映射（把模型乱输出的类别纠正到三类）
NORMALIZE_MAP = {
    "法律": "法律类",
    "法务类": "法律类",

    "违规": "违规类",
    "敏感": "违规类",
    "违法类": "违规类",
    "非法类": "违规类",

    "闲聊": "闲聊类",
    "聊天": "闲聊类",
}

# 如果模型把 intent 留空，怎么处理：
# - True：把空预测当作“闲聊类”（你的样本里很多空值本质上是闲聊）
# - False：空预测保持为空（会被当作错误预测）
EMPTY_PRED_AS_CHAT = True


# =========================
# 1) 工具函数
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_label(x: str) -> str:
    """把预测/真实标签归一化到三类（或空）"""
    if x is None:
        return ""
    s = str(x).strip().replace(" ", "")
    if s in NORMALIZE_MAP:
        s = NORMALIZE_MAP[s]
    if s == "" and EMPTY_PRED_AS_CHAT:
        s = "闲聊类"
    return s


def load_pairs(pred_path: str) -> Tuple[List[str], List[str], int]:
    """读取 y_true, y_pred"""
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


def build_cm(y_true: List[str], y_pred: List[str], classes: List[str]) -> Dict[str, Dict[str, int]]:
    """混淆矩阵 cm[gt][pred]"""
    cm = {gt: {pr: 0 for pr in classes} for gt in classes}
    for gt, pr in zip(y_true, y_pred):
        if gt not in classes:
            continue
        if pr not in classes:
            # 预测不在三类内：记为漏判（只影响 Recall/Accuracy），不计入任何预测列
            # 若你想让 Precision 也更严格，可把它强行映射到“闲聊类/违规类”
            continue
        cm[gt][pr] += 1
    return cm


def calc_accuracy(y_true: List[str], y_pred: List[str], classes: List[str]) -> float:
    correct, total = 0, 0
    for gt, pr in zip(y_true, y_pred):
        if gt not in classes:
            continue
        total += 1
        if pr == gt:
            correct += 1
    return correct / total if total > 0 else 0.0


def calc_prf(cm: Dict[str, Dict[str, int]], classes: List[str]) -> Tuple[Dict[str, Dict[str, float]], float, float, float]:
    """由混淆矩阵计算 per-class 和 macro"""
    per_class = {}
    ps, rs, fs = [], [], []

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[gt][c] for gt in classes if gt != c)
        fn = sum(cm[c][pr] for pr in classes if pr != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c] = {"Precision": precision, "Recall": recall, "F1": f1}
        ps.append(precision)
        rs.append(recall)
        fs.append(f1)

    macro_p = sum(ps) / len(classes)
    macro_r = sum(rs) / len(classes)
    macro_f1 = sum(fs) / len(classes)

    return per_class, macro_p, macro_r, macro_f1


# =========================
# 2) 主流程
# =========================

def main():
    print(f"[INFO] PRED_PATH = {PRED_PATH}")
    print(f"[INFO] OUT_PATH  = {OUT_PATH}")
    print(f"[INFO] MODEL_TAG = {MODEL_TAG}")
    print(f"[INFO] EMPTY_PRED_AS_CHAT = {EMPTY_PRED_AS_CHAT}")

    y_true, y_pred, total_lines = load_pairs(PRED_PATH)

    acc = calc_accuracy(y_true, y_pred, CLASSES)
    cm = build_cm(y_true, y_pred, CLASSES)
    per_class, macro_p, macro_r, macro_f1 = calc_prf(cm, CLASSES)

    metrics = {
        "model_tag": MODEL_TAG,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class
    }

    safe_mkdir(OUT_DIR)

    # 输出为一个 JSON 对象（你要求的格式）
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    print(f"[DONE] Read {total_lines} lines.")
    print(f"[DONE] Metrics saved to: {OUT_PATH}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
