import json
import random
from collections import Counter
from pathlib import Path

from common.paths import data_path

# 目标分布
TARGET = {"法律类": 1200, "违规类": 400, "闲聊类": 400}
# raw 文件命名规则（jsonl）
PATTERN = {"法律类": "law_*.jsonl", "违规类": "violate_*.jsonl", "闲聊类": "chat_*.jsonl"}

def read_jsonl(files, label):
    """读取 jsonl（不去重）。只保留字段齐全且类型匹配的样本。"""
    rows = []
    for fp in files:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = str(obj.get("问题", "")).strip()
            t = str(obj.get("类型", "")).strip()
            if q and t == label:
                rows.append({"问题": q, "类型": t})
    return rows

def main():
    raw = data_path("intent", "raw")
    out = data_path("intent", "intent_2k.json")

    print(f"[INFO] raw dir: {raw}")

    final = []
    for label, need in TARGET.items():
        files = sorted(raw.glob(PATTERN[label]))
        print(f"{label}: 扫描到 {len(files)} 个文件")

        data = read_jsonl(files, label)
        have = len(data)
        if have < need:
            raise ValueError(f"{label} 数量不足：需要 {need}，实际 {have}，还差 {need - have}")

        final.extend(random.sample(data, need))
        print(f"  -> 抽样 {need}（可用 {have}）")

    random.shuffle(final)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    # 最终统计
    c = Counter(x["类型"] for x in final)
    print("\n=== Final Summary ===")
    for k in ["法律类", "违规类", "闲聊类"]:
        print(f"{k}: {c.get(k, 0)}")
    print(f"总计: {len(final)}")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
