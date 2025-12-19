# -*- coding: utf-8 -*-
"""
Merge:
- data/rewrite/synth/self_instruct_qa_300.json
- data/rewrite/rewrite_200_base.json
into:
- data/rewrite/qa_testset_500.json

Then shuffle the merged dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


RANDOM_SEED = 42   # 固定 seed，保证实验可复现


def load_json_list(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")

    out = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = str(obj.get("query", "")).strip()
        a = str(obj.get("answer", "")).strip()
        if not q or not a:
            continue
        out.append({"query": q, "answer": a})
    return out


def main():
    root = Path(__file__).resolve().parent.parent

    gen_300_path = root / "data" / "rewrite" / "synth" / "self_instruct_qa_300.json"
    base_200_path = root / "data" / "rewrite" / "rewrite_200_base.json"
    out_path = root / "data" / "rewrite" / "qa_testset_500.json"

    gen_300 = load_json_list(gen_300_path)
    base_200 = load_json_list(base_200_path)

    print(f"[LOAD] self_instruct_qa_300.json: {len(gen_300)}")
    print(f"[LOAD] rewrite_200_base.json: {len(base_200)}")

    merged = gen_300 + base_200

    if len(merged) != 500:
        raise ValueError(
            f"Expected 500 QA pairs, but got {len(merged)} "
            f"(300_file={len(gen_300)}, 200_file={len(base_200)})"
        )

    # ✅ 打乱顺序（关键）
    random.seed(RANDOM_SEED)
    random.shuffle(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[DONE] Shuffled & saved 500 QA pairs to: {out_path}")
    print(f"[INFO] Shuffle seed = {RANDOM_SEED}")


if __name__ == "__main__":
    main()

