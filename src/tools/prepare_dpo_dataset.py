# -*- coding: utf-8 -*-
"""
Prepare DPO dataset for LLaMA-Factory:
1) Convert jsonl -> json(list) : dpo_law.json
2) Token length stats with a 3B tokenizer (default: Qwen2.5-3B-Instruct)
3) Print summary + save stats json

Input jsonl format (each line):
{"instruction": "...", "input": "", "chosen": "...", "rejected": "..."}

Usage (Windows PowerShell):
  python tools/prepare_dpo_dataset.py ^
    --in_jsonl experiments\exp3\data\dpo_pairs_qwen32b.jsonl ^
    --out_json experiments\exp3\data\dpo_law.json ^
    --stats_json experiments\exp3\data\dpo_law_token_stats.json ^
    --tokenizer_model Qwen/Qwen2.5-3B-Instruct

If you train with another base model, set --tokenizer_model to that model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


REQUIRED_KEYS = ["instruction", "input", "chosen", "rejected"]


@dataclass
class ItemTokens:
    prompt: int
    chosen: int
    rejected: int
    total_chosen: int
    total_rejected: int


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = (line or "").strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def percentile(sorted_vals: List[int], p: float) -> int:
    """Nearest-rank percentile, p in [0,1]."""
    if not sorted_vals:
        return 0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    k = math.ceil(p * len(sorted_vals)) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


def summarize(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    sv = sorted(vals)
    n = len(sv)
    avg = sum(sv) / n
    return {
        "count": n,
        "min": sv[0],
        "p50": percentile(sv, 0.50),
        "p90": percentile(sv, 0.90),
        "p95": percentile(sv, 0.95),
        "p99": percentile(sv, 0.99),
        "max": sv[-1],
        "mean": round(avg, 2),
    }


def hist(vals: List[int], bins: List[int]) -> Dict[str, int]:
    """
    bins example: [128, 256, 512, 1024, 2048, 4096]
    return counts for <=bin and >lastbin
    """
    out: Dict[str, int] = {}
    for b in bins:
        out[f"<= {b}"] = 0
    out[f"> {bins[-1]}"] = 0

    for v in vals:
        placed = False
        for b in bins:
            if v <= b:
                out[f"<= {b}"] += 1
                placed = True
                break
        if not placed:
            out[f"> {bins[-1]}"] += 1
    return out


def make_prompt(instruction: str, input_text: str) -> str:
    """
    LLaMA-Factory ranking dataset uses:
      prompt = instruction
      query  = input
    但为了 token 统计更接近训练时“输入给模型的内容”，这里把二者拼起来统计。
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="path to dpo_pairs_qwen32b.jsonl")
    parser.add_argument("--out_json", required=True, help="output json(list) path, e.g. dpo_law.json")
    parser.add_argument("--stats_json", required=True, help="output token stats json path")
    parser.add_argument(
        "--tokenizer_model",
        default=os.getenv("TOKENIZER_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        help="HF tokenizer model id (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = ensure_dir(Path(args.out_json))
    stats_path = ensure_dir(Path(args.stats_json))

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"[INFO] input : {in_path}")
    print(f"[INFO] output: {out_path}")
    print(f"[INFO] stats : {stats_path}")
    print(f"[INFO] tokenizer_model: {args.tokenizer_model}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Read jsonl -> list
    data: List[Dict[str, Any]] = []
    bad = 0
    missing = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            obj = safe_json_loads(line)
            if obj is None:
                bad += 1
                continue

            if not all(k in obj for k in REQUIRED_KEYS):
                missing += 1
                continue

            item = {
                "instruction": obj.get("instruction", ""),
                "input": obj.get("input", ""),
                "chosen": obj.get("chosen", ""),
                "rejected": obj.get("rejected", ""),
            }
            data.append(item)

            if args.max_samples and len(data) >= args.max_samples:
                break

    if not data:
        raise ValueError("No valid items loaded. Check jsonl format/keys.")

    # Save json(list)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved json(list): {out_path} | n={len(data)} | bad_lines={bad} | missing_keys={missing}")

    # Token stats
    prompt_lens: List[int] = []
    chosen_lens: List[int] = []
    rejected_lens: List[int] = []
    total_chosen_lens: List[int] = []
    total_rejected_lens: List[int] = []

    for it in data:
        prompt = make_prompt(it["instruction"], it["input"])
        chosen = (it["chosen"] or "").strip()
        rejected = (it["rejected"] or "").strip()

        p_ids = tok.encode(prompt, add_special_tokens=False)
        c_ids = tok.encode(chosen, add_special_tokens=False)
        r_ids = tok.encode(rejected, add_special_tokens=False)

        prompt_lens.append(len(p_ids))
        chosen_lens.append(len(c_ids))
        rejected_lens.append(len(r_ids))
        total_chosen_lens.append(len(p_ids) + len(c_ids))
        total_rejected_lens.append(len(p_ids) + len(r_ids))

    bins = [128, 256, 512, 1024, 2048, 4096]
    stats = {
        "tokenizer_model": args.tokenizer_model,
        "num_samples": len(data),
        "bad_lines": bad,
        "missing_keys": missing,
        "lengths": {
            "prompt": summarize(prompt_lens),
            "chosen": summarize(chosen_lens),
            "rejected": summarize(rejected_lens),
            "total_prompt_plus_chosen": summarize(total_chosen_lens),
            "total_prompt_plus_rejected": summarize(total_rejected_lens),
        },
        "histogram_bins": bins,
        "histograms": {
            "prompt": hist(prompt_lens, bins),
            "chosen": hist(chosen_lens, bins),
            "rejected": hist(rejected_lens, bins),
            "total_prompt_plus_chosen": hist(total_chosen_lens, bins),
            "total_prompt_plus_rejected": hist(total_rejected_lens, bins),
        },
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print short summary
    print("\n" + "=" * 80)
    print("[SUMMARY] Token length stats")
    for k, v in stats["lengths"].items():
        print(f"- {k}: {v}")
    print("=" * 80)
    print(f"[DONE] saved token stats: {stats_path}")


if __name__ == "__main__":
    main()
