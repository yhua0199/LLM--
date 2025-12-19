'''
Qwen3
0.5B~1G
1.5B~3G
3B~6G
7B~14G
'''


# -*- coding: utf-8 -*-
"""
rewrite_generate.py
- Read testset json
- Use Qwen Instruct model to rewrite queries
- Save ONLY final outputs to results/rewrite/*.jsonl

Directory (your screenshot):
exp1_intent_rewrite/
  data/rewrite/qa_testset_500.json
  data/rewrite/rewrite_200_base.json
  prompts/rewrite.txt
  results/rewrite/
  src/rewrite_generate.py
"""

# =========================
# ====== CONFIG ==========
# =========================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model (use a valid HF repo id or local folder path)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Input
INPUT_JSON = os.path.join(BASE_DIR, "data", "rewrite", "qa_testset_500.json")
# INPUT_JSON = os.path.join(BASE_DIR, "data", "rewrite", "rewrite_200_base.json")

# Prompt
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "rewrite.txt")

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "rewrite")
OUTPUT_FILE = "qwen2.5_1.5b_rewrite.jsonl"  # 按你要求也可改成 qwen3_0.5b_rewrite.jsonl

# Generation params
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.2
TOP_P = 0.9

# Runtime
DEVICE = "auto"          # auto / cuda / cpu
LOG_EVERY = 20           # 每多少条打印一次进度
PREVIEW_N = 3            # 预览前多少条的输出示例（不会影响保存）
FAIL_SAVE_EMPTY = True   # 解析失败时仍保存 {"原始问题":..., "改写问题":""}

# =========================
# ====== IMPORTS =========
# =========================
import json
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# ====== UTILS ===========
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_query(item: Any) -> Optional[str]:
    """
    兼容常见字段：
    - {"query": "..."} / {"问题": "..."} / {"原始问题": "..."} / {"question": "..."} / {"input": "..."}
    - 或 item 直接是字符串
    """
    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return None

    for k in ["query", "问题", "原始问题", "question", "input"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 兜底：如果有 messages 结构，取最后一个 user
    msgs = item.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()

    return None


def build_chat_input(tokenizer, rewrite_prompt: str, query: str) -> str:
    """
    使用 Qwen chat template
    rewrite_prompt 是你 prompts/rewrite.txt 的内容
    """
    messages = [
        {"role": "system", "content": "你是一个严谨的中文法律问题改写系统。"},
        {"role": "user", "content": f"{rewrite_prompt}\n{query}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从模型输出提取 JSON（严格）
    """
    t = text.strip()
    # 1) 整段就是 JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) 截取第一个 { 到最后一个 }
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = t[l:r+1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


@torch.inference_mode()
def generate_rewrite(model, tokenizer, rewrite_prompt: str, query: str) -> Dict[str, str]:
    """
    返回最终需要保存的结构：
    {"原始问题": "...", "改写问题": "..."}
    """
    chat_input = build_chat_input(tokenizer, rewrite_prompt, query)
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    obj = parse_json_from_text(raw_text)

    if isinstance(obj, dict):
        orig = obj.get("原始问题", query)
        rew = obj.get("改写问题", "")
        return {
            "原始问题": str(orig),
            "改写问题": str(rew).strip(),
            "_raw": raw_text,  # 仅用于预览打印，落盘前会移除
        }

    # 解析失败兜底
    return {
        "原始问题": query,
        "改写问题": "",
        "_raw": raw_text,
    }


def normalize_dataset(data: Any) -> List[Any]:
    """
    允许输入是：
    - list
    - dict 包 list（data/items/samples/examples）
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ["data", "items", "samples", "examples"]:
            v = data.get(k)
            if isinstance(v, list):
                return v

    raise ValueError(f"Unsupported JSON structure: {type(data)}")


# =========================
# ====== MAIN ============
# =========================

def main():
    # ---- path checks
    print("[INFO] BASE_DIR    =", BASE_DIR)
    print("[INFO] INPUT_JSON  =", INPUT_JSON)
    print("[INFO] PROMPT_PATH =", PROMPT_PATH)
    print("[INFO] OUTPUT_DIR  =", OUTPUT_DIR)

    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # ---- device
    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] DEVICE      =", device)

    # ---- load model
    print("[INFO] Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    # ---- load prompt & data
    rewrite_prompt = load_text(PROMPT_PATH)
    data = normalize_dataset(load_json(INPUT_JSON))

    total = len(data)
    processed = 0
    ok = 0

    print(f"[INFO] Total samples: {total}")
    print(f"[INFO] Saving to: {out_path}")
    print("=" * 60)

    with open(out_path, "w", encoding="utf-8") as wf:
        for idx, item in enumerate(data):
            q = extract_query(item)
            if not q:
                continue

            result = generate_rewrite(model, tokenizer, rewrite_prompt, q)
            raw_text = result.pop("_raw", "")  # 不落盘，只用于预览

            processed += 1
            if result["改写问题"]:
                ok += 1

            # ---- preview examples
            if processed <= PREVIEW_N:
                print(f"[PREVIEW {processed}] 原始问题: {result['原始问题']}")
                print(f"[PREVIEW {processed}] 模型原始输出: {raw_text[:300]}{'...' if len(raw_text) > 300 else ''}")
                print(f"[PREVIEW {processed}] 解析后改写问题: {result['改写问题']}")
                print("-" * 60)

            # ---- save
            if (not result["改写问题"]) and (not FAIL_SAVE_EMPTY):
                # 可选：解析失败就不保存
                continue

            wf.write(json.dumps(result, ensure_ascii=False) + "\n")

            # ---- progress log
            if processed % LOG_EVERY == 0:
                rate = ok / processed if processed else 0.0
                print(f"[PROGRESS] {processed}/{total} | parsed_ok={ok} | ok_rate={rate:.2%}")

    print("=" * 60)
    print(f"[DONE] Saved: {out_path}")
    print(f"[STATS] processed={processed}, parsed_ok={ok}, ok_rate={(ok/processed if processed else 0):.2%}")


if __name__ == "__main__":
    main()
