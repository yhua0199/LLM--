# src/exp2/rag/rewrite_qwen_make_qa_eval.py
# -*- coding: utf-8 -*-
"""
实验2：用 Qwen(3B/7B) 对 qa_50.json 做“问题改写（rewrite）”，生成 qa_eval_500.json

目标：
- 输入：experiments/exp2/data/qa_50.json
- 输出：experiments/exp2/data/qa_eval_500.json（JSON数组）
  每条格式：
  {
    "query": "...",
    "answer": "...",
    "rewrite": "..."
  }

特点：
1) 环境变量控制（3b/7b、模型路径、设备、生成参数）
2) 云端路径友好：支持直接指定本地模型目录（如 /mnt/workspace/model_cache/...）
3) 只改写 query，不改动 answer
4) 输出为 JSON（不是 jsonl）
5) 带进度和预览输出
"""

from __future__ import annotations

# =========================================================
# 0) 必须先锁定实验目录（paths.py import 时读取环境变量）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp2")

from src.common.paths import data_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 配置：全部可用环境变量覆盖
# =========================================================
import argparse
import json
import re
import time
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# 输入/输出
# -------------------------
IN_FILE = os.getenv("REWRITE_IN_FILE", "qa_50.json")                 # experiments/exp2/data/qa_50.json
OUT_FILE = os.getenv("REWRITE_OUT_FILE", "qa_eval_500.json")        # experiments/exp2/data/qa_eval_500.json

# -------------------------
# LLM：3B/7B 切换
# -------------------------
QWEN_SIZE = os.getenv("QWEN_SIZE", "3b").strip().lower()            # 3b / 7b
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto").strip().lower()

# 可直接指定本地模型路径（推荐云端用，避免下载）
# 例：/mnt/workspace/model_cache/Qwen/Qwen2.5-7B-Instruct
QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID",
    "Qwen/Qwen2.5-3B-Instruct" if QWEN_SIZE == "3b" else "Qwen/Qwen2.5-7B-Instruct"
)

# -------------------------
# 生成参数（rewrite 建议更稳定：低温、不采样）
# -------------------------
REWRITE_MAX_NEW_TOKENS = int(os.getenv("REWRITE_MAX_NEW_TOKENS", "64"))
REWRITE_TEMPERATURE = float(os.getenv("REWRITE_TEMPERATURE", "0.0"))
REWRITE_TOP_P = float(os.getenv("REWRITE_TOP_P", "0.9"))
REWRITE_DO_SAMPLE = os.getenv("REWRITE_DO_SAMPLE", "0") == "1"      # 默认不采样

# -------------------------
# 进度/预览
# -------------------------
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
PREVIEW_EVERY = int(os.getenv("PREVIEW_EVERY", "10"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "160"))

# sanity：只跑前 N 条（跑全量就 unset / 设为 0）
SANITY_N = int(os.getenv("REWRITE_SANITY_N", "0"))

# -------------------------
# Prompt：专用于改写
# -------------------------
DEFAULT_REWRITE_PROMPT = """你是一个中文法律问答助手的“问题改写器”。

任务：把用户原始法律咨询问题改写成更清晰、无歧义、便于检索的版本。

要求：
1) 保持原问题的法律意图不变，不增加新事实，不删除关键事实。
2) 用更规范/更书面但仍自然的中文表达。
3) 不要输出任何解释、不要加引号、不要加编号。
4) 只输出改写后的问题文本。

原始问题：
{query}

改写后问题：
"""


# =========================================================
# 2) 工具函数
# =========================================================
def _pick_torch_device(device_env: str) -> str:
    if device_env != "auto":
        return device_env
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_json_list(p) -> List[Dict[str, Any]]:
    if not p.exists():
        raise FileNotFoundError(f"未找到输入文件：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"输入必须是 JSON 数组（list）：{p}")
    return [x for x in data if isinstance(x, dict)]


def normalize_rewrite(text: str) -> str:
    """清理模型输出：去空白、去多余引号/代码块等"""
    t = (text or "").strip()
    # 去掉可能的 ``` 包裹
    t = re.sub(r"^```.*?\n", "", t, flags=re.S)
    t = re.sub(r"\n```$", "", t)
    # 去掉首尾引号
    t = t.strip().strip('"').strip("“”").strip()
    # 防止模型输出多行：只取第一行非空
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def build_llm_input(tokenizer, prompt_text: str, query: str) -> str:
    user_text = prompt_text.format(query=query)

    # 优先使用 chat template（Qwen Instruct 更稳）
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_text


@torch.inference_mode()
def generate_rewrite(model, tokenizer, input_text: str) -> str:
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=REWRITE_MAX_NEW_TOKENS,
        do_sample=REWRITE_DO_SAMPLE,
        temperature=REWRITE_TEMPERATURE,
        top_p=REWRITE_TOP_P,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return normalize_rewrite(text)


# =========================================================
# 3) 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Exp2 Rewrite QA (Qwen 3B/7B) -> qa_eval_500.json")
    parser.add_argument("--in_file", type=str, default=IN_FILE, help="input json under experiments/<exp>/data/")
    parser.add_argument("--out_file", type=str, default=OUT_FILE, help="output json under experiments/<exp>/data/")
    args = parser.parse_args()

    in_path = data_path(args.in_file)
    out_path = data_path(args.out_file)
    ensure_dir(out_path.parent)

    items = load_json_list(in_path)

    # sanity
    if SANITY_N > 0:
        items = items[:SANITY_N]

    device = _pick_torch_device(LLM_DEVICE)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print("=" * 90)
    print("[INFO] Exp2 Rewrite（生成 rewrite 字段）")
    print(f"[INFO] LLM_EXPERIMENT = {os.environ.get('LLM_EXPERIMENT')}")
    print(f"[INFO] input : {in_path} | n={len(items)}")
    print(f"[INFO] output: {out_path}")
    print(f"[INFO] model : {QWEN_MODEL_ID} (QWEN_SIZE={QWEN_SIZE})")
    print(f"[INFO] device: {device} | dtype={torch_dtype}")
    print(f"[INFO] gen   : max_new_tokens={REWRITE_MAX_NEW_TOKENS}, temp={REWRITE_TEMPERATURE}, top_p={REWRITE_TOP_P}, do_sample={REWRITE_DO_SAMPLE}")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.eval()

    prompt_text = DEFAULT_REWRITE_PROMPT

    results: List[Dict[str, str]] = []
    t0 = time.time()

    for i, obj in enumerate(items, start=1):
        query = (obj.get("query", "") or "").strip()
        answer = (obj.get("answer", "") or "").strip()
        if not query:
            continue

        input_text = build_llm_input(tokenizer, prompt_text, query)
        rewrite = generate_rewrite(model, tokenizer, input_text)

        # 兜底：如果模型输出空，至少回退原 query
        if not rewrite:
            rewrite = query

        results.append({"query": query, "answer": answer, "rewrite": rewrite})

        if i % LOG_EVERY == 0 or i == len(items):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {i}/{len(items)} | {speed:.2f} 条/秒 | elapsed {elapsed:.1f}s")

        if i == 1 or i % PREVIEW_EVERY == 0:
            print("\n[PREVIEW]")
            print("query  :", query[:PREVIEW_CHARS] + ("…" if len(query) > PREVIEW_CHARS else ""))
            print("rewrite:", rewrite[:PREVIEW_CHARS] + ("…" if len(rewrite) > PREVIEW_CHARS else ""))
            print()

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved: {out_path}")


if __name__ == "__main__":
    main()
