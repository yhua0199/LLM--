# -*- coding: utf-8 -*-
"""
实验2 Baseline 推理脚本（不使用RAG）

需求对齐：
1) 路径不写死：统一使用 src.common.paths（并自动选择 exp2）
2) 只读取 qa_testset_500.json 的 "query" 做推理
3) 阶段输出进度 + 示例回答
4) 输入/输出/模型配置/参数/路径等前置，中文注释
5) 结果保存到 experiments/exp2/results，字段：query / answer / LLM_answer
6) 输出文件命名：answer_模型参数（这里：answer_qwen2.5_3b.json）

运行方式（在项目根目录）：
  python -m src.exp2.baseline.infer
  或
  python src/exp2/baseline/infer.py
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】先设置实验选择（因为 paths.py 在 import 时会读取环境变量）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp2")  # 保证 data_path / results_path 指向 experiments/exp2/...

# 现在再 import paths（非常关键）
from src.common.paths import data_path, prompt_path, results_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 【配置前置区】输入/输出/模型配置/参数/路径/日志 全部放这里
# =========================================================
import argparse
import json
import time
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# 输入数据配置
# -------------------------
# 你截图中 data 文件名就是 qa_testset_500.json
DATA_FILE = "qa_testset_500.json"   # -> experiments/exp2/data/qa_testset_500.json

# -------------------------
# Prompt 配置（后续你放到 experiments/exp2/prompts）
# -------------------------
PROMPT_FILE = "baseline_qa.txt"     # -> experiments/exp2/prompts/baseline_qa.txt

DEFAULT_PROMPT = """你是一名法律咨询助手。

要求：
1) 只解释一般性的法律原则与建议，不编造具体事实。
2) 不提供违法/规避监管/危险行为的具体操作步骤。
4) 表达清晰、简洁，结构：原则说明 + 建议方向。

用户问题：
{query}

请给出回答：
"""

# -------------------------
# 模型配置（本题要求：用 Qwen2.5 的 3B 模型）
# -------------------------
MODEL_CONFIG = {
    "model_tag": "qwen2.5_3b",
    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    "dtype": "float16",          # 云端/显卡通常用 float16；CPU 会自动回落到 float32
    "use_chat_template": True,   # Qwen instruct 建议用 chat template
}

# -------------------------
# 生成参数（可在这里统一调优）
# -------------------------
GEN_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "do_sample": True,
}

# -------------------------
# 日志/进度/示例输出
# -------------------------
LOG_EVERY = 10
PREVIEW_EVERY = 50
PREVIEW_CHARS = 220


# =========================================================
# 2) 工具函数：读取prompt / 读取数据 / 构造输入 / 生成回答
# =========================================================
def load_prompt_text() -> str:
    """
    优先从 experiments/exp2/prompts/baseline_qa.txt 读取 prompt；
    如果文件不存在，则使用 DEFAULT_PROMPT。
    """
    p = prompt_path(PROMPT_FILE)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if "{query}" not in txt:
            raise ValueError(f"Prompt 文件必须包含 '{{query}}' 占位符：{p}")
        return txt
    return DEFAULT_PROMPT


def load_testset() -> List[Dict]:
    """
    读取 qa_testset_500.json（JSON数组）
    - 每条至少包含 query
    - answer 作为参考答案保留，但不会作为模型输入（严格满足要求2）
    """
    p = data_path(DATA_FILE)
    if not p.exists():
        raise FileNotFoundError(f"未找到测试集文件：{p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("qa_testset_500.json 必须是 JSON 数组（list）。")

    items = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = obj.get("query", "")
        if isinstance(q, str) and q.strip():
            items.append({
                "query": q.strip(),
                "answer": obj.get("answer", "")  # 参考答案可为空
            })

    if not items:
        raise ValueError("测试集中没有找到有效的 'query' 字段。")
    return items


def build_model_input(tokenizer, prompt_tpl: str, query: str, use_chat_template: bool) -> str:
    """
    构造模型输入：
    - 如果模型/Tokenizer 支持 chat_template，则用对话模板（更稳）
    - 否则退化为纯文本 prompt
    """
    user_text = prompt_tpl.format(query=query)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 若某些环境/版本不兼容，自动回退
            pass

    return user_text


@torch.inference_mode()
def generate_one(model, tokenizer, input_text: str) -> str:
    """
    单条推理：
    只解码模型“新生成”的 token，
    避免把 system / user prompt 一起当成回答。
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=GEN_CONFIG["max_new_tokens"],
        do_sample=GEN_CONFIG["do_sample"],
        temperature=GEN_CONFIG["temperature"],
        top_p=GEN_CONFIG["top_p"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 🔑 关键：切掉输入部分，只保留新生成内容
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return answer.strip()



# =========================================================
# 3) 主流程：加载 -> 推理 -> 日志 -> 保存
# =========================================================
def run_baseline(model_id: str, model_tag: str, use_chat_template: bool) -> str:
    # ---------- 读取数据与prompt ----------
    items = load_testset()
    prompt_tpl = load_prompt_text()

    # ---------- 输出路径（确保目录存在） ----------
    out_dir = ensure_dir(results_path())  # -> experiments/exp2/results
    out_file = out_dir / f"answer_{model_tag}.json"  # 要求6：answer_模型参数

    # ---------- 设备与dtype ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = MODEL_CONFIG.get("dtype", "float16")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    # ---------- 打印关键配置（实验报告/复现需要） ----------
    print("=" * 80)
    print("[INFO] 实验：Exp2 Baseline（无RAG）")
    print(f"[INFO] LLM_EXPERIMENT = {os.environ.get('LLM_EXPERIMENT')}")
    print(f"[INFO] 数据集路径：{data_path(DATA_FILE)} | 样本数：{len(items)}")
    prompt_file_path = prompt_path(PROMPT_FILE)
    print(f"[INFO] Prompt：{prompt_file_path if prompt_file_path.exists() else 'DEFAULT_PROMPT'}")
    print(f"[INFO] 模型：{model_id} | tag：{model_tag}")
    print(f"[INFO] 设备：{device} | dtype：{dtype}")
    print(f"[INFO] 生成参数：{GEN_CONFIG}")
    print(f"[INFO] 输出文件：{out_file}")
    print("=" * 80)

    # ---------- 加载模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    # ---------- 推理循环 ----------
    results: List[Dict] = []
    start_time = time.time()

    for idx, obj in enumerate(items, start=1):
        query = obj["query"]
        ref_answer = obj.get("answer", "")

        # 要求2：只使用 query（不使用 ref_answer）
        input_text = build_model_input(tokenizer, prompt_tpl, query, use_chat_template=use_chat_template)
        llm_answer = generate_one(model, tokenizer, input_text)

        results.append({
            "query": query,
            "answer": ref_answer,
            "LLM_answer": llm_answer
        })

        # 要求3：进度输出
        if idx % LOG_EVERY == 0 or idx == len(items):
            elapsed = time.time() - start_time
            speed = idx / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {idx}/{len(items)} | {speed:.2f} 条/秒 | elapsed {elapsed:.1f}s")

        # 要求3：示例输出
        if idx == 1 or idx % PREVIEW_EVERY == 0:
            pq = query[:PREVIEW_CHARS] + ("…" if len(query) > PREVIEW_CHARS else "")
            pa = llm_answer[:PREVIEW_CHARS] + ("…" if len(llm_answer) > PREVIEW_CHARS else "")
            print("\n[示例输出]")
            print("Q:", pq)
            print("A:", pa)
            print()

    # ---------- 保存结果 ----------
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 已保存：{out_file}")
    return str(out_file)


# =========================================================
# 4) CLI 入口：默认使用 Qwen2.5-3B-Instruct
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Exp2 Baseline Inference (Qwen2.5 3B)")
    # 允许你将来覆盖，但默认就是 Qwen2.5-3B-Instruct
    parser.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"], help="HuggingFace model id")
    parser.add_argument("--model_tag", type=str, default=MODEL_CONFIG["model_tag"], help="output filename tag")
    args = parser.parse_args()

    run_baseline(
        model_id=args.model_id,
        model_tag=args.model_tag,
        use_chat_template=MODEL_CONFIG.get("use_chat_template", True),
    )


if __name__ == "__main__":
    main()
