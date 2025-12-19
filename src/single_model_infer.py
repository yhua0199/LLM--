# -*- coding: utf-8 -*-
"""
single_model_infer.py

单模型：意图识别 + Query 改写（融合推理）
输入：data/intent/raw/intent_2k.json 中的 "问题"
输出：模型返回 JSON：{"intent":"","rewrite_query":""}
保存：results/rewrite/qwen2.5_3b_intent_rewrite.jsonl
每 20 条打印一次进度 + 样例
所有配置集中在文件最前面
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 0) 配置区（全部放前面）
# =========================

# 路径配置
DATA_PATH = "data/intent/intent_2k.json"
PROMPT_PATH = "prompts/intent_rewrite_infer.txt"

OUT_DIR = "results/rewrite"
OUT_FILE = "qwen2.5_1.5b_intent_rewrite.jsonl"  # 你指定的命名
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

# 模型配置（你可按需改，但文件名固定为 qwen2.5_3b_intent_rewrite.jsonl）
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 推理参数
MAX_NEW_TOKENS = 80
DO_SAMPLE = True
TEMPERATURE = 0.2
TOP_P = 0.9

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 打印进度
PRINT_EVERY_N = 10

# JSON 解析容错次数
JSON_PARSE_RETRY = 2


# =========================
# 1) 工具函数
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_first_json_obj(text: str) -> Optional[str]:
    """
    从文本中提取第一个完整的 JSON 对象块（按 {} 配对）
    """
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i + 1]
    return None


def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    """
    尝试解析 JSON，做轻量容错（去 codefence、替换中文引号、截断到最后一个 }）
    """
    if not s:
        return None
    s = s.strip()

    # 去掉 ```json ... ``` 包裹
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 替换中文引号
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 截断到最后一个 }
    if "}" in s:
        s = s[: s.rfind("}") + 1]

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 若完全是单引号风格，尝试替换
        if "'" in s and '"' not in s:
            s2 = s.replace("'", '"')
            try:
                return json.loads(s2)
            except json.JSONDecodeError:
                return None
        return None


def fill_prompt(prompt_tmpl: str, query: str) -> str:
    """
    用 {query} 占位符填充
    """
    if "{query}" in prompt_tmpl:
        return prompt_tmpl.replace("{query}", query)
    # 若模板没有 {query}，就追加到末尾
    return prompt_tmpl.rstrip() + "\n\n【用户问题】\n" + query


# =========================
# 2) 主逻辑
# =========================

def main():
    print(f"[INFO] DEVICE={DEVICE}, DTYPE={DTYPE}")
    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] DATA_PATH={DATA_PATH}")
    print(f"[INFO] PROMPT_PATH={PROMPT_PATH}")
    print(f"[INFO] OUT_PATH={OUT_PATH}")

    # 读 prompt
    prompt_tmpl = load_prompt(PROMPT_PATH)

    # 读数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("intent_2k.json 预期是 list[dict]")

    total = len(data)
    print(f"[INFO] Loaded {total} samples.")

    # 创建输出目录
    safe_mkdir(OUT_DIR)

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    t0 = time.time()
    written = 0

    with open(OUT_PATH, "w", encoding="utf-8") as w:
        for idx, sample in enumerate(data, start=1):
            query = sample.get("问题", "")
            gt_type = sample.get("类型", "")

            # 空问题处理：也写一行占位，保证行数对齐
            if not query:
                out_obj = {
                    "原始问题": "",
                    "改写问题": "",
                    "预测类型": "",
                    "真实类型": gt_type
                }
                w.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1
                continue

            prompt = fill_prompt(prompt_tmpl, query)

            # 编码
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else None,
                    top_p=TOP_P if DO_SAMPLE else None,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 解码
            full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # 尽量截取生成增量部分，避免 prompt 干扰 JSON 解析
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            gen_text = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.strip()

            # 解析 JSON
            parsed = None
            cur_text = gen_text

            for _ in range(JSON_PARSE_RETRY):
                json_str = extract_first_json_obj(cur_text)
                if json_str:
                    parsed = try_parse_json(json_str)
                else:
                    parsed = try_parse_json(cur_text)

                if parsed and isinstance(parsed, dict):
                    break

                # 容错：去掉前置说明，强行从第一个 { 开始
                cur_text = re.sub(r"^[\s\S]*?(\{)", r"{", cur_text, count=1).strip()

            intent_pred = ""
            rewrite_query = ""

            if parsed and isinstance(parsed, dict):
                intent_pred = str(parsed.get("intent", "")).strip()
                rewrite_query = str(parsed.get("rewrite_query", "")).strip()

            # 写入你指定格式
            out_obj = {
                "原始问题": query,
                "改写问题": rewrite_query,
                "预测类型": intent_pred,
                "真实类型": gt_type
            }
            w.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

            # 每 20 条打印一次进度 + 样例
            if idx % PRINT_EVERY_N == 0 or idx == 1 or idx == total:
                elapsed = time.time() - t0
                speed = written / elapsed if elapsed > 0 else 0.0
                print(f"\n[PROGRESS] {idx}/{total} | written={written} | {speed:.2f} samples/s")
                print("[SAMPLE] 原始问题:", query)
                print("[SAMPLE] 模型输出(截取):", (gen_text[:300] + "..." if len(gen_text) > 300 else gen_text))
                print("[SAMPLE] 保存JSON:", json.dumps(out_obj, ensure_ascii=False))

    print(f"\n[DONE] Wrote {written} lines to: {OUT_PATH}")


if __name__ == "__main__":
    main()
