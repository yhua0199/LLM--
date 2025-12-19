# -*- coding: utf-8 -*-
"""
Self-instruct generation of legal QA pairs (query, answer)
Grounded on law_articles.json

Output:
  data/rewrite/synth/self_instruct_qa_300.jsonl
"""

# -*- coding: utf-8 -*-
"""
Self-instruct generation of legal QA pairs (query, answer)
Grounded on law_articles.json

Output:
  data/rewrite/synth/self_instruct_qa_300.jsonl
"""

# =========================================================
# ================ 0. 全局配置（只改这里） =================
# =========================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # 使用的模型
TARGET_NUM = 240                      # 目标生成数量

# generation parameters
MAX_NEW_TOKENS = 80
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0


# runtime control
RANDOM_SEED = 42
LOG_EVERY = 1         # 每生成多少条输出一次进度
SHOW_EXAMPLE_EVERY = 1 # 每生成多少条打印一个示例

# =========================================================
# ===================== 1. 依赖导入 ========================
# =========================================================

import json
import random
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# ===================== 2. 路径配置 ========================
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

LAW_PATH = ROOT_DIR / "data" / "rewrite" / "seed" / "law_articles.json"
PROMPT_PATH = ROOT_DIR / "prompts" / "rewrite_self_instruct_qa.txt"

OUT_DIR = ROOT_DIR / "data" / "rewrite" / "synth"
OUT_PATH = OUT_DIR / "self_instruct_qa_300.jsonl"


# =========================================================
# ===================== 3. 工具函数 ========================
# =========================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def render_prompt(template: str, law_title: str, law_content: str) -> str:
    return (
        template
        .replace("{{law_title}}", law_title)
        .replace("{{law_content}}", law_content)
    )

def extract_json(text: str) -> Optional[Dict]:
    """
    从模型输出中提取第一个 JSON 对象
    """
    if not text:
        return None

    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None

    try:
        return json.loads(text[l:r+1])
    except Exception:
        return None

def normalize_qa(obj: Dict) -> Optional[Dict[str, str]]:
    if not isinstance(obj, dict):
        return None
    if "query" not in obj or "answer" not in obj:
        return None

    query = str(obj["query"]).strip()
    answer = str(obj["answer"]).strip()

    if not query or not answer:
        return None

    query = re.sub(r"\s+", " ", query)
    answer = re.sub(r"\s+", " ", answer)

    return {"query": query, "answer": answer}


# =========================================================
# ===================== 4. 主流程 ==========================
# =========================================================

def main():
    print("=" * 60)
    print("Self-instruct QA Generation Started")
    print(f"Model: {MODEL_ID}")
    print(f"Target samples: {TARGET_NUM}")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    law_articles = load_json(LAW_PATH)
    prompt_template = load_text(PROMPT_PATH)

    random.shuffle(law_articles)

    print(f"[INFO] Loaded {len(law_articles)} law articles")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    print("[DEBUG] first_param_device:", next(model.parameters()).device)

    results = []
    seen_queries = set()

    idx = 0
    while len(results) < TARGET_NUM and idx < len(law_articles):
        item = law_articles[idx]
        idx += 1

        law_title = item.get("law_title", "").strip()
        law_content = item.get("content", "").strip()

        if not law_title or not law_content:
            continue

        prompt = render_prompt(prompt_template, law_title, law_content)

        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(chat_text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        obj = extract_json(raw)
        qa = normalize_qa(obj) if obj else None

        if qa is None:
            continue
        if qa["query"] in seen_queries:
            continue

        seen_queries.add(qa["query"])

        results.append({
            "query": qa["query"],
            "answer": qa["answer"]
        })

        # ===================== 日志输出 =====================
        cur = len(results)

        if cur % LOG_EVERY == 0:
            print(f"[PROGRESS] {cur}/{TARGET_NUM} samples generated")

        if cur % SHOW_EXAMPLE_EVERY == 0:
            print("-" * 50)
            print("[EXAMPLE]")
            print("Query :", qa["query"])
            print("Answer:", qa["answer"])
            print("-" * 50)

    # ===================== 写文件 =====================
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"[DONE] Generated {len(results)} samples")
    print(f"[SAVED] {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

