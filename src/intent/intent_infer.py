# 用Qwen2.5，因为稳定性强，更听话，同时和当前任务匹配度高，Qwen3可能会用力过猛；为什么不是GPT，而是Qwen，Qwen开源，中文理解更好
# 大小与现存：参数量*2+上框架开销、临时张量、KV cache
# 4-bit 约为FP16的1/4
'''
0.5B~1G
1.5B~3G
3B~6G
7B~14G
'''

"""
intent_infer.py

功能：
- 使用 Qwen2.5-0.5B-Instruct 模型进行【意图识别】推理
- 输入：experiments/<exp>/data/intent/intent_2k.json
- 输出：experiments/<exp>/results/intent/intent_pred_qwen2.5_0.5b.jsonl

说明：
- 这是【推理脚本】，不是训练脚本
- 只做 inference，不更新任何模型参数
"""


import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.paths import data_path, ensure_dir, prompt_path, results_path


# =========================
# 1. 模型与任务配置
# =========================

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = "qwen2.5_7b"


LABELS = ["法律类", "违规类", "闲聊类"]

# 小模型有时会先吐空白/换行再吐标签，给多一点 token 更稳
MAX_NEW_TOKENS = 3

# ---------- 途中检查开关 ----------
LOG_EVERY = 10           # 每50条打印一次进度
SAMPLE_EVERY = 400       # 每200条打印一次样例
SHOW_SAMPLES = 3         # 每次打印样例条数
DEBUG_FIRST_N = 20       # 前N条详细打印（sanity check）
WARN_UNKNOWN_RATE = 0.30 # UNKNOWN占比超过30%就报警


# =========================
# 3. Prompt 相关函数
# =========================

def load_prompt(path: Path) -> str:
    """读取意图识别 Prompt 模板（intent_infer.txt）"""
    return path.read_text(encoding="utf-8")


def build_prompt(prompt_tpl: str, question: str) -> str:
    """
    将具体问题填充进 Prompt 模板。
    支持两种写法：
    1) Prompt 中包含 {question} 占位符
    2) Prompt 中不包含，占位时自动拼接到末尾
    """
    if "{question}" in prompt_tpl:
        return prompt_tpl.format(question=question)
    return prompt_tpl.rstrip() + "\n\n用户输入：\n" + question + "\n"


def build_chat_prompt(tokenizer, prompt_text: str) -> str:
    """
    对 Qwen Instruct 模型更稳：用 chat template 包一层 system/user 格式。
    这样模型更容易遵循“只输出标签”的指令。
    """
    messages = [
        {"role": "system", "content": "你是一个严格的文本分类器。"},
        {"role": "user", "content": prompt_text},
    ]
    # add_generation_prompt=True 会加上 assistant 起始标记，利于生成
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# =========================
# 4. 模型输出解析
# =========================

def parse_label(text: str) -> str:
    """
    从模型生成的文本中提取意图类别。
    - 先做精确包含匹配
    - 再做轻量兜底（小模型常省略“类”）
    """
    t = (text or "").strip()

    # 精确匹配（最可靠）
    for lab in LABELS:
        if lab in t:
            return lab

    # 兜底（避免大量UNKNOWN）
    if "法律" in t:
        return "法律类"
    if "违规" in t or "敏感" in t or "政治" in t or "色情" in t or "暴恐" in t:
        return "违规类"
    if "闲聊" in t or "聊天" in t or "日常" in t or "问候" in t or "天气" in t:
        return "闲聊类"

    return "UNKNOWN"


# =========================
# 5. 单条推理函数
# =========================

@torch.inference_mode()
def infer_one(model, tokenizer, prompt: str) -> str:
    """
    对单条输入进行模型推理。
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # 分类任务：关闭采样，保证稳定
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# =========================
# 6. 主流程
# =========================

def main():
    data_file = data_path("intent", "intent_2k.json")
    prompt_file = prompt_path("intent_infer.txt")
    out_dir = results_path("intent")
    ensure_dir(out_dir)
    out_file = out_dir / f"intent_pred_{MODEL_TAG}.jsonl"

    print(f"[INFO] Model : {MODEL_ID}")
    print(f"[INFO] Data  : {data_file}")
    print(f"[INFO] Prompt: {prompt_file}")
    print(f"[INFO] Output: {out_file}")

    # ---------- 加载数据 ----------
    data = json.loads(data_file.read_text(encoding="utf-8"))

    # ---------- 加载 Prompt ----------
    prompt_tpl = load_prompt(prompt_file)

    # ---------- 加载模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("[DEBUG] model.device =", next(model.parameters()).device)
    print("[DEBUG] cuda_available =", torch.cuda.is_available())

    # ---------- 推理循环 ----------
    results = []
    pred_counter = Counter()
    correct = 0

    # 用于途中打印的样例缓存（优先收集 wrong/UNKNOWN）
    samples = []

    total = len(data)

    for idx, item in enumerate(data, 1):
        question = item["问题"]
        gt = item["类型"]

        # 1) build plain prompt text
        plain_prompt = build_prompt(prompt_tpl, question)

        # 2) wrap to chat template (more stable for Qwen Instruct)
        prompt = build_chat_prompt(tokenizer, plain_prompt)

        raw_output = infer_one(model, tokenizer, prompt)
        pred = parse_label(raw_output)

        pred_counter[pred] += 1
        if pred == gt:
            correct += 1

        # 保存结果（强烈建议把 raw_output 也存下来）
        results.append({
            "问题": question,
            "真实类型": gt,
            "预测类型": pred,
            "raw_output": raw_output
        })

        # 前N条详细打印：快速 sanity check
        if idx <= DEBUG_FIRST_N:
            print(f"[DEBUG {idx}] GT={gt} | PRED={pred} | RAW='{raw_output}'")

        # 收集样例（优先错的/UNKNOWN）
        if (pred != gt) or (pred == "UNKNOWN"):
            if len(samples) < SHOW_SAMPLES:
                samples.append((question, gt, pred, raw_output))

        # 进度打印 + 滚动统计 + 样例
        if idx % LOG_EVERY == 0:
            acc_so_far = correct / idx
            unk_so_far = pred_counter.get("UNKNOWN", 0) / idx
            print(f"[RUN] {idx}/{total} | acc={acc_so_far:.4f} | unk={unk_so_far:.2%}")

            if unk_so_far > WARN_UNKNOWN_RATE:
                print("[WARN] UNKNOWN rate is high -> check prompt format / increase MAX_NEW_TOKENS / improve parse_label.")

            if (idx % SAMPLE_EVERY == 0) and samples:
                print("---- samples (wrong/UNKNOWN) ----")
                for q, gt_, pred_, raw_ in samples:
                    print(f"Q: {q}")
                    print(f"GT: {gt_} | PRED: {pred_} | RAW: {raw_}")
                    print("----")
                samples.clear()

    # ---------- 保存结果 ----------
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 输出统计 ----------
    print("\n=== Inference Summary (0.5B) ===")
    print(f"Total samples : {total}")
    print(f"Accuracy (rough): {correct / total:.4f}")
    print("Prediction distribution:")
    for k, v in pred_counter.items():
        print(f"  {k}: {v}")

    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
