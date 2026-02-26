# -*- coding: utf-8 -*-
"""
实验2：法律问答一致性评测（LLM-as-a-Judge，API 版本）

判定任务（你给的 prompt）：
- 对比【标准答案 answer】与【待测答案 LLM_answer】
- Judge 只输出 1 或 0（不输出解释）

输入：
- experiments/exp2/results/<pred_file>
  每条包含：query / answer / LLM_answer

输出（仍在 experiments/exp2/results）：
1) 逐条明细：
   eval_consistency_<pred_tag>__judge_<judge_tag>.json
2) 汇总指标（包含一致性准确率）：
   metrics_consistency_<pred_tag>__judge_<judge_tag>.json

使用方式示例：
- OpenAI:
  set OPENAI_API_KEY=你的key
  python -m src.exp2.baseline.eval --provider openai --pred_file answer_qwen2.5_3b.json --judge_model gpt-4o-mini --judge_tag gpt-4o-mini

- DashScope（阿里云 Model Studio OpenAI 兼容）:
  set DASHSCOPE_API_KEY=你的key
  python -m src.exp2.baseline.eval --provider dashscope --pred_file answer_qwen2.5_3b.json --judge_model qwen3-32b --judge_tag qwen3-32b --dashscope_region intl
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】锁定实验目录为 exp2（避免 paths.py 默认指向 exp1）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp2")

from src.common.paths import results_path, prompt_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 全部设置前置：Prompt / API / 重试 / 日志
# =========================================================
import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# OpenAI SDK（也用于 DashScope 的 OpenAI 兼容接口）
from openai import OpenAI


JUDGE_PROMPT_FILE = "judge_consistency.txt"

DEFAULT_JUDGE_PROMPT = """任务：法律问答一致性评测

你是一位资深的法律专家。请对比【真实回答】和【模型生成回答】，判断待测答案是否在法律结论、核心逻辑和建议上与标准答案保持一致。

【问题】: {query}

【标准答案】: {answer}

【待测答案】: {LLM_answer}

输出标准：

1. 如果待测答案的法律适用正确，结论与标准答案一致请输出：1

2. 如果待测答案存在关键性事实错误、法律引用错误或结论错误，请输出：0

输出格式：

只输出数字 1 或 0，不要输出任何解释说明和其他信息。
"""

# API 调用参数（为了“更稳定”，建议低温/不采样）
API_TEMPERATURE = 0.0
API_MAX_TOKENS = 8

# 重试参数（网络波动/限速时有用）
MAX_RETRIES = 6
RETRY_BASE_SLEEP_SEC = 2.0

# 进度输出
LOG_EVERY = 10
PREVIEW_EVERY = 50
PREVIEW_CHARS = 160


@dataclass
class EvalConfig:
    provider: str               # openai | dashscope
    pred_file: str              # answer_xxx.json
    pred_tag: str               # 用于输出命名
    judge_model: str            # gpt-4o-mini / qwen3-32b ...
    judge_tag: str              # 用于输出命名
    dashscope_region: str       # cn | intl（仅 dashscope 用）


# =========================================================
# 2) 工具函数：prompt / 数据 / 解析 / API client
# =========================================================
def load_judge_prompt() -> str:
    """
    读取 judge prompt：
    - 若 experiments/exp2/prompts/judge_consistency.txt 存在，则读取
    - 否则使用 DEFAULT_JUDGE_PROMPT（保证优先跑通）
    """
    p = prompt_path(JUDGE_PROMPT_FILE)
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        for k in ("{query}", "{answer}", "{LLM_answer}"):
            if k not in text:
                raise ValueError(f"Judge prompt 缺少占位符 {k}：{p}")
        return text
    return DEFAULT_JUDGE_PROMPT


def load_predictions(pred_file: str) -> List[Dict]:
    """
    从 experiments/exp2/results/<pred_file> 读取预测结果 JSON（数组）
    每条需要包含：query / answer / LLM_answer
    """
    p = results_path(pred_file)
    if not p.exists():
        raise FileNotFoundError(f"未找到预测文件：{p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("预测文件必须是 JSON 数组（list）。")

    items = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = obj.get("query", "")
        a = obj.get("answer", "")
        la = obj.get("LLM_answer", "")
        if isinstance(q, str) and q.strip():
            items.append({"query": q, "answer": a, "LLM_answer": la})

    if not items:
        raise ValueError("预测文件中没有有效样本（至少需要 query 字段）。")
    return items


def parse_label(text: str) -> Optional[int]:
    """
    Judge 可能会输出多余空白/换行。这里鲁棒解析：抓第一个 0 或 1。
    """
    m = re.search(r"[01]", text or "")
    return int(m.group(0)) if m else None


def build_client(cfg: EvalConfig) -> OpenAI:
    """
    构建 OpenAI SDK client：
    - provider=openai：使用 OPENAI_API_KEY
    - provider=dashscope：使用 DASHSCOPE_API_KEY + base_url（OpenAI 兼容）
      文档说明只需改 API key / base_url / model name 即可迁移。:contentReference[oaicite:5]{index=5}
    """
    provider = cfg.provider.lower().strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("未检测到环境变量 OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    if provider == "dashscope":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("未检测到环境变量 DASHSCOPE_API_KEY")

        # 阿里云 Model Studio OpenAI 兼容接口 base_url
        # Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        # China (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1 :contentReference[oaicite:6]{index=6}
        region = (cfg.dashscope_region or "intl").lower().strip()
        base_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if region == "cn"
            else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        return OpenAI(api_key=api_key, base_url=base_url)

    raise ValueError("provider 只能是 openai 或 dashscope")


def judge_once(client: OpenAI, model: str, prompt_text: str) -> str:
    """
    单次调用 Judge（使用 chat.completions，返回文本）
    为了稳定：temperature=0，max_tokens 很小。
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        temperature=API_TEMPERATURE,
        max_tokens=API_MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def judge_with_retry(client: OpenAI, model: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    """
    带重试的调用：
    - 遇到网络/限速等异常：指数退避重试
    返回：(raw_text, error_str)
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = judge_once(client, model, prompt_text)
            return raw, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            sleep_sec = RETRY_BASE_SLEEP_SEC * (2 ** (attempt - 1))
            time.sleep(sleep_sec)
    return "", last_err


# =========================================================
# 3) 主流程：逐条评测 -> 统计一致性准确率 -> 保存
# =========================================================
def run_eval(cfg: EvalConfig) -> Tuple[str, str]:
    items = load_predictions(cfg.pred_file)
    judge_prompt_tpl = load_judge_prompt()

    out_dir = ensure_dir(results_path())  # experiments/exp2/results
    detail_path = out_dir / f"eval_consistency_{cfg.pred_tag}__judge_{cfg.judge_tag}.json"
    metrics_path = out_dir / f"metrics_consistency_{cfg.pred_tag}__judge_{cfg.judge_tag}.json"

    print("=" * 80)
    print("[INFO] Exp2 一致性评测（API Judge）")
    print(f"[INFO] provider: {cfg.provider}")
    print(f"[INFO] pred_file: {results_path(cfg.pred_file)} | n={len(items)}")
    pf = prompt_path(JUDGE_PROMPT_FILE)
    print(f"[INFO] judge_prompt: {pf if pf.exists() else 'DEFAULT_JUDGE_PROMPT'}")
    print(f"[INFO] judge_model: {cfg.judge_model} | judge_tag: {cfg.judge_tag}")
    if cfg.provider == "dashscope":
        print(f"[INFO] dashscope_region: {cfg.dashscope_region}")
    print(f"[INFO] output_detail: {detail_path}")
    print(f"[INFO] output_metrics: {metrics_path}")
    print("=" * 80)

    client = build_client(cfg)

    t0 = time.time()
    details: List[Dict] = []

    n_valid = 0
    n_correct = 0
    n_parse_fail = 0
    n_api_fail = 0

    for i, obj in enumerate(items, start=1):
        query = obj["query"]
        answer = obj.get("answer", "")
        llm_answer = obj.get("LLM_answer", "")

        prompt_text = judge_prompt_tpl.format(query=query, answer=answer, LLM_answer=llm_answer)

        raw, err = judge_with_retry(client, cfg.judge_model, prompt_text)
        label = parse_label(raw)

        if err is not None:
            n_api_fail += 1
        if label is None:
            n_parse_fail += 1
        else:
            n_valid += 1
            n_correct += int(label == 1)

        details.append({
            "query": query,
            "answer": answer,
            "LLM_answer": llm_answer,
            "judge_raw": raw,
            "judge_label": label,       # 0/1/None
            "judge_error": err,         # None 或错误字符串
        })

        if i % LOG_EVERY == 0 or i == len(items):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {i}/{len(items)} | {speed:.2f} 条/秒 | api_fail={n_api_fail} | parse_fail={n_parse_fail}")

        if i == 1 or i % PREVIEW_EVERY == 0:
            print("\n[PREVIEW]")
            print("Q:", (query[:PREVIEW_CHARS] + "…") if len(query) > PREVIEW_CHARS else query)
            print("Judge raw:", raw)
            print("Parsed label:", label)
            if err:
                print("API error:", err)
            print()

    accuracy = (n_correct / n_valid) if n_valid > 0 else 0.0
    elapsed = time.time() - t0

    metrics = {
        "provider": cfg.provider,
        "pred_file": cfg.pred_file,
        "pred_tag": cfg.pred_tag,
        "judge_model": cfg.judge_model,
        "judge_tag": cfg.judge_tag,
        "num_samples": len(items),
        "num_valid_labels": n_valid,
        "num_label_1": n_correct,
        "num_parse_fail": n_parse_fail,
        "num_api_fail": n_api_fail,
        "accuracy_consistency": accuracy,
        "time_sec": elapsed,
        "api_params": {
            "temperature": API_TEMPERATURE,
            "max_tokens": API_MAX_TOKENS,
            "max_retries": MAX_RETRIES,
        },
    }

    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"[DONE] accuracy_consistency = {accuracy:.4f} (valid={n_valid}, api_fail={n_api_fail}, parse_fail={n_parse_fail})")
    print(f"[DONE] saved detail:  {detail_path}")
    print(f"[DONE] saved metrics: {metrics_path}")
    print("=" * 80)

    return str(detail_path), str(metrics_path)


# =========================================================
# 4) CLI：可换 pred_file / judge_model / provider
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Exp2 Consistency Eval (API Judge)")

    parser.add_argument("--provider", choices=["openai", "dashscope"], required=True,
                        help="使用哪个 API：openai 或 dashscope（阿里云 Model Studio OpenAI 兼容）")
    parser.add_argument("--pred_file", required=True,
                        help="exp2/results 下的预测文件名，例如 answer_qwen2.5_3b.json / answer_qwen2.5_7b.json")
    parser.add_argument("--pred_tag", default=None,
                        help="用于输出命名的 tag；默认从 pred_file 推断（answer_xxx.json -> xxx）")

    parser.add_argument("--judge_model", required=True,
                        help="Judge 模型名：如 openai:gpt-4o-mini / dashscope:qwen3-32b")
    parser.add_argument("--judge_tag", default=None,
                        help="用于输出命名的 judge_tag；默认用 judge_model（会做安全清洗）")

    parser.add_argument("--dashscope_region", choices=["cn", "intl"], default="intl",
                        help="仅 provider=dashscope 用：cn=北京，intl=新加坡")

    args = parser.parse_args()

    # pred_tag 推断
    pred_tag = args.pred_tag
    if not pred_tag:
        base = os.path.basename(args.pred_file)
        m = re.match(r"answer_(.+)\.json$", base)
        pred_tag = m.group(1) if m else os.path.splitext(base)[0]

    # judge_tag 推断（避免文件名里有 / : 等）
    judge_tag = args.judge_tag
    if not judge_tag:
        judge_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.judge_model)

    cfg = EvalConfig(
        provider=args.provider,
        pred_file=args.pred_file,
        pred_tag=pred_tag,
        judge_model=args.judge_model,
        judge_tag=judge_tag,
        dashscope_region=args.dashscope_region,
    )

    run_eval(cfg)


if __name__ == "__main__":
    main()
