# -*- coding: utf-8 -*-
"""
实验2：RAG 输出一致性评测（LLM-as-a-Judge，API 版本）
- 仅对“已生成的 RAG 结果文件”做评估（不做推理、不生成RAG结果）

判定任务：
- 对比【标准答案 answer】与【待测答案 LLM_answer】
- Judge 只输出 1 或 0（不输出解释）

输入：
- RAG 预测文件（JSON 数组），每条包含：query / answer / LLM_answer
  典型路径：experiments/exp2/results/rag/answer_rag_xxx.json
  也支持绝对路径（Windows / Linux / 云端）

输出：
- 目录：experiments/exp2/results/<RAG_EVAL_OUT_SUBDIR>   (默认 rag_eval)
1) 逐条明细：
   eval_consistency_rag_<pred_tag>__judge_<judge_tag>.json
2) 汇总指标：
   metrics_consistency_rag_<pred_tag>__judge_<judge_tag>.json

用法示例（项目根目录）：
  # 锁定实验
  $env:LLM_EXPERIMENT="exp2"

  # OpenAI
  $env:OPENAI_API_KEY="sk-xxx"
  python -m src.exp2.rag.eval_rag_consistency --provider openai ^
    --pred_file "D:\PythonProject\LLM\experiments\exp2\results\rag\answer_rag_Qwen2.5-3B-Instruct_bm10_bg10_rrf20_ctx10.json" ^
    --judge_model gpt-4o-mini --judge_tag gpt-4o-mini

  # DashScope（阿里云 OpenAI 兼容）
  $env:DASHSCOPE_API_KEY="xxx"
  python -m src.exp2.rag.eval_rag_consistency --provider dashscope ^
    --pred_file experiments/exp2/results/rag/answer_rag_xxx.json ^
    --judge_model qwen3-32b --judge_tag qwen3-32b --dashscope_region intl

可选环境变量：
- RAG_EVAL_OUT_SUBDIR=rag_eval
- RAG_EVAL_PRED_FILE=...（不传 --pred_file 时使用）
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


JUDGE_PROMPT_FILE = "judge_consistency.txt"

DEFAULT_JUDGE_PROMPT = """任务：法律问答一致性评测
你是一位资深的法律专家。请对比【真实回答】和【模型生成回答】，判断待测答案是否在法律结论、核心逻辑和建议上与标准答案保持一致。

【问题】: {query}
【标准答案】: {answer}
【待测答案】: {LLM_answer}

输出标准：
1. 如果待测答案的法律适用和结论与标准答案一致，或更好地回答了问题，输出：1
2. 如果待测答案存在关键性事实错误、法律错误或结论错误，输出：0

输出格式：
只输出数字 1 或 0，否则视为 0，不要输出任何解释说明和其他信息。
"""

# API 调用参数（为了稳定：低温、短输出）
API_TEMPERATURE = 0.0
API_MAX_TOKENS = 8

# 重试参数
MAX_RETRIES = 6
RETRY_BASE_SLEEP_SEC = 2.0

# 进度输出
LOG_EVERY = 10
PREVIEW_EVERY = 50
PREVIEW_CHARS = 160

# 输出子目录（可用 env 覆盖）
OUT_SUBDIR = os.getenv("RAG_EVAL_OUT_SUBDIR", "rag_eval").strip()


@dataclass
class EvalConfig:
    provider: str               # openai | dashscope
    pred_file: str              # 预测文件（可相对 experiments/exp2/results，也可绝对路径）
    pred_tag: str               # 用于输出命名
    judge_model: str            # gpt-4o-mini / qwen3-32b ...
    judge_tag: str              # 用于输出命名
    dashscope_region: str       # cn | intl（仅 dashscope 用）


# =========================================================
# 2) 工具函数：prompt / 数据 / 解析 / API client / 路径处理
# =========================================================
def load_judge_prompt() -> str:
    """
    读取 judge prompt：
    - 若 experiments/exp2/prompts/judge_consistency.txt 存在，则读取
    - 否则使用 DEFAULT_JUDGE_PROMPT
    """
    p = prompt_path(JUDGE_PROMPT_FILE)
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        for k in ("{query}", "{answer}", "{LLM_answer}"):
            if k not in text:
                raise ValueError(f"Judge prompt 缺少占位符 {k}：{p}")
        return text
    return DEFAULT_JUDGE_PROMPT


def resolve_pred_path(pred_file: str) -> Path:
    """
    路径友好：
    - 如果 pred_file 是绝对路径：直接用
    - 如果是相对路径：
        1) 若它已经包含 experiments/exp2/results/... 这样的前缀，直接用
        2) 否则默认相对 experiments/exp2/results 下寻找（results_path(pred_file)）
    """
    pf = Path(pred_file)

    # 绝对路径（Windows: D:\... / Linux: /...）
    if pf.is_absolute():
        return pf

    # 用户可能传 experiments/exp2/results/rag/xxx.json 这种相对路径
    if "experiments" in pf.parts:
        return pf

    # 默认：exp2/results/<pred_file>
    return results_path(pred_file)


def load_predictions(pred_file: str) -> List[Dict]:
    """
    读取预测结果 JSON（数组）
    每条需要包含：query / answer / LLM_answer
    """
    p = resolve_pred_path(pred_file)
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
    """鲁棒解析：抓第一个 0 或 1。"""
    m = re.search(r"[01]", text or "")
    return int(m.group(0)) if m else None


def build_client(cfg: EvalConfig) -> OpenAI:
    """
    构建 OpenAI SDK client：
    - provider=openai：使用 OPENAI_API_KEY
    - provider=dashscope：使用 DASHSCOPE_API_KEY + base_url（OpenAI 兼容）
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

        region = (cfg.dashscope_region or "intl").lower().strip()
        base_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if region == "cn"
            else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        return OpenAI(api_key=api_key, base_url=base_url)

    raise ValueError("provider 只能是 openai 或 dashscope")


def judge_once(client: OpenAI, model: str, prompt_text: str) -> str:
    """单次调用 Judge（chat.completions）。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=API_TEMPERATURE,
        max_tokens=API_MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def judge_with_retry(client: OpenAI, model: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    """带指数退避重试。返回：(raw_text, error_str)。"""
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


def infer_pred_tag(pred_file: str) -> str:
    """
    用于输出命名的 pred_tag：
    - 取文件名去掉扩展名
    - 如果以 answer_ 开头，去掉 answer_
    """
    name = Path(pred_file).name
    stem = Path(name).stem  # 去掉 .json
    m = re.match(r"answer_(.+)$", stem)
    return m.group(1) if m else stem


def safe_tag(s: str) -> str:
    """清洗 tag 避免文件名非法字符。"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_") or "tag"


# =========================================================
# 3) 主流程：逐条评测 -> 统计一致性准确率 -> 保存
# =========================================================
def run_eval(cfg: EvalConfig) -> Tuple[str, str]:
    items = load_predictions(cfg.pred_file)
    judge_prompt_tpl = load_judge_prompt()

    out_dir = ensure_dir(results_path() / OUT_SUBDIR)  # experiments/exp2/results/<OUT_SUBDIR>
    detail_path = out_dir / f"eval_consistency_rag_{cfg.pred_tag}__judge_{cfg.judge_tag}.json"
    metrics_path = out_dir / f"metrics_consistency_rag_{cfg.pred_tag}__judge_{cfg.judge_tag}.json"

    pred_path = resolve_pred_path(cfg.pred_file)

    print("=" * 90)
    print("[INFO] Exp2 RAG 一致性评测（API Judge，仅评估结果文件）")
    print(f"[INFO] provider: {cfg.provider}")
    print(f"[INFO] pred_file: {pred_path} | n={len(items)}")
    pf = prompt_path(JUDGE_PROMPT_FILE)
    print(f"[INFO] judge_prompt: {pf if pf.exists() else 'DEFAULT_JUDGE_PROMPT'}")
    print(f"[INFO] judge_model: {cfg.judge_model} | judge_tag: {cfg.judge_tag}")
    if cfg.provider == "dashscope":
        print(f"[INFO] dashscope_region: {cfg.dashscope_region}")
    print(f"[INFO] output_detail:  {detail_path}")
    print(f"[INFO] output_metrics: {metrics_path}")
    print("=" * 90)

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
        "task": "rag_consistency_eval",
        "provider": cfg.provider,
        "pred_file": str(pred_path),
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
        "output_dir": str(out_dir),
    }

    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 90)
    print(f"[DONE] accuracy_consistency = {accuracy:.4f} (valid={n_valid}, api_fail={n_api_fail}, parse_fail={n_parse_fail})")
    print(f"[DONE] saved detail:  {detail_path}")
    print(f"[DONE] saved metrics: {metrics_path}")
    print("=" * 90)

    return str(detail_path), str(metrics_path)


# =========================================================
# 4) CLI：可换 pred_file / judge_model / provider
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Exp2 RAG Consistency Eval (API Judge, eval-only)")

    parser.add_argument("--provider", choices=["openai", "dashscope"], required=True,
                        help="使用哪个 API：openai 或 dashscope（阿里云 OpenAI 兼容）")

    parser.add_argument("--pred_file", default=os.getenv("RAG_EVAL_PRED_FILE", ""),
                        help="RAG 预测文件：可为绝对路径，或相对 experiments/exp2/results 的路径")
    parser.add_argument("--pred_tag", default=None,
                        help="用于输出命名的 tag；默认从 pred_file 推断（去掉 answer_ 前缀）")

    parser.add_argument("--judge_model", required=True,
                        help="Judge 模型名：如 openai:gpt-4o-mini / dashscope:qwen3-32b")
    parser.add_argument("--judge_tag", default=None,
                        help="用于输出命名的 judge_tag；默认用 judge_model（会做安全清洗）")

    parser.add_argument("--dashscope_region", choices=["cn", "intl"], default="intl",
                        help="仅 provider=dashscope 用：cn=北京，intl=新加坡")

    args = parser.parse_args()

    if not args.pred_file:
        raise ValueError("请提供 --pred_file 或设置环境变量 RAG_EVAL_PRED_FILE")

    pred_tag = args.pred_tag or infer_pred_tag(args.pred_file)
    judge_tag = args.judge_tag or safe_tag(args.judge_model)

    cfg = EvalConfig(
        provider=args.provider,
        pred_file=args.pred_file,
        pred_tag=safe_tag(pred_tag),
        judge_model=args.judge_model,
        judge_tag=judge_tag,
        dashscope_region=args.dashscope_region,
    )

    run_eval(cfg)


if __name__ == "__main__":
    main()
