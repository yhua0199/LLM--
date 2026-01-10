# src/exp2/rag/infer_rag_rewrite_qwen.py
# -*- coding: utf-8 -*-
"""
实验2：RAG + rewrite + Qwen 推理脚本（生成最终回答，不做RAGAS评估、不保存中间RAG结果）

与 infer_rag_qwen.py 的区别：
- 输入：qa_50_with_rewrite.json（每条包含 query/answer/rewrite）
- 用 rewrite 作为“检索query + LLM提问”（若 rewrite 为空则回退到 query）
- 输出：experiments/exp2/results/rag/
  字段：query / rewrite / answer / LLM_answer

运行方式（项目根目录）：
  $env:LLM_EXPERIMENT="exp2"

  # ES
  $env:ES_URL="http://127.0.0.1:9200"
  $env:ES_BM25_INDEX="law_bm25"
  $env:ES_BGE_LARGE_INDEX="law_bge_large"
  $env:ES_VECTOR_FIELD="vector"

  # RAG 参数
  $env:BM25_TOPK="10"
  $env:BGE_LARGE_TOPK="10"
  $env:RRF_TOPK="20"
  $env:RRF_K="60"
  $env:RAG_CTX_TOPK="10"

  # Embedding
  $env:BGE_LARGE_MODEL="BAAI/bge-large-zh-v1.5"
  $env:EMB_DEVICE="cuda"

  # LLM（3B/7B 切换）
  $env:QWEN_SIZE="3b"   # or 7b
  $env:LLM_DEVICE="cuda"
  $env:LLM_MAX_NEW_TOKENS="256"
  $env:LLM_TEMPERATURE="0.2"
  $env:LLM_TOP_P="0.8"

  python -m src.exp2.rag.infer_rag_rewrite_qwen
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】先设置实验选择（因为 paths.py 在 import 时会读取环境变量）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp2")

from src.common.paths import data_path, prompt_path, results_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 配置前置区：输入/输出/ES/RAG/LLM/日志 全部放这里
# =========================================================
import argparse
import json
import re
import time
from typing import Any, Dict, List

from elasticsearch import Elasticsearch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# 输入数据配置
# -------------------------
DEFAULT_DATA_FILE = os.getenv("RAG_QA_FILE", "qa_50_with_rewrite.json")  # -> experiments/exp2/data/

# -------------------------
# 输出配置
# -------------------------
OUT_SUBDIR = os.getenv("RAG_OUT_RESULTS_SUBDIR", "rag")  # -> experiments/exp2/results/rag/

# -------------------------
# ES 配置
# -------------------------
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
BM25_INDEX = os.getenv("ES_BM25_INDEX", "law_bm25")
BGE_LARGE_INDEX = os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large")
ES_VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "vector")

# -------------------------
# RAG 参数
# -------------------------
BM25_TOPK = int(os.getenv("BM25_TOPK", "10"))
BGE_LARGE_TOPK = int(os.getenv("BGE_LARGE_TOPK", "10"))
RRF_TOPK = int(os.getenv("RRF_TOPK", "20"))        # RRF 融合后候选数
RRF_K = int(os.getenv("RRF_K", "60"))              # 常用 60
RAG_CTX_TOPK = int(os.getenv("RAG_CTX_TOPK", "10"))  # 最终喂给LLM的context条数
HYBRID_DEDUP = os.getenv("RAG_HYBRID_DEDUP", "1") == "1"

# Sanity
SANITY_N = int(os.getenv("RAG_SANITY_N", "0"))

# -------------------------
# Embedding 配置（query embedding）
# -------------------------
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto").strip().lower()
BGE_LARGE_MODEL = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-large-zh-v1.5")
EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "16"))
EMB_NORMALIZE = os.getenv("EMB_NORMALIZE", "1") == "1"

# -------------------------
# LLM 配置：Qwen2.5 3B / 7B
# -------------------------
QWEN_SIZE = os.getenv("QWEN_SIZE", "3b").strip().lower()   # 3b / 7b
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto").strip().lower()

QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID",
    "Qwen/Qwen2.5-3B-Instruct" if QWEN_SIZE == "3b" else "Qwen/Qwen2.5-7B-Instruct"
)

LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.8"))
LLM_DO_SAMPLE = os.getenv("LLM_DO_SAMPLE", "1") == "1"

# -------------------------
# Prompt
# -------------------------
RAG_PROMPT_FILE = os.getenv("RAG_PROMPT_FILE", "rag_qa.txt")

DEFAULT_RAG_PROMPT = """你是一名法律咨询助手。

要求：
1) 仅基于给定的“参考资料（contexts）”与常识性的法律原则进行解释，不编造不存在的法条细节。
2) 不输出违法/危险行为的具体操作步骤。
3) 若参考资料不足以支持结论，请明确说明“信息不足”，并给出需要补充的信息点。
4) 输出简洁清晰，结构建议：结论/原则 + 依据（引用资料要点） + 建议。

【用户问题】
{query}

【参考资料（contexts）】
{contexts}

请给出回答：
"""

# -------------------------
# 日志/进度
# -------------------------
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
PREVIEW_EVERY = int(os.getenv("PREVIEW_EVERY", "20"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "220"))


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


def _safe_tag(s: str) -> str:
    s = s.split("/")[-1]
    s = re.sub(r"[^a-zA-Z0-9._\-]+", "_", s)
    return s or "tag"


def load_prompt_text() -> str:
    p = prompt_path(RAG_PROMPT_FILE)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if "{query}" not in txt or "{contexts}" not in txt:
            raise ValueError(f"RAG Prompt 必须包含 {{query}} 和 {{contexts}} 占位符：{p}")
        return txt
    return DEFAULT_RAG_PROMPT


def load_testset_with_rewrite(qa_file: str) -> List[Dict[str, str]]:
    """
    读取 qa_file（JSON数组）
    - query: 原始问题（落盘对照）
    - rewrite: 改写问题（用于检索&提问；为空则回退 query）
    - answer: 参考答案（仅落盘，不喂给LLM）
    """
    p = data_path(qa_file)
    if not p.exists():
        raise FileNotFoundError(f"未找到测试集文件：{p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{qa_file} 必须是 JSON 数组（list）。")

    items: List[Dict[str, str]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = (obj.get("query", "") or "").strip()
        if not q:
            continue
        r = (obj.get("rewrite", "") or "").strip()
        a = (obj.get("answer", "") or "").strip()
        items.append({"query": q, "rewrite": r, "answer": a})

    if not items:
        raise ValueError("测试集中没有找到有效的 'query' 字段。")
    return items


# -------------------------
# Embedding
# -------------------------
class STEmbedder:
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=EMB_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=EMB_NORMALIZE,
        )
        return [v.tolist() for v in vecs]


# -------------------------
# ES retrieval
# -------------------------
def bm25_search(es: Elasticsearch, index: str, query: str, k: int) -> List[str]:
    body = {"size": k, "query": {"match": {"text": query}}}
    resp = es.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    return [h["_source"]["text"] for h in hits]


def knn_search(es: Elasticsearch, index: str, query_vec: List[float], k: int) -> List[str]:
    body = {
        "knn": {
            "field": ES_VECTOR_FIELD,
            "query_vector": query_vec,
            "k": k,
            "num_candidates": max(50, k * 10),
        },
        "_source": ["text"],
    }
    resp = es.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    return [h["_source"]["text"] for h in hits]


# -------------------------
# RRF fusion
# -------------------------
def rrf_fuse(
    bm25_ctxs: List[str],
    dense_ctxs: List[str],
    k_final: int,
    rrf_k: int,
    dedup: bool,
) -> List[str]:
    scores: Dict[str, float] = {}

    def add_list(ctxs: List[str]):
        for i, t in enumerate(ctxs, start=1):
            key = (t or "").strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + i)

    add_list(bm25_ctxs)
    add_list(dense_ctxs)

    items = list(scores.items())
    items.sort(key=lambda kv: kv[1], reverse=True)

    fused: List[str] = []
    seen = set()
    for key, _ in items:
        if dedup and key in seen:
            continue
        seen.add(key)
        fused.append(key)
        if len(fused) >= k_final:
            break
    return fused


def format_contexts(ctxs: List[str]) -> str:
    lines = []
    for i, c in enumerate(ctxs, start=1):
        c = (c or "").strip()
        if c:
            lines.append(f"{i}. {c}")
    return "\n".join(lines) if lines else "(无)"


def build_llm_input(tokenizer, prompt_tpl: str, question: str, ctxs: List[str]) -> str:
    """
    question: 这里传 rewrite（或回退 query）
    """
    contexts_text = format_contexts(ctxs)
    user_text = prompt_tpl.format(query=question, contexts=contexts_text)

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
def generate_one(model, tokenizer, input_text: str) -> str:
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=LLM_DO_SAMPLE,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


# =========================================================
# 3) 主流程
# =========================================================
def run_rag_rewrite_qwen(qa_file: str) -> str:
    items = load_testset_with_rewrite(qa_file)
    prompt_tpl = load_prompt_text()

    if SANITY_N > 0:
        items = items[:SANITY_N]

    out_dir = ensure_dir(results_path() / OUT_SUBDIR)

    model_tag = _safe_tag(QWEN_MODEL_ID)
    rag_tag = (
        f"rag_rewrite_{model_tag}"
        f"_bm{BM25_TOPK}_bg{BGE_LARGE_TOPK}_rrf{RRF_TOPK}_ctx{RAG_CTX_TOPK}"
    )
    out_file = out_dir / f"answer_{rag_tag}.json"

    emb_device = _pick_torch_device(EMB_DEVICE)
    llm_device = _pick_torch_device(LLM_DEVICE)

    print("=" * 90)
    print("[INFO] 实验：Exp2 RAG + rewrite + Qwen 推理（不评估，不保存中间RAG）")
    print(f"[INFO] LLM_EXPERIMENT = {os.environ.get('LLM_EXPERIMENT')}")
    print(f"[INFO] 数据集：{data_path(qa_file)} | 样本数：{len(items)}")
    pfile = prompt_path(RAG_PROMPT_FILE)
    print(f"[INFO] RAG Prompt：{pfile if pfile.exists() else 'DEFAULT_RAG_PROMPT'}")
    print(f"[INFO] ES_URL={ES_URL}")
    print(f"[INFO] Indices: bm25={BM25_INDEX}, dense={BGE_LARGE_INDEX}, vector_field={ES_VECTOR_FIELD}")
    print(f"[INFO] RAG: BM25_TOPK={BM25_TOPK}, BGE_TOPK={BGE_LARGE_TOPK}, RRF_TOPK={RRF_TOPK}, RRF_K={RRF_K}, CTX_TOPK={RAG_CTX_TOPK}, dedup={HYBRID_DEDUP}")
    print(f"[INFO] Embedding: model={BGE_LARGE_MODEL}, device={emb_device}, normalize={EMB_NORMALIZE}")
    print(f"[INFO] LLM: model={QWEN_MODEL_ID}, device={llm_device}")
    print(f"[INFO] GEN: max_new_tokens={LLM_MAX_NEW_TOKENS}, temp={LLM_TEMPERATURE}, top_p={LLM_TOP_P}, do_sample={LLM_DO_SAMPLE}")
    print(f"[INFO] 输出文件：{out_file}")
    print("=" * 90)

    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[INFO] Connected ES: version={info['version']['number']}, url={ES_URL}")

    embedder = STEmbedder(BGE_LARGE_MODEL, device=emb_device)

    torch_dtype = torch.float16 if llm_device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if llm_device.startswith("cuda") else None,
    )
    model.eval()

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for idx, obj in enumerate(items, start=1):
        query = obj["query"]
        rewrite = obj.get("rewrite", "").strip()
        answer = obj.get("answer", "")

        # 用 rewrite 作为检索/提问；为空则回退 query
        question = rewrite if rewrite else query

        bm25_ctxs = bm25_search(es, BM25_INDEX, question, BM25_TOPK)
        q_vec = embedder.embed([question])[0]
        bge_ctxs = knn_search(es, BGE_LARGE_INDEX, q_vec, BGE_LARGE_TOPK)

        fused = rrf_fuse(
            bm25_ctxs=bm25_ctxs,
            dense_ctxs=bge_ctxs,
            k_final=RRF_TOPK,
            rrf_k=RRF_K,
            dedup=HYBRID_DEDUP,
        )
        ctxs_for_llm = fused[:RAG_CTX_TOPK]

        input_text = build_llm_input(tokenizer, prompt_tpl, question, ctxs_for_llm)
        llm_answer = generate_one(model, tokenizer, input_text)

        results.append({
            "query": query,
            "rewrite": rewrite,
            "answer": answer,      # 仅用于结果对照，绝不喂给LLM
            "LLM_answer": llm_answer,
        })

        if idx % LOG_EVERY == 0 or idx == len(items):
            elapsed = time.time() - start_time
            speed = idx / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {idx}/{len(items)} | {speed:.2f} 条/秒 | elapsed {elapsed:.1f}s")

        if idx == 1 or idx % PREVIEW_EVERY == 0:
            pq = question[:PREVIEW_CHARS] + ("…" if len(question) > PREVIEW_CHARS else "")
            pa = llm_answer[:PREVIEW_CHARS] + ("…" if len(llm_answer) > PREVIEW_CHARS else "")
            print("\n[示例输出]")
            print("Q(rewrite):", pq)
            print("A:", pa)
            print()

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 已保存：{out_file}")
    return str(out_file)


# =========================================================
# 4) CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Exp2 RAG + Rewrite + Qwen Inference (no RAGAS)")
    parser.add_argument(
        "--qa_file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="input QA json filename under experiments/<exp>/data/ (e.g., qa_50_with_rewrite.json)",
    )
    args = parser.parse_args()
    run_rag_rewrite_qwen(qa_file=args.qa_file)


if __name__ == "__main__":
    main()
