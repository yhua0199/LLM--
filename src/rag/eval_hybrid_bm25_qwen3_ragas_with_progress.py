# src/exp2/rag/eval_hybrid_bm25_qwen3_ragas_with_progress.py
"""
【功能】Hybrid Retrieval 评估：BM25 TopN + Qwen3 向量 TopM 合并为 Top(N+M)，再用 RAGAS 评估
- 先检索并合并 contexts（TopN + TopM，可选去重），写入缓存 jsonl
- 再计算 RAGAS 的 context_recall + context_precision（with reference）
- 检索带 tqdm 进度条；RAGAS 阶段由 ragas 自己显示进度（版本相关）
- 支持 SANITY（先跑10条/200条）通过环境变量控制
- 输出路径云端友好：experiments/<exp>/data/<RAG_OUT_SUBDIR>/

【输入】
- ES 索引：
  - BM25:  law_bm25（默认）
  - Qwen3: law_qwen3_emb（默认）
- Ground truth QA（list JSON）：
  experiments/<exp>/data/<RAG_GROUND_TRUTH>

【输出】（默认 experiments/<exp>/data/rag/）
- retrieval_hybrid_bm25_qwen3_topk{BM25}+{QWEN3}.jsonl   # 检索+合并缓存
- ragas_hybrid_bm25_qwen3_topk{BM25}+{QWEN3}.json        # RAGAS 指标结果

【运行示例】PowerShell
  $env:LLM_EXPERIMENT="exp2"
  $env:ES_URL="http://127.0.0.1:9200"
  $env:OPENAI_API_KEY="sk-..."
  $env:RAGAS_JUDGE_MODEL="gpt-4o-mini"

  # 只跑前10条 sanity
  $env:RAG_SANITY_N="10"

  # Hybrid 配置
  $env:BM25_TOPK="10"
  $env:QWEN3_TOPK="10"
  $env:ES_VECTOR_FIELD="embedding"   # 你的 law_qwen3_emb mapping 里 dense_vector 字段名是 embedding
  $env:RAG_HYBRID_DEDUP="1"          # 合并去重（默认1）

  # Embedding 设备
  $env:EMB_DEVICE="cuda"             # auto/cpu/cuda/cuda:0

  python -m src.exp2.rag.eval_hybrid_bm25_qwen3_ragas_with_progress

  # 跑全量：取消 SANITY
  Remove-Item Env:RAG_SANITY_N
  python -m src.exp2.rag.eval_hybrid_bm25_qwen3_ragas_with_progress
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from datasets import Dataset
from elasticsearch import Elasticsearch

from src.common.paths import data_path, ensure_dir

# RAGAS
from ragas import evaluate
from ragas.metrics import context_recall, context_precision


# --------------------
# Config
# --------------------
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")

# I/O
RAG_OUT_SUBDIR = os.getenv("RAG_OUT_SUBDIR", "rag")
GT_QA_FILE = os.getenv("RAG_GROUND_TRUTH", "qa_testset_500.json")

# Indices
BM25_INDEX = os.getenv("ES_BM25_INDEX", "law_bm25")
QWEN3_INDEX = os.getenv("ES_QWEN3_INDEX", "law_qwen3_emb")

# TopK
BM25_TOPK = int(os.getenv("BM25_TOPK", "10"))
QWEN3_TOPK = int(os.getenv("QWEN3_TOPK", "10"))

# Hybrid behavior
HYBRID_DEDUP = os.getenv("RAG_HYBRID_DEDUP", "1") == "1"  # 合并后去重（按文本完全一致）

# Sanity (0=全量)
SANITY_N = int(os.getenv("RAG_SANITY_N", "0"))

# ES dense vector field name (Qwen3 index often uses "embedding")
ES_VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "vector")

# Embedding model config (Qwen3)
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto").strip().lower()
QWEN3_MODEL = os.getenv("QWEN3_EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "16"))
EMB_MAX_LEN = int(os.getenv("EMB_MAX_LEN", "1024"))
EMB_NORMALIZE = os.getenv("EMB_NORMALIZE", "1") == "1"

# RAGAS judge model (仅提示；真正用哪个由 ragas 内部/环境决定)
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")


# --------------------
# IO helpers
# --------------------
def read_json(path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _result_to_jsonable(results: Any) -> Dict[str, Any]:
    """
    把 ragas.evaluate 的返回值（EvaluationResult）转换成可 JSON 序列化 dict。
    兼容不同 ragas 版本：优先 to_dict，其次 dict(results)，最后提取关键指标。
    """
    if hasattr(results, "to_dict"):
        try:
            d = results.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    try:
        d = dict(results)
        if isinstance(d, dict) and d:
            out: Dict[str, Any] = {}
            for k, v in d.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    out[str(k)] = v
            return out
    except Exception:
        pass

    out = {}
    for key in ("context_recall", "context_precision"):
        try:
            out[key] = float(results[key])
        except Exception:
            pass

    return out if out else {"raw": str(results)}


# --------------------
# Load ground truth QA
# --------------------
def load_ground_truth() -> List[dict]:
    gt_path = data_path(GT_QA_FILE)
    if not gt_path.exists():
        raise FileNotFoundError(f"QA 真值文件不存在: {gt_path}")
    data = read_json(gt_path)
    if not isinstance(data, list):
        raise ValueError("QA 真值文件应为 list JSON（不是 jsonl）")
    return data


# --------------------
# Qwen3 embedder
# --------------------
def _pick_torch_device() -> str:
    if EMB_DEVICE != "auto":
        return EMB_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class Qwen3HFEmbedder:
    """
    HuggingFace Transformers embedding（用于 Qwen3-Embedding）
    - mean pooling + normalize（适配 cosine）
    """
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.device = device
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden, attention_mask, torch_mod):
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def embed(self, texts: List[str]) -> List[List[float]]:
        torch = self.torch
        out: List[List[float]] = []

        for i in range(0, len(texts), EMB_BATCH_SIZE):
            batch = texts[i:i + EMB_BATCH_SIZE]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=EMB_MAX_LEN,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}

            with torch.no_grad():
                outputs = self.model(**tok)
                last_hidden = outputs.last_hidden_state
                pooled = self._mean_pool(last_hidden, tok["attention_mask"], torch)

                if EMB_NORMALIZE:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            out.extend(pooled.detach().cpu().tolist())

        return out


# --------------------
# ES retrieval
# --------------------
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


# --------------------
# Merge
# --------------------
def merge_contexts(bm25_ctxs: List[str], qwen_ctxs: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    merged_raw = bm25_ctxs + qwen_ctxs
    info: Dict[str, Any] = {
        "bm25_count": len(bm25_ctxs),
        "qwen3_count": len(qwen_ctxs),
        "merged_raw_count": len(merged_raw),
        "dedup_enabled": HYBRID_DEDUP,
    }

    if not HYBRID_DEDUP:
        info["merged_final_count"] = len(merged_raw)
        info["dedup_removed"] = 0
        return merged_raw, info

    seen = set()
    merged: List[str] = []
    removed = 0
    for t in merged_raw:
        key = t.strip()
        if not key:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        merged.append(t)

    info["merged_final_count"] = len(merged)
    info["dedup_removed"] = removed
    return merged, info


# --------------------
# Build ragas dataset + cache
# --------------------
def build_hybrid_dataset_and_cache(
    qa_list: List[dict],
    es: Elasticsearch,
    qwen_embedder: Qwen3HFEmbedder,
) -> Dataset:
    questions: List[str] = []
    contexts_all: List[List[str]] = []
    references: List[str] = []

    cache_rows: List[Dict[str, Any]] = []

    print(
        f"\n[hybrid_eval] Hybrid(bm25+qwen3): "
        f"bm25_index={BM25_INDEX}, qwen3_index={QWEN3_INDEX}, "
        f"bm25_topk={BM25_TOPK}, qwen3_topk={QWEN3_TOPK}, "
        f"vector_field={ES_VECTOR_FIELD}, dedup={HYBRID_DEDUP}"
    )
    print("[hybrid_eval] Retrieving & merging contexts for each query ...")

    for item in tqdm(qa_list, desc="Retrieving(Hybrid bm25+qwen3)", ncols=100):
        q = (item.get("query") or item.get("question") or "").strip()
        ref = (item.get("answer") or "").strip()
        if not q:
            continue

        bm25_ctxs = bm25_search(es, BM25_INDEX, q, BM25_TOPK)
        q_vec = qwen_embedder.embed([q])[0]
        qwen_ctxs = knn_search(es, QWEN3_INDEX, q_vec, QWEN3_TOPK)

        merged_ctxs, merge_info = merge_contexts(bm25_ctxs, qwen_ctxs)

        questions.append(q)
        contexts_all.append(merged_ctxs)
        references.append(ref)

        cache_rows.append({
            "retriever": "hybrid_bm25_qwen3",
            "bm25_index": BM25_INDEX,
            "qwen3_index": QWEN3_INDEX,
            "bm25_topk": BM25_TOPK,
            "qwen3_topk": QWEN3_TOPK,
            "topk_total": BM25_TOPK + QWEN3_TOPK,
            "question": q,
            "reference": ref,
            "contexts": merged_ctxs,     # ✅ 最终 Top20（dedup 可能 < 20）
            "merge_info": merge_info,
        })

    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    cache_path = out_dir / f"retrieval_hybrid_bm25_qwen3_topk{BM25_TOPK}+{QWEN3_TOPK}.jsonl"
    write_jsonl(cache_path, cache_rows)
    print(f"[hybrid_eval] Retrieval cache saved: {cache_path}")

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts_all,
        "reference": references,
    })


# --------------------
# Main
# --------------------
def main() -> None:
    # 1) QA
    qa_list = load_ground_truth()
    print(f"[hybrid_eval] Loaded QA: {len(qa_list)}")

    if SANITY_N > 0:
        qa_list = qa_list[:SANITY_N]
        print(f"[hybrid_eval] Sanity mode: use first {len(qa_list)} samples")

    # 2) ES connect
    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[hybrid_eval] Connected ES: version={info['version']['number']}, url={ES_URL}")

    # 3) Init Qwen3 embedder
    device = _pick_torch_device()
    print("[hybrid_eval] ===== Device Check (Qwen3 embedding) =====")
    print(f"[hybrid_eval] EMB_DEVICE env={EMB_DEVICE} -> using device={device}")
    try:
        import torch
        print(f"[hybrid_eval] torch={torch.__version__}, cuda_ver={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[hybrid_eval] gpu_name={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[hybrid_eval] torch diagnostics unavailable: {e}")
    print("[hybrid_eval] ==========================================")

    qwen_embedder = Qwen3HFEmbedder(QWEN3_MODEL, device=device)

    # 4) Retrieve+Merge => build dataset + cache jsonl
    ds = build_hybrid_dataset_and_cache(qa_list, es, qwen_embedder)

    # 5) RAGAS eval
    print("\n[hybrid_eval] Running RAGAS metrics ...")
    print(f"[hybrid_eval] Judge model suggestion: {RAGAS_JUDGE_MODEL} (set via env RAGAS_JUDGE_MODEL)")
    metrics = [context_recall, context_precision]
    results = evaluate(ds, metrics=metrics)

    results_json = _result_to_jsonable(results)

    # 6) Save results
    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    out_path = out_dir / f"ragas_hybrid_bm25_qwen3_topk{BM25_TOPK}+{QWEN3_TOPK}.json"
    write_json(out_path, results_json)

    print(f"\n[hybrid_eval] DONE ✅ saved: {out_path}")
    print(json.dumps(results_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
