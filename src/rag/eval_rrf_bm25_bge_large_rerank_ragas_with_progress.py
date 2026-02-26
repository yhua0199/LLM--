# src/exp2/rag/eval_rrf_bm25_bge_large_rerank_ragas_with_progress.py
"""
【功能】RRF Hybrid Retrieval + Cross-Encoder Rerank 评估：
BM25 TopN + BGE-large TopM -> RRF 融合排序 (TopK_rrf) -> Reranker 二次排序 -> TopK_final
再用 RAGAS 评估 context_recall + context_precision（with reference）

与 eval_rrf_bm25_bge_large_ragas_with_progress.py 的区别：
- 增加 reranker（bge-reranker-base / bge-reranker-large-zh），由环境变量控制
- retrieval 不保存中间结果（直接流入 rerank）
- 最终输出两个文件：
  1) 重排后 contexts 结果 jsonl
  2) ragas 评估结果 json

【输入】
- ES 索引：
  - BM25:      law_bm25（默认）
  - BGE-large: law_bge_large（默认）
- Ground truth QA（list JSON）：
  experiments/<exp>/data/<RAG_GROUND_TRUTH>

【输出】（默认 experiments/<exp>/data/rag/）
- retrieval_rrf_bm25_bge_large_rerank_<RERANK_TAG>_topk{TOPK_FINAL}_bm{BM25_TOPK}_bg{BGE_TOPK}_rrf{RRF_TOPK}.jsonl
- ragas_rrf_bm25_bge_large_rerank_<RERANK_TAG>_topk{TOPK_FINAL}_bm{BM25_TOPK}_bg{BGE_TOPK}_rrf{RRF_TOPK}.json

【运行示例】PowerShell
  $env:LLM_EXPERIMENT="exp2"
  $env:ES_URL="http://127.0.0.1:9200"
  $env:OPENAI_API_KEY="sk-..."
  $env:RAGAS_JUDGE_MODEL="gpt-4o-mini"

  # 先跑 30 条 sanity
  $env:RAG_SANITY_N="30"

  # 召回候选 + RRF
  $env:BM25_TOPK="10"
  $env:BGE_LARGE_TOPK="10"
  $env:RRF_TOPK="20"        # RRF 后进入 rerank 的候选数（建议 >= final）
  $env:RRF_K="60"

  # rerank 最终 TopK（默认 10；可改 5）
  $env:RERANK_TOPK_FINAL="10"

  # reranker 选择（二选一）
  $env:RERANK_MODEL_ID="BAAI/bge-reranker-base"
  # $env:RERANK_MODEL_ID="BAAI/bge-reranker-large-zh"

  # 可选去重
  $env:RAG_HYBRID_DEDUP="1"

  # 索引与向量字段
  $env:ES_BM25_INDEX="law_bm25"
  $env:ES_BGE_LARGE_INDEX="law_bge_large"
  $env:ES_VECTOR_FIELD="vector"

  # Embedding 设备（BGE-large query embedding）+ Reranker 设备
  $env:EMB_DEVICE="cuda"
  $env:RERANK_DEVICE="cuda"     # auto/cpu/cuda/cuda:0

  python -m src.exp2.rag.eval_rrf_bm25_bge_large_rerank_ragas_with_progress

  # 跑全量：取消 SANITY
  Remove-Item Env:RAG_SANITY_N
  python -m src.exp2.rag.eval_rrf_bm25_bge_large_rerank_ragas_with_progress
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

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
BGE_LARGE_INDEX = os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large")

# Candidate TopK
BM25_TOPK = int(os.getenv("BM25_TOPK", "10"))
BGE_LARGE_TOPK = int(os.getenv("BGE_LARGE_TOPK", "10"))

# RRF stage
RRF_TOPK = int(os.getenv("RRF_TOPK", "20"))          # RRF 后进入 rerank 的候选数
RRF_K = int(os.getenv("RRF_K", "60"))                # 常用 60

# Rerank final TopK
RERANK_TOPK_FINAL = int(os.getenv("RERANK_TOPK_FINAL", "10"))  # 默认10，可改5

# Hybrid behavior
HYBRID_DEDUP = os.getenv("RAG_HYBRID_DEDUP", "1") == "1"

# Sanity
SANITY_N = int(os.getenv("RAG_SANITY_N", "0"))

# ES dense vector field name
ES_VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "vector")

# RAGAS judge model (仅提示；实际由 ragas/openai 环境决定)
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")


# --------------------
# Embedding (BGE-large via SentenceTransformer)
# --------------------
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto").strip().lower()
BGE_LARGE_MODEL = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-large-zh-v1.5")
EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "16"))
EMB_NORMALIZE = os.getenv("EMB_NORMALIZE", "1") == "1"


def _pick_torch_device(device_env: str) -> str:
    if device_env != "auto":
        return device_env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class STEmbedder:
    """SentenceTransformer embedding（用于 BGE-large query embedding）"""
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


# --------------------
# Reranker (Cross-Encoder)
# --------------------
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "BAAI/bge-reranker-base")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto").strip().lower()
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))


def _safe_tag(model_id: str) -> str:
    """
    将模型名转成文件名友好 tag：
    BAAI/bge-reranker-large-zh -> bge-reranker-large-zh
    """
    name = model_id.split("/")[-1].strip()
    name = re.sub(r"[^a-zA-Z0-9._\-]+", "_", name)
    return name or "reranker"


class CrossEncoderReranker:
    """
    使用 sentence-transformers CrossEncoder 进行重排
    注意：FlagEmbedding 的 bge-reranker 系列可直接用 CrossEncoder 加载
    """
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, ctxs: List[str]) -> List[Tuple[str, float]]:
        if not ctxs:
            return []
        pairs = [(query, c) for c in ctxs]
        scores = self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
        # scores: np.ndarray / list
        scored = list(zip(ctxs, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


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
    """EvaluationResult -> JSON dict（解决不可序列化问题）"""
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
# RRF Fusion
# --------------------
def rrf_fuse(
    bm25_ctxs: List[str],
    dense_ctxs: List[str],
    k_final: int,
    rrf_k: int,
    dedup: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    RRF: score(d) = Σ 1 / (rrf_k + rank_d_in_list)
    rank 从 1 开始
    """
    scores: Dict[str, float] = {}
    ranks: Dict[str, Dict[str, int]] = {}

    def add_list(ctxs: List[str], source: str):
        for i, t in enumerate(ctxs, start=1):
            key = t.strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + i)
            ranks.setdefault(key, {})[source] = i

    add_list(bm25_ctxs, "bm25")
    add_list(dense_ctxs, "bge_large")

    items = list(scores.items())
    items.sort(key=lambda kv: kv[1], reverse=True)

    fused: List[str] = []
    removed = 0
    seen = set()

    for key, _score in items:
        if dedup:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        fused.append(key)
        if len(fused) >= k_final:
            break

    info = {
        "bm25_count": len(bm25_ctxs),
        "bge_large_count": len(dense_ctxs),
        "rrf_k": rrf_k,
        "k_final": k_final,
        "dedup_enabled": dedup,
        "dedup_removed": removed,
        "fused_count": len(fused),
        "top_debug": [
            {"text": fused[i], "ranks": ranks.get(fused[i], {}), "score": scores.get(fused[i], 0.0)}
            for i in range(min(5, len(fused)))
        ],
    }
    return fused, info


# --------------------
# Build dataset (retrieval -> RRF -> rerank) + save final cache
# --------------------
def build_rrf_rerank_dataset_and_save(
    qa_list: List[dict],
    es: Elasticsearch,
    embedder: STEmbedder,
    reranker: CrossEncoderReranker,
    rerank_tag: str,
) -> Dataset:
    questions: List[str] = []
    contexts_all: List[List[str]] = []
    references: List[str] = []
    cache_rows: List[Dict[str, Any]] = []

    print(
        f"\n[rrf_rerank_eval] RRF(bm25+bge_large) + Rerank({rerank_tag}):\n"
        f"  bm25_index={BM25_INDEX}, bge_large_index={BGE_LARGE_INDEX}\n"
        f"  bm25_topk={BM25_TOPK}, bge_large_topk={BGE_LARGE_TOPK}\n"
        f"  rrf_k={RRF_K}, rrf_topk={RRF_TOPK}\n"
        f"  rerank_topk_final={RERANK_TOPK_FINAL}\n"
        f"  vector_field={ES_VECTOR_FIELD}, dedup={HYBRID_DEDUP}\n"
        f"  embed_model={BGE_LARGE_MODEL}, rerank_model={RERANK_MODEL_ID}"
    )
    print("[rrf_rerank_eval] Retrieving candidates -> RRF -> Rerank ...")

    for item in tqdm(qa_list, desc=f"Retrieving+Rerank({rerank_tag})", ncols=100):
        q = (item.get("query") or item.get("question") or "").strip()
        ref = (item.get("answer") or "").strip()
        if not q:
            continue

        # 1) retrieve
        bm25_ctxs = bm25_search(es, BM25_INDEX, q, BM25_TOPK)
        q_vec = embedder.embed([q])[0]
        bge_ctxs = knn_search(es, BGE_LARGE_INDEX, q_vec, BGE_LARGE_TOPK)

        # 2) RRF fuse -> candidate list for rerank
        fused_ctxs, fuse_info = rrf_fuse(
            bm25_ctxs=bm25_ctxs,
            dense_ctxs=bge_ctxs,
            k_final=RRF_TOPK,
            rrf_k=RRF_K,
            dedup=HYBRID_DEDUP,
        )

        # 3) rerank -> final topk
        scored = reranker.rerank(q, fused_ctxs)
        top_ctxs = [t for t, _s in scored[:RERANK_TOPK_FINAL]]

        questions.append(q)
        contexts_all.append(top_ctxs)
        references.append(ref)

        cache_rows.append({
            "retriever": "rrf_bm25_bge_large",
            "reranker": rerank_tag,
            "rerank_model_id": RERANK_MODEL_ID,
            "bm25_index": BM25_INDEX,
            "bge_large_index": BGE_LARGE_INDEX,
            "bm25_topk": BM25_TOPK,
            "bge_large_topk": BGE_LARGE_TOPK,
            "rrf_k": RRF_K,
            "rrf_topk": RRF_TOPK,
            "final_topk": RERANK_TOPK_FINAL,
            "question": q,
            "reference": ref,
            "contexts": top_ctxs,
            "fuse_info": fuse_info,
            "rerank_top_debug": [
                {"text": t, "score": s}
                for t, s in scored[: min(5, len(scored))]
            ],
        })

    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    cache_path = out_dir / (
        f"retrieval_rrf_bm25_bge_large_rerank_{rerank_tag}_topk{RERANK_TOPK_FINAL}"
        f"_bm{BM25_TOPK}_bg{BGE_LARGE_TOPK}_rrf{RRF_TOPK}.jsonl"
    )
    write_jsonl(cache_path, cache_rows)
    print(f"[rrf_rerank_eval] Reranked cache saved: {cache_path}")

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts_all,
        "reference": references,
    })


# --------------------
# Main
# --------------------
def main() -> None:
    qa_list = load_ground_truth()
    print(f"[rrf_rerank_eval] Loaded QA: {len(qa_list)}")

    if SANITY_N > 0:
        qa_list = qa_list[:SANITY_N]
        print(f"[rrf_rerank_eval] Sanity mode: use first {len(qa_list)} samples")

    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[rrf_rerank_eval] Connected ES: version={info['version']['number']}, url={ES_URL}")

    # devices
    emb_device = _pick_torch_device(EMB_DEVICE)
    rerank_device = _pick_torch_device(RERANK_DEVICE)

    print("[rrf_rerank_eval] ===== Device Check =====")
    print(f"[rrf_rerank_eval] EMB_DEVICE env={EMB_DEVICE} -> using emb_device={emb_device}")
    print(f"[rrf_rerank_eval] RERANK_DEVICE env={RERANK_DEVICE} -> using rerank_device={rerank_device}")
    try:
        import torch
        print(f"[rrf_rerank_eval] torch={torch.__version__}, cuda_ver={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[rrf_rerank_eval] gpu_name={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[rrf_rerank_eval] torch diagnostics unavailable: {e}")
    print("[rrf_rerank_eval] ========================")

    # init models
    embedder = STEmbedder(BGE_LARGE_MODEL, device=emb_device)
    rerank_tag = _safe_tag(RERANK_MODEL_ID)
    reranker = CrossEncoderReranker(RERANK_MODEL_ID, device=rerank_device)

    # build dataset + save reranked cache
    ds = build_rrf_rerank_dataset_and_save(
        qa_list=qa_list,
        es=es,
        embedder=embedder,
        reranker=reranker,
        rerank_tag=rerank_tag,
    )

    # ragas
    print("\n[rrf_rerank_eval] Running RAGAS metrics ...")
    print(f"[rrf_rerank_eval] Judge model suggestion: {RAGAS_JUDGE_MODEL} (set via env RAGAS_JUDGE_MODEL)")
    results = evaluate(ds, metrics=[context_recall, context_precision])
    results_json = _result_to_jsonable(results)

    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    out_path = out_dir / (
        f"ragas_rrf_bm25_bge_large_rerank_{rerank_tag}_topk{RERANK_TOPK_FINAL}"
        f"_bm{BM25_TOPK}_bg{BGE_LARGE_TOPK}_rrf{RRF_TOPK}.json"
    )
    write_json(out_path, results_json)

    print(f"\n[rrf_rerank_eval] DONE ✅ saved: {out_path}")
    print(json.dumps(results_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
