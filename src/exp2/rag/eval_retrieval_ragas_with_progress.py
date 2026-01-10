# src/exp2/rag/eval_retrieval_ragas_with_progress.py
"""
【功能】统一评估检索阶段（Retrieval）效果：BM25 / BGE / BGE-large / Qwen3 embedding
- 仅计算 RAGAS 的 context_recall + context_precision（with reference）
- 带检索进度条；并缓存检索结果，方便后续重复评估不再打 ES

【输入】
1) ES 索引：
   - BM25:       law_bm25
   - BGE:        law_bge
   - BGE-large:  law_bge_large
   - Qwen3 emb:  law_qwen3_emb
2) Ground truth QA（list JSON）：
   experiments/<exp>/data/<RAG_GROUND_TRUTH>

【输出】（都在 experiments/<exp>/data/<RAG_OUT_SUBDIR>/ 下）
- retrieval_<retriever>_topk{K}.jsonl          # 每条 query 的检索结果缓存
- ragas_<retriever>_topk{K}.json              # RAGAS 指标结果

【运行示例】
PowerShell:
  $env:LLM_EXPERIMENT="exp2"
  $env:ES_URL="http://127.0.0.1:9200"
  $env:RAG_RETRIEVER="bm25"       # bm25 / bge / bge_large / qwen3
  $env:RAG_EVAL_TOPK="10"
  $env:OPENAI_API_KEY="sk-..."    # 如果用到 LLM 指标（通常需要）
  $env:RAGAS_JUDGE_MODEL="gpt-4o-mini"
  python -m src.exp2.rag.eval_retrieval_ragas_with_progress

【可配置环境变量（配置前置）】
- ES_URL                    默认 http://127.0.0.1:9200
- RAG_RETRIEVER             bm25 / bge / bge_large / qwen3
- RAG_EVAL_TOPK             默认 10
- RAG_OUT_SUBDIR            默认 rag（输出目录：experiments/<exp>/data/rag）
- RAG_GROUND_TRUTH          默认 qa_testset_500.json（相对 experiments/<exp>/data）
- ES_BM25_INDEX             默认 law_bm25
- ES_BGE_INDEX              默认 law_bge
- ES_BGE_LARGE_INDEX        默认 law_bge_large
- ES_QWEN3_INDEX             默认 law_qwen3_emb

向量检索（query embedding）相关：
- EMB_DEVICE                auto / cpu / cuda / cuda:0   （默认 auto）
- BGE_MODEL                 默认 BAAI/bge-base-zh-v1.5
- BGE_LARGE_MODEL           默认 BAAI/bge-large-zh-v1.5
- QWEN3_EMB_MODEL           默认 Qwen/Qwen3-Embedding-0.6B
- EMB_BATCH_SIZE            默认 16
- EMB_MAX_LEN               默认 768
- EMB_NORMALIZE             默认 1（推荐 cosine）

RAGAS 裁判（judge）相关：
- RAGAS_JUDGE_MODEL         默认 gpt-4o-mini
"""


from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

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

RETRIEVER = os.getenv("RAG_RETRIEVER", "bm25").strip().lower()
TOP_K = int(os.getenv("RAG_EVAL_TOPK", "10"))

RAG_OUT_SUBDIR = os.getenv("RAG_OUT_SUBDIR", "rag")
GT_QA_FILE = os.getenv("RAG_GROUND_TRUTH", "qa_testset_500.json")

BM25_INDEX = os.getenv("ES_BM25_INDEX", "law_bm25")
BGE_INDEX = os.getenv("ES_BGE_INDEX", "law_bge")
BGE_LARGE_INDEX = os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large")
QWEN3_INDEX = os.getenv("ES_QWEN3_INDEX", "law_qwen3_emb")
ES_VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "vector")

# embedding model config
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto").strip().lower()
BGE_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-base-zh-v1.5")
BGE_LARGE_MODEL = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-large-zh-v1.5")
QWEN3_MODEL = os.getenv("QWEN3_EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")

EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "16"))
EMB_MAX_LEN = int(os.getenv("EMB_MAX_LEN", "1024"))
EMB_NORMALIZE = os.getenv("EMB_NORMALIZE", "1") == "1"

# RAGAS judge model (仅打印提示；真正用哪个由 ragas 内部/环境变量决定)
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")

# 可选：小样本 sanity check（不设则 0=全量）
SANITY_N = int(os.getenv("RAG_SANITY_N", "0"))


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
    把 ragas.evaluate 的返回值（EvaluationResult）转换成可 JSON 序列化的 dict。
    兼容不同 ragas 版本：优先 to_dict，其次转成 dict，最后提取常用分数。
    """
    # 1) 新版/部分版本
    if hasattr(results, "to_dict"):
        try:
            d = results.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    # 2) 有些版本结果可迭代/可 dict()
    try:
        d = dict(results)
        if isinstance(d, dict) and len(d) > 0:
            # 确保是纯 python 类型
            out: Dict[str, Any] = {}
            for k, v in d.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    out[str(k)] = v
            return out
    except Exception:
        pass

    # 3) 兜底：只取最关键的两个指标（如果能下标访问）
    out = {}
    for key in ("context_recall", "context_precision"):
        try:
            out[key] = float(results[key])
        except Exception:
            pass
    if out:
        return out

    # 4) 最终兜底：字符串化（至少能落盘，方便排查）
    return {"raw": str(results)}


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
# Embedding backends (for dense retrieval)
# --------------------
class Embedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

def _pick_torch_device() -> str:
    if EMB_DEVICE != "auto":
        return EMB_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class STEmbedder(Embedder):
    """SentenceTransformer embedding（用于 BGE/BGE-large）"""
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.device = device
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=EMB_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=EMB_NORMALIZE,
        )
        return [v.tolist() for v in vecs]

class Qwen3HFEmbedder(Embedder):
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


def build_embedder() -> Optional[Embedder]:
    device = _pick_torch_device()

    print("[eval_retrieval] ===== Device Check =====")
    print(f"[eval_retrieval] EMB_DEVICE env={EMB_DEVICE} -> using device={device}")
    try:
        import torch
        print(f"[eval_retrieval] torch={torch.__version__}, cuda_ver={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[eval_retrieval] gpu_name={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[eval_retrieval] torch diagnostics unavailable: {e}")
    print("[eval_retrieval] ========================")

    if RETRIEVER == "bm25":
        return None
    if RETRIEVER == "bge":
        return STEmbedder(BGE_MODEL, device=device)
    if RETRIEVER in ("bge_large", "bge-large", "large"):
        return STEmbedder(BGE_LARGE_MODEL, device=device)
    if RETRIEVER == "qwen3":
        return Qwen3HFEmbedder(QWEN3_MODEL, device=device)

    raise ValueError(f"未知 RAG_RETRIEVER: {RETRIEVER}")


# --------------------
# ES retrieval
# --------------------
def pick_index_name() -> str:
    if RETRIEVER == "bm25":
        return BM25_INDEX
    if RETRIEVER == "bge":
        return BGE_INDEX
    if RETRIEVER in ("bge_large", "bge-large", "large"):
        return BGE_LARGE_INDEX
    if RETRIEVER == "qwen3":
        return QWEN3_INDEX
    raise ValueError(f"未知 RAG_RETRIEVER: {RETRIEVER}")

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
# Build ragas dataset
# --------------------
def build_ragas_dataset(
    qa_list: List[dict],
    es: Elasticsearch,
    index_name: str,
    embedder: Optional[Embedder],
) -> Dataset:
    questions: List[str] = []
    contexts_all: List[List[str]] = []
    references: List[str] = []

    cache_rows: List[Dict[str, Any]] = []

    print(f"\n[eval_retrieval] Retriever={RETRIEVER}, index={index_name}, topk={TOP_K}")
    print("[eval_retrieval] Retrieving contexts for each query ...")

    for item in tqdm(qa_list, desc=f"Retrieving({RETRIEVER})", ncols=100):
        q = (item.get("query") or item.get("question") or "").strip()
        ref = (item.get("answer") or "").strip()

        if not q:
            continue

        if RETRIEVER == "bm25":
            ctxs = bm25_search(es, index_name, q, TOP_K)
        else:
            assert embedder is not None
            q_vec = embedder.embed([q])[0]
            ctxs = knn_search(es, index_name, q_vec, TOP_K)

        questions.append(q)
        contexts_all.append(ctxs)
        references.append(ref)

        cache_rows.append({
            "retriever": RETRIEVER,
            "index": index_name,
            "topk": TOP_K,
            "question": q,
            "reference": ref,
            "contexts": ctxs,
        })

    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    cache_path = out_dir / f"retrieval_{RETRIEVER}_topk{TOP_K}.jsonl"
    write_jsonl(cache_path, cache_rows)
    print(f"[eval_retrieval] Retrieval cache saved: {cache_path}")

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
    print(f"[eval_retrieval] Loaded QA: {len(qa_list)}")

    # 可选：环境变量控制小样本
    if SANITY_N > 0:
        qa_list = qa_list[:SANITY_N]
        print(f"[eval_retrieval] Sanity mode: use first {len(qa_list)} samples")

    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[eval_retrieval] Connected ES: version={info['version']['number']}, url={ES_URL}")

    index_name = pick_index_name()
    embedder = build_embedder()

    ds = build_ragas_dataset(qa_list, es, index_name, embedder)

    print("\n[eval_retrieval] Running RAGAS metrics ...")
    print(f"[eval_retrieval] Judge model suggestion: {RAGAS_JUDGE_MODEL} (set via env RAGAS_JUDGE_MODEL)")
    metrics = [context_recall, context_precision]

    results = evaluate(ds, metrics=metrics)

    # ✅ 关键：转成可 JSON 序列化结构
    results_json = _result_to_jsonable(results)

    out_dir = ensure_dir(data_path(RAG_OUT_SUBDIR))
    out_path = out_dir / f"ragas_{RETRIEVER}_topk{TOP_K}.json"
    write_json(out_path, results_json)

    print(f"\n[eval_retrieval] DONE ✅ saved: {out_path}")
    print(json.dumps(results_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
