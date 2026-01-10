# src/exp2/es/index_bge.py
"""
【功能】用 BGE / BGE-large 构建 ES 向量索引（dense_vector）
输入：experiments/<exp>/data/rag/chunks.jsonl
输出：ES index（law_bge / law_bge_large）

运行：
  $env:LLM_EXPERIMENT="exp2"
  $env:ES_URL="http://127.0.0.1:9200"
  python -m src.exp2.es.index_bge

可选配置：
  $env:RAG_CHUNKS_FILE="rag/chunks.jsonl"     # 相对 experiments/<exp>/data 的路径
  $env:ES_BGE_INDEX="law_bge"
  $env:ES_BGE_LARGE_INDEX="law_bge_large"
  $env:BGE_MODEL="BAAI/bge-base-zh-v1.5"
  $env:BGE_LARGE_MODEL="BAAI/bge-large-zh-v1.5"
  $env:INDEX_MODEL="bge"                     # bge / bge_large
  $env:EMB_BATCH_SIZE="32"
  $env:ES_BULK_SIZE="500"
  $env:ES_RECREATE="0"                       # 1=删除并重建索引（谨慎）
  $env:EMB_DEVICE="auto"                     # auto / cpu / cuda / cuda:0
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List

from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from src.common.paths import data_path

# ---------- Config ----------
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
CHUNKS_FILE = os.getenv("RAG_CHUNKS_FILE", "rag/chunks.jsonl")

BGE_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-base-zh-v1.5")
BGE_LARGE_MODEL = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-large-zh-v1.5")

ES_BGE_INDEX = os.getenv("ES_BGE_INDEX", "law_bge")
ES_BGE_LARGE_INDEX = os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large")

# 选择要建哪个索引：bge / bge_large
INDEX_MODEL = os.getenv("INDEX_MODEL", "bge").strip().lower()

EMB_BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "32"))
ES_BULK_SIZE = int(os.getenv("ES_BULK_SIZE", "500"))
RECREATE = os.getenv("ES_RECREATE", "0") == "1"

# 设备选择：auto / cpu / cuda / cuda:0
EMB_DEVICE = os.getenv("EMB_DEVICE", "auto").strip().lower()


def read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_model_and_index() -> tuple[str, str, int]:
    """返回：(model_name, index_name, dims)"""
    if INDEX_MODEL in ("bge_large", "bge-large", "large"):
        return BGE_LARGE_MODEL, ES_BGE_LARGE_INDEX, 1024
    return BGE_MODEL, ES_BGE_INDEX, 768


def create_index(es: Elasticsearch, index_name: str, dims: int) -> None:
    """创建向量索引 mapping（dense_vector + cosine）"""
    if es.indices.exists(index=index_name):
        if RECREATE:
            es.indices.delete(index=index_name)
            print(f"[index_bge] deleted existing index={index_name}")
        else:
            print(f"[index_bge] index 已存在，跳过创建：{index_name}")
            return

    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "law_group": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "source": {"type": "text"},
                "text": {"type": "text"},
                "text_len": {"type": "integer"},
                # 向量字段：用于 knn
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }
    es.indices.create(index=index_name, **mapping)
    print(f"[index_bge] created index={index_name}, dims={dims}")


def bulk_actions(index_name: str, rows: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for r in rows:
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": r["chunk_id"],   # 用 chunk_id 做主键，方便覆盖/去重
            "_source": r,
        }


def pick_device_for_sentence_transformers() -> str:
    """
    决定 embedding 用什么设备：
    - EMB_DEVICE=cpu/cuda/cuda:0 直接用
    - EMB_DEVICE=auto：如果 torch.cuda.is_available() True 就 cuda，否则 cpu
    """
    if EMB_DEVICE != "auto":
        return EMB_DEVICE

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        # 没装 torch 或 torch 异常时，退回 cpu
        return "cpu"


def print_device_diagnostics(device: str) -> None:
    """打印当前环境的 CUDA/GPU 信息，方便你确认是不是 GPU 在跑"""
    print("[index_bge] ===== Device Check =====")
    print(f"[index_bge] EMB_DEVICE env = {EMB_DEVICE}")
    print(f"[index_bge] Using device   = {device}")

    try:
        import torch
        print(f"[index_bge] torch.__version__      = {torch.__version__}")
        print(f"[index_bge] torch.version.cuda    = {torch.version.cuda}")
        print(f"[index_bge] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[index_bge] torch.cuda.device_count() = {torch.cuda.device_count()}")
            print(f"[index_bge] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[index_bge] torch diagnostics unavailable: {e}")

    print("[index_bge] ========================")


def main() -> None:
    # 1) 选择模型和索引
    model_name, index_name, dims = pick_model_and_index()
    print(f"[index_bge] ES_URL={ES_URL}")
    print(f"[index_bge] INDEX_MODEL={INDEX_MODEL}")
    print(f"[index_bge] model={model_name}")
    print(f"[index_bge] index={index_name}, dims={dims}")

    # 2) 连接 ES
    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[index_bge] connected, es_version={info['version']['number']}")

    # 3) 创建 index mapping
    create_index(es, index_name, dims)

    # 4) 确定设备 + 加载 embedding 模型
    device = pick_device_for_sentence_transformers()
    print_device_diagnostics(device)

    print("[index_bge] loading embedding model ...")
    # SentenceTransformer 支持 device 参数（强制在指定设备运行）
    emb = SentenceTransformer(model_name, device=device)

    # 5) 读取 chunks.jsonl
    chunks_path = data_path(CHUNKS_FILE)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks 文件不存在：{chunks_path}")

    rows = list(read_jsonl(chunks_path))
    print(f"[index_bge] loaded chunks: {len(rows)}")

    # 6) 分批 embedding + bulk 写入 ES
    buffer_docs: List[Dict[str, Any]] = []
    texts_buffer: List[str] = []

    indexed = 0
    failed = 0

    pbar = tqdm(total=len(rows), desc=f"Indexing({INDEX_MODEL})", ncols=100)

    def flush():
        nonlocal indexed, failed, buffer_docs, texts_buffer
        if not buffer_docs:
            return

        vectors = emb.encode(
            texts_buffer,
            batch_size=EMB_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine 检索推荐 normalize
        )

        for d, v in zip(buffer_docs, vectors):
            d["vector"] = v.tolist()

        try:
            helpers.bulk(es, bulk_actions(index_name, buffer_docs), chunk_size=ES_BULK_SIZE)
            indexed += len(buffer_docs)
        except Exception as e:
            failed += len(buffer_docs)
            print(f"\n[index_bge] bulk failed: {e}")

        buffer_docs = []
        texts_buffer = []

    for r in rows:
        doc = {
            "chunk_id": r["chunk_id"],
            "doc_id": r.get("doc_id", ""),
            "title": r.get("title", ""),
            "law_group": r.get("law_group", ""),
            "rel_path": r.get("rel_path", ""),
            "source": r.get("source", ""),
            "text": r.get("text", ""),
            "text_len": int(r.get("text_len", len(r.get("text", "")))),
        }

        buffer_docs.append(doc)
        texts_buffer.append(doc["text"])

        if len(buffer_docs) >= ES_BULK_SIZE:
            flush()

        pbar.update(1)

    flush()
    pbar.close()

    es.indices.refresh(index=index_name)
    print(f"\n[index_bge] DONE ✅ indexed={indexed}, failed={failed}, index={index_name}")


if __name__ == "__main__":
    main()
