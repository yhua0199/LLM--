# src/exp2/es/index_bm25.py
# -*- coding: utf-8 -*-
"""
【功能】构建 ES BM25（稀疏检索）索引
输入：experiments/<exp>/data/rag/chunks.jsonl
输出：ES index（law_bm25）

运行（Linux / DSW）：
  export LLM_EXPERIMENT=exp2
  export ES_URL=http://127.0.0.1:9200
  export ES_BM25_INDEX=law_bm25
  export ES_RECREATE=1   # 首次建议 1，后续增量可设 0
  python -m src.exp2.es.index_bm25

可选配置：
  export RAG_CHUNKS_FILE="rag/chunks.jsonl"   # 相对 experiments/<exp>/data 的路径
  export ES_BULK_SIZE="500"
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】先确保 LLM_EXPERIMENT 在 import paths 前已就位
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", os.getenv("LLM_EXPERIMENT", "exp2"))

import json
from typing import Any, Dict, Iterable, List

from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

from src.common.paths import data_path

# =========================================================
# Config
# =========================================================
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200").strip()

CHUNKS_FILE = os.getenv("RAG_CHUNKS_FILE", "rag/chunks.jsonl").strip()

BM25_INDEX = os.getenv("ES_BM25_INDEX", os.getenv("ES_BM25_INDEX", "law_bm25")).strip()
# 兼容你有时写的 ES_RECREATE / ES_BM25_RECREATE
RECREATE = (os.getenv("ES_RECREATE", "0") == "1") or (os.getenv("ES_BM25_RECREATE", "0") == "1")

ES_BULK_SIZE = int(os.getenv("ES_BULK_SIZE", "500"))


def read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def create_index(es: Elasticsearch, index_name: str) -> None:
    """
    创建 BM25 索引 mapping
    - text: 用于 match 查询
    - 其余字段用于追溯/展示
    """
    if es.indices.exists(index=index_name):
        if RECREATE:
            es.indices.delete(index=index_name)
            print(f"[index_bm25] deleted existing index={index_name}")
        else:
            print(f"[index_bm25] index already exists, skip create: {index_name}")
            return

    mapping = {
        "settings": {
            # 可以按需扩展 analyzer；默认 standard 也能跑
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "law_group": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "source": {"type": "text"},
                "text": {"type": "text"},        # BM25 检索主字段
                "text_len": {"type": "integer"},
            }
        },
    }
    es.indices.create(index=index_name, **mapping)
    print(f"[index_bm25] created index={index_name}")


def bulk_actions(index_name: str, rows: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for r in rows:
        chunk_id = r.get("chunk_id")
        if not chunk_id:
            continue
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": chunk_id,     # 用 chunk_id 做主键，重复跑会覆盖
            "_source": r,
        }


def main() -> None:
    print(f"[index_bm25] LLM_EXPERIMENT={os.environ.get('LLM_EXPERIMENT')}")
    print(f"[index_bm25] ES_URL={ES_URL}")
    print(f"[index_bm25] BM25_INDEX={BM25_INDEX}")
    print(f"[index_bm25] CHUNKS_FILE={CHUNKS_FILE}")
    print(f"[index_bm25] ES_BULK_SIZE={ES_BULK_SIZE}")
    print(f"[index_bm25] RECREATE={RECREATE}")

    # 1) connect ES
    es = Elasticsearch(ES_URL)
    info = es.info()
    print(f"[index_bm25] connected, es_version={info['version']['number']}")

    # 2) create mapping
    create_index(es, BM25_INDEX)

    # 3) load chunks
    chunks_path = data_path(CHUNKS_FILE)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks 文件不存在：{chunks_path}")

    rows_raw = list(read_jsonl(chunks_path))
    print(f"[index_bm25] loaded chunks: {len(rows_raw)}")

    # 4) prepare docs (只保留需要的字段，避免写入冗余)
    rows: List[Dict[str, Any]] = []
    for r in rows_raw:
        rows.append({
            "chunk_id": r.get("chunk_id", ""),
            "doc_id": r.get("doc_id", ""),
            "title": r.get("title", ""),
            "law_group": r.get("law_group", ""),
            "rel_path": r.get("rel_path", ""),
            "source": r.get("source", ""),
            "text": r.get("text", ""),
            "text_len": int(r.get("text_len", len(r.get("text", "") or ""))),
        })

    # 5) bulk index with progress
    indexed = 0
    failed = 0

    pbar = tqdm(total=len(rows), desc=f"Indexing(BM25:{BM25_INDEX})", ncols=100)

    buf: List[Dict[str, Any]] = []

    def flush():
        nonlocal indexed, failed, buf
        if not buf:
            return
        try:
            helpers.bulk(es, bulk_actions(BM25_INDEX, buf), chunk_size=min(ES_BULK_SIZE, len(buf)))
            indexed += len(buf)
        except Exception as e:
            failed += len(buf)
            print(f"\n[index_bm25] bulk failed: {e}")
        buf = []

    for d in rows:
        # text 为空的跳过
        if not (d.get("text") or "").strip():
            pbar.update(1)
            continue

        buf.append(d)
        if len(buf) >= ES_BULK_SIZE:
            flush()
        pbar.update(1)

    flush()
    pbar.close()

    es.indices.refresh(index=BM25_INDEX)
    print(f"\n[index_bm25] DONE ✅ indexed={indexed}, failed={failed}, index={BM25_INDEX}")


if __name__ == "__main__":
    main()
