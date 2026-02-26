# src/exp2/es/index_qwen3_embedding.py
"""
【功能】用 Qwen3 Embedding（GPU 优先）对 chunks.jsonl 生成向量并写入 Elasticsearch

【输入】
experiments/<exp>/data/rag/chunks.jsonl   (由 chunk_corpus.py 生成)

【输出】
写入 ES 索引（默认 law_qwen3_emb），每个 chunk 一条 doc，包含：
- chunk_id/doc_id/source/title/law_group/rel_path/text/text_len
- embedding: dense_vector（用于向量检索）

【运行方式】
PowerShell（在项目根目录，和 src 同级）：
  $env:LLM_EXPERIMENT="exp2"
  $env:ES_URL="http://127.0.0.1:9200"
  python -m src.exp2.es.index_qwen3_embedding

可选参数：
  $env:ES_QWEN3_INDEX="law_qwen3_emb"
  $env:ES_RECREATE="1"              # 1=删掉旧索引后重建；0=不删
  $env:QWEN3_EMB_MODEL="Qwen/Qwen3-Embedding-0.6B"   # 你也可以换成自己的模型名
  $env:EMB_BATCH_SIZE="16"          # GPU 显存小就调低（例如 8）
  $env:EMB_MAX_LEN="512"            # 每条文本截断长度
  $env:ES_BULK_SIZE="256"           # 每次 bulk 写入条数
  $env:EMB_NORMALIZE="1"            # 1=向量归一化（推荐）
"""

from __future__ import annotations

import os
import json
import time
import math
from typing import Dict, Any, Iterable, List

import requests
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from src.common.paths import data_path, ensure_dir


# =========================
# 0) 环境变量配置（不写死路径）
# =========================
ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
INDEX_NAME = os.getenv("ES_QWEN3_INDEX", "law_qwen3_emb")
RECREATE = os.getenv("ES_RECREATE", "0") == "1"

# 输入 chunks 文件位置：experiments/<exp>/data/rag/chunks.jsonl
RAG_SUBDIR = os.getenv("RAG_OUT_SUBDIR", "rag")
CHUNKS_FILE = os.getenv("RAG_CHUNKS_FILE", "chunks.jsonl")

# Qwen3 embedding 模型（可换）
MODEL_ID = os.getenv("QWEN3_EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# embedding 参数
BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "8"))
MAX_LEN = int(os.getenv("EMB_MAX_LEN", "768"))
NORMALIZE = os.getenv("EMB_NORMALIZE", "1") == "1"

# ES bulk 参数
BULK_SIZE = int(os.getenv("ES_BULK_SIZE", "256"))
TIMEOUT = int(os.getenv("ES_TIMEOUT", "60"))


# =========================
# 1) IO：读取 JSONL
# =========================
def read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# =========================
# 2) ES 工具：检查、删除、建索引、bulk 写入
# =========================
def es_get(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def es_head(url: str) -> int:
    r = requests.head(url, timeout=TIMEOUT)
    return r.status_code

def es_delete(url: str) -> Dict[str, Any]:
    r = requests.delete(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def es_put(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.put(url, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def es_post(url: str, body: Any, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    r = requests.post(url, data=body, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def es_index_exists(index: str) -> bool:
    return es_head(f"{ES_URL}/{index}") == 200

def create_index_if_needed(index: str, dim: int) -> None:
    """
    创建用于向量检索的 ES 索引：
    - text: 用于 BM25 / match
    - embedding: dense_vector 用于 knn
    """
    if es_index_exists(index):
        if RECREATE:
            print(f"[index_qwen3] ES_RECREATE=1，删除旧索引：{index}")
            es_delete(f"{ES_URL}/{index}")
        else:
            print(f"[index_qwen3] 索引已存在且不重建：{index}")
            return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            # 如果你后续要用 ES 的 knn / HNSW，可以在这里加 index 配置
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "law_group": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "source": {"type": "text"},
                "piece_idx": {"type": "integer"},
                "chunk_idx": {"type": "integer"},
                "text": {"type": "text"},
                "text_len": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim
                }
            }
        }
    }
    print(f"[index_qwen3] 创建索引：{index} (dims={dim})")
    es_put(f"{ES_URL}/{index}", mapping)

def bulk_upsert(index: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    使用 ES _bulk 写入（upsert 行为）：
    - _id 设为 chunk_id，重复写会覆盖同 id（这对“重跑”是安全的）
    """
    lines = []
    for d in docs:
        _id = d["chunk_id"]
        meta = {"index": {"_index": index, "_id": _id}}
        lines.append(json.dumps(meta, ensure_ascii=False))
        lines.append(json.dumps(d, ensure_ascii=False))
    body = "\n".join(lines) + "\n"

    headers = {"Content-Type": "application/x-ndjson"}
    res = es_post(f"{ES_URL}/_bulk", body=body.encode("utf-8"), headers=headers)
    return res


# =========================
# 3) Embedding：mean pooling + 可选归一化
# =========================
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: (B, T, H)
    attention_mask:   (B, T)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B,H)
    counts = mask.sum(dim=1).clamp(min=1e-6)                         # (B,1)
    return summed / counts

def embed_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device
) -> torch.Tensor:
    """
    返回 shape: (B, dim)
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        # 自动混精（只在 GPU 有用）
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)

        vec = mean_pooling(out.last_hidden_state, enc["attention_mask"])

        if NORMALIZE:
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)

    return vec


# =========================
# 4) 主流程
# =========================
def main():
    # ---- 4.1 输入文件检查 ----
    rag_dir = ensure_dir(data_path(RAG_SUBDIR))
    chunks_path = rag_dir / CHUNKS_FILE
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"缺少 chunks 文件：{chunks_path}\n"
            "请先运行：python -m src.exp2.rag.chunk_corpus"
        )

    # ---- 4.2 选择 device，并打印 GPU 使用信息（你要的输出）----
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print("\n[index_qwen3] ===== Device Check =====")
    print(f"[index_qwen3] torch.cuda.is_available() = {cuda_ok}")
    if cuda_ok:
        print(f"[index_qwen3] GPU = {torch.cuda.get_device_name(0)}")
        # 仅展示当前进程可见显存（刚启动通常很小）
        print(f"[index_qwen3] GPU mem allocated = {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")
    print(f"[index_qwen3] Using device = {device}")
    print("[index_qwen3] =========================\n")

    # ---- 4.3 加载模型到 device，并再次确认模型在 GPU 上 ----
    print(f"[index_qwen3] Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    model.to(device)

    # 再打印一次：确认模型参数在哪个设备上（关键）
    first_param = next(model.parameters())
    print(f"[index_qwen3] Model param device = {first_param.device}")
    if device.type == "cuda":
        print(f"[index_qwen3] GPU mem after model loaded = {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # embedding 维度（用隐藏层维度作为向量维度）
    dim = getattr(model.config, "hidden_size", None)
    if dim is None:
        # 兜底：用一次 dummy forward 推断维度
        dummy = embed_texts(model, tokenizer, ["测试"], device)
        dim = int(dummy.shape[1])
    print(f"[index_qwen3] Embedding dim = {dim}")

    # ---- 4.4 建索引（必要时删除并重建）----
    create_index_if_needed(INDEX_NAME, dim)

    # ---- 4.5 读取 chunks，批量 embedding，再 bulk 写入 ----
    chunks_iter = list(read_jsonl(str(chunks_path)))
    total = len(chunks_iter)
    print(f"[index_qwen3] chunks loaded = {total} from {chunks_path}")

    # 进度条：按 embedding batch 来
    num_batches = math.ceil(total / BATCH_SIZE)
    pbar = tqdm(total=num_batches, desc="[index_qwen3] Embedding+Indexing", ncols=100)

    bulk_buffer: List[Dict[str, Any]] = []
    indexed = 0
    failed = 0

    t0 = time.time()

    for b in range(num_batches):
        batch = chunks_iter[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
        texts = [x["text"] for x in batch]

        # 1) embedding
        vecs = embed_texts(model, tokenizer, texts, device)  # (B,dim)
        vecs = vecs.detach().cpu().tolist()

        # 2) 组装 doc
        for item, emb in zip(batch, vecs):
            doc = {
                "chunk_id": item["chunk_id"],
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "law_group": item.get("law_group", ""),
                "rel_path": item.get("rel_path", ""),
                "source": item.get("source", ""),
                "piece_idx": int(item.get("piece_idx", 0)),
                "chunk_idx": int(item.get("chunk_idx", 0)),
                "text": item["text"],
                "text_len": int(item.get("text_len", len(item["text"]))),
                "embedding": emb,
            }
            bulk_buffer.append(doc)

        # 3) bulk 写入（缓冲到 BULK_SIZE）
        if len(bulk_buffer) >= BULK_SIZE:
            try:
                res = bulk_upsert(INDEX_NAME, bulk_buffer)
                if res.get("errors"):
                    # 如果 bulk 有失败项，统计一下
                    for it in res.get("items", []):
                        status = it.get("index", {}).get("status", 200)
                        if status >= 300:
                            failed += 1
                    indexed += (len(bulk_buffer) - failed)
                else:
                    indexed += len(bulk_buffer)
            except Exception as e:
                failed += len(bulk_buffer)
                print(f"\n[index_qwen3] BULK ERROR: {e}\n")
            finally:
                bulk_buffer.clear()

        # 4) 更新进度条 & GPU 监控输出
        pbar.update(1)
        if device.type == "cuda" and (b + 1) % 20 == 0:
            mem = torch.cuda.memory_allocated(0) / 1024 / 1024
            pbar.set_postfix({"indexed": indexed, "failed": failed, "gpuMB": f"{mem:.0f}"})

    # flush 剩余 bulk
    if bulk_buffer:
        try:
            res = bulk_upsert(INDEX_NAME, bulk_buffer)
            if res.get("errors"):
                for it in res.get("items", []):
                    status = it.get("index", {}).get("status", 200)
                    if status >= 300:
                        failed += 1
                indexed += (len(bulk_buffer) - failed)
            else:
                indexed += len(bulk_buffer)
        except Exception as e:
            failed += len(bulk_buffer)
            print(f"\n[index_qwen3] BULK ERROR (final flush): {e}\n")
        finally:
            bulk_buffer.clear()

    pbar.close()

    # refresh index
    try:
        requests.post(f"{ES_URL}/{INDEX_NAME}/_refresh", timeout=TIMEOUT).raise_for_status()
    except Exception:
        pass

    t1 = time.time()
    print("\n[index_qwen3] DONE ✅")
    print(f"[index_qwen3] index = {INDEX_NAME}")
    print(f"[index_qwen3] indexed = {indexed}, failed = {failed}")
    print(f"[index_qwen3] time_sec = {t1 - t0:.2f}")

    # 最后再输出一次 GPU 占用（你可以拍图写报告）
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated(0) / 1024 / 1024
        print(f"[index_qwen3] GPU mem end = {mem:.2f} MB")


if __name__ == "__main__":
    main()
