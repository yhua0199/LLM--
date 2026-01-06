# src/exp2/rag/chunk_corpus.py
"""
【功能】RAG 第一步分块（chunking）
将 prepare_corpus.py 产出的 corpus.jsonl（每行=一个md全文文档）
处理为 chunks.jsonl（每行=一个chunk，后续用于向量化检索）。

【输入】
experiments/<exp>/data/<RAG_OUT_SUBDIR>/corpus.jsonl

【输出】
experiments/<exp>/data/<RAG_OUT_SUBDIR>/chunks.jsonl
experiments/<exp>/data/<RAG_OUT_SUBDIR>/chunks_stats.json

【分块逻辑（推荐法律文本）】
1) 先按“第X条”做条文级切分（piece），尽量避免chunk跨条
2) 再对每个piece用递归分块器按自然边界切分，控制chunk长度
3) 使用小幅 overlap 缓解边界截断（如 50 字符）

【运行方式】
PowerShell（项目根目录）：
  $env:LLM_EXPERIMENT="exp2"
  python -m src.exp2.rag.chunk_corpus

【可配置环境变量】
- LLM_EXPERIMENT            默认 exp1；这里要设为 exp2
- RAG_OUT_SUBDIR            默认 rag（输出目录：experiments/<exp>/data/rag）
- RAG_CORPUS_FILE           默认 corpus.jsonl（注意：这里只写文件名，不写路径）
- RAG_CHUNKS_FILE           默认 chunks.jsonl
- RAG_CHUNK_SIZE            默认 600（每个chunk最大字符数，字符数不是词数）
- RAG_CHUNK_OVERLAP         默认 50（相邻chunk重叠字符数）
- RAG_SPLIT_BY_ARTICLE      默认 1（先按“第X条”切分，建议开启）
- RAG_MIN_CHUNK_LEN         默认 60（过滤过短chunk，避免噪音）
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List

from src.common.paths import data_path, ensure_dir
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# 1) 配置区（全用环境变量，可跨设备）
# =========================
# 输出/输入所在子目录：experiments/<exp>/data/<RAG_DIR>/
RAG_DIR = os.getenv("RAG_OUT_SUBDIR", "rag")

# 注意：这里必须是“文件名”，不要写路径，否则会和 rag_dir 拼坏路径
CORPUS_FILE = os.getenv("RAG_CORPUS_FILE", "corpus.jsonl")
CHUNKS_FILE = os.getenv("RAG_CHUNKS_FILE", "chunks.jsonl")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

SPLIT_BY_ARTICLE = os.getenv("RAG_SPLIT_BY_ARTICLE", "1") == "1"
MIN_CHUNK_LEN = int(os.getenv("RAG_MIN_CHUNK_LEN", "60"))


# =========================
# 2) 正则：识别法条边界
# =========================
# “第X条”零宽切分：保留“第X条”在piece开头
ARTICLE_SPLIT = re.compile(r"(?=第[一二三四五六七八九十百千万零〇两]+条)")
ARTICLE_HEAD = re.compile(r"^(第[一二三四五六七八九十百千万零〇两]+条)")


# =========================
# 3) 通用工具函数
# =========================
def sha1_short(s: str, n: int = 10) -> str:
    """生成短hash，用于chunk_id，避免ID过长且能基本稳定标识内容。"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """按行读取 JSONL：每行一个 JSON 对象。适合大文件流式处理。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """按行写入 JSONL。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_text_by_article(text: str) -> List[str]:
    """
    将法律文本按“第X条”切分为多个 piece，降低 chunk 跨法条的概率。
    若全文没有出现“第X条”，则不切分，返回整体作为一个 piece。
    """
    pieces = [p.strip() for p in ARTICLE_SPLIT.split(text) if p.strip()]
    return pieces if len(pieces) > 1 else [text.strip()]


def infer_source(title: str, piece_text: str) -> str:
    """
    为 chunk 生成可追溯的来源标识（便于 debug、写报告、引用）。
    示例： "全民所有制工业企业法 第五十六条"
    """
    piece_text = piece_text.strip()
    m = ARTICLE_HEAD.match(piece_text)
    if m:
        return f"{title} {m.group(1)}" if title else m.group(1)
    return title or "unknown"


def make_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """
    构建递归分块器：
    - 优先按段落/换行切
    - 再按中文标点切（。！？；，）
    - 最后兜底按字符硬切（""）
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",  # 段落
            "\n",    # 换行
            "。", "！", "？",
            "；", "，",
            " ",
            ""       # 最终兜底：字符级切分
        ],
    )


# =========================
# 4) 主流程
# =========================
def main() -> None:
    # rag_dir: experiments/<exp>/data/<RAG_DIR>/
    rag_dir = ensure_dir(data_path(RAG_DIR))

    # 输入：prepare_corpus 产物（注意：只拼文件名）
    corpus_path = rag_dir / CORPUS_FILE

    # 输出：chunk 产物
    chunks_path = rag_dir / CHUNKS_FILE
    stats_path = rag_dir / "chunks_stats.json"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"缺少输入文件：{corpus_path}\n"
            f"请确认已生成 {CORPUS_FILE}，并先运行：python -m src.exp2.rag.prepare_corpus\n"
            f"当前实验：LLM_EXPERIMENT={os.getenv('LLM_EXPERIMENT')}"
        )

    splitter = make_recursive_splitter()

    out_rows: List[Dict[str, Any]] = []

    # 统计信息
    docs = 0
    pieces_total = 0
    chunks_total = 0
    filtered_too_short = 0

    # 逐文档处理（每行一个 doc）
    for doc in read_jsonl(corpus_path):
        docs += 1

        doc_id = doc["doc_id"]
        title = doc.get("title", "")
        rel_path = doc.get("rel_path", "")
        law_group = doc.get("law_group", "")
        text = (doc.get("text") or "").strip()

        if not text:
            continue

        # A) 先按“第X条”切（推荐）
        pieces = split_text_by_article(text) if SPLIT_BY_ARTICLE else [text]
        pieces_total += len(pieces)

        # B) 再对每个 piece 递归分块
        for piece_idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue

            source = infer_source(title, piece)
            chunk_texts = splitter.split_text(piece)

            for chunk_idx, chunk in enumerate(chunk_texts):
                chunk = chunk.strip()

                # 过滤过短 chunk
                if len(chunk) < MIN_CHUNK_LEN:
                    filtered_too_short += 1
                    continue

                content_sig = sha1_short(chunk, 10)
                chunk_id = f"{doc_id}_p{piece_idx:04d}_c{chunk_idx:03d}_{content_sig}"

                out_rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "law_group": law_group,
                    "rel_path": rel_path,
                    "source": source,
                    "piece_idx": piece_idx,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                    "text_len": len(chunk),
                })
                chunks_total += 1

        if docs % 100 == 0:
            print(f"[chunk_corpus] 已处理文档: {docs}，当前chunk数: {chunks_total}")

    # 可复现排序
    out_rows.sort(key=lambda x: (x["law_group"], x["rel_path"], x["piece_idx"], x["chunk_idx"]))

    # 写出 chunks
    write_jsonl(chunks_path, out_rows)

    # 写出统计信息
    stats = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "split_by_article": SPLIT_BY_ARTICLE,
        "min_chunk_len": MIN_CHUNK_LEN,
        "input_docs": docs,
        "article_pieces": pieces_total,
        "output_chunks": chunks_total,
        "filtered_too_short": filtered_too_short,
        "in_file": str(corpus_path).replace("\\", "/"),
        "out_file": str(chunks_path).replace("\\", "/"),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[chunk_corpus] 完成 ✅")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n[chunk_corpus] 前2条chunk预览：")
    for r in out_rows[:2]:
        snippet = r["text"][:200].replace("\n", " ")
        print(f"- {r['chunk_id']} | {r['source']} | len={r['text_len']}")
        print(f"  {snippet}...\n")


if __name__ == "__main__":
    main()
