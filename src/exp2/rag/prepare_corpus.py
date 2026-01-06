# src/exp2/rag/prepare_corpus.py
from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

from src.common.paths import data_path, ensure_dir, experiment_root


# -----------------------------
# Config (no hard-coded paths)
# -----------------------------
DEFAULT_INPUT_SUBDIR = os.getenv("LAW_BOOK_SUBDIR", "law_book")  # under exp2/data/
DEFAULT_OUT_SUBDIR = os.getenv("RAG_OUT_SUBDIR", "rag")          # under exp2/data/
DEFAULT_OUT_FILE = os.getenv("RAG_CORPUS_FILE", "corpus.jsonl")

# Some repos store as exp2/data/law_book/Law-Book/...
# This script will auto-detect nested "Law-Book" if present.
AUTO_NESTED_LAWBOOK_DIRNAME = "Law-Book"


# -----------------------------
# Text cleaning helpers
# -----------------------------
_MD_NOISE_PATTERNS = [
    # GitHub footer-like or navigation noise (rare, safe)
    r"^\s*Back to top\s*$",
    r"^\s*Raw\s*$",
]

def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def strip_md_code_fences(text: str) -> str:
    # Remove triple-backtick code blocks if any (laws shouldn't have them, but just in case)
    return re.sub(r"```.*?```", "", text, flags=re.S)

def remove_md_links(text: str) -> str:
    # Keep link text, drop URL: [text](url) -> text
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

def remove_extra_whitespace(text: str) -> str:
    # Collapse multiple spaces/tabs; keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def drop_noise_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        bad = False
        for p in _MD_NOISE_PATTERNS:
            if re.match(p, line):
                bad = True
                break
        if not bad:
            lines.append(line)
    return "\n".join(lines).strip()

def basic_clean_md(text: str) -> str:
    text = normalize_newlines(text)
    text = strip_md_code_fences(text)
    text = remove_md_links(text)
    text = drop_noise_lines(text)
    text = remove_extra_whitespace(text)
    return text


# -----------------------------
# Metadata helpers
# -----------------------------
def sha1_short(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def infer_title_from_filename(filename: str) -> str:
    # Example: "全民所有制工业企业法（2009-08-27）.md" -> "全民所有制工业企业法"
    name = filename
    if name.lower().endswith(".md"):
        name = name[:-3]
    # Remove date parentheses (Chinese fullwidth or normal)
    name = re.sub(r"[（(]\d{4}-\d{2}-\d{2}[）)]", "", name).strip()
    return name

def rel_to_data_dir(p: Path) -> str:
    # Store portable path relative to exp2/data
    data_dir = data_path()  # exp2/data
    try:
        return str(p.relative_to(data_dir)).replace("\\", "/")
    except Exception:
        # fallback to absolute (shouldn't happen if under data)
        return str(p).replace("\\", "/")


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    rel_path: str
    law_group: str  # e.g., "1-宪法" / "2-宪法相关法"
    text: str
    text_len: int
    file_sha1: str


def find_law_book_root() -> Path:
    """
    Find exp2/data/law_book, and if contains nested Law-Book, use that.
    """
    lb = data_path(DEFAULT_INPUT_SUBDIR)
    if not lb.exists():
        raise FileNotFoundError(
            f"law_book directory not found: {lb}\n"
            f"Check your EXPERIMENT_ROOT/LAW_BOOK_SUBDIR env or folder structure."
        )
    nested = lb / AUTO_NESTED_LAWBOOK_DIRNAME
    if nested.exists() and nested.is_dir():
        return nested
    return lb


def iter_md_files(root: Path) -> Iterable[Path]:
    # Skip hidden dirs like .git
    for p in root.rglob("*.md"):
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def build_doc_id(rel_path: str) -> str:
    # Stable doc_id based on relative path within exp2/data
    # Example: "law_book/3-民法商法/全民所有制工业企业法（2009-08-27）.md"
    return sha1_short(rel_path, 12)


def infer_law_group(rel_path: str) -> str:
    # Try get the first folder under law_book/...
    # rel_path looks like: "law_book/2-宪法相关法/xxx.md"
    parts = rel_path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    law_root = find_law_book_root()
    out_dir = ensure_dir(data_path(DEFAULT_OUT_SUBDIR))
    out_file = out_dir / DEFAULT_OUT_FILE
    stats_file = out_dir / "corpus_stats.json"

    md_files = list(iter_md_files(law_root))
    if not md_files:
        raise RuntimeError(f"No .md files found under: {law_root}")

    rows: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}  # text_hash -> doc_id (for optional dedup)
    num_skipped_empty = 0
    num_deduped = 0

    for idx, fp in enumerate(md_files, start=1):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        cleaned = basic_clean_md(raw)

        if len(cleaned) < 20:  # too short after cleaning
            num_skipped_empty += 1
            continue

        relp = rel_to_data_dir(fp)
        doc_id = build_doc_id(relp)
        title = infer_title_from_filename(fp.name)
        law_group = infer_law_group(relp)
        fsha1 = file_sha1(fp)

        # Optional dedup by cleaned text hash (avoid identical files)
        text_hash = sha1_short(cleaned, 16)
        if text_hash in seen_hashes:
            num_deduped += 1
            continue
        seen_hashes[text_hash] = doc_id

        doc = CorpusDoc(
            doc_id=doc_id,
            title=title,
            rel_path=relp,
            law_group=law_group,
            text=cleaned,
            text_len=len(cleaned),
            file_sha1=fsha1,
        )
        rows.append(asdict(doc))

        if idx % 200 == 0:
            print(f"[prepare_corpus] processed {idx}/{len(md_files)} files...")

    # Sort for reproducibility
    rows.sort(key=lambda x: (x["law_group"], x["rel_path"]))

    write_jsonl(out_file, rows)

    # Stats
    lengths = [r["text_len"] for r in rows]
    stats = {
        "experiment_root": str(experiment_root()).replace("\\", "/"),
        "law_root": str(law_root).replace("\\", "/"),
        "input_md_files": len(md_files),
        "output_docs": len(rows),
        "skipped_too_short": num_skipped_empty,
        "deduped_by_text_hash": num_deduped,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": (sum(lengths) / len(lengths)) if lengths else 0,
        "out_file": str(out_file).replace("\\", "/"),
    }

    ensure_dir(stats_file.parent)
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Preview a few docs
    print("\n[prepare_corpus] DONE")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n[prepare_corpus] Preview (first 2 docs):")
    for r in rows[:2]:
        snippet = r["text"][:200].replace("\n", " ")
        print(f"- doc_id={r['doc_id']} group={r['law_group']} title={r['title']} path={r['rel_path']}")
        print(f"  text[:200]={snippet}...\n")


if __name__ == "__main__":
    main()
