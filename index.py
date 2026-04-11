#!/usr/bin/env python3
"""
index.py — Parse FAO-56 PDF, generate embeddings, store in SQLite.

Usage:
    python index.py /path/to/fao-56.pdf [--db fao56.db]
"""

import argparse
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pdfplumber
import sqlite_vec
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~90MB, fast, no API key needed
CHUNK_CHARS = 900                   # target chars per chunk
OVERLAP_CHARS = 180                 # overlap between consecutive chunks
BATCH_SIZE = 64                     # sentences per embedding batch

# ── Helpers ───────────────────────────────────────────────────────────────────

SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s{1,4}[A-Z][A-Za-z ,\-]{3,70}$")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) for all pages."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append((i, text))
            print(f"\r  Extracting pages {i}/{total}...", end="", flush=True)
    print()
    return pages


def build_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """
    Merge all page text, split into overlapping chunks.
    Detect section headings to tag each chunk.
    """
    # Join all pages preserving double-newlines between pages
    full_text = "\n\n".join(text for _, text in pages)

    # Normalize whitespace
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    paragraphs = re.split(r"\n{2,}", full_text)

    chunks = []
    current_section: str | None = None
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if SECTION_RE.match(para):
            current_section = para

        candidate = (buffer + "\n" + para).strip() if buffer else para

        if len(candidate) > CHUNK_CHARS and buffer:
            chunks.append({"section": current_section, "content": buffer.strip()})
            # Overlap: take the tail of current buffer
            buffer = buffer[-OVERLAP_CHARS:].strip() + "\n" + para
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append({"section": current_section, "content": buffer.strip()})

    return chunks


def serialize_vector(v: np.ndarray) -> bytes:
    """Pack float32 array as little-endian bytes for sqlite-vec."""
    return struct.pack(f"{len(v)}f", *v.tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Index FAO-56 PDF into SQLite RAG database")
    parser.add_argument("pdf", help="Path to fao-56.pdf")
    parser.add_argument("--db", default="fao56.db", help="Output SQLite database path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    print(f"[1/4] Extracting text from {pdf_path.name}...")
    pages = extract_pages(str(pdf_path))
    print(f"      {len(pages)} pages extracted.")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    print("[2/4] Chunking text...")
    chunks = build_chunks(pages)
    print(f"      {len(chunks)} chunks created.")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print(f"[3/4] Loading model '{MODEL_NAME}' and generating embeddings...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity = dot product
        device="cpu",
    )

    dim = embeddings.shape[1]
    print(f"      {len(embeddings)} embeddings generated ({dim} dims).")

    # ── Step 4: Store in SQLite ───────────────────────────────────────────────
    print(f"[4/4] Writing to {db_path}...")
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(f"""
        CREATE TABLE chunks (
            id      INTEGER PRIMARY KEY,
            section TEXT,
            content TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            id      INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        );
    """)

    conn.executemany(
        "INSERT INTO chunks (id, section, content) VALUES (?, ?, ?)",
        [(i, c["section"], c["content"]) for i, c in enumerate(chunks)],
    )

    conn.executemany(
        "INSERT INTO chunk_embeddings (id, embedding) VALUES (?, ?)",
        [(i, serialize_vector(embeddings[i])) for i in range(len(chunks))],
    )

    # Store metadata
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('model', ?)", (MODEL_NAME,))
    conn.execute("INSERT INTO meta VALUES ('dim', ?)", (str(dim),))
    conn.execute("INSERT INTO meta VALUES ('chunks', ?)", (str(len(chunks)),))
    conn.execute("INSERT INTO meta VALUES ('pdf', ?)", (pdf_path.name,))

    conn.commit()
    conn.close()

    print(f"\nDone. Database: {db_path.resolve()}")
    print(f"  {len(chunks)} chunks | {dim}-dim embeddings | model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
