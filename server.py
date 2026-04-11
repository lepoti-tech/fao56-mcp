#!/usr/bin/env python3
"""
server.py — MCP server exposing FAO-56 RAG as a Claude Code tool.

Configure in ~/.claude/claude_code_config.json:
    {
      "mcpServers": {
        "fao56": {
          "command": "/path/to/fao56-rag/.venv/bin/python",
          "args": ["/path/to/fao56-rag/server.py"],
          "env": { "FAO56_DB": "/path/to/fao56-rag/fao56.db" }
        }
      }
    }
"""

import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import sqlite_vec
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("FAO56_DB", str(Path(__file__).parent / "fao56.db"))
TOP_K = 5
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# ── Load resources once at startup ───────────────────────────────────────────

_model: SentenceTransformer | None = None
_conn: sqlite3.Connection | None = None
_dim: int = 384


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        if not Path(DB_PATH).exists():
            raise FileNotFoundError(
                f"Database not found: {DB_PATH}\n"
                "Run: python index.py /path/to/fao-56.pdf first."
            )
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.enable_load_extension(True)
        sqlite_vec.load(_conn)
        _conn.enable_load_extension(False)

        meta = dict(_conn.execute("SELECT key, value FROM meta").fetchall())
        global _dim
        _dim = int(meta.get("dim", 384))

    return _conn


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        conn = _get_conn()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        model_name = meta.get("model", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name, device="cpu")
    return _model


def _serialize(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("fao56-rag", host=MCP_HOST, port=MCP_PORT)


@mcp.tool()
def query_fao56(question: str, top_k: int = TOP_K) -> str:
    """
    Search the FAO-56 (Crop Evapotranspiration — Guidelines for computing crop
    water requirements, 2025 revised edition) knowledge base and return the most
    relevant passages.

    Use this tool when you need authoritative information about:
    - Reference evapotranspiration (ETo) and the Penman-Monteith equation
    - Crop coefficients (Kc, Kcb) for specific crops
    - Crop evapotranspiration under standard conditions (ETc)
    - Crop evapotranspiration under non-standard conditions (ETc act)
    - Soil water balance and depletion
    - Crop growth stages and phenological parameters
    - Irrigation scheduling and water requirements
    - Soil evaporation and dual crop coefficient approach
    - Climate data processing for ET calculations

    Args:
        question: Natural language question or technical term to search for.
        top_k: Number of passages to return (default 5, max 10).
    """
    top_k = min(int(top_k), 10)

    try:
        conn = _get_conn()
        model = _get_model()
    except FileNotFoundError as e:
        return str(e)

    # Embed the query
    q_vec = model.encode(question, normalize_embeddings=True)
    q_bytes = _serialize(q_vec)

    # Vector similarity search via sqlite-vec
    rows = conn.execute(
        """
        SELECT c.section, c.content, e.distance
        FROM chunk_embeddings e
        JOIN chunks c ON c.id = e.id
        WHERE e.embedding MATCH ?
          AND k = ?
        ORDER BY e.distance
        """,
        (q_bytes, top_k),
    ).fetchall()

    if not rows:
        return "No relevant passages found in FAO-56 for this query."

    parts = []
    for section, content, distance in rows:
        header = f"[FAO-56 — {section}]" if section else "[FAO-56]"
        similarity = round(1 - distance, 3)  # sqlite-vec returns L2 distance for cosine-normalized vecs
        parts.append(f"{header} (relevance: {similarity})\n{content}")

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def fao56_stats() -> str:
    """
    Return metadata about the indexed FAO-56 database
    (number of chunks, embedding model, source file).
    """
    try:
        conn = _get_conn()
    except FileNotFoundError as e:
        return str(e)

    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    return (
        f"FAO-56 RAG database\n"
        f"  Source:  {meta.get('pdf', 'unknown')}\n"
        f"  Chunks:  {meta.get('chunks', '?')}\n"
        f"  Model:   {meta.get('model', '?')}\n"
        f"  Dims:    {meta.get('dim', '?')}\n"
        f"  DB path: {DB_PATH}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()

    if args.transport == "sse" and MCP_MOUNT_PATH and MCP_MOUNT_PATH != "/":
        # Mount the SSE app at MCP_MOUNT_PATH so the endpoint event includes the
        # full path prefix (e.g. /fao-56/messages/), enabling reverse-proxy setups.
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        sse_app = mcp.sse_app(mount_path=MCP_MOUNT_PATH)
        app = Starlette(routes=[Mount(MCP_MOUNT_PATH, app=sse_app)])
        uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
    else:
        mcp.run(transport=args.transport)
