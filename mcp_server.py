#!/usr/bin/env python3
"""MCP server for RAG knowledge base. Exposes search/add/remove/list tools to Claude Code."""

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Add project dir to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from embedding import get_embedding_function
from query import load_config, get_client, search as rag_search

mcp = FastMCP("rag-knowledge", instructions="""
RAG Knowledge Base with 3 collections:
- my-work: Published papers, profile, CV, resume (who I am and what I've done)
- active-research: Manuscript drafts and analysis notes (work in progress)
- literature: Academic papers from PhD through postdocs (~700 papers)

Use search_knowledge to find grounded information before making claims about
the user's work, papers, or research. Use list_collections to see what's available.
Results are reranked by a cross-encoder for relevance quality.
""")

cfg = load_config()
client = get_client(cfg)
ef = get_embedding_function()


@mcp.tool()
def search_knowledge(collection: str, query: str, n_results: int = 5,
                     project: str = None, source: str = None,
                     era: str = None) -> str:
    """Search a knowledge base collection with semantic search.

    Args:
        collection: Collection name (my-work, active-research, literature)
        query: Natural language search query
        n_results: Number of results to return (default 5)
        project: Filter by project (e.g. "project-a", "project-b")
        source: Filter by source (e.g. "zotero", "endnote-phd")
        era: Filter by era (e.g. "phd", "purdue", "postdoc")
    """
    where = {}
    if project:
        where["project"] = project
    if source:
        where["source"] = source
    if era:
        where["era"] = era

    results = rag_search(collection, query, n_results=n_results,
                         where=where if where else None)

    if not results:
        return f"No results found in '{collection}' for: {query}"

    output = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        header = f"Result {i} (distance: {r['distance']:.4f})"
        source_info = meta.get("filename", "unknown")
        if "project" in meta:
            source_info += f" | project: {meta['project']}"
        if "page" in meta:
            source_info += f" | page: {meta['page']}"
        if "header" in meta:
            source_info += f" | section: {meta['header']}"
        output.append(f"### {header}\n**Source:** {source_info}\n\n{r['text']}")

    return "\n\n---\n\n".join(output)


@mcp.tool()
def list_collections() -> str:
    """List all knowledge base collections with their descriptions and chunk counts."""
    lines = []
    for name, col_cfg in cfg["collections"].items():
        try:
            col = client.get_collection(name, embedding_function=ef)
            count = col.count()
        except Exception:
            count = 0
        lines.append(f"- **{name}** ({count} chunks): {col_cfg['description']}")
    return "\n".join(lines)


@mcp.tool()
def add_document(collection: str, text: str, source_name: str,
                 metadata: dict = None) -> str:
    """Add a new document to a collection.

    Args:
        collection: Collection name
        text: Document text to add
        source_name: Human-readable source name (e.g. "new_paper.pdf")
        metadata: Optional metadata dict (e.g. {"project": "project-a"})
    """
    import hashlib
    from ingest import chunk_text

    try:
        col = client.get_or_create_collection(collection, embedding_function=ef)
    except Exception as e:
        return f"Error accessing collection '{collection}': {e}"

    chunks = chunk_text(text)
    meta = metadata or {}
    meta["filename"] = source_name

    ids = []
    for i, chunk in enumerate(chunks):
        cid = hashlib.sha256(f"{collection}:{source_name}:{i}".encode()).hexdigest()[:16]
        ids.append(cid)

    col.upsert(ids=ids, documents=chunks, metadatas=[meta] * len(chunks))
    return f"Added {len(chunks)} chunks from '{source_name}' to '{collection}'"


@mcp.tool()
def remove_documents(collection: str, project: str = None,
                     source: str = None, filename: str = None) -> str:
    """Remove documents from a collection by metadata filter.

    Args:
        collection: Collection name
        project: Remove by project (e.g. "project-b")
        source: Remove by source (e.g. "endnote-phd")
        filename: Remove by filename
    """
    where = {}
    if project:
        where["project"] = project
    if source:
        where["source"] = source
    if filename:
        where["filename"] = filename

    if not where:
        return "Must specify at least one filter (project, source, or filename)"

    try:
        col = client.get_collection(collection, embedding_function=ef)
        before = col.count()
        col.delete(where=where)
        after = col.count()
        return f"Deleted {before - after} chunks from '{collection}' (was {before}, now {after})"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def collection_stats(collection: str) -> str:
    """Show statistics for a collection: chunk counts by metadata fields."""
    try:
        col = client.get_collection(collection, embedding_function=ef)
    except Exception:
        return f"Collection '{collection}' not found."

    data = col.get(include=["metadatas"])
    count = len(data["ids"])

    keys = {}
    for meta in data["metadatas"]:
        for k, v in meta.items():
            if k == "filepath":
                continue
            keys.setdefault(k, {})
            keys[k][str(v)] = keys[k].get(str(v), 0) + 1

    lines = [f"**{collection}**: {count} chunks\n"]
    for k, vals in sorted(keys.items()):
        lines.append(f"**{k}:**")
        for v, c in sorted(vals.items(), key=lambda x: -x[1]):
            lines.append(f"  - {v}: {c}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
