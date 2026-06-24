#!/usr/bin/env python3
"""Query the RAG knowledge base from terminal or Python."""

import argparse
import os
import sys
from pathlib import Path

import chromadb
import yaml

from embedding import get_embedding_function

# Lazy-loaded reranker
_reranker = None
_RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"
_RERANKER_MODEL_FILE = "flashrank-TinyBERT-L-2-v2.onnx"
_RERANKER_CACHE_DIR = Path(
    os.environ.get("FLASHRANK_CACHE_DIR", "~/.cache/flashrank")
).expanduser()
INTERNAL_METADATA_KEYS = {"filepath", "file_size", "file_mtime_ns", "source_metadata_hash"}


def _get_reranker():
    global _reranker
    if _reranker is None:
        import shutil
        from flashrank import Ranker

        model_dir = _RERANKER_CACHE_DIR / _RERANKER_MODEL
        model_file = model_dir / _RERANKER_MODEL_FILE
        if model_dir.exists() and not model_file.exists():
            shutil.rmtree(model_dir)
        _reranker = Ranker(
            model_name=_RERANKER_MODEL,
            cache_dir=str(_RERANKER_CACHE_DIR),
        )
    return _reranker


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["chroma_path"] = os.path.expanduser(cfg["chroma_path"])
    return cfg


def get_client(cfg: dict) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=cfg["chroma_path"])


def search(collection_name: str, query: str, n_results: int = 5,
           where: dict = None, config_path: str = None,
           rerank: bool = True) -> list[dict]:
    """Search a collection with semantic search + cross-encoder reranking.

    Usable from Python:
        from query import search
        results = search("my-work", "activation barriers for ADH")
    """
    cfg = load_config(config_path)
    client = get_client(cfg)
    ef = get_embedding_function()

    try:
        collection = client.get_collection(collection_name, embedding_function=ef)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        return []

    # Fetch more candidates than needed for reranking
    fetch_n = min(n_results * 4, 20) if rerank else n_results
    kwargs = {"query_texts": [query], "n_results": fetch_n}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    # Rerank with cross-encoder
    if rerank and items:
        try:
            from flashrank import RerankRequest
            reranker = _get_reranker()
            passages = [{"id": i, "text": item["text"]} for i, item in enumerate(items)]
            request = RerankRequest(query=query, passages=passages)
            ranked = reranker.rerank(request)
            reranked_items = []
            for r in ranked[:n_results]:
                idx = r["id"]
                item = items[idx]
                item["rerank_score"] = r["score"]
                reranked_items.append(item)
            return reranked_items
        except Exception as e:
            print(f"  WARN: Reranking failed ({e}), returning unranked results")
            return items[:n_results]

    return items[:n_results]


def list_collections(config_path: str = None):
    """List all collections with chunk counts."""
    cfg = load_config(config_path)
    client = get_client(cfg)
    ef = get_embedding_function()
    for name, col_cfg in cfg["collections"].items():
        try:
            col = client.get_collection(name, embedding_function=ef)
            count = col.count()
        except Exception:
            count = 0
        print(f"  {name}: {col_cfg['description']} ({count} chunks)")


def delete_documents(collection_name: str, where: dict, config_path: str = None):
    """Delete documents matching a metadata filter.

    Example: delete_documents("active-research", {"project": "project-b"})
    """
    cfg = load_config(config_path)
    client = get_client(cfg)
    ef = get_embedding_function()
    collection = client.get_collection(collection_name, embedding_function=ef)
    before = collection.count()
    collection.delete(where=where)
    after = collection.count()
    print(f"Deleted {before - after} chunks from '{collection_name}' "
          f"(was {before}, now {after})")


def main():
    parser = argparse.ArgumentParser(description="Query RAG knowledge base")
    sub = parser.add_subparsers(dest="command")

    # search
    s = sub.add_parser("search", help="Search a collection")
    s.add_argument("collection", help="Collection name")
    s.add_argument("query", help="Search query")
    s.add_argument("-n", type=int, default=5, help="Number of results")
    s.add_argument("--project", help="Filter by project metadata")
    s.add_argument("--source", help="Filter by source metadata")
    s.add_argument("--era", help="Filter by era (phd/postdoc/purdue)")
    s.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")

    # list
    sub.add_parser("list", help="List collections")

    # delete
    d = sub.add_parser("delete", help="Delete documents by metadata filter")
    d.add_argument("collection", help="Collection name")
    d.add_argument("--project", help="Delete by project")
    d.add_argument("--source", help="Delete by source")

    # stats
    st = sub.add_parser("stats", help="Show collection statistics")
    st.add_argument("collection", help="Collection name")

    args = parser.parse_args()

    if args.command == "list":
        list_collections()
    elif args.command == "search":
        where = {}
        if args.project:
            where["project"] = args.project
        if args.source:
            where["source"] = args.source
        if args.era:
            where["era"] = args.era
        results = search(args.collection, args.query, n_results=args.n,
                         where=where if where else None,
                         rerank=not args.no_rerank)
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            score_str = (f", rerank: {r['rerank_score']:.4f}"
                         if "rerank_score" in r else "")
            print(f"\n--- Result {i} (distance: {r['distance']:.4f}{score_str}) ---")
            print(f"Source: {meta.get('filename', 'unknown')}")
            if "project" in meta:
                print(f"Project: {meta['project']}")
            if "page" in meta:
                print(f"Page: {meta['page']}")
            if "section" in meta:
                print(f"Section: {meta['section']}")
            elif "header" in meta:
                print(f"Section: {meta['header']}")
            print(f"\n{r['text'][:500]}{'...' if len(r['text']) > 500 else ''}")
    elif args.command == "delete":
        where = {}
        if args.project:
            where["project"] = args.project
        if args.source:
            where["source"] = args.source
        if not where:
            print("Must specify at least one filter (--project or --source)")
            return
        delete_documents(args.collection, where)
    elif args.command == "stats":
        cfg = load_config()
        client = get_client(cfg)
        ef = get_embedding_function()
        try:
            col = client.get_collection(args.collection, embedding_function=ef)
        except Exception:
            print(f"Collection '{args.collection}' not found.")
            return
        data = col.get(include=["metadatas"])
        count = len(data["ids"])
        print(f"Collection: {args.collection} ({count} chunks)")
        keys = {}
        for meta in data["metadatas"]:
            for k, v in meta.items():
                if k in INTERNAL_METADATA_KEYS:
                    continue
                keys.setdefault(k, {})
                keys[k][str(v)] = keys[k].get(str(v), 0) + 1
        for k, vals in sorted(keys.items()):
            print(f"\n  {k}:")
            for v, c in sorted(vals.items(), key=lambda x: -x[1]):
                print(f"    {v}: {c}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
