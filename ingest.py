#!/usr/bin/env python3
"""Ingest documents into RAG knowledge base collections."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
import yaml
from tqdm import tqdm

from embedding import get_embedding_function

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64  # chunks per upsert/update batch
METADATA_PAGE_SIZE = 5000

FILEPATH_KEY = "filepath"
FILE_SIZE_KEY = "file_size"
FILE_MTIME_NS_KEY = "file_mtime_ns"
SOURCE_METADATA_HASH_KEY = "source_metadata_hash"
FILE_SIGNATURE_KEYS = (FILE_SIZE_KEY, FILE_MTIME_NS_KEY, SOURCE_METADATA_HASH_KEY)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["chroma_path"] = os.path.expanduser(cfg["chroma_path"])
    return cfg


def get_client(cfg: dict) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=cfg["chroma_path"])


def stable_id(collection_name: str, filepath: str, chunk_idx: int) -> str:
    """Deterministic ID from collection + filepath + chunk index."""
    key = f"{collection_name}:{filepath}:{chunk_idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def source_metadata_hash(metadata: dict) -> str:
    """Stable hash of configured source metadata for incremental invalidation."""
    payload = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def file_signature(filepath: str, source_metadata: dict) -> dict:
    """Return metadata fields that identify the current file contents/config."""
    st = os.stat(filepath)
    return {
        FILEPATH_KEY: filepath,
        FILE_SIZE_KEY: st.st_size,
        FILE_MTIME_NS_KEY: st.st_mtime_ns,
        SOURCE_METADATA_HASH_KEY: source_metadata_hash(source_metadata),
    }


# --- Reference section detection ---

REFERENCE_PATTERNS = [
    r"^\s*References\s*$",
    r"^\s*REFERENCES\s*$",
    r"^\s*Bibliography\s*$",
    r"^\s*Works\s+Cited\s*$",
    r"^\s*Literature\s+Cited\s*$",
    r"^\[1\]\s+[A-Z]",  # numbered reference list start like [1] A. Author
]
REFERENCE_RE = re.compile("|".join(REFERENCE_PATTERNS), re.MULTILINE)


def strip_references(text: str) -> str:
    """Remove reference/bibliography section from extracted text."""
    match = REFERENCE_RE.search(text)
    if match:
        # Keep everything before the references heading
        return text[:match.start()].strip()
    return text


def strip_pdf_boilerplate(text: str) -> str:
    """Remove common journal boilerplate from PDF text."""
    # Remove "Downloaded by ... on ..." lines
    text = re.sub(r"Downloaded by .+ on .+\.", "", text)
    # Remove "Published on ... " lines
    text = re.sub(r"Published on \d+ .+\.", "", text)
    # Remove "View Article Online" etc
    text = re.sub(r"View Article Online", "", text)
    # Remove "This journal is ©" lines
    text = re.sub(r"This journal is ©.*$", "", text, flags=re.MULTILINE)
    # Remove DOI lines
    text = re.sub(r"DOI:\s*10\.\d+/\S+", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Text extraction ---

def extract_pdf(filepath: str) -> list[dict]:
    """Extract text from PDF, one chunk per page. Tags bibliography pages with metadata."""
    pages = []
    try:
        doc = fitz.open(filepath)
        in_references = False
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text and len(text) > 50:
                # Detect start of references section
                if not in_references and REFERENCE_RE.search(text):
                    in_references = True
                pages.append({
                    "text": strip_pdf_boilerplate(text),
                    "page": i + 1,
                    "is_bibliography": in_references,
                })
        doc.close()

    except Exception as e:
        print(f"  WARN: Failed to extract {filepath}: {e}")
    return [p for p in pages if p["text"] and len(p["text"]) > 50]


def extract_text_file(filepath: str) -> str:
    """Read a text file (.md, .tex, .txt)."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"  WARN: Failed to read {filepath}: {e}")
        return ""


# --- TeX processing ---

def strip_tex_commands(text: str) -> str:
    """Lightly clean TeX for better embedding quality."""
    # Remove comments
    text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
    # Remove common environments we don't need
    text = re.sub(r"\\begin\{(figure|table|equation)\*?\}.*?\\end\{\1\*?\}",
                  "", text, flags=re.DOTALL)
    # Extract text from commands: \textbf{X} -> X, \emph{X} -> X
    text = re.sub(r"\\(?:textbf|textit|emph|underline|text)\{([^}]*)\}", r"\1", text)
    # Remove \cite{...}, \ref{...}, \label{...}
    text = re.sub(r"\\(?:cite|ref|label|eqref|cref)\w*\{[^}]*\}", "", text)
    # Remove remaining commands but keep their content
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    # Remove commands without arguments
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_tex_by_section(text: str, max_chars: int = 1200) -> list[dict]:
    """Split TeX by \\section/\\subsection boundaries, preserving section names."""
    # Split on \section or \subsection
    pattern = r"(\\(?:sub)*section\*?\{[^}]+\})"
    parts = re.split(pattern, text)

    sections = []
    current_section = ""
    current_text = ""

    for part in parts:
        sec_match = re.match(r"\\(?:sub)*section\*?\{([^}]+)\}", part)
        if sec_match:
            # Save previous section
            if current_text.strip():
                sections.append({"section": current_section, "text": current_text.strip()})
            current_section = sec_match.group(1).strip()
            current_text = ""
        else:
            current_text += part

    # Save last section
    if current_text.strip():
        sections.append({"section": current_section, "text": current_text.strip()})

    # Now clean and chunk each section
    chunks = []
    for sec in sections:
        cleaned = strip_tex_commands(sec["text"])
        if not cleaned or len(cleaned) < 30:
            continue
        sub_chunks = chunk_text(cleaned, max_chars=max_chars)
        for sc in sub_chunks:
            chunks.append({"text": sc, "section": sec["section"]})

    # Fallback: if no sections found, chunk the whole thing
    if not chunks:
        cleaned = strip_tex_commands(text)
        if cleaned:
            sub_chunks = chunk_text(cleaned, max_chars=max_chars)
            for sc in sub_chunks:
                chunks.append({"text": sc, "section": ""})

    return chunks


# --- Chunking ---

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_text = " ".join(words[-overlap // 5:]) if words else ""
            current = overlap_text + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_markdown(text: str, max_chars: int = 1000) -> list[dict]:
    """Split markdown by headers, then by size if needed."""
    sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r"^(#{1,3})\s+(.+)$", section, re.MULTILINE)
        header = header_match.group(2).strip() if header_match else ""
        sub_chunks = chunk_text(section, max_chars=max_chars)
        for sc in sub_chunks:
            chunks.append({"text": sc, "header": header})
    return chunks if chunks else [{"text": text, "header": ""}]


# --- File discovery ---

def resolve_files(source: dict) -> list[str]:
    """Resolve source config to list of file paths."""
    import glob as globmod

    path = os.path.expanduser(source["path"])
    glob_pattern = source.get("glob", "")
    excludes = source.get("exclude", [])

    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        print(f"  WARN: Path not found: {path}")
        return []

    if glob_pattern:
        if "{" in glob_pattern:
            exts = re.findall(r"\{([^}]+)\}", glob_pattern)
            patterns = []
            for ext_group in exts:
                for ext in ext_group.split(","):
                    p = re.sub(r"\{[^}]+\}", ext.strip(), glob_pattern)
                    patterns.append(p)
        else:
            patterns = [glob_pattern]

        files = []
        for p in patterns:
            full = os.path.join(path, p)
            files.extend(globmod.glob(full, recursive=True))
    else:
        files = [path] if os.path.isfile(path) else []

    result = []
    for f in files:
        rel = os.path.relpath(f, path)
        skip = False
        for exc in excludes:
            if globmod.fnmatch.fnmatch(rel, exc):
                skip = True
                break
        if not skip:
            result.append(f)

    return sorted(set(result))


# --- Ingestion ---

def _metadata_values_equal(left, right) -> bool:
    """Compare Chroma metadata values while tolerating int/string round-trips."""
    return left == right or (left is not None and right is not None and str(left) == str(right))


def existing_file_manifest(collection) -> dict:
    """Build a filepath -> existing chunk metadata map from Chroma."""
    count = collection.count()
    if count == 0:
        return {}

    print(f"  Reading existing metadata from {count} chunks...")
    manifest = {}

    for offset in tqdm(range(0, count, METADATA_PAGE_SIZE),
                       desc="  metadata", unit="batch"):
        data = collection.get(
            include=["metadatas"],
            limit=METADATA_PAGE_SIZE,
            offset=offset,
        )
        for cid, meta in zip(data.get("ids", []), data.get("metadatas", [])):
            if not meta:
                continue
            filepath = meta.get(FILEPATH_KEY)
            if not filepath:
                continue

            entry = manifest.setdefault(filepath, {
                "ids": [],
                "metadatas": [],
                "chunk_count": 0,
                "sample_metadata": meta,
                "inconsistent_signature": False,
            })
            entry["ids"].append(cid)
            entry["metadatas"].append(meta)
            entry["chunk_count"] += 1

            for key in FILE_SIGNATURE_KEYS:
                value = meta.get(key)
                if key not in entry:
                    entry[key] = value
                elif not _metadata_values_equal(entry[key], value):
                    entry["inconsistent_signature"] = True

    return manifest


def manifest_matches_signature(entry: dict, signature: dict) -> bool:
    """True when existing chunks match the current file signature."""
    if entry.get("inconsistent_signature"):
        return False
    return all(
        entry.get(key) is not None and _metadata_values_equal(entry.get(key), signature[key])
        for key in FILE_SIGNATURE_KEYS
    )


def legacy_manifest_can_be_backfilled(entry: dict, source_metadata: dict) -> bool:
    """True for pre-incremental chunks whose source metadata still matches config."""
    if any(entry.get(key) is not None for key in FILE_SIGNATURE_KEYS):
        return False

    sample = entry.get("sample_metadata") or {}
    return all(_metadata_values_equal(sample.get(k), v) for k, v in source_metadata.items())


def backfill_file_metadata(collection, manifest: dict, backfill_items: list[tuple[str, dict]]) -> int:
    """Add file signature fields to legacy chunks without re-embedding them."""
    updated_chunks = 0
    for filepath, signature in tqdm(backfill_items, desc="  backfill metadata", unit="file"):
        entry = manifest[filepath]
        ids = entry["ids"]
        metadatas = [{**meta, **signature} for meta in entry["metadatas"]]
        for i in range(0, len(ids), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(ids))
            collection.update(
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            updated_chunks += batch_end - i
    return updated_chunks


def print_dry_run_summary(name: str, stats: dict, prune: bool,
                          force: bool, missing_sources: list[str]) -> None:
    """Print a read-only ingest plan."""
    print(f"  Dry run for '{name}': no changes written")
    print(f"  Existing indexed files: {stats['existing_files']}")
    print(f"  Discovered source files: {stats['discovered']}")
    print(f"  New files: {stats['planned_new']}")
    print(f"  Changed files: {stats['planned_updated']}")
    print(f"  Unchanged files: {stats['unchanged']}")
    if stats["backfilled_files"]:
        print(f"  Legacy metadata backfill: {stats['backfilled_files']} files, "
              f"{stats['backfilled_chunks']} chunks")
    if stats["errors"]:
        print(f"  File stat errors: {stats['errors']}")

    if force:
        print("  Force rebuild: would delete and recreate the collection before indexing")
        return

    print(f"  Stale files: {stats['stale_files']} "
          f"({stats['stale_chunks']} existing chunks)")
    if prune and missing_sources:
        print("  Prune would be skipped because configured source paths are missing:")
        for path in missing_sources:
            print(f"        {path}")
    elif prune:
        print(f"  Would prune: {stats['stale_files']} files, "
              f"{stats['stale_chunks']} chunks")
    elif stats["stale_files"]:
        print("  Stale files would be kept. Add --prune to remove them.")


def _process_file(args: tuple) -> dict:
    """Process a single file into chunks + metadata. Runs in worker processes."""
    filepath, base_metadata, collection_name, signature = args
    ext = os.path.splitext(filepath)[1].lower()
    file_meta = {**base_metadata, **signature}
    result = {"filepath": filepath, "ids": [], "documents": [], "metadatas": [],
              "error": None, "skipped": False}

    try:
        if ext == ".pdf":
            pages = extract_pdf(filepath)
            if not pages:
                result["error"] = "no text extracted"
                return result
            for page_data in pages:
                chunks = chunk_text(page_data["text"])
                for i, chunk in enumerate(chunks):
                    cid = stable_id(collection_name, filepath, page_data["page"] * 100 + i)
                    meta = {**file_meta, "page": page_data["page"],
                            "filename": os.path.basename(filepath),
                            "is_bibliography": page_data.get("is_bibliography", False)}
                    result["ids"].append(cid)
                    result["documents"].append(chunk)
                    result["metadatas"].append(meta)

        elif ext == ".tex":
            raw = extract_text_file(filepath)
            if not raw:
                result["error"] = "empty file"
                return result
            tex_chunks = chunk_tex_by_section(raw)
            for i, tc in enumerate(tex_chunks):
                cid = stable_id(collection_name, filepath, i)
                meta = {**file_meta, "filename": os.path.basename(filepath),
                        "section": tc.get("section", "")}
                result["ids"].append(cid)
                result["documents"].append(tc["text"])
                result["metadatas"].append(meta)

        elif ext in (".md", ".txt"):
            raw = extract_text_file(filepath)
            if not raw:
                result["error"] = "empty file"
                return result
            md_chunks = chunk_markdown(raw)
            for i, mc in enumerate(md_chunks):
                cid = stable_id(collection_name, filepath, i)
                meta = {**file_meta, "filename": os.path.basename(filepath),
                        "header": mc.get("header", "")}
                result["ids"].append(cid)
                result["documents"].append(mc["text"])
                result["metadatas"].append(meta)
        else:
            result["skipped"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def ingest_collection(client: chromadb.PersistentClient, name: str,
                      col_config: dict, force: bool = False,
                      prune: bool = False, dry_run: bool = False,
                      workers: int = 4) -> dict:
    """Incrementally ingest sources into a collection."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ef = get_embedding_function()
    collection = None
    existing_manifest = {}

    if dry_run:
        try:
            collection = client.get_collection(name, embedding_function=ef)
        except Exception:
            collection = None
        if collection is not None:
            existing_manifest = existing_file_manifest(collection)
    elif force:
        try:
            client.delete_collection(name)
            print(f"  Cleared existing collection '{name}' for re-index")
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=name,
            metadata={"description": col_config.get("description", "")},
            embedding_function=ef,
        )
    else:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"description": col_config.get("description", "")},
            embedding_function=ef,
        )
        existing_manifest = existing_file_manifest(collection)

    stats = {
        "discovered": 0,
        "existing_files": len(existing_manifest),
        "planned_new": 0,
        "planned_updated": 0,
        "processed": 0,
        "new": 0,
        "updated": 0,
        "chunks": 0,
        "unchanged": 0,
        "backfilled_files": 0,
        "backfilled_chunks": 0,
        "unsupported": 0,
        "errors": 0,
        "pruned_files": 0,
        "pruned_chunks": 0,
        "stale_files": 0,
        "stale_chunks": 0,
    }

    # Gather all files across sources with their metadata
    all_files = []
    discovered_paths = set()
    missing_sources = []
    for source in col_config.get("sources", []):
        base_metadata = source.get("metadata", {})
        source_path = os.path.expanduser(source["path"])
        if not os.path.exists(source_path):
            missing_sources.append(source_path)
        files = resolve_files(source)
        print(f"  Source: {source_path} -> {len(files)} files")
        for f in files:
            try:
                signature = file_signature(f, base_metadata)
            except OSError as e:
                print(f"  WARN: Failed to stat {f}: {e}")
                stats["errors"] += 1
                continue
            all_files.append((f, base_metadata, name, signature))
            discovered_paths.add(f)

    stats["discovered"] = len(all_files)
    comparison_manifest = {} if force else existing_manifest

    # Plan the incremental work before starting expensive PDF extraction/embedding.
    files_to_process = []
    backfill_items = []
    for args in all_files:
        filepath, base_metadata, _, signature = args
        existing = comparison_manifest.get(filepath)
        if existing and manifest_matches_signature(existing, signature):
            stats["unchanged"] += 1
            continue
        if existing and legacy_manifest_can_be_backfilled(existing, base_metadata):
            stats["unchanged"] += 1
            stats["backfilled_files"] += 1
            backfill_items.append((filepath, signature))
            continue
        if existing:
            stats["planned_updated"] += 1
        else:
            stats["planned_new"] += 1
        files_to_process.append(args)

    stale_paths = [] if force else sorted(set(existing_manifest) - discovered_paths)
    stats["stale_files"] = len(stale_paths)
    stats["stale_chunks"] = sum(existing_manifest[p]["chunk_count"] for p in stale_paths)
    if backfill_items:
        stats["backfilled_chunks"] = sum(
            existing_manifest[filepath]["chunk_count"]
            for filepath, _ in backfill_items
        )

    print(f"  Incremental plan: {len(files_to_process)} to process "
          f"({stats['planned_new']} new, {stats['planned_updated']} changed), "
          f"{stats['unchanged']} unchanged")

    if dry_run:
        print_dry_run_summary(name, stats, prune, force, missing_sources)
        return stats

    if backfill_items:
        stats["backfilled_chunks"] = backfill_file_metadata(
            collection, existing_manifest, backfill_items
        )

    if prune and missing_sources:
        print("  WARN: Skipping prune because configured source paths are missing:")
        for path in missing_sources:
            print(f"        {path}")
    elif prune:
        if stale_paths:
            print(f"  Pruning {len(stale_paths)} stale files...")
        for filepath in tqdm(stale_paths, desc=f"  {name} prune", unit="file"):
            stats["pruned_files"] += 1
            stats["pruned_chunks"] += existing_manifest[filepath]["chunk_count"]
            collection.delete(where={FILEPATH_KEY: filepath})

    # Phase 1: Parallel file extraction (CPU + I/O bound)
    print(f"  Extracting text from {len(files_to_process)} files ({workers} workers)...")
    pending_ids = []
    pending_docs = []
    pending_metas = []
    replace_paths = set()

    if files_to_process and workers <= 1:
        for args in tqdm(files_to_process, desc=f"  {name} extract", unit="file"):
            result = _process_file(args)
            filepath = result["filepath"]
            if result["skipped"]:
                stats["unsupported"] += 1
            elif result["error"]:
                stats["errors"] += 1
                print(f"  WARN: Failed to process {filepath}: {result['error']}")
            elif result["ids"]:
                pending_ids.extend(result["ids"])
                pending_docs.extend(result["documents"])
                pending_metas.extend(result["metadatas"])
                stats["processed"] += 1
                if filepath in comparison_manifest:
                    stats["updated"] += 1
                    replace_paths.add(filepath)
                else:
                    stats["new"] += 1
    elif files_to_process:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_file, args): args[0] for args in files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  {name} extract", unit="file"):
                result = future.result()
                filepath = result["filepath"]
                if result["skipped"]:
                    stats["unsupported"] += 1
                elif result["error"]:
                    stats["errors"] += 1
                    print(f"  WARN: Failed to process {filepath}: {result['error']}")
                elif result["ids"]:
                    pending_ids.extend(result["ids"])
                    pending_docs.extend(result["documents"])
                    pending_metas.extend(result["metadatas"])
                    stats["processed"] += 1
                    if filepath in comparison_manifest:
                        stats["updated"] += 1
                        replace_paths.add(filepath)
                    else:
                        stats["new"] += 1

    # Phase 2: Pre-compute embeddings with fastembed (large batch, much faster)
    total_chunks = len(pending_ids)
    if total_chunks == 0:
        print(f"  Collection '{name}': {stats['processed']} files processed, "
              f"0 chunks upserted ({stats['unchanged']} unchanged, "
              f"{stats['pruned_files']} pruned, {stats['errors']} errors)")
        return stats

    print(f"  Embedding {total_chunks} chunks with fastembed...")
    from fastembed import TextEmbedding
    from embedding import MODEL_NAME
    model = TextEmbedding(model_name=MODEL_NAME)
    all_embeddings = list(tqdm(
        model.embed(pending_docs, batch_size=64),
        total=total_chunks, desc=f"  {name} embed", unit="chunk"
    ))
    all_embeddings = [e.tolist() for e in all_embeddings]

    if replace_paths:
        print(f"  Removing old chunks for {len(replace_paths)} changed files...")
        for filepath in tqdm(sorted(replace_paths), desc=f"  {name} replace", unit="file"):
            collection.delete(where={FILEPATH_KEY: filepath})

    # Phase 3: Upsert with pre-computed embeddings (no re-embedding by ChromaDB)
    print(f"  Upserting {total_chunks} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, total_chunks, BATCH_SIZE),
                  desc=f"  {name} upsert", unit="batch"):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        collection.upsert(
            ids=pending_ids[i:batch_end],
            documents=pending_docs[i:batch_end],
            metadatas=pending_metas[i:batch_end],
            embeddings=all_embeddings[i:batch_end],
        )
        stats["chunks"] += batch_end - i

    print(f"  Collection '{name}': {stats['processed']} files processed "
          f"({stats['new']} new, {stats['updated']} updated), "
          f"{stats['chunks']} chunks upserted, {stats['unchanged']} unchanged, "
          f"{stats['pruned_files']} pruned, {stats['unsupported']} unsupported, "
          f"{stats['errors']} errors")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG knowledge base")
    parser.add_argument("collections", nargs="*",
                        help="Collection names to ingest (default: all)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Force re-index (deletes and rebuilds collections)")
    parser.add_argument("--prune", action="store_true",
                        help="Remove chunks for files no longer found in configured sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned ingest/prune changes without writing anything")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel extraction workers (use 1 for serial ingest)")
    parser.add_argument("--list", action="store_true", help="List collections and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = get_client(cfg)

    if args.list:
        for name, col_cfg in cfg["collections"].items():
            try:
                ef = get_embedding_function()
                col = client.get_collection(name, embedding_function=ef)
                count = col.count()
            except Exception:
                count = 0
            print(f"  {name}: {col_cfg['description']} ({count} chunks)")
        return

    targets = args.collections or list(cfg["collections"].keys())

    for name in targets:
        if name not in cfg["collections"]:
            print(f"Unknown collection: {name}")
            continue
        print(f"\nIngesting '{name}'...")
        ingest_collection(client, name, cfg["collections"][name],
                          force=args.force, prune=args.prune, dry_run=args.dry_run,
                          workers=max(1, args.workers))

    print("\nDone.")


if __name__ == "__main__":
    main()
