#!/usr/bin/env python3
"""Ingest documents into RAG knowledge base collections."""

import argparse
import hashlib
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

BATCH_SIZE = 64  # chunks per upsert batch


def _process_file(args: tuple) -> dict:
    """Process a single file into chunks + metadata. Runs in worker processes."""
    filepath, base_metadata, collection_name = args
    ext = os.path.splitext(filepath)[1].lower()
    file_meta = {**base_metadata, "filepath": filepath}
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
                      workers: int = 4) -> dict:
    """Ingest all sources into a collection with parallel extraction and batched upserts."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ef = get_embedding_function()

    if force:
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

    stats = {"files": 0, "chunks": 0, "skipped": 0, "errors": 0}

    # Gather all files across sources with their metadata
    all_files = []
    for source in col_config.get("sources", []):
        base_metadata = source.get("metadata", {})
        files = resolve_files(source)
        print(f"  Source: {os.path.expanduser(source['path'])} -> {len(files)} files")
        for f in files:
            all_files.append((f, base_metadata, name))

    # Phase 1: Parallel file extraction (CPU + I/O bound)
    print(f"  Extracting text from {len(all_files)} files ({workers} workers)...")
    pending_ids = []
    pending_docs = []
    pending_metas = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_file, args): args[0] for args in all_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  {name} extract", unit="file"):
            result = future.result()
            if result["skipped"]:
                stats["skipped"] += 1
            elif result["error"]:
                stats["errors"] += 1
            elif result["ids"]:
                pending_ids.extend(result["ids"])
                pending_docs.extend(result["documents"])
                pending_metas.extend(result["metadatas"])
                stats["files"] += 1

    # Phase 2: Pre-compute embeddings with fastembed (large batch, much faster)
    total_chunks = len(pending_ids)
    print(f"  Embedding {total_chunks} chunks with fastembed...")
    from fastembed import TextEmbedding
    from embedding import MODEL_NAME
    model = TextEmbedding(model_name=MODEL_NAME)
    all_embeddings = list(tqdm(
        model.embed(pending_docs, batch_size=64),
        total=total_chunks, desc=f"  {name} embed", unit="chunk"
    ))
    all_embeddings = [e.tolist() for e in all_embeddings]

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

    print(f"  Collection '{name}': {stats['files']} files, {stats['chunks']} chunks "
          f"({stats['skipped']} skipped, {stats['errors']} errors)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG knowledge base")
    parser.add_argument("collections", nargs="*",
                        help="Collection names to ingest (default: all)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Force re-index (deletes and rebuilds collections)")
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
        ingest_collection(client, name, cfg["collections"][name], force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
