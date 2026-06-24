---
summary: "Complete usage guide for the RAG knowledge base — querying, adding, removing, re-indexing across all interfaces"
read_when:
  - "using the RAG knowledge base"
  - "searching personal knowledge base"
  - "adding or removing documents from collections"
  - "re-indexing the knowledge base"
---

# RAG Knowledge Base — Usage Guide

Your knowledge base has 3 collections:

| Collection | What's in it | Use when |
|---|---|---|
| `my-work` | Profile, CV, resumes, 8 published papers + SI | Writing applications, cover letters, checking your own publication details |
| `active-research` | 5 manuscript repos (drafts, notes, data) | Working on manuscripts, cross-referencing across projects |
| `literature` | 697 PDFs from Zotero + EndNote (PhD through postdocs) | Finding papers you've read, checking what the literature says about a topic |

---

## 1. Claude Code (MCP — automatic)

The MCP server is registered globally. In any Claude Code session, you can just ask naturally and Claude will use the tools behind the scenes.

### Asking Claude directly

```
"What did my 2019 BiVO4 paper find about charge transport?"
"Find papers in my library about MLIP for transition states"
"What activation barriers did we report in manuscript 01?"
"Which of my publications used CatMAP?"
```

Claude will call `search_knowledge` automatically when it needs grounded information.

### Explicit tool requests

You can also be explicit about which collection to search:

```
"Search my-work for CatEnergy tool"
"Search active-research for NEB calculations in project project-a"
"Search literature for constant potential DFT methods"
```

### Available MCP tools (Claude Code only)

These are MCP tools — they only exist inside Claude Code sessions where the MCP server runs. You never call them directly; Claude calls them behind the scenes when your questions need grounded information. Outside Claude Code, use the terminal CLI or Python API (sections 2 and 3 below).

| Tool | What it does |
|---|---|
| `search_knowledge` | Semantic search with optional filters |
| `list_collections` | Show all collections with chunk counts |
| `collection_stats` | Metadata breakdown for a collection |
| `add_document` | Add new text to a collection |
| `remove_documents` | Remove by project, source, or filename |

---

## 2. Terminal (CLI)

Activate the conda env first:

```bash
conda activate rag
cd ~/rag-knowledge
```

### Search

```bash
# Basic search (returns top 5 results with reranking)
python query.py search my-work "free energy barriers"

# More results
python query.py search literature "machine learning interatomic potentials" -n 10

# Filter by metadata
python query.py search active-research "NEB transition state" --project project-a
python query.py search literature "bismuth vanadate" --era phd
python query.py search literature "water splitting" --source zotero

# Disable reranking (faster, less accurate)
python query.py search literature "proton transport" --no-rerank
```

### Available filters

| Flag | Values | Works on |
|---|---|---|
| `--project` | `project-a`, `project-b`, `project-c`, `project-d`, `project-e` | active-research |
| `--era` | `phd`, `purdue`, `postdoc` | active-research, literature |
| `--source` | `zotero`, `endnote-phd` | literature |

### List collections

```bash
python query.py list
```

Output:
```
  my-work: Published work, profile, CV, resume (957 chunks)
  active-research: Manuscript drafts, analysis notes, research data (727 chunks)
  literature: Academic literature from PhD through postdocs (83861 chunks)
```

### Collection stats (metadata breakdown)

```bash
python query.py stats active-research
```

Shows chunk counts grouped by project, era, stage, etc.

### Delete documents

```bash
# Remove all S-BVO chunks (e.g., after manuscript is published)
python query.py delete active-research --project project-b

# Remove all EndNote PhD papers
python query.py delete literature --source endnote-phd
```

---

## 3. Python (import in scripts, notebooks, REPL)

```python
# Add the repo to your path (or run from ~/rag-knowledge/)
import sys
sys.path.insert(0, "/path/to/rag-knowledge")

from query import search, list_collections, delete_documents
```

### Search

```python
# Basic search — returns list of dicts
results = search("my-work", "activation barriers for ADH")

for r in results:
    print(r["metadata"]["filename"], r["rerank_score"])
    print(r["text"][:200])
    print()
```

Each result dict has:
```python
{
    "text": "the matched chunk text...",
    "metadata": {
        "filename": "Main Text.pdf",
        "page": 3,              # for PDFs
        "section": "Results",   # for .tex files
        "header": "Publications", # for .md files
        "project": "project-a", # for active-research
        "source": "zotero",     # for literature
        "era": "purdue",
        "is_bibliography": False,
        "filepath": "/full/path/to/file",
    },
    "distance": 0.5248,         # embedding distance (lower = closer)
    "rerank_score": 0.9585,     # cross-encoder relevance (higher = better)
}
```

### Search with filters

```python
# Only search manuscript 01
results = search("active-research", "entropy contributions",
                 where={"project": "project-a"})

# Only PhD-era literature
results = search("literature", "band gap calculations",
                 where={"era": "phd"})

# More results, no reranking
results = search("literature", "DFT+U", n_results=20, rerank=False)
```

### Delete

```python
delete_documents("active-research", where={"project": "project-b"})
```

---

## 4. Adding new documents

### Via Claude Code

```
"Add this text to my-work: [paste text]"
```

Claude calls `add_document` — it chunks and embeds automatically.

### Via terminal (re-index a collection)

If you've added new papers to Zotero or updated manuscript files:

```bash
conda activate rag
cd ~/rag-knowledge

# Incremental ingest: skips unchanged files, adds new files, refreshes changed files
python ingest.py active-research

# Also remove chunks for files that disappeared from configured sources
python ingest.py literature --prune

# Force full rebuild (deletes and re-creates — use after changing config)
python ingest.py literature --force
```

Normal ingest compares each discovered filepath against stored file metadata
(`file_size`, `file_mtime_ns`, and source metadata hash). Unchanged files are
not re-extracted or re-embedded. The first run after upgrading from older
indexes backfills this metadata onto existing chunks when source metadata still
matches, so a Zotero add-only update does not require rebuilding the full
literature collection.

### Via Python

```python
from query import load_config, get_client
from embedding import get_embedding_function
from ingest import chunk_text, stable_id

cfg = load_config()
client = get_client(cfg)
ef = get_embedding_function()
col = client.get_collection("my-work", embedding_function=ef)

# Add a single document
text = "Your new document text here..."
chunks = chunk_text(text)
for i, chunk in enumerate(chunks):
    col.upsert(
        ids=[f"manual-doc-{i}"],
        documents=[chunk],
        metadatas=[{"filename": "new_paper.pdf", "type": "published-paper"}],
    )
```

---

## 5. Understanding results

### Distance vs rerank score

Every result has two relevance measures:

- **distance** (0 to 2): Embedding cosine distance. Lower = semantically closer. Computed by the vector search. Fast but coarse — finds topically related chunks.
- **rerank_score** (0 to 1): Cross-encoder relevance score. Higher = more relevant to your specific query. Computed after retrieval. Slower but much more accurate.

**Trust the rerank_score more than distance.** A chunk with distance=0.6 and rerank=0.95 is more relevant than one with distance=0.4 and rerank=0.02.

### When results seem off

- **Getting bibliography hits?** The reranker should push them down. If not, try a more specific query.
- **Missing a paper you know is there?** Try different phrasing. Semantic search matches meaning, not exact words. "proton hopping in BiVO4" and "charge transport in bismuth vanadate" both work.
- **Too many results from one source?** Use metadata filters: `--project`, `--source`, `--era`.
- **Want exact keyword match?** Current system is semantic-only. For exact string matching, use `grep` on the source files directly.

---

## 6. Maintenance

### Config file

`~/rag-knowledge/config.yaml` — defines collections, source paths, and metadata tags. Edit this to add new sources or change metadata.

### Data location

- ChromaDB data: `~/.local/share/rag-knowledge/chroma/`
- Do NOT put this on a cloud-synced folder (corrupts SQLite)
- To reset everything: delete that directory and re-run `python ingest.py --force`

### Conda environment

```bash
conda activate rag   # Python 3.12, all dependencies pre-installed
```

### Lifecycle of a manuscript

1. Start writing → chunks are in `active-research` (tagged with project name)
2. Manuscript published → remove from active-research, add published PDF to my-work
   ```bash
   python query.py delete active-research --project project-b
   # Copy published PDF to Box > My Publications, then:
   python ingest.py my-work --force
   ```
3. Add to Zotero → re-index literature
   ```bash
   python ingest.py literature
   ```
   Use `--prune` only when you also want to remove chunks for PDFs that were
   deleted or moved out of the configured Zotero/EndNote sources.
