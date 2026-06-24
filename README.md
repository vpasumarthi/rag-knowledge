# rag-knowledge

**Personal RAG over ~700 research papers**, with a Claude MCP integration for grounded retrieval during writing and analysis.

A local retrieval-augmented knowledge base over the author's research corpus — published work, manuscript drafts, and the literature accumulated through PhD and three postdoc appointments.

## What's in it

Three ChromaDB collections, separated by purpose:

| Collection | Contents | ~Chunks |
|---|---|---|
| `my-work` | Published papers, profile, CV, resume | 957 |
| `active-research` | Manuscript drafts, analysis notes, project data | 727 |
| `literature` | ~700 PDFs from Zotero + EndNote (PhD → postdoc) | 83,861 |

Embedding search via ChromaDB, with a cross-encoder reranker on top for relevance quality. Exposed as a Claude MCP server (`mcp_server.py`) so Claude Code can retrieve directly during conversation, plus a CLI (`query.py`) for terminal use and a Python API for scripts/notebooks.

## Why three collections (and not one)

The retrieval need differs enough across collections that mixing them costs precision:

- A question about *my own prior work* (`my-work`) should not be diluted by surrounding literature.
- A question about *what the field says* (`literature`) should not be polluted by in-progress drafts.
- A question about *an active manuscript* (`active-research`) is scoped to that project, often with a `--project` filter on top.

Keeping them separate trades one extra routing decision for a much cleaner result set per query.

## How it's used

Daily, in three places:

- **Writing manuscripts** — grounded recall of methods/results from prior papers without re-reading PDFs.
- **Drafting cover letters and applications** — citing my own work accurately, with chunk-level references rather than recollection.
- **Reviewing literature** — semantic search across a 700-paper corpus that's otherwise effectively unindexed.

The MCP integration is the highest-leverage piece: Claude can pull grounded text from the corpus mid-conversation rather than hallucinating citations.

## Quick start

See `docs/usage.md` for the full guide. Short version:

```bash
conda activate rag
cd ~/rag-knowledge

# Search
python query.py search my-work "free energy barriers"
python query.py search literature "MLIP transition states" --era postdoc

# Re-index after adding new papers
python ingest.py literature
```

In Claude Code, the MCP server is registered globally — just ask naturally and the `search_knowledge` tool is invoked automatically.

## Repo layout

```
rag-knowledge/
  ingest.py        chunking + embedding + upsert into ChromaDB
  query.py         search/list/stats/delete CLI + Python API
  embedding.py     embedding function setup
  mcp_server.py    MCP server for Claude integration
  config.yaml      collection definitions, source paths, metadata tags
  docs/usage.md    complete usage guide
```

## Acknowledgment

Architecture, collection structure, MCP-vs-CLI integration choice, and metadata schema are design decisions by the author. Implementation was built in collaboration with AI coding assistants (Claude Code). This is an AI-augmented personal project — design intent is the contribution; code is its realization.
