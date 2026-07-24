---
title: "rag-knowledge: Local Retrieval and Grounding for Scientific Literature"
date: ""
documentclass: article
fontsize: 11pt
mainfont: "SourceSansPro-Regular.otf"
mainfontoptions:
  - BoldFont=SourceSansPro-Bold.otf
  - ItalicFont=SourceSansPro-RegularIt.otf
  - BoldItalicFont=SourceSansPro-BoldIt.otf
sansfont: "SourceSansPro-Regular.otf"
sansfontoptions:
  - BoldFont=SourceSansPro-Bold.otf
  - ItalicFont=SourceSansPro-RegularIt.otf
  - BoldItalicFont=SourceSansPro-BoldIt.otf
monofont: "Menlo"
geometry: margin=0.68in
colorlinks: true
linkcolor: blue
urlcolor: blue
linestretch: 1.05
header-includes:
  - \usepackage{microtype}
  - \usepackage{enumitem}
  - \usepackage{tikz}
  - \usepackage{titlesec}
  - \usepackage{ragged2e}
  - \usetikzlibrary{positioning}
  - \titlespacing*{\section}{0pt}{1.4ex plus .3ex minus .2ex}{0.6ex}
  - \makeatletter
  - '\renewcommand{\maketitle}{\begin{center}{\Large\bfseries rag-knowledge: Local Retrieval and Grounding\par for Scientific Literature\par}\end{center}\vspace{0.6em}}'
  - \makeatother
  - \setlist{nosep,leftmargin=*}
  - \microtypesetup{protrusion=true,expansion=true}
  - \AtBeginDocument{\setlength{\JustifyingParindent}{0pt}\justifying\setlength{\parindent}{0pt}\setlength{\emergencystretch}{1em}\tolerance=800\hyphenpenalty=200\exhyphenpenalty=500}
  - \setlength{\parindent}{0pt}
  - \setlength{\parskip}{3pt}
---

**Repository:** <https://github.com/vpasumarthi/rag-knowledge>

## Overview

`rag-knowledge` is a local retrieval and grounding system for a heterogeneous scientific research corpus. It preserves source traceability and makes the retrieval layer available through MCP-compatible agentic environments such as Codex and Claude Code, as well as command-line and Python interfaces.

A calling application selects the relevant collection, submits a query with optional metadata filters, and receives a limited set of reranked, source-linked passages. Planning, synthesis, and response generation are handled by the calling application.

## System architecture and implementation

\begin{center}
\begin{tikzpicture}[
  stage/.style={
    draw=black!45,
    rounded corners=2pt,
    fill=black!3,
    minimum height=9mm,
    inner xsep=4pt,
    font=\normalsize,
    align=center
  },
  flow/.style={->, semithick, draw=black!60},
  lane/.style={font=\normalsize\bfseries, anchor=east}
]
\node[lane] (ingest-label) {Ingestion};
\node[stage, right=4mm of ingest-label] (sources) {Sources};
\node[stage, right=4mm of sources] (parse) {Parse and segment};
\node[stage, right=4mm of parse] (embed) {Embeddings};
\node[stage, right=4mm of embed] (store) {ChromaDB};
\draw[flow] (sources) -- (parse);
\draw[flow] (parse) -- (embed);
\draw[flow] (embed) -- (store);

\node[lane, below=8mm of ingest-label] (query-label) {Retrieval};
\node[stage, right=4mm of query-label] (interface) {MCP / CLI / Python};
\node[stage, right=4mm of interface] (candidates) {Vector search};
\node[stage, right=4mm of candidates] (rerank) {Cross-encoder\\reranking};
\node[stage, right=4mm of rerank] (evidence) {Source-linked\\passages};
\draw[flow] (interface) -- (candidates);
\draw[flow] (candidates) -- (rerank);
\draw[flow] (rerank) -- (evidence);
\end{tikzpicture}
\end{center}

**Corpus organization.** The index contains approximately 87,000 passages, including material extracted from approximately 700 literature PDFs. It is divided into three ChromaDB collections covering published work, active research material, and broader scientific literature. Separating these sources allows retrieval to be scoped by purpose and prevents material with different roles from competing within the same result set.

**Document processing and provenance.** The ingestion pipeline supports PDF, LaTeX, Markdown, and plain-text sources. PDFs are processed at the page level, with longer pages divided further while retaining page metadata. LaTeX and Markdown sources use section or heading boundaries where available. Retrieved passages retain source metadata for verification.

**Two-stage retrieval.** Semantic search first retrieves a broader candidate set from ChromaDB. A cross-encoder then reranks those candidates before the highest-ranked passages are returned. The system uses `BAAI/bge-base-en-v1.5` for embeddings and `ms-marco-TinyBERT-L-2-v2` for reranking, with both models running locally through ONNX-based implementations.

**Interfaces.** A shared Python retrieval layer is exposed through an MCP server, a command-line interface, and direct Python access. This allows the same corpus to support MCP-compatible agentic environments, terminal workflows, notebooks, and custom analysis pipelines.

**Incremental ingestion.** Passage identifiers remain stable across updates, and a manifest tracks source changes. Unchanged documents are skipped, modified documents are reprocessed, and records associated with removed sources can be pruned without rebuilding the complete index. Ingestion and database updates are batched to keep updates practical at the current corpus scale.

## Current limitations

- Retrieval operates at the passage level and is less effective for questions requiring whole-document synthesis.
- Query rewriting and automatic collection routing are not implemented.
- Page-level PDF processing can be coarse for dense two-column documents.
- Bibliography pages are identified during ingestion but are not yet treated differently during ranking.
- Evaluation is currently qualitative rather than based on a labeled retrieval benchmark.
