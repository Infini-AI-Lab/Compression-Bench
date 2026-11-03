---
name: compression-papers
description: Search and read an offline collection of 1394 research papers on compressing large language models across techniques, from the Awesome-LLM-Compression catalogue plus a supplement of arXiv papers (categories, alphabetically - Distillation, Efficient Prompting, KV Cache Compression, Layer and Depth Reduction, Lossless and Entropy Coding, Low-Rank Factorization, Mixture-of-Experts Compression, Other, Pruning and Sparsity, Quantization, Survey, Weight Sharing and Tensor Decomposition). Full-text search ranked by relevance to a query; a paper's text page by page, the passages matching a pattern, or its PDF.
---

# Compression papers

An offline, read-only collection of 1394 research papers on compressing large
language models across techniques: the papers of the Awesome-LLM-Compression
catalogue (https://github.com/HuangOwen/Awesome-LLM-Compression), a few related
papers, and a supplement of papers from arXiv, each as a PDF and its text
extract. A paper's categories are the catalogue's section names, or for a
supplement paper names of the same kind given to it when it was added; all of
them, alphabetically: Distillation, Efficient Prompting, KV Cache Compression,
Layer and Depth Reduction, Lossless and Entropy Coding, Low-Rank Factorization,
Mixture-of-Experts Compression, Other, Pruning and Sparsity, Quantization,
Survey, Weight Sharing and Tensor Decomposition. A paper's year is that of its
arXiv id where it has one.

    /opt/compression-papers/catalog.json      one row per paper: id, title, year, categories, url, files
    /opt/compression-papers/papers/ID.pdf     the paper
    /opt/compression-papers/papers/ID.txt     its text extract
    /opt/compression-papers/index.sqlite      the full-text index (SQLite FTS5)

An ID is the paper's arXiv id (`YYMM.NNNNN`), or the catalogue's own id for a
paper without one (`openreview-...`, `aclanthology-...`).

## Search

    python3 /opt/compression-papers/skill/scripts/search.py "QUERY" [--limit N] [--category C] [--since YEAR] [--all]
    python3 /opt/compression-papers/skill/scripts/search.py --categories

Results are ordered by BM25 relevance to QUERY over the title and the full
text, and by id where relevance is equal; no other ordering is applied. Each
word of QUERY is a term (a hyphenated word is a phrase), and a double-quoted
part is one phrase; a paper matches when it holds any term, or every term with
`--all`. `--category C` keeps papers whose category contains C
(case-insensitive), `--since YEAR` papers of that year or later; with either
and no QUERY, the matching papers are listed by id. Each result prints its id,
year, title, categories and a passage with the matched terms in [brackets].
`--limit` defaults to 10. `--categories` prints every category with its number
of papers.

## Read

    python3 /opt/compression-papers/skill/scripts/read.py ID                  catalogue entry and page 1
    python3 /opt/compression-papers/skill/scripts/read.py ID --page N         page N
    python3 /opt/compression-papers/skill/scripts/read.py ID --pages          page count, each page's size and first line
    python3 /opt/compression-papers/skill/scripts/read.py ID --grep PATTERN [--context K] [--max-matches M]
    python3 /opt/compression-papers/skill/scripts/read.py ID --pdf            the PDF's path

A page is one page of the text extract; a longer page, or an extract without
page breaks, is cut at line ends into parts of at most 12000 characters,
numbered as pages. `--grep` takes a case-insensitive regular expression and
prints each matching line (marked `>`) with K lines of context (default 2),
labelled by page and line; `--max-matches` (default 20) bounds the passages
printed.

The scripts use only the Python standard library and run with any `python3`.
