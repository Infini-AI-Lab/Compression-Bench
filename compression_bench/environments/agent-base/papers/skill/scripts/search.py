#!/usr/bin/env python3
"""Search the paper collection's titles and full text.

    search.py QUERY [--limit N] [--category C] [--since YEAR] [--all]
    search.py --categories

Results are ranked by BM25 relevance to QUERY over title and full text (ties
by id). Each whitespace-separated word of QUERY is a term (a hyphenated word
"a-b" is the phrase "a b"); a double-quoted part is one phrase. A paper matches when it
holds any term, or every term with --all. --category keeps papers whose
catalogue category contains C (case-insensitive); --since keeps papers of that
year or later. With --category or --since and no QUERY, the matching papers
are listed by id.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperlib  # noqa: E402

#: Dropped from a query that has other terms: they match nearly every paper.
STOPWORDS = frozenset("a an and are as at be by can do does for from how in into is it its of on or "
                      "that the this to via we what when which with".split())


def fts_query(query: str, require_all: bool = False) -> str:
    terms = [re.findall(r"\w+", phrase.lower()) for phrase in re.findall(r'"([^"]*)"', query)]
    terms += [re.findall(r"\w+", chunk.lower()) for chunk in re.sub(r'"[^"]*"', " ", query).split()]
    terms = [t for t in terms if t]
    meaningful = [t for t in terms if not (len(t) == 1 and t[0] in STOPWORDS)]
    terms = meaningful or terms
    # Quoted, a term is never read as an FTS5 operator (AND, NOT, NEAR, column filters).
    return (" AND " if require_all else " OR ").join('"' + " ".join(t) + '"' for t in terms)


def ranked(db, query: str) -> list[tuple[int, str]]:
    weights = ", ".join(str(w) for w in paperlib.WEIGHTS)
    rows = db.execute(f"SELECT rowid, id, bm25({paperlib.TABLE}, {weights}) AS score FROM {paperlib.TABLE} "
                      f"WHERE {paperlib.TABLE} MATCH ? ORDER BY score, id", (query,)).fetchall()
    return [(rowid, paper_id) for rowid, paper_id, _ in rows]


def snippet(db, query: str, rowid: int) -> str:
    got = db.execute(f"SELECT snippet({paperlib.TABLE}, 2, '[', ']', ' ... ', 24) FROM {paperlib.TABLE} "
                     f"WHERE {paperlib.TABLE} MATCH ? AND rowid = ?", (query, rowid)).fetchone()
    return paperlib.compact(got[0]) if got and got[0] else ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("query", nargs="*", help="words or \"quoted phrases\"")
    parser.add_argument("--limit", type=int, default=10, help="results to print (default 10)")
    parser.add_argument("--category", help="keep papers whose category contains this text")
    parser.add_argument("--since", type=int, metavar="YEAR", help="keep papers of YEAR or later")
    parser.add_argument("--all", action="store_true", help="require every term")
    parser.add_argument("--categories", action="store_true", help="list the catalogue's categories")
    args = parser.parse_args(argv)
    base = paperlib.root()
    papers = {p["id"]: p for p in paperlib.load_catalog(base)}

    if args.categories:
        counts = Counter(c for p in papers.values() for c in p["categories"])
        for name in sorted(counts, key=str.lower):
            print(f"{counts[name]:5d}  {name}")
        print(f"{len(papers)} papers")
        return 0

    def kept(paper: dict) -> bool:
        if args.category and not any(args.category.lower() in c.lower() for c in paper["categories"]):
            return False
        return not (args.since and (paper.get("year") or 0) < args.since)

    if args.category and not any(args.category.lower() in c.lower()
                                 for p in papers.values() for c in p["categories"]):
        known = sorted({c for p in papers.values() for c in p["categories"]}, key=str.lower)
        print(f"no category contains {args.category!r}; categories: {', '.join(known)}", file=sys.stderr)
        return 2
    text = " ".join(args.query).strip()
    query = fts_query(text, args.all) if text else ""
    if text and not query:
        print("the query holds no searchable word", file=sys.stderr)
        return 2
    if not query and not (args.category or args.since):
        parser.print_usage(sys.stderr)
        return 2

    if query:
        db = paperlib.connect(base)
        hits = [(rowid, pid) for rowid, pid in ranked(db, query) if pid in papers and kept(papers[pid])]
    else:
        db, hits = None, [(0, pid) for pid in sorted(papers) if kept(papers[pid])]
    shown = hits[:max(args.limit, 0)]
    for rank, (rowid, pid) in enumerate(shown, 1):
        paper = papers[pid]
        print(f"{rank}. {pid} ({paper.get('year') or '-'}) {paper['title']}")
        print(f"   categories: {', '.join(paper['categories'])}")
        if db is not None:
            found = snippet(db, query, rowid)
            if found:
                print(f"   {found}")
    reader = Path(__file__).resolve().parent / "read.py"
    print(f"{len(shown)} of {len(hits)} matching papers shown. Text: python3 {reader} ID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
