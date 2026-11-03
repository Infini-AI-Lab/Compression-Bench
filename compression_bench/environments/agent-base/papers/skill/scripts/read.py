#!/usr/bin/env python3
"""Read one paper of the collection.

    read.py ID                 the paper's catalogue entry and its first page
    read.py ID --page N        page N of the text extract
    read.py ID --pages         the number of pages, and each page's size and first line
    read.py ID --grep PATTERN [--context K] [--max-matches M]
                               lines matching PATTERN (a case-insensitive regular
                               expression), with K lines of context (default 2)
    read.py ID --pdf           the path of the paper's PDF

ID is the id search.py prints (an arXiv id YYMM.NNNNN, with or without an
`arXiv:` prefix or a version suffix, or the catalogue's own id). A page is one page of the text extract;
a longer page, or an extract without page breaks, is cut at line ends into
parts of at most 12000 characters, numbered as pages.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperlib  # noqa: E402


def header(paper: dict) -> str:
    return (f"{paper['id']}  {paper['title']}\n"
            f"year {paper.get('year') or '-'} | categories: {', '.join(paper['categories'])} | {paper['url']}")


def show_page(paper: dict, pages: list[str], number: int) -> int:
    total = len(pages)
    if not 1 <= number <= total:
        print(f"{paper['id']} has pages 1-{total}", file=sys.stderr)
        return 2
    print(f"== {paper['id']} page {number} of {total}")
    print(pages[number - 1])
    nxt = f"; next: --page {number + 1}" if number < total else ""
    print(f"== end of page {number} of {total}{nxt}")
    return 0


def grep(paper: dict, pages: list[str], pattern: str, context: int, limit: int) -> int:
    try:
        regex = re.compile(pattern, re.I)
    except re.error:
        regex = re.compile(re.escape(pattern), re.I)
    blocks, matches = [], 0
    for number, page in enumerate(pages, 1):
        lines = page.split("\n")
        hits = [i for i, line in enumerate(lines) if regex.search(line)]
        matches += len(hits)
        spans: list[list[int]] = []
        for i in hits:
            lo, hi = max(0, i - context), min(len(lines) - 1, i + context)
            if spans and lo <= spans[-1][1] + 1:
                spans[-1][1] = hi
                spans[-1][2].append(i)
            else:
                spans.append([lo, hi, [i]])
        for lo, hi, marked in spans:
            body = [(">" if i in marked else " ") + " " + " ".join(lines[i].split())
                    for i in range(lo, hi + 1)]
            blocks.append(f"-- page {number}, line {lo + 1}\n" + "\n".join(body))
    print(f"== {paper['id']}: {matches} matching lines in {len(blocks)} passages for /{pattern}/"
          + (f" (first {limit} shown)" if len(blocks) > limit else ""))
    for block in blocks[:limit]:
        print(block)
    return 0 if matches else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("id")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--page", type=int, metavar="N")
    mode.add_argument("--pages", action="store_true")
    mode.add_argument("--grep", metavar="PATTERN")
    mode.add_argument("--pdf", action="store_true")
    parser.add_argument("--context", type=int, default=2, metavar="K")
    parser.add_argument("--max-matches", type=int, default=20, metavar="M",
                        help="passages to print with --grep (default 20)")
    args = parser.parse_args(argv)
    base = paperlib.root()
    paper = paperlib.find(paperlib.load_catalog(base), args.id)
    if paper is None:
        print(f"no paper with id {args.id!r}; search.py prints the ids", file=sys.stderr)
        return 2
    if args.pdf:
        print(base / paper["files"]["pdf"])
        return 0
    pages = paperlib.pages(paperlib.read_text(paper, base))
    if args.pages:
        print(header(paper))
        print(f"{len(pages)} pages")
        for number, page in enumerate(pages, 1):
            first = next((" ".join(line.split()) for line in page.split("\n") if line.strip()), "")
            print(f"{number:4d}  {len(page):6d} chars  {first[:90]}")
        return 0
    if args.grep is not None:
        return grep(paper, pages, args.grep, max(args.context, 0), max(args.max_matches, 1))
    if args.page is None:
        print(header(paper))
    return show_page(paper, pages, args.page or 1)


if __name__ == "__main__":
    sys.exit(main())
