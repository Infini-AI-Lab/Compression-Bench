"""The paper collection in the agent image and the `compression-papers` skill that
searches it (compression_bench/environments/agent-base/papers/).

The container holds no third-party source code, no toolkit interpreter and no
quantizer binary; of the library only the papers are shipped, offered through a
skill whose scripts search and read them. Nothing the
agent sees may name, rank, highlight or favour a method: results are ordered by
query relevance only, ties by id.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "compression_bench" / "environments" / "agent-base" / "papers"
SKILL = PAPERS / "skill"
LIBRARY = REPO / "compression_bench" / "environments" / "agent-base" / "library"
EXCLUDE = REPO / "compression_bench" / "environments" / "agent-base" / "library-exclude.txt"

#: Method, format and recipe names (beyond the instruction test's own list) that
#: nothing the skill prints or states may carry.
METHOD_WORDS = re.compile(
    r"\b(gptq|gptaq|awq|autoawq|quip|qtip|aqlm|hqq|imatrix|smoothquant|spinquant|quarot|bitsandbytes|"
    r"nf4|fp4|fp8|int4|int8|marlin|gguf|exl2|exl3|vptq|sparsegpt|wanda|omniquant|squeezellm|autoround|"
    r"llmcompressor|llama\.cpp|i?q[2-8]_[a-z0-9_]+)\b|group[ -]size|\b\d+[- ]bits?\b", re.I)


def load(directory: Path, name: str):
    """Import a module of the image's build context without leaving a __pycache__ in it."""
    import importlib
    sys.path.insert(0, str(directory))
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = before


def page(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs) + "\n"


def make_library(root: Path) -> Path:
    """Four catalogue papers (one withheld by the guard, one not retrieved), two
    references entries (one the catalogue reuses, one it does not list) and a
    supplement of two arXiv papers (one the catalogue already holds). Returns
    the supplement's list."""
    def put(path: str, text: str) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        target.with_suffix(".pdf").write_bytes(b"%PDF-1.4 " + path.encode())

    same = page("Shared words about tensor sketches and their sketching budgets.")
    put("quantization/papers/arxiv-2401.00002.txt", same)
    put("quantization/papers/arxiv-2401.00001.txt", same)
    put("pruning/papers/arxiv-2402.00003.txt",
        page("Sketches of Matrices", "A heading.") + "\f" +
        page("     indented body line one", "     indented body line two mentions sketches") + "\f" +
        "x" * 30000)
    put("distillation/papers/openreview-AbC-.txt", page("Teacher and student, nothing else."))
    put("distillation/papers/arxiv-2403.00004.txt", page("Withheld text about sketches."))
    put("other/method/paper.txt", page("Reused reference copy about sketches."))
    put("other/albert/paper.txt", page("A paper on sharing parameters across layers."))
    put("supplement/arxiv-2404.00006.txt", page("Factored weights of a language model, one factor pair per layer."))
    put("supplement/arxiv-2401.00002.txt", page("The supplement's copy of a catalogue paper."))
    entries = [
        ("arxiv-2401.00001", "Paper A", 2024, ["Quantization"], "quantization/papers/arxiv-2401.00001"),
        ("arxiv-2401.00002", "Paper B", 2024, ["Quantization"], "quantization/papers/arxiv-2401.00002"),
        ("arxiv-2402.00003", "Sketches of Matrices", 2024, ["Pruning and Sparsity"], "pruning/papers/arxiv-2402.00003"),
        ("openreview-AbC-", "Teacher Student", 2023, ["Distillation"], "distillation/papers/openreview-AbC-"),
        ("arxiv-2403.00004", "Withheld", 2024, ["Distillation"], "distillation/papers/arxiv-2403.00004"),
        ("arxiv-2305.00005", "Reused", 2023, ["Other"], "references/../other/method/paper"),
    ]
    catalogue = [{"id": i, "title": t, "url": f"https://example.org/{i}", "year": y, "upstream_categories": c,
                  "pdf_path": f"{p}.pdf", "text_path": f"{p}.txt", "status": "available"}
                 for i, t, y, c, p in entries]
    catalogue.append({"id": "openreview-gone", "title": "Not retrieved", "url": "u", "year": 2023,
                      "upstream_categories": ["Other"], "pdf_path": "x.pdf", "text_path": "x.txt",
                      "status": "failed"})
    (root / "external/awesome-llm-compression").mkdir(parents=True)
    (root / "external/awesome-llm-compression/papers.json").write_text(json.dumps({"papers": catalogue}))
    references = [
        {"name": "method", "title": None, "url": "https://example.org/m", "categories": ["low_rank"],
         "pdf_path": "../other/method/paper.pdf", "text_path": "../other/method/paper.txt"},
        {"name": "albert", "title": None, "url": "https://example.org/albert", "categories": ["parameter_sharing"],
         "pdf_path": "../other/albert/paper.pdf", "text_path": "../other/albert/paper.txt"},
    ]
    (root / "references").mkdir()
    (root / "references/papers.json").write_text(json.dumps({"papers": references}))
    supplement = [{"arxiv_id": i, "version": "v1", "title": t, "year": 2024, "categories": c,
                   "abs_url": f"https://arxiv.org/abs/{i}v1", "pdf_url": f"https://arxiv.org/pdf/{i}v1"}
                  for i, t, c in (("2404.00006", "Factored Weights", ["Low-Rank Factorization"]),
                                  ("2401.00002", "Paper B again", ["Pruning and Sparsity"]))]
    listing = root.parent / "supplement.json"
    listing.write_text(json.dumps({"schema_version": 1, "source": "arXiv", "papers": supplement}))
    return listing


def build(library: Path, exclude: Path, skill: Path, out: Path, supplement: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PAPERS / "build_collection.py"), "--library", str(library),
                           "--exclude", str(exclude), "--skill", str(skill), "--out", str(out),
                           "--supplement", str(supplement)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("papers")
    library = tmp / "library"
    supplement = make_library(library)
    (library / "pruning/papers/upstream").mkdir(parents=True)
    exclude = tmp / "exclude.txt"
    exclude.write_text("# header\ndistillation/papers/arxiv-2403.00004.txt\t1 benchmark question(s)\n")
    skill = tmp / "skill"
    shutil.copytree(SKILL, skill, ignore=shutil.ignore_patterns("__pycache__"))
    text = (skill / "SKILL.md").read_text()
    (skill / "SKILL.md").write_text(re.sub(r"\d+ research papers", "7 research papers", text))
    out = tmp / "out"
    got = build(library, exclude, skill, out, supplement)
    assert got.returncode == 0, got.stderr
    assert "7 papers" in got.stdout and "1 withheld" in got.stdout
    return out


def run(collection: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(collection / "skill" / "scripts" / script), *args],
                          capture_output=True, text=True,
                          env={**os.environ, "COMPRESSION_PAPERS_ROOT": str(collection)})


def ids_of(output: str) -> list[str]:
    return re.findall(r"^\d+\. (\S+) ", output, re.M)


def test_the_collection_holds_papers_only_and_withholds_the_guarded(collection):
    catalog = json.loads((collection / "catalog.json").read_text())
    ids = [p["id"] for p in catalog["papers"]]
    assert ids == sorted(ids) == ["1909.11942", "2305.00005", "2401.00001", "2401.00002", "2402.00003",
                                  "2404.00006", "openreview-AbC-"]
    assert set(catalog["papers"][0]) == {"id", "title", "year", "categories", "url", "files"}
    albert = catalog["papers"][0]
    assert albert["categories"] == ["Other"] and albert["year"] == 2019 and albert["url"].endswith("/albert")
    # A supplement paper the catalogue already holds is the catalogue's.
    by_id = {p["id"]: p for p in catalog["papers"]}
    assert by_id["2401.00002"]["title"] == "Paper B" and by_id["2401.00002"]["categories"] == ["Quantization"]
    assert by_id["2404.00006"]["categories"] == ["Low-Rank Factorization"]
    assert by_id["2404.00006"]["url"] == "https://arxiv.org/abs/2404.00006v1"
    assert catalog["supplement"] == "https://arxiv.org"
    files = sorted(p.name for p in (collection / "papers").iterdir())
    assert files == sorted(f"{i}.{x}" for i in ids for x in ("pdf", "txt"))
    assert not any("2403.00004" in p.name for p in collection.rglob("*"))
    assert sorted(p.name for p in collection.iterdir()) == ["catalog.json", "index.sqlite", "papers", "skill"]
    assert not list(collection.rglob("upstream*")) and not list(collection.rglob("__pycache__"))


def test_the_builder_refuses_a_skill_that_misstates_the_count(tmp_path):
    library = tmp_path / "library"
    supplement = make_library(library)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("")
    got = build(library, exclude, SKILL, tmp_path / "out", supplement)
    assert got.returncode != 0 and "research papers" in got.stderr
    assert not (tmp_path / "out").exists()


def test_the_builder_refuses_a_missing_or_duplicate_supplement_paper(tmp_path):
    library = tmp_path / "library"
    supplement = make_library(library)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("")
    listing = json.loads(supplement.read_text())
    (library / "supplement/arxiv-2404.00006.pdf").unlink()
    got = build(library, exclude, SKILL, tmp_path / "out", supplement)
    assert got.returncode != 0 and "2404.00006: missing pdf" in got.stderr
    (library / "supplement/arxiv-2404.00006.pdf").write_bytes(b"%PDF-1.4")
    listing["papers"].append(dict(listing["papers"][0]))
    supplement.write_text(json.dumps(listing))
    got = build(library, exclude, SKILL, tmp_path / "out", supplement)
    assert got.returncode != 0 and "duplicate ids ['2404.00006', '2404.00006']" in got.stderr
    assert not (tmp_path / "out").exists()


def test_search_ranks_by_relevance_and_breaks_ties_by_id(collection):
    got = run(collection, "search.py", "sketches")
    assert got.returncode == 0, got.stderr
    ranked = ids_of(got.stdout)
    # The title match ranks first; two identical texts tie and come in id order.
    assert ranked[0] == "2402.00003"
    assert ranked.index("2401.00001") + 1 == ranked.index("2401.00002")
    assert "openreview-AbC-" not in ranked and "[sketches]" in got.stdout
    assert f"{len(ranked)} of {len(ranked)} matching papers shown" in got.stdout


def test_search_filters_phrases_and_operators(collection):
    assert ids_of(run(collection, "search.py", "sketches", "--category", "prun").stdout) == ["2402.00003"]
    assert ids_of(run(collection, "search.py", "sketches", "--since", "2024", "--limit", "1").stdout) == ["2402.00003"]
    assert set(ids_of(run(collection, "search.py", "teacher", "matrices").stdout)) == {"2402.00003", "openreview-AbC-"}
    assert ids_of(run(collection, "search.py", "teacher", "matrices", "--all").stdout) == []
    assert ids_of(run(collection, "search.py", '"student, nothing"').stdout) == ["openreview-AbC-"]
    assert ids_of(run(collection, "search.py", "parameters-sharing").stdout) == []      # a phrase, in order
    assert ids_of(run(collection, "search.py", "sharing-parameters").stdout) == ["1909.11942"]
    # FTS5 syntax in a query is text, never an operator or an error.
    for query in ('NOT AND', 'title:x', '"', 'a*', 'NEAR(a b)'):
        assert run(collection, "search.py", query).returncode in (0, 2), query
    listed = run(collection, "search.py", "--category", "quantization")
    assert ids_of(listed.stdout) == ["2401.00001", "2401.00002"]
    assert run(collection, "search.py", "x", "--category", "none-such").returncode == 2
    assert run(collection, "search.py").returncode == 2
    counts = run(collection, "search.py", "--categories").stdout.splitlines()
    assert counts[-1] == "7 papers"
    names = [line.split(None, 1)[1] for line in counts[:-1]]
    assert names == sorted(names, key=str.lower)


def test_read_pages_greps_and_names_the_pdf(collection):
    first = run(collection, "read.py", "arXiv:2402.00003v3")
    assert first.returncode == 0, first.stderr
    assert first.stdout.startswith("2402.00003  Sketches of Matrices\n")
    assert "== 2402.00003 page 1 of " in first.stdout and "next: --page 2" in first.stdout
    second = run(collection, "read.py", "2402.00003", "--page", "2").stdout
    assert "\nindented body line one\n" in second            # the page's left margin is removed
    pages = run(collection, "read.py", "2402.00003", "--pages").stdout
    total = int(re.search(r"^(\d+) pages$", pages, re.M).group(1))
    paperlib = load(SKILL / "scripts", "paperlib")
    assert total == 2 + -(-30000 // paperlib.PAGE_CHARS)       # a long page is cut into parts
    assert str(paperlib.PAGE_CHARS) in (SKILL / "SKILL.md").read_text()
    assert run(collection, "read.py", "2402.00003", "--page", str(total + 1)).returncode == 2
    grep = run(collection, "read.py", "2402.00003", "--grep", "SKETCH", "--context", "1")
    assert grep.returncode == 0
    assert "-- page 1, line 1" in grep.stdout and "> Sketches of Matrices" in grep.stdout
    assert "-- page 2" in grep.stdout and "> indented body line two mentions sketches" in grep.stdout
    assert run(collection, "read.py", "2402.00003", "--grep", "(unbalanced").returncode == 1
    pdf = run(collection, "read.py", "openreview-abc-", "--pdf").stdout.strip()
    assert pdf == str(collection / "papers" / "openreview-AbC-.pdf")
    assert run(collection, "read.py", "9999.99999").returncode == 2


def test_the_skill_md_is_a_claude_code_user_skill():
    text = (SKILL / "SKILL.md").read_text()
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md opens with YAML frontmatter"
    import yaml
    from lifecycle import workspace
    front = yaml.safe_load(match.group(1))
    assert front["name"] == workspace.PAPERS_SKILL_NAME == "compression-papers"
    assert workspace.PAPERS_SKILL == SKILL
    assert re.fullmatch(r"[a-z0-9-]{1,64}", front["name"])
    assert 0 < len(front["description"]) <= 1024
    assert "/opt/compression-papers/skill/scripts/search.py" in text
    assert "/opt/compression-papers/skill/scripts/read.py" in text


def test_nothing_the_skill_states_or_prints_names_or_favours_a_method(collection):
    import test_the_instruction_only_states_facts as facts
    help_text = [run(collection, script, "--help").stdout for script in ("search.py", "read.py")]
    sources = [p.read_text() for p in (SKILL / "scripts").glob("*.py")]
    for text in [(SKILL / "SKILL.md").read_text(), *help_text, *sources]:
        assert not [p for p in facts.METHODS if re.search(p, text)], text[:80]
        assert not [p for p, _ in facts.FORBIDDEN if re.search(p, text, re.I)], text[:80]
        assert not [s for s in facts.GATE_SECRETS if s in text.lower()], text[:80]
        assert not METHOD_WORDS.search(text), METHOD_WORDS.search(text).group(0)
        for word in ("recommend", "popular", "best", "start here", "state-of-the-art", "sota", "default method"):
            assert word not in text.lower(), word
        # No paper is named by its id as an example.
        assert not re.search(r"\b\d{4}\.\d{4,5}\b", text)


def test_the_skill_states_the_real_collections_size():
    if not (LIBRARY / "external/awesome-llm-compression/papers.json").is_file() or \
            not any((LIBRARY / "quantization" / "papers").glob("*.txt")):
        pytest.skip("library papers not restored on this host (prepare_references.py)")
    build_collection = load(PAPERS, "build_collection")
    excluded = {line.split("\t", 1)[0] for line in EXCLUDE.read_text().splitlines()
                if line and not line.startswith("#")}
    rows, withheld = build_collection.collect(LIBRARY, excluded)
    assert withheld == len({p.rsplit(".", 1)[0] for p in excluded})
    stated = set(re.findall(r"(\d+) research papers", (SKILL / "SKILL.md").read_text()))
    assert stated == {str(len(rows))}
    assert not {r["source"]["text"] for r in rows} & excluded
    assert not [r for r in rows if "upstream" in r["source"]["text"]]
