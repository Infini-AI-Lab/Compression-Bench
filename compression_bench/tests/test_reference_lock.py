"""The agent image's paper library: untracked, listed paper by paper with its source.

`/opt/compression-papers` is built from the papers in `compression_bench/environments/agent-base/library/`.
They are gitignored; `compression_bench/environments/reference-lock.json` (the
catalogue and curated papers) and `agent-base/papers/supplement.json` (the arXiv
supplement) name each one and its publisher's URL, and `prepare_references.py`
downloads them and makes their text extracts. These hold the two lists, the
restore and the ignore rules to the same set of files."""
import importlib.util
import io
import json
import posixpath
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "compression_bench" / "environments" / "reference-lock.json"
LIBRARY = REPO / "compression_bench" / "environments" / "agent-base" / "library"
SUPPLEMENT = REPO / "compression_bench" / "environments" / "agent-base" / "papers" / "supplement.json"
PDF = (b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>"
       b"endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 100]/Contents 4 0 R/Resources<</Font<<"
       b"/F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 10 50 Td (Pinned words) Tj ET\n"
       b"endstream endobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R>>\n")


def load(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_references", REPO / "compression_bench/scripts/deploy/prepare_references.py")


def untracked(path: str) -> bool:
    """The .gitignore rules for the library: **/papers/, **/*.pdf, **/paper.txt, supplement/."""
    parts = PurePosixPath(path).parts
    return "papers" in parts[:-1] or path.endswith(".pdf") or parts[-1] == "paper.txt" or parts[0] == "supplement"


def library_files(papers) -> list[str]:
    return [name for p in papers for name in (p["pdf"], prepare.text_of(p["pdf"]))]


def git_ignored(names) -> list[str]:
    return subprocess.run(["git", "-C", str(REPO), "check-ignore", "--stdin"],
                          input="\n".join(f"{LIBRARY.relative_to(REPO)}/{n}" for n in names),
                          capture_output=True, text=True).stdout.split()


def test_the_lock_names_every_paper_and_its_source():
    papers = prepare.load_lock(LOCK)
    assert prepare.DEFAULT_LOCK == LOCK and len(papers) > 1000
    assert all(untracked(n) for n in library_files(papers))
    assert len(git_ignored(library_files(papers))) == 2 * len(papers), "a library paper is not gitignored"
    for paper in papers:
        if "arxiv.org" in paper["url"]:
            assert re.fullmatch(r"https://arxiv\.org/pdf/\d{4}\.\d{4,5}v\d+", paper["url"]), paper["url"]


def test_the_catalogue_and_references_are_the_locked_papers():
    """Every paper the collection reads (build_collection.py) is a paper the lock downloads."""
    locked = {p["pdf"] for p in prepare.load_lock(LOCK)}
    catalogue = json.loads((LIBRARY / "external/awesome-llm-compression/papers.json").read_text())["papers"]
    wanted = {posixpath.normpath(p["pdf_path"]) for p in catalogue if p["status"] == "available"}
    references = json.loads((LIBRARY / "references/papers.json").read_text())["papers"]
    wanted |= {posixpath.normpath(posixpath.join("references", p["pdf_path"])) for p in references}
    assert wanted == locked


def test_every_untracked_library_file_on_disk_is_listed():
    listed = set(library_files(prepare.load_lock(LOCK) + prepare.load_supplement(SUPPLEMENT)))
    on_disk = [p.relative_to(LIBRARY).as_posix() for p in LIBRARY.rglob("*") if p.is_file()]
    stray = [p for p in on_disk if untracked(p) and p not in listed]
    assert not stray, f"untracked and unlisted (would be lost from a fresh clone): {stray[:5]}"


def test_the_supplement_is_valid_ignored_and_disjoint_from_the_lock():
    papers = prepare.load_supplement(SUPPLEMENT)
    assert prepare.DEFAULT_SUPPLEMENT == SUPPLEMENT and papers
    names = library_files(papers)
    assert all(untracked(n) for n in names)
    assert not set(names) & set(library_files(prepare.load_lock(LOCK)))
    assert len(git_ignored(names)) == len(names), "a supplement file is not gitignored"
    data = json.loads(SUPPLEMENT.read_text())
    assert data["source"] == "arXiv" and data["queries"]
    for entry in data["papers"]:
        assert entry["abs_url"] == f"https://arxiv.org/abs/{entry['arxiv_id']}{entry['version']}"
        assert entry["categories"] and set(entry["categories"]) <= set(data["queries"])


def test_the_restore_downloads_extracts_and_keeps_what_is_present(tmp_path, monkeypatch):
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext (poppler) not installed")
    papers = [{"pdf": "quantization/papers/arxiv-2401.00001.pdf", "url": "https://arxiv.org/pdf/2401.00001v2"},
              {"pdf": "quantization/m/paper.pdf", "url": "https://proceedings.example.org/m.pdf"}]
    served, body = [], [PDF]

    def fake_urlopen(request, timeout):
        served.append(request.full_url)
        return io.BytesIO(body[0])
    monkeypatch.setattr(prepare.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prepare, "REQUEST_INTERVAL_S", 0)
    library = tmp_path / "library"
    library.mkdir()

    with pytest.raises(ValueError, match="missing"):
        prepare.restore(papers, library, verify=True)
    assert not served
    assert prepare.restore(papers, library) == 2
    assert served == [p["url"] for p in papers]
    for paper in papers:
        assert (library / paper["pdf"]).read_bytes() == PDF
        assert b"Pinned words" in (library / prepare.text_of(paper["pdf"])).read_bytes()
    prepare.restore(papers, library, verify=True)                        # all present: nothing to do
    prepare.restore(papers, library)
    assert len(served) == 2

    text = library / prepare.text_of(papers[0]["pdf"])
    text.unlink()                                                        # a missing extract is remade
    prepare.restore(papers, library)
    assert len(served) == 2 and b"Pinned words" in text.read_bytes()

    (library / papers[1]["pdf"]).unlink()                                # a response that is not a PDF
    body[0] = b"<html>moved</html>"
    with pytest.raises(ValueError, match="could not be downloaded"):
        prepare.restore(papers, library)
    assert not (library / papers[1]["pdf"]).exists()
    assert not [p for p in (library / "quantization" / "m").iterdir() if p.name.startswith(".")]


def test_the_lists_refuse_unsafe_or_insecure_entries(tmp_path):
    for paper, error in (({"pdf": "../x.pdf", "url": "https://a.org/x.pdf"}, "Unsafe"),
                         ({"pdf": "a/x.pdf", "url": "http://a.org/x.pdf"}, "https"),
                         ({"pdf": "a/x.txt", "url": "https://a.org/x.pdf"}, "Invalid")):
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({"schema_version": 2, "papers": [paper]}))
        with pytest.raises(ValueError, match=error):
            prepare.load_lock(lock)
