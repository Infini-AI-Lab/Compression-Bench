import hashlib
import json

from lifecycle import boundary


def make_submission(root):
    (root / "model").mkdir(parents=True)
    (root / "inference.py").write_text("INFERENCE_API_VERSION = 1\n")
    (root / "model" / "config.json").write_text('{"a": 1}')
    (root / "model" / "w.safetensors").write_bytes(b"\0" * 1024)
    return root


def test_capture_records_full_sha256_and_copies_only_small_code(tmp_path):
    sub = make_submission(tmp_path / "submission")
    lineage = boundary.Lineage(tmp_path / "proto", "L1")
    rec = lineage.capture(sub, note="first", run_id="r1", source="agent", background=False)
    assert rec["version"] == "v1" and rec["files"] == 3 and rec["total_bytes"] == 1024 + 26 + 8
    version = tmp_path / "proto" / "L1" / "L1_v1"
    assert (version / "files" / "inference.py").exists()
    assert (version / "files" / "model" / "config.json").exists()
    assert not (version / "files" / "model" / "w.safetensors").exists()
    manifest = json.loads((version / "MANIFEST.json").read_text())
    shard = next(e for e in manifest if e["path"] == "model/w.safetensors")
    assert shard["sha256"] == hashlib.sha256(b"\0" * 1024).hexdigest()
    done = json.loads((version / "VERSION.json").read_text())
    assert done["manifest_status"] == "complete" and done["manifest_identity"]
    assert [row["version"] for row in lineage.versions()] == ["v1"]


def test_unchanged_files_are_not_reread(tmp_path, monkeypatch):
    sub = make_submission(tmp_path / "submission")
    lineage = boundary.Lineage(tmp_path / "proto", "L")
    lineage.capture(sub, note="a", run_id="r", source="agent", background=False)
    calls = []
    real = boundary.sha256_file
    monkeypatch.setattr(boundary, "sha256_file", lambda p: calls.append(p.name) or real(p))
    (sub / "inference.py").write_text("INFERENCE_API_VERSION = 1\n# v2\n")
    lineage.capture(sub, note="b", run_id="r", source="agent", background=False)
    assert calls == ["inference.py"]


def test_identity_matches_identical_trees_and_differs_on_change(tmp_path):
    sub = make_submission(tmp_path / "s")
    m1 = boundary.hash_listing(sub, boundary.listing(sub))
    m2 = boundary.hash_listing(sub, boundary.listing(sub))
    assert boundary.identity_of(m1) == boundary.identity_of(m2)
    (sub / "model" / "w.safetensors").write_bytes(b"\1" * 1024)
    m3 = boundary.hash_listing(sub, boundary.listing(sub))
    assert boundary.identity_of(m3) != boundary.identity_of(m1)


def test_symlinks_are_recorded_not_followed(tmp_path):
    sub = make_submission(tmp_path / "s")
    (sub / "link").symlink_to("/etc/passwd")
    entries = boundary.listing(sub)
    link = next(e for e in entries if e["path"] == "link")
    assert link == {"path": "link", "type": "symlink", "target": "/etc/passwd"}


def test_requests_are_found_validated_and_acknowledged(tmp_path):
    folder = tmp_path / "work" / ".versions"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text(json.dumps({"action": "publish", "note": "n" * 5000}))
    (folder / "b.json").write_text("not json")
    (folder / ".c.tmp").write_text("{}")
    pending = boundary.pending_requests(tmp_path)
    assert [p.name for p in pending] == ["a.json", "b.json"]
    assert boundary.read_request(pending[0]) == {"action": "publish", "note": "n" * 4000}
    assert boundary.read_request(pending[1])["action"] == "invalid"
    boundary.acknowledge(pending[0], {"status": "captured"})
    assert [p.name for p in boundary.pending_requests(tmp_path)] == ["b.json"]


def test_snapshot_status_is_recorded_and_the_manifest_matches_either_way(tmp_path, monkeypatch):
    """Reflink may or may not be available on the test filesystem; both
    outcomes must give a complete version that says which one happened."""
    monkeypatch.setenv("COMPRESSION_BENCH_SNAPSHOT_MIN_FREE_GB", "0")
    sub = make_submission(tmp_path / "s")
    lineage = boundary.Lineage(tmp_path / "p", "L")
    rec = lineage.capture(sub, note="a", run_id="r", source="agent", background=False)
    assert rec["snapshot"]["status"] in ("reflink", "none")
    if rec["snapshot"]["status"] == "reflink":
        assert (tmp_path / "p" / ".objects" / "L" / "v1" / "model" / "w.safetensors").exists()
    else:
        assert not (tmp_path / "p" / ".objects" / "L" / "v1").exists()
    data = json.loads((lineage.dir / "L_v1" / "VERSION.json").read_text())
    assert data["manifest_status"] == "complete"


def test_low_disk_skips_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPRESSION_BENCH_SNAPSHOT_MIN_FREE_GB", "1000000000")
    sub = make_submission(tmp_path / "s")
    rec = boundary.Lineage(tmp_path / "p", "L").capture(sub, note="a", run_id="r",
                                                        source="agent", background=False)
    assert rec["snapshot"]["status"] == "skipped" and "free" in rec["snapshot"]["why"]
