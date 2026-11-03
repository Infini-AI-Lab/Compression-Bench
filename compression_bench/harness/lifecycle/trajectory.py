"""The agent's trajectory: what it was told, what it thought, said and ran.

Recorded while the agent works (agent.py) and never touched by cleaning:

  prompts/<n>.<kind>.txt  what the harness sent each launch on stdin
  agent.stream.jsonl      the CLI's event stream, every launch appended, each
                          event stamped by the host: {"t", "agent_cli", "event"}
  agent.<kind>.<n>.log    each launch's raw output, stream and stray lines
  work/.agent-home/       the CLI's own transcript: Claude's
                          .claude/projects/<cwd>/<session>.jsonl (subagents
                          under <session>/subagents/) and its memory notes;
                          Codex's .codex/sessions/**/rollout-*.jsonl

`export` turns them into `<run>/trajectory/`, the record analysis reads:

  steps.jsonl       one row per step, in order: launch (with its prompt),
                    thinking, message, tool_call (tool, input), tool_result
                    (output, error, seconds since its call), background (a
                    background task starting or ending), error, result (a
                    launch's turns, tokens and cost)
  transcripts/      copies of the CLI's own transcripts and memory notes
  notes/            the agent's own top-level notes (RECIPE.md, EXPERIMENTS.md, ...)
  TRAJECTORY.json   counts (steps by kind; calls, errors and seconds per tool;
                    thinking with and without text), the launches, the files
                    the agent wrote, the workspace's logs, and a checksum of
                    every source

and copies TRAJECTORY.json, a gzipped steps.jsonl and the notes to the
lineage (`<lineage>/TRAJECTORY/<run id>/`), the durable record. A subagent's
steps carry `parent`: the tool call that started it.

Thinking is only as complete as the CLI made it: Claude in print mode omits
it unless asked (agent.CLAUDE_TRACE_FLAGS); thinking blocks without text are
counted as `thinking.without_text`.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path

from . import record, telemetry
from .utils import read_json, seconds_between, sha256_file, utc_now, write_json_atomically

SCHEMA = "compression_bench.trajectory/1"
FOLDER = "trajectory"
#: A tool's output or a string in its input longer than this keeps its head and tail.
OUTPUT_CAP = 64 << 10
INPUT_CAP = 256 << 10
#: Stream events that carry no decision or action (token tickers, progress pings).
NOISE = {("system", "thinking_tokens"), ("tool_progress", None), ("rate_limit_event", None),
         ("turn.started", None), ("item.updated", None)}
#: Workspace files listed (not copied) as the agent's logs.
LOG_SUFFIXES = {".log", ".txt", ".jsonl", ".csv", ".out", ".err"}
#: Files the harness wrote into the workspace; not the agent's notes.
HARNESS_NOTES = {"AGENT_TASK.md"}
MAX_LISTED_LOGS = 2000
NOTE_MAX_BYTES = 1 << 20


def _clip(text: str, cap: int) -> tuple[str, bool]:
    if len(text) <= cap:
        return text, False
    half = cap // 2
    return f"{text[:half]}\n...[{len(text) - 2 * half} chars omitted]...\n{text[-half:]}", True


def _clip_tree(value, cap: int = INPUT_CAP):
    if isinstance(value, str):
        return _clip(value, cap)[0]
    if isinstance(value, dict):
        return {k: _clip_tree(v, cap) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_tree(v, cap) for v in value]
    return value


def _result_text(content) -> str:
    """A tool_result's content as text (images become a placeholder)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(f"[{block.get('type', 'block')}]")
        return "\n".join(parts)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


# --- the stream, one row per step ---------------------------------------------------

class Steps:
    """Builds rows from stream events; pairs each tool result with its call."""

    def __init__(self, launches: list[dict], prompts: dict[int, str]):
        self.rows: list[dict] = []
        self.launches = launches
        self.prompts = prompts
        self.calls: dict[str, dict] = {}
        self.skipped: dict[str, int] = {}
        self.inits = 0

    def launch_of(self, t: str | None) -> int | None:
        found = None
        for index, launch in enumerate(self.launches):
            if t and (launch.get("utc") or "") <= t:
                found = index
        return found

    def add(self, t: str | None, kind: str, parent=None, **fields) -> dict:
        row = {"step": len(self.rows), "t": t, "launch": self.launch_of(t), "kind": kind}
        if parent:
            row["parent"] = parent
        row.update(fields)
        self.rows.append(row)
        return row

    def skip(self, label: str) -> None:
        self.skipped[label] = self.skipped.get(label, 0) + 1

    def call(self, t, parent, call_id, tool, tool_input) -> None:
        self.add(t, "tool_call", parent, id=call_id, tool=tool, input=_clip_tree(tool_input))
        self.calls[str(call_id)] = {"t": t, "tool": tool}

    def result(self, t, parent, call_id, output: str, is_error: bool, **extra) -> None:
        call = self.calls.get(str(call_id), {})
        text, clipped = _clip(output, OUTPUT_CAP)
        fields = {"id": call_id, "tool": call.get("tool"), "is_error": bool(is_error),
                  "output": text, "output_chars": len(output),
                  "seconds": seconds_between(call.get("t"), t), **extra}
        if clipped:
            fields["output_truncated"] = True
        self.add(t, "tool_result", parent, **fields)

    def launch_row(self, t, **fields) -> None:
        index = self.launch_of(t)
        launch = self.launches[index] if index is not None else {}
        self.add(t, "launch", launch_kind=launch.get("kind"),
                 prompt=self.prompts.get(index) if index is not None else None,
                 prompt_sha256=launch.get("prompt_sha256"), **fields)

    # Claude Code: --output-format stream-json --verbose
    def claude(self, t: str | None, event: dict) -> None:
        kind, subtype = event.get("type"), event.get("subtype")
        parent = event.get("parent_tool_use_id")
        if (kind, subtype) in NOISE or (kind, None) in NOISE:
            self.skip(f"{kind}:{subtype}" if subtype else str(kind))
            return
        if kind == "system" and subtype == "init":
            self.inits += 1
            self.launch_row(t, session_id=event.get("session_id"), model=event.get("model"),
                            cwd=event.get("cwd"), tools=event.get("tools"),
                            permission_mode=event.get("permissionMode"))
            return
        if kind in ("assistant", "user"):
            content = (event.get("message") or {}).get("content")
            if isinstance(content, str):
                self.add(t, "message" if kind == "assistant" else "input", parent,
                         role=kind, text=content)
                return
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    self.add(t, "thinking", parent, text=block.get("thinking") or "")
                elif btype == "redacted_thinking":
                    self.add(t, "thinking", parent, text="", redacted=True)
                elif btype == "text":
                    self.add(t, "message" if kind == "assistant" else "input", parent,
                             role=kind, text=block.get("text") or "")
                elif btype in ("tool_use", "server_tool_use"):
                    self.call(t, parent, block.get("id"), block.get("name"), block.get("input"))
                elif btype == "tool_result":
                    self.result(t, parent, block.get("tool_use_id"),
                                _result_text(block.get("content")), block.get("is_error"))
                else:
                    self.skip(f"{kind}:{btype}")
            return
        if kind == "system" and subtype in ("task_started", "task_notification", "task_updated",
                                           "background_tasks_changed"):
            keep = {k: v for k, v in event.items()
                    if k not in ("type", "subtype", "uuid", "session_id")}
            self.add(t, "background", parent, subtype=subtype,
                     detail=_clip_tree(keep, OUTPUT_CAP))
            return
        if kind == "result":
            usage = event.get("usage") or {}
            self.add(t, "result", subtype=subtype, is_error=event.get("is_error"),
                     turns=event.get("num_turns"),
                     duration_s=round((event.get("duration_ms") or 0) / 1000, 1),
                     session_cost_usd=event.get("total_cost_usd"),
                     output_tokens=usage.get("output_tokens"),
                     stop_reason=event.get("stop_reason"),
                     text=_clip(str(event.get("result") or ""), OUTPUT_CAP)[0])
            return
        self.skip(f"{kind}:{subtype}" if subtype else str(kind))

    # Codex: codex exec --json
    def codex(self, t: str | None, event: dict) -> None:
        kind = event.get("type")
        if (kind, None) in NOISE:
            self.skip(str(kind))
            return
        if kind == "thread.started":
            self.inits += 1
            self.launch_row(t, session_id=event.get("thread_id"))
            return
        if kind in ("item.started", "item.completed"):
            item = event.get("item") or {}
            itype = item.get("type") or item.get("item_type")
            item_id = item.get("id")
            if kind == "item.started":
                if itype in ("command_execution", "mcp_tool_call", "web_search", "file_change"):
                    self._codex_call(t, item_id, itype, item)
                return
            if itype == "reasoning":
                self.add(t, "thinking", text=item.get("text") or "")
            elif itype == "agent_message":
                self.add(t, "message", role="assistant", text=item.get("text") or "")
            elif itype in ("command_execution", "mcp_tool_call", "web_search", "file_change"):
                if str(item_id) not in self.calls:
                    self._codex_call(t, item_id, itype, item)
                if itype == "command_execution":
                    code = item.get("exit_code")
                    self.result(t, None, item_id, str(item.get("aggregated_output") or ""),
                                item.get("status") == "failed" or code not in (0, None),
                                exit_code=code)
                elif itype == "mcp_tool_call":
                    self.result(t, None, item_id,
                                _result_text(item.get("result") or item.get("error")),
                                item.get("status") == "failed" or bool(item.get("error")))
                else:
                    self.result(t, None, item_id, str(item.get("status") or ""),
                                item.get("status") == "failed")
            elif itype == "todo_list":
                self.add(t, "plan", items=item.get("items"))
            elif itype == "error":
                self.add(t, "error", text=item.get("message") or "")
            else:
                self.skip(f"item:{itype}")
            return
        if kind in ("turn.completed", "turn.failed"):
            usage = event.get("usage") or {}
            self.add(t, "result", subtype=kind, is_error=kind == "turn.failed",
                     output_tokens=usage.get("output_tokens"),
                     reasoning_tokens=usage.get("reasoning_output_tokens"),
                     error=(event.get("error") or {}).get("message"))
            return
        if kind == "error":
            self.add(t, "error", text=event.get("message") or "")
            return
        self.skip(str(kind))

    def _codex_call(self, t, item_id, itype, item) -> None:
        if itype == "command_execution":
            self.call(t, None, item_id, "shell", {"command": item.get("command")})
        elif itype == "mcp_tool_call":
            self.call(t, None, item_id, f"mcp:{item.get('server')}.{item.get('tool')}",
                      item.get("arguments"))
        elif itype == "web_search":
            self.call(t, None, item_id, "web_search", {"query": item.get("query")})
        else:
            self.call(t, None, item_id, "apply_patch", {"changes": item.get("changes")})


def steps_from_stream(stream: Path, launches: list[dict],
                      prompts: dict[int, str] | None = None) -> Steps:
    steps = Steps(launches, prompts or {})
    if not stream.exists():
        return steps
    with stream.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                wrapped = json.loads(line)
            except json.JSONDecodeError:
                steps.skip("<not json>")
                continue
            event = wrapped.get("event") if isinstance(wrapped.get("event"), dict) else None
            if event is None:
                steps.skip("<unwrapped>")
                continue
            if wrapped.get("agent_cli") == "codex":
                steps.codex(wrapped.get("t"), event)
            else:
                steps.claude(wrapped.get("t"), event)
    return steps


# --- the summary --------------------------------------------------------------------

WRITE_TOOLS = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path",
               "NotebookEdit": "notebook_path"}


def summarize(rows: list[dict], skipped: dict[str, int]) -> dict:
    by_kind: dict[str, int] = {}
    tools: dict[str, dict] = {}
    thinking = {"blocks": 0, "with_text": 0, "without_text": 0, "chars": 0}
    messages = {"count": 0, "chars": 0}
    written: list[str] = []
    for row in rows:
        kind = row["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if kind == "thinking":
            thinking["blocks"] += 1
            text = row.get("text") or ""
            thinking["with_text" if text else "without_text"] += 1
            thinking["chars"] += len(text)
        elif kind == "message":
            messages["count"] += 1
            messages["chars"] += len(row.get("text") or "")
        elif kind == "tool_call":
            entry = tools.setdefault(row.get("tool") or "?", {"calls": 0, "errors": 0,
                                                               "seconds": 0.0, "max_seconds": 0.0})
            entry["calls"] += 1
            field = WRITE_TOOLS.get(row.get("tool"))
            data = row.get("input") if isinstance(row.get("input"), dict) else {}
            if field and data.get(field):
                written.append(data[field])
            for change in (data.get("changes") or []) if row.get("tool") == "apply_patch" else []:
                if isinstance(change, dict) and change.get("path"):
                    written.append(change["path"])
        elif kind == "tool_result":
            entry = tools.setdefault(row.get("tool") or "?", {"calls": 0, "errors": 0,
                                                               "seconds": 0.0, "max_seconds": 0.0})
            entry["errors"] += bool(row.get("is_error"))
            if isinstance(row.get("seconds"), (int, float)):
                entry["seconds"] = round(entry["seconds"] + row["seconds"], 1)
                entry["max_seconds"] = max(entry["max_seconds"], row["seconds"])
    unique_written = list(dict.fromkeys(written))
    return {"steps": len(rows), "by_kind": by_kind,
            "subagent_steps": sum(1 for r in rows if r.get("parent")),
            "tool_calls": sum(t["calls"] for t in tools.values()),
            "tool_errors": sum(t["errors"] for t in tools.values()),
            "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1]["calls"])),
            "thinking": thinking, "messages": messages,
            "files_written": unique_written[:500],
            "files_written_count": len(unique_written),
            "first_utc": next((r["t"] for r in rows if r.get("t")), None),
            "last_utc": next((r["t"] for r in reversed(rows) if r.get("t")), None),
            "skipped_events": dict(sorted(skipped.items(), key=lambda kv: -kv[1]))}


def _source(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def workspace_logs(work: Path) -> list[dict]:
    """The agent's log-like files in its workspace (they stay there; listed here)."""
    found = []
    skip = {work / ".agent-home", work / ".tmp", work / ".versions"}
    for dirpath, dirnames, filenames in os.walk(work):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if here / d not in skip]
        for name in filenames:
            path = here / name
            if path.suffix.lower() in LOG_SUFFIXES and not path.is_symlink():
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found.append({"path": str(path.relative_to(work)), "bytes": stat.st_size,
                              "mtime": stat.st_mtime})
    found.sort(key=lambda r: -r["mtime"])
    for row in found:
        row.pop("mtime")
    return found[:MAX_LISTED_LOGS]


# --- the whole step -----------------------------------------------------------------

def transcript_dirs(work: Path) -> dict[str, Path]:
    home = work / ".agent-home"
    return {"claude": home / ".claude" / "projects", "codex": home / ".codex" / "sessions"}


def launch_prompts(root: Path) -> dict[int, str]:
    prompts = {}
    for path in sorted((root / "prompts").glob("*.txt")):
        head = path.name.split(".", 1)[0]
        if head.isdigit():
            prompts[int(head)] = path.read_text(encoding="utf-8", errors="replace")
    return prompts


def export(root: Path | str, lineage: Path | str | None = None) -> dict:
    """Write `<run>/trajectory/` (and the lineage's copy); return the summary."""
    root = Path(root)
    events = list(record.Journal(root))
    launches = [e for e in events if e.get("event") == "agent_launched"]
    out = root / FOLDER
    out.mkdir(exist_ok=True)

    identity = read_json(root / "identity.json", {}) or {}
    prompts = launch_prompts(root)
    steps = steps_from_stream(root / "agent.stream.jsonl", launches, prompts)
    steps_path = out / "steps.jsonl"
    partial = steps_path.with_suffix(".jsonl.partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in steps.rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(partial, steps_path)

    copied = []
    for cli, folder in transcript_dirs(root / "work").items():
        if not folder.is_dir():
            continue
        target = out / "transcripts" / cli
        for dirpath, _, filenames in os.walk(folder):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink() or name.endswith(".lock"):
                    continue
                dest = target / path.relative_to(folder)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                copied.append(_source(dest, root))

    notes = []
    work = root / "work"
    for path in sorted(work.glob("*.md")) if work.is_dir() else []:
        if path.name in HARNESS_NOTES or path.is_symlink() or path.stat().st_size > NOTE_MAX_BYTES:
            continue
        dest = out / "notes" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        notes.append(_source(dest, root))

    memory = telemetry.copy_memory(root, out / "memory")

    sources = {"stream": _source(root / "agent.stream.jsonl", root)
               if (root / "agent.stream.jsonl").exists() else None,
               "launch_logs": [_source(p, root) for p in sorted(root.glob("agent.*.log"))],
               "prompts": [_source(p, root) for p in sorted((root / "prompts").glob("*.txt"))],
               "transcripts": copied, "notes": notes,
               "steps": _source(steps_path, root)}
    agent_cli = next((e.get("agent_cli") for e in launches if e.get("agent_cli")), None)
    summary = {
        "kind": "compression_bench_trajectory", "schema": SCHEMA, "exported_utc": utc_now(),
        "run_id": identity.get("run_id"),
        "agent_cli": agent_cli,
        "agent_model": next((e.get("agent_model") for e in launches if e.get("agent_model")), None),
        "agent_effort": next((e.get("agent_effort") for e in launches if e.get("agent_effort")),
                             None),
        "launches": [{"index": i, "kind": e.get("kind"), "utc": e.get("utc"),
                      "session_id": e.get("session_id"), "continued": e.get("continued"),
                      "prompt_chars": e.get("prompt_chars"),
                      "prompt_saved": i in prompts} for i, e in enumerate(launches)],
        "launches_seen_in_stream": steps.inits,
        **summarize(steps.rows, steps.skipped),
        "workspace_logs": workspace_logs(work) if work.is_dir() else [],
        "memory": memory,
        "sources": sources,
    }
    summary["timing"] = telemetry.write_timing(root, summary)
    summary["resources"] = telemetry.resource_summary(root)
    write_json_atomically(out / "TRAJECTORY.json", summary)
    back = read_json(out / "TRAJECTORY.json", {}) or {}
    lines = sum(1 for _ in steps_path.open(encoding="utf-8"))
    if back.get("steps") != len(steps.rows) or lines != len(steps.rows):
        raise RuntimeError(f"{out} did not read back")

    if lineage is not None:
        durable = Path(lineage) / "TRAJECTORY" / str(summary["run_id"] or root.name)
        durable.mkdir(parents=True, exist_ok=True)
        with steps_path.open("rb") as source, gzip.GzipFile(
                durable / "steps.jsonl.gz.partial", "wb", mtime=0) as target:
            shutil.copyfileobj(source, target)
        os.replace(durable / "steps.jsonl.gz.partial", durable / "steps.jsonl.gz")
        for note in notes:
            dest = durable / "notes" / Path(note["path"]).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / note["path"], dest)
        if (out / "memory").is_dir():
            shutil.copytree(out / "memory", durable / "memory", dirs_exist_ok=True)
        shutil.copy2(root / telemetry.TIMING, durable / telemetry.TIMING)
        if (root / telemetry.RESOURCES).is_file():
            with (root / telemetry.RESOURCES).open("rb") as source, gzip.GzipFile(
                    durable / (telemetry.RESOURCES + ".gz"), "wb", mtime=0) as target:
                shutil.copyfileobj(source, target)
        write_json_atomically(durable / "TRAJECTORY.json", summary)
        summary["lineage_copy"] = str(durable)
    return summary


def brief(summary: dict) -> dict:
    """The part of TRAJECTORY.json that METRICS.json carries."""
    keys = ("schema", "exported_utc", "agent_cli", "launches_seen_in_stream", "steps",
            "by_kind", "subagent_steps", "tool_calls", "tool_errors", "tools", "thinking",
            "messages", "files_written_count", "first_utc", "last_utc", "lineage_copy",
            "memory", "resources")
    out = {k: summary.get(k) for k in keys}
    timing = summary.get("timing") or {}
    out["timing"] = {k: timing.get(k) for k in ("phases", "public_check_totals")}
    out["timing"]["verdict_stages"] = {s.get("stage"): s.get("seconds")
                                       for s in timing.get("verdict_stages") or []}
    out["files"] = {"run": f"{FOLDER}/TRAJECTORY.json", "steps": f"{FOLDER}/steps.jsonl"}
    return out
