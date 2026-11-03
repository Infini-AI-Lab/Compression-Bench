"""What a run did, in one file: `SUMMARY.json`.

Read from the journal and the agent's stream only; nothing here measures the
submission (that is the verdict's job) or re-derives a state the journal does
not hold.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import pricing
from .record import Journal, RunIdentity, current_state
from .utils import parse_utc, read_json, utc_now, write_json_atomically


def usage_totals(stream_path: Path | str) -> dict:
    """Token usage and cost from the CLI's `result` events (Claude) or
    `turn.completed` events (Codex), summed across launches."""
    totals = {"turns": 0, "input_tokens": 0, "output_tokens": 0,
              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
              "total_cost_usd": 0.0, "tool_calls": 0, "subagent_tool_calls": 0,
              "result_events": 0}
    # Claude's result `total_cost_usd` is the session's running total (a resumed
    # session reports its earlier cost again); turns and tokens are per launch.
    # So cost is each session's largest report, summed over sessions. Likewise
    # `modelUsage` (per model, subagents included) is a session's running total:
    # the last one per session, summed over sessions, is what `pricing` prices.
    # A launch ended mid-turn (deadline, cancel) writes no `result`: what its session
    # used after its last one is the tail, counted from the messages (pricing.tail_cost).
    session_cost: dict[str, float] = {}
    session_models: dict[str, dict] = {}
    tails: dict[str, dict] = {}
    path = Path(stream_path)
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line).get("event", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        kind = event.get("type")
        if kind in ("assistant", "system") and event.get("session_id"):
            tail_message(tails.setdefault(str(event["session_id"]), {"messages": {}, "thinking": 0}), event)
        if kind == "assistant":
            # A subagent's forwarded calls carry the call that started it.
            counter = "subagent_tool_calls" if event.get("parent_tool_use_id") else "tool_calls"
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    totals[counter] += 1
        elif kind == "item.started":
            totals["tool_calls"] += 1
        if kind in ("result", "turn.completed"):
            totals["result_events"] += 1
            usage = event.get("usage") or {}
            for key in ("input_tokens", "output_tokens",
                        "cache_read_input_tokens", "cache_creation_input_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
            cached = usage.get("cached_input_tokens")
            if isinstance(cached, (int, float)):
                totals["cache_read_input_tokens"] += int(cached)
            cost = event.get("total_cost_usd")
            session = str(event.get("session_id") or "")
            tails.pop(session, None)
            if isinstance(cost, (int, float)):
                session_cost[session] = max(session_cost.get(session, 0.0), float(cost))
            if isinstance(event.get("modelUsage"), dict):
                session_models[session] = event["modelUsage"]
            turns = event.get("num_turns")
            if isinstance(turns, int):
                totals["turns"] += turns
    totals["total_cost_usd"] = round(sum(session_cost.values()), 6)
    totals["sessions"] = len(session_cost)
    models: dict[str, dict] = {}

    def model_slot(model: str) -> dict:
        return models.setdefault(model, {"input_tokens": 0, "output_tokens": 0,
                                         "cache_read_input_tokens": 0,
                                         "cache_creation_input_tokens": 0,
                                         "cli_cost_usd": 0.0, "cost_basis": None})
    for per_model in session_models.values():
        for model, used in per_model.items():
            into = model_slot(model)
            for key, source in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"),
                                ("cache_read_input_tokens", "cacheReadInputTokens"),
                                ("cache_creation_input_tokens", "cacheCreationInputTokens")):
                into[key] += int(used.get(source) or 0)
            into["cli_cost_usd"] = round(into["cli_cost_usd"] + float(used.get("costUSD") or 0), 6)
            into["cost_basis"] = used.get("costBasis") or into["cost_basis"]
    for tail in tails.values():
        for model, used in tail_usage(tail).items():
            whole = model_slot(model).setdefault("tail", {})
            for key, value in used.items():
                whole[key] = whole.get(key, 0) + value
    totals["models"] = models
    totals.update(pricing.run_cost(models))
    return totals


#: Output tokens per visible character (text and tool input) of Claude's messages,
#: an estimate (0.51-0.55 on finished sessions, whose output counts are exact).
VISIBLE_TOKENS_PER_CHAR = 0.52


def tail_message(tail: dict, event: dict) -> None:
    """Fold one streamed event into its session's unpriced tail."""
    if event.get("type") == "system":
        if event.get("subtype") == "thinking_tokens":
            tail["thinking"] += int(event.get("estimated_tokens_delta") or 0)
        return
    message = event.get("message") or {}
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return
    entry = tail["messages"].setdefault(str(message.get("id") or len(tail["messages"])),
                                        {"model": message.get("model") or "", "usage": usage, "chars": 0})
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            entry["chars"] += len(block.get("text") or "")
        elif isinstance(block, dict) and block.get("type") == "tool_use":
            entry["chars"] += len(json.dumps(block.get("input")))


def tail_usage(tail: dict) -> dict[str, dict]:
    """Per model: the tail's tokens, input and cache exact, output estimated (the
    session's thinking goes to its most-used model)."""
    out: dict[str, dict] = {}
    for entry in tail["messages"].values():
        usage, model = entry["usage"], entry["model"]
        used = out.setdefault(model, {"messages": 0, "input_tokens": 0, "output_tokens": 0,
                                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                                      "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0})
        split = usage.get("cache_creation") or {}
        written = int(usage.get("cache_creation_input_tokens") or 0)
        one_hour = int(split.get("ephemeral_1h_input_tokens") or 0)
        used["messages"] += 1
        used["input_tokens"] += int(usage.get("input_tokens") or 0)
        used["cache_read_input_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
        used["cache_creation_input_tokens"] += written
        used["cache_write_1h_tokens"] += one_hour
        used["cache_write_5m_tokens"] += written - one_hour
        used["output_tokens"] += max(int(usage.get("output_tokens") or 0),
                                     round(entry["chars"] * VISIBLE_TOKENS_PER_CHAR))
    if out and tail["thinking"]:
        main = max(out, key=lambda m: out[m]["messages"])
        out[main]["output_tokens"] += tail["thinking"]
    return out


def unfinished_seconds(launched_utc: str | None, stream: Path) -> float:
    """Seconds from a launch to the last event its stream recorded (0 if unknown)."""
    if not launched_utc or not stream.exists():
        return 0.0
    last = None
    with stream.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        for line in handle.read().decode("utf-8", errors="ignore").splitlines():
            try:
                last = json.loads(line).get("t") or last
            except (json.JSONDecodeError, AttributeError):
                continue
    if not last:
        return 0.0
    try:
        return max(0.0, (parse_utc(last) - parse_utc(launched_utc)).total_seconds())
    except ValueError:
        return 0.0


def write_summary(identity: RunIdentity, journal: Journal, **extra) -> dict:
    root = identity.root
    launches = journal.events("agent_launched")
    finished = journal.events("agent_finished")
    versions = journal.events("version_captured")
    seconds = sum(e.get("seconds", 0) for e in finished)
    if len(launches) > len(finished):
        # The last launch was ended without an `agent_finished` (a cancel): it ran
        # until its stream's last event.
        seconds += unfinished_seconds(launches[-1].get("utc"), root / "agent.stream.jsonl")
    summary = {
        "kind": "compression_bench_run_summary", "schema_version": 1,
        "written_utc": utc_now(),
        "run_id": identity.run_id, "task_id": identity.task_id,
        "task_hash": identity.task_hash, "lineage_id": identity.lineage_id,
        "namespace": identity.namespace, "created_utc": identity.created_at,
        "deadline_utc": identity.deadline_utc,
        "budget_hours": identity.budget_hours,
        "state": current_state(journal),
        "agent": {
            "launches": len(launches),
            "seconds": round(seconds, 1),
            "agent_cli": launches[0].get("agent_cli") if launches else None,
            "agent_model": launches[0].get("agent_model") if launches else None,
            "agent_effort": launches[0].get("agent_effort") if launches else None,
            "usage": usage_totals(root / "agent.stream.jsonl"),
        },
        "versions": [{"version": v.get("version"), "note": v.get("note"),
                      "utc": v.get("utc")} for v in versions],
        "verdict": (journal.last("final_validation") or {}).get("verdict"),
    }
    previous = read_json(root / "SUMMARY.json", {}) or {}
    for key, value in previous.items():
        summary.setdefault(key, value)
    summary.update(extra)
    write_json_atomically(root / "SUMMARY.json", summary)
    return summary
