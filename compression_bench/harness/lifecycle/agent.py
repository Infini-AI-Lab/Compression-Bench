"""The coding agent: launched once, continued in place until the deadline.

As in ServerBench, these rules hold:

* The prompt travels on stdin from a file staged in the container, never on
  the argv: a `pkill -f <task text>` by the agent must not match its own CLI.
* Continuation is real or it is not claimed. Claude gets a session id chosen
  by the harness and is resumed with `--resume`; Codex reports its thread id
  and is resumed with `codex exec resume`. A session the platform will not
  resume is a cold restart, journalled as one.
* The budget is measured against the ORIGINAL deadline, never re-based -- except
  across a credit hold: a provider that refuses for want of credit (an API key's
  spending limit) holds the run, its container paused and every file kept, and the
  agent gets back the time it had left when the hold began (`credit_hold`).
* Tools that promise a later callback (`ScheduleWakeup`, cron) are disallowed:
  nothing here delivers one, and an agent waiting for it would idle the run.
* Tools that reach the internet from the provider's side (`WebSearch`,
  `WebFetch`, `RemoteTrigger`) are disallowed: they would bypass the
  container's egress allowlist, and every task states that it runs offline.
* A launch that could never reach its model (bad key, unknown model, billing)
  is terminal; retrying cannot help. One refused for want of credit holds the
  run until credit returns.
* The agent's reasoning is recorded, not only its actions: Claude runs with
  summarized thinking (`--thinking-display summarized`; print mode otherwise
  omits it) and forwards its subagents' text, Codex with detailed reasoning
  summaries. `trajectory.py` turns the stream into the run's trajectory.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from . import backend, instructions
from .containers import BASE_URL_KEY, read_env_file
from .record import Journal, RunIdentity
from .utils import parse_utc, until_space, utc_after, utc_now, write_json_atomically

#: Budget below this is not worth resuming into.
EARLY_STOP_MARGIN_S = 300.0
#: Runaway cap on launches in one run.
MAX_LAUNCHES = 200
#: A launch shorter than this, with at most IDLE_LAUNCH_CALLS tool calls, did
#: nothing; it is resumed with the idle-restart prompt after a backoff of
#: IDLE_BACKOFF_S per launch in the streak, at most MAX_IDLE_BACKOFF_S. There is
#: no idle cap: the agent is resumed until the deadline.
IDLE_LAUNCH_S = 180.0
IDLE_LAUNCH_CALLS = 4
IDLE_BACKOFF_S = 30.0
MAX_IDLE_BACKOFF_S = 300.0

NO_CALLBACK_TOOLS = ("ScheduleWakeup", "CronCreate", "CronDelete", "CronList")
#: Server-side network tools: the egress allowlist cannot see them.
OFFLINE_TOOLS = ("WebSearch", "WebFetch", "RemoteTrigger")
#: What makes the stream carry the agent's reasoning (see the module docstring).
CLAUDE_TRACE_FLAGS = ("--thinking-display", "summarized", "--forward-subagent-text")
CODEX_TRACE_CONFIG = ('model_reasoning_summary="detailed"',)

TERMINAL_MARKERS = (
    '"error":"model_not_found"', '"api_error_status":404',
    "It may not exist or you may not have access to it", "Invalid API key",
    '"api_error_status":403', "Key limit exceeded", "Failed to authenticate",
    "insufficient_quota", "exceeded your current quota",
    "OAuth access token has been revoked",
)
SESSION_LIMIT_MARKERS = ("hit your session limit", "session limit reached",
                         "usage limit reached")
SESSION_LIMIT_FALLBACK_S = 30 * 60

#: A provider's refusal that more credit cures (e.g. an API key's spending limit).
#: It holds the run instead of ending it: the resumes would fail until the deadline,
#: then the verdict and the cleanup would take the container and the workspace's
#: weights with them.
CREDIT_MARKERS = ("Key limit exceeded", "Insufficient credits", "requires more credits",
                  "credit balance is too low")
CREDIT_STATUS = re.compile(r'"api_error_status"\s*:\s*402\b')
#: Credit left on the route's key (USD, when the provider can say) to resume into.
CREDIT_RESUME_USD = 5.0
#: A hold looks for credit every CREDIT_POLL_S; a resume that met no credit again,
#: or a provider that cannot say, waits CREDIT_RETRY_S before the next try.
CREDIT_POLL_S = 60.0
CREDIT_RETRY_S = 1800.0
#: A hold ends without credit after this long, or when the operator writes
#: `<run>/END_CREDIT_HOLD`; the agent phase then ends ("credit") and the run is
#: verdicted as at its deadline.
CREDIT_HOLD_MAX_S = float(os.environ.get("COMPRESSION_BENCH_CREDIT_HOLD_MAX_H", "72")) * 3600
#: Present while a run holds (the janitor keeps off its weights); removed at the end.
CREDIT_HOLD_FILE = "CREDIT_HOLD.json"
END_CREDIT_HOLD_FILE = "END_CREDIT_HOLD"

PROMPT_IN_CONTAINER = "/tmp/.compression-bench-prompt.{n}.txt"


class ContinuationUnavailable(RuntimeError):
    """The platform could not continue the recorded session."""


@dataclasses.dataclass(frozen=True)
class AgentRun:
    session_id: str
    continued: bool
    kind: str
    exit_code: int
    seconds: float
    budget_seconds: float
    stdout_path: str
    stopped_early: bool
    tool_calls: int
    terminal_because: str = ""
    session_limited_s: float | None = None
    credit_because: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind_of_record"] = "agent_run"
        return d

    @property
    def idle(self) -> bool:
        return self.seconds < IDLE_LAUNCH_S and self.tool_calls <= IDLE_LAUNCH_CALLS


def remaining_seconds(identity: RunIdentity, *, now: float | None = None) -> float:
    deadline = parse_utc(identity.deadline_utc)
    current = (datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
               if now is not None else datetime.datetime.now(datetime.timezone.utc))
    return max(0.0, (deadline - current).total_seconds())


def tool_calls(stdout_path: Path | str) -> int:
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return 0
    calls = len(re.findall(r'"type"\s*:\s*"tool_use"', body))
    calls += len(re.findall(r'"type"\s*:\s*"item\.started"', body))   # codex
    for name in NO_CALLBACK_TOOLS:
        calls -= body.count(f'"name":"{name}"')
    return max(calls, 0)


def _reached_its_model(body: str) -> bool:
    """The model answered at least once. A long session is ONE turn, so its only `result` line is the last
    one, and a transient authentication failure late in the session must not read as "could not reach
    its model". Any assistant message proves the model was reached."""
    result = re.compile(r'"type"\s*:\s*"(result|turn\.completed)"')
    errored = re.compile(r'"is_error"\s*:\s*true')
    answered = re.compile(r'"type"\s*:\s*"(assistant|item\.completed)"')
    return any((result.search(line) and not errored.search(line)) or answered.search(line)
               for line in body.splitlines())


def terminal_failure(stdout_path: Path | str) -> str:
    """Why this agent can never start, or "" if it merely stopped. A launch
    that completed a turn reached its model and is never terminal; a marker in
    a subagent's own error record is that subagent's, not the session's."""
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return ""
    if _reached_its_model(body):
        return ""
    body = "\n".join(line for line in body.splitlines()
                     if '"subagent_type"' not in line)
    hits = [marker for marker in TERMINAL_MARKERS if marker in body]
    if not hits:
        return ""
    return (f"the agent could not reach its model at all ({', '.join(hits)}); "
            f"retrying cannot help")


_RESET_AT = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap])m?\b",
                       re.IGNORECASE)


def session_limited(stdout_path: Path | str, *, now=None) -> float | None:
    """Seconds to wait if the CLI hit a usage/session limit, else None."""
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return None
    if not any(marker in body for marker in SESSION_LIMIT_MARKERS):
        return None
    latest = None
    for match in _RESET_AT.finditer(body):
        latest = match
    if latest is None:
        return float(SESSION_LIMIT_FALLBACK_S)
    hour = int(latest.group(1)) % 12 + (12 if latest.group(3).lower() == "p" else 0)
    minute = int(latest.group(2) or 0)
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        return 30.0
    return max(30.0, (target - moment).total_seconds())


_API_ERROR = re.compile(r'"is_api_error_message"\s*:\s*true')
_RESULT = re.compile(r'"type"\s*:\s*"result"')


def credit_exhausted(stdout_path: Path | str) -> str:
    """Why the provider refused this launch for want of credit, or "". Only how the
    launch ENDED counts: the CLI's last own record (an API-error message or the
    result), never a tool's output and never a subagent's record."""
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return ""
    last = ""
    for line in body.splitlines():
        if '"subagent_type"' not in line and (_API_ERROR.search(line) or _RESULT.search(line)):
            last = line
    hits = [marker for marker in CREDIT_MARKERS if marker in last]
    if CREDIT_STATUS.search(last):
        hits.append("HTTP 402")
    return ", ".join(hits)


def credit_left(agent_env: Path | str | None, *, timeout: float = 30.0) -> float | None:
    """USD left on the route's key, when its provider can say (OpenRouter's
    `/api/v1/key`; no limit is infinite), else None."""
    if not agent_env:
        return None
    try:
        env = read_env_file(agent_env)
    except OSError:
        return None
    token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")
    if urlparse(env.get(BASE_URL_KEY, "")).hostname != "openrouter.ai" or not token:
        return None
    request = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                     headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response).get("data") or {}
    except Exception:                                   # noqa: BLE001 - unknown, not none
        return None
    left = data.get("limit_remaining")
    return float("inf") if left is None else float(left)


def session_of(journal: Journal) -> str:
    started = journal.last("agent_session")
    return started["session_id"] if started else ""


def ensure_session(journal: Journal) -> str:
    existing = session_of(journal)
    if existing:
        return existing
    session_id = str(uuid.uuid4())
    journal.append("agent_session", session_id=session_id, minted_utc=utc_now())
    return session_id


def new_session(journal: Journal, agent_cli: str) -> str:
    session_id = str(uuid.uuid4()) if agent_cli == "claude" else ""
    journal.append("agent_session", session_id=session_id, minted_utc=utc_now(),
                   cold_start=True, replaces=session_of(journal))
    return session_id


def tee_stream(pipe, raw_path: Path, stream_path: Path, *, agent_cli: str,
               on_session=None, on_dropped=None) -> int:
    """Copy the CLI's stream to the per-launch raw log and, timestamped, to the
    run's `agent.stream.jsonl` (APPENDED across launches, so a continued
    session reads as one trajectory). Returns the number of events.

    The pipe is always drained to its end: a full disk (`OSError` on a write)
    only loses the lines it cannot hold -- reported once through
    `on_dropped(error)`, the next line retried -- because a reader that stopped
    would leave the CLI blocked on its stdout and the run would never see it
    finish."""
    events = 0

    class Sink:
        failed = False

        def __init__(self, handle):
            self.handle = handle

        def write(self, text):
            try:
                if self.failed:
                    self.handle.write("\n")      # a failed write may have left half a line
                self.handle.write(text)
                self.handle.flush()
                self.failed = False
            except (OSError, ValueError) as exc:
                if not self.failed and on_dropped is not None:
                    try:
                        on_dropped(f"{type(exc).__name__}: {exc}")
                    except OSError:
                        pass                      # the journal is on the same full disk
                self.failed = True

    with raw_path.open("a", encoding="utf-8") as raw, \
            stream_path.open("a", encoding="utf-8") as stream_file:
        raw_sink, stream = Sink(raw), Sink(stream_file)
        for line in pipe:
            raw_sink.write(line)
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            events += 1
            if (agent_cli == "codex" and on_session is not None
                    and event.get("type") == "thread.started"
                    and event.get("thread_id")):
                on_session(str(event["thread_id"]))
            stream.write(json.dumps({"t": utc_now(), "agent_cli": agent_cli,
                                     "event": event}) + "\n")
    return events


def stage_prompt(*, container: str, text: str, index: int) -> str:
    path = PROMPT_IN_CONTAINER.format(n=index)
    try:
        backend.put_file(container, text, path)
    except RuntimeError as exc:
        raise ContinuationUnavailable(
            f"could not stage the agent prompt into {container}: {exc}") from exc
    return path


def command_for(*, agent_cli: str, prompt_path: str, session_id: str,
                continued: bool, model: str, effort: str, env_file: str = "") -> str:
    """The shell command that runs one launch inside main. With `env_file`
    (an API-key provider route) the CLI's provider settings are read from that
    bound file inside the container, never passed on a command line."""
    source = (f"set -a; . {shlex.quote(env_file)}; set +a; " if env_file else "")
    if agent_cli == "codex":
        common = ["--skip-git-repo-check",
                  "--dangerously-bypass-approvals-and-sandbox"]
        if model:
            common += ["--model", model]
        if effort:
            common += ["-c", f'model_reasoning_effort="{effort}"']
        for setting in CODEX_TRACE_CONFIG:
            common += ["-c", setting]
        if continued:
            argv = ["codex", "exec", "resume", *common, "--json", session_id, "-"]
        else:
            argv = ["codex", "exec", *common, "-C", backend.WORKDIR, "--json", "-"]
        return f"{source}{shlex.join(argv)} < {shlex.quote(prompt_path)}"
    argv = ["claude", "--dangerously-skip-permissions"]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    argv += CLAUDE_TRACE_FLAGS
    argv += (["--resume", session_id] if continued else ["--session-id", session_id])
    argv += ["--disallowed-tools", *NO_CALLBACK_TOOLS, *OFFLINE_TOOLS,
             "--output-format", "stream-json", "--verbose", "--print"]
    return f"{source}{shlex.join(argv)} < {shlex.quote(prompt_path)}"


def launch(*, identity: RunIdentity, journal: Journal, container: str,
           task_dir: Path, agent_cli: str, model: str = "", effort: str = "",
           kind: str = "start", poll: float = 5.0, on_tick=None,
           env_file: str = "") -> AgentRun:
    """Start or continue the one coding agent; block until it exits or the
    deadline passes. `kind` is start | continuation | cold-restart |
    idle-restart."""
    budget = remaining_seconds(identity)
    if budget <= 0:
        raise ContinuationUnavailable(
            f"the run's deadline {identity.deadline_utc} has passed")
    if kind == "cold-restart":
        session_id = new_session(journal, agent_cli)
        continued = False
    elif kind == "start":
        session_id = ensure_session(journal) if agent_cli == "claude" else ""
        continued = False
    else:
        session_id = session_of(journal)
        continued = True
        if not session_id:
            raise ContinuationUnavailable("no recorded session to continue")
    if kind == "start":
        # The specification as delivered: the budget and deadline are this run's
        # data (`WORKSPACE.json`, `./clock`) and change no word of it.
        prompt = instructions.brief(task_dir)
    else:
        prompt = instructions.session_prompt(
            task_dir, "continuation" if kind == "continuation" else kind,
            budget_seconds=budget, deadline_utc=identity.deadline_utc)
    index = len(journal.events("agent_launched"))
    stdout_path = identity.root / f"agent.{kind}.{index}.log"
    # The staged copy lives in the container's /tmp and goes with it; this one
    # is the trajectory's record of what the agent was told.
    (identity.root / "prompts").mkdir(exist_ok=True)
    until_space(lambda: (identity.root / "prompts" / f"{index}.{kind}.txt").write_text(prompt, encoding="utf-8"),
                f"writing the {kind} prompt")
    prompt_path = stage_prompt(container=container, text=prompt, index=index)
    command = command_for(agent_cli=agent_cli, prompt_path=prompt_path,
                          session_id=session_id, continued=continued,
                          model=model, effort=effort, env_file=env_file)
    argv = backend.exec_argv(container, command)
    journal.append("agent_launched", kind=kind, session_id=session_id,
                   continued=continued, agent_cli=agent_cli, agent_model=model,
                   agent_effort=effort, budget_seconds=round(budget, 1),
                   container=container, command=command, prompt_path=prompt_path,
                   prompt_chars=len(prompt),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())

    started = time.monotonic()
    stream_path = identity.root / "agent.stream.jsonl"
    process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               stdin=subprocess.DEVNULL)
    resolved = {"session_id": session_id}

    def bind_session(found: str) -> None:
        if resolved["session_id"]:
            return
        resolved["session_id"] = found
        journal.append("agent_session", session_id=found,
                       discovered_utc=utc_now(), provider="codex")

    reader = threading.Thread(target=tee_stream,
                              args=(process.stdout, stdout_path, stream_path),
                              kwargs={"agent_cli": agent_cli,
                                      "on_session": bind_session,
                                      "on_dropped": lambda error: journal.append(
                                          "agent_stream_write_failed", error=error,
                                          utc=utc_now())},
                              daemon=True, name="agent-stream")
    reader.start()
    journal.append("owned_process", pid=process.pid, what="agent exec client")
    while True:
        code = process.poll()
        if code is not None:
            break
        if time.monotonic() - started >= budget:
            # Ending the docker-exec client does not end the CLI inside the
            # container; the caller stops the whole environment at the
            # deadline, which does.
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            code = process.returncode if process.returncode is not None else -15
            journal.append("agent_budget_expired",
                           seconds=round(time.monotonic() - started, 1))
            break
        if on_tick is not None:
            on_tick()
        time.sleep(poll)
    reader.join(timeout=30)
    seconds = round(time.monotonic() - started, 1)
    left = remaining_seconds(identity)
    run = AgentRun(session_id=resolved["session_id"], continued=continued,
                   kind=kind, exit_code=int(code), seconds=seconds,
                   budget_seconds=round(budget, 1), stdout_path=str(stdout_path),
                   stopped_early=left > EARLY_STOP_MARGIN_S,
                   tool_calls=tool_calls(stdout_path),
                   terminal_because=terminal_failure(stdout_path),
                   session_limited_s=session_limited(stdout_path),
                   credit_because=credit_exhausted(stdout_path))
    journal.append("agent_finished", **run.to_dict(),
                   remaining_seconds=round(left, 1))
    return run


def can_continue(*, container: str, session_id: str, agent_cli: str) -> bool:
    """Ask the platform whether the recorded session can actually be resumed."""
    if not session_id:
        return False
    if agent_cli == "codex":
        got = backend.run(
            container,
            f"find -L {backend.AGENT_HOME}/.codex/sessions -type f "
            f"-name {shlex.quote('*' + session_id + '*.jsonl')} -print -quit",
            timeout=60)
        return got.returncode == 0 and bool(got.stdout.strip())
    # The session's transcript on disk, not a model call: a probe turn would send no
    # provider settings (on an API-key route it fails `Not logged in`, turning every
    # resume into a cold restart), cost a full-context request and write itself into
    # the agent's own session.
    got = backend.run(
        container,
        f"find -L {backend.AGENT_HOME}/.claude/projects -type f "
        f"-name {shlex.quote(session_id + '.jsonl')} -size +0 -print -quit",
        timeout=60)
    return got.returncode == 0 and bool(got.stdout.strip())


def develop(*, identity: RunIdentity, journal: Journal, container: str,
            task_dir: Path, agent_cli: str, model: str, effort: str,
            on_tick=None, sleep=time.sleep, launch_fn=None,
            can_continue_fn=None, env_file: str = "", agent_env: Path | str | None = None,
            extend_deadline=None, credit_left_fn=None) -> str:
    """The continuation loop: the agent is resumed until the deadline. Returns why
    it ended: deadline | terminal (it can never reach its model) | credit (a credit
    hold ended without credit) | launch_cap (a runaway guard). `agent_env` is the
    host's copy of the route's env file (asked for the credit left);
    `extend_deadline(deadline_utc)` records a credit hold's new deadline and returns
    the identity that carries it."""
    launch_fn = launch_fn or launch
    can_continue_fn = can_continue_fn or can_continue
    kind = "start"
    idle_streak = 0
    credit_streak = 0
    for _ in range(MAX_LAUNCHES):
        run = launch_fn(identity=identity, journal=journal, container=container,
                        task_dir=task_dir, agent_cli=agent_cli, model=model,
                        effort=effort, kind=kind, on_tick=on_tick,
                        **({"env_file": env_file} if env_file else {}))
        left = remaining_seconds(identity)
        if left <= EARLY_STOP_MARGIN_S:
            return "deadline"
        if run.credit_because:
            deadline = credit_hold(identity=identity, journal=journal, container=container,
                                   why=run.credit_because, agent_seconds_left=left,
                                   agent_env=agent_env, retry=credit_streak > 0, sleep=sleep,
                                   on_tick=on_tick, credit_left_fn=credit_left_fn)
            credit_streak += 1
            if deadline is None:
                return "credit"
            identity = (extend_deadline(deadline) if extend_deadline is not None
                        else dataclasses.replace(identity, deadline_utc=deadline))
            idle_streak = 0
            kind = ("continuation" if can_continue_fn(container=container, session_id=session_of(journal),
                                                      agent_cli=agent_cli) else "cold-restart")
            continue
        credit_streak = 0
        if run.terminal_because:
            journal.append("agent_cannot_start", why=run.terminal_because)
            return "terminal"
        # A launch the platform cut short (a session limit) is waited out, not
        # counted as the agent's idleness.
        idle = run.idle and run.session_limited_s is None
        if run.session_limited_s is not None:
            wait = min(run.session_limited_s, max(0.0, left - EARLY_STOP_MARGIN_S))
            journal.append("agent_session_limited", wait_seconds=round(wait, 1))
            _sleep_ticking(wait, sleep=sleep, on_tick=on_tick)
        if idle:
            idle_streak += 1
            _sleep_ticking(min(IDLE_BACKOFF_S * idle_streak, MAX_IDLE_BACKOFF_S),
                           sleep=sleep, on_tick=on_tick)
        else:
            idle_streak = 0
        if remaining_seconds(identity) <= EARLY_STOP_MARGIN_S:
            return "deadline"
        session = session_of(journal)
        if can_continue_fn(container=container, session_id=session,
                           agent_cli=agent_cli):
            kind = "idle-restart" if idle else "continuation"
        else:
            kind = "cold-restart"
    journal.append("continuation_declined", why="launch cap", cap=MAX_LAUNCHES)
    return "launch_cap"


def credit_hold(*, identity: RunIdentity, journal: Journal, container: str, why: str,
                agent_seconds_left: float, agent_env: Path | str | None = None, retry: bool = False,
                sleep=time.sleep, on_tick=None, credit_left_fn=None) -> str | None:
    """Hold the run while its provider refuses for want of credit: the container is
    paused (every process frozen, its files, GPU memory and card claim kept) and the
    janitor keeps off (`CREDIT_HOLD.json`). Credit back on the key (or, for a
    provider that cannot say, every CREDIT_RETRY_S) ends it: the container resumes
    and the new deadline returned is now + the agent's time left when the hold
    began, so the outage costs the agent nothing and gives it nothing. None: the
    hold ended without credit (CREDIT_HOLD_MAX_S, or `<run>/END_CREDIT_HOLD`)."""
    credit_left_fn = credit_left_fn or credit_left
    root = identity.root
    marker = root / CREDIT_HOLD_FILE
    paused = backend.pause(container)
    since = utc_now()
    write_json_atomically(marker, {"since_utc": since, "why": why, "deadline_utc": identity.deadline_utc,
                                   "agent_seconds_left": round(agent_seconds_left, 1), "paused": paused,
                                   "end_without_credit": f"touch {root / END_CREDIT_HOLD_FILE}"})
    journal.append("agent_credit_hold", why=why, agent_seconds_left=round(agent_seconds_left, 1),
                   deadline_utc=identity.deadline_utc, paused=paused)
    first_try = CREDIT_RETRY_S if retry else CREDIT_POLL_S
    held, outcome, left_usd = 0.0, "", None
    try:
        while not outcome:
            _sleep_ticking(CREDIT_POLL_S, sleep=sleep, on_tick=on_tick)
            held += CREDIT_POLL_S
            if (root / END_CREDIT_HOLD_FILE).exists() or held >= CREDIT_HOLD_MAX_S:
                outcome = "released"
            elif held >= first_try:
                left_usd = credit_left_fn(agent_env)
                if left_usd is None:
                    outcome = "retry" if held >= CREDIT_RETRY_S else ""
                elif left_usd >= CREDIT_RESUME_USD:
                    outcome = "credit"
    finally:
        resumed = backend.unpause(container) if paused else False
        marker.unlink(missing_ok=True)
    if outcome == "released":
        journal.append("agent_credit_hold_ended", outcome=outcome, since_utc=since, resumed=resumed,
                       credit_left_usd=left_usd)
        return None
    deadline = utc_after(agent_seconds_left)
    journal.append("agent_credit_hold_ended", outcome=outcome, since_utc=since, resumed=resumed,
                   credit_left_usd=None if left_usd is None or left_usd == float("inf") else round(left_usd, 2),
                   deadline_from=identity.deadline_utc, deadline_to=deadline)
    return deadline


def _sleep_ticking(seconds: float, *, sleep, on_tick, step: float = 5.0) -> None:
    """Wait `seconds` in `step` slices, ticking between them (counted, not
    clock-read, so an injected `sleep` makes this instantaneous in tests)."""
    left = max(0.0, float(seconds))
    while left > 0:
        if on_tick is not None:
            on_tick()
        slice_ = min(step, left)
        sleep(slice_)
        left -= slice_
