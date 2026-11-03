#!/usr/bin/env python3
"""Serialized Unix-socket runner for the isolated public Checker."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 65536
TOKENIZER_NOTICE = (
    "Candidate tokenizer files are not used for scoring. "
    "For calibration and local tests, use the original model's official tokenizer: "
    "/input/model for formal tasks, or /input/debug-model for debug-9b. "
    "Scores obtained with a different tokenizer are not comparable."
)


def receive_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_REQUEST_BYTES:
            raise ValueError("request exceeded 64 KiB")
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def send_response(connection: socket.socket, response: dict[str, Any]) -> None:
    response = {**response, "tokenizer_notice": TOKENIZER_NOTICE}
    try:
        connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
    except ConnectionError as exc:
        # A disconnected client must not stop the service.
        print(f"Checker client disconnected; response not delivered: {exc}",
              file=sys.stderr, flush=True)


def submission_listing(root: Path) -> list[list[Any]]:
    """[relative path, size] of every file: enough to tell which version was checked."""
    listing: list[list[Any]] = []
    try:
        for path in sorted(root.rglob("*")):
            try:
                if path.is_file():
                    listing.append([path.relative_to(root).as_posix(), path.stat().st_size])
            except OSError:
                continue                        # the agent may change it while we look
    except OSError:
        pass
    return listing


def append_history(history: Path | None, row: dict[str, Any]) -> None:
    """One line per request, for the run's record. The agent's container does not
    mount this file; a failure to write it never affects the response."""
    if history is None:
        return
    try:
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Checker history not written: {exc}", file=sys.stderr, flush=True)


def run_evaluator(
    evaluator: Path,
    submission: Path,
    package_only: bool,
    timeout: int,
    profile: str = "formal",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="quant-checker-") as temp_dir:
        result_path = Path(temp_dir) / "result.json"
        command = [
            sys.executable,
            str(evaluator),
            str(submission),
            "--output",
            str(result_path),
            "--profile",
            profile,
        ]
        if package_only:
            command.append("--package-only")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            result = {
                "valid": False,
                "error": "public evaluator returned invalid JSON",
            }
    if completed.returncode != 0 and result.get("valid", True):
        result = {"valid": False, "error": "public evaluator failed"}
    if not result.get("valid", False) and completed.stderr:
        result["diagnostic"] = completed.stderr[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluation-timeout", type=int, required=True)
    parser.add_argument("--debug-submission", type=Path, default=Path("/debug-submission"))
    parser.add_argument("--history", type=Path,
                        help="append every request and response here (JSON lines)")
    args = parser.parse_args()

    if args.evaluation_timeout <= 0:
        raise SystemExit("Checker evaluation timeout must be positive")
    if not args.evaluator.is_file():
        raise SystemExit(f"public evaluator does not exist: {args.evaluator}")

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def stop(_signum: int, _frame: object) -> None:
        listener.close()
        if args.socket.exists():
            args.socket.unlink()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    listener.bind(str(args.socket))
    os.chmod(args.socket, 0o666)
    listener.listen(1)

    full_checks = 0
    while True:
        connection, _ = listener.accept()
        with connection:
            started = time.time()
            row: dict[str, Any] = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
            try:
                request = json.loads(receive_line(connection))
                row["request"] = request
                if not isinstance(request, dict):
                    raise ValueError("Checker request must be an object")
                if request.get("protocol") != 1:
                    raise ValueError("unsupported Checker protocol")
                action = request.get("action")
                mode = request.get("mode", "formal")
                if not isinstance(mode, str) or mode not in {"formal", "debug-9b"}:
                    raise ValueError("unsupported Checker mode")
                if not isinstance(action, str) or action not in {"package-only", "full-check"}:
                    raise ValueError("unsupported Checker action")
                if action == "full-check":
                    full_checks += 1
                response = run_evaluator(
                    args.evaluator,
                    args.debug_submission if mode == "debug-9b" else args.submission,
                    action == "package-only",
                    args.evaluation_timeout,
                    profile=mode,
                )
                response["full_checks_used"] = full_checks
                if mode == "debug-9b":
                    response.update(debug_only=True, is_official_score=False)
                    for key in ("reward", "public_reward", "target_met"):
                        response.pop(key, None)
                send_response(connection, response)
                row.update(submission=submission_listing(
                    args.debug_submission if mode == "debug-9b" else args.submission),
                    response=response)
            except ConnectionError as exc:
                print(f"Checker client disconnected while receiving: {exc}",
                      file=sys.stderr, flush=True)
            except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                send_response(connection, {"valid": False, "error": str(exc)})
                row["response"] = {"valid": False, "error": str(exc)}
            row["seconds"] = round(time.time() - started, 3)
            append_history(args.history, row)


if __name__ == "__main__":
    main()
