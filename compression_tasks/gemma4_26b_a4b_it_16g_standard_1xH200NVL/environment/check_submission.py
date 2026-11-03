#!/usr/bin/env python3
"""Client for the isolated public compression Checker."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


SOCKET_PATH = Path("/run/compression_bench/checker.sock")
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def receive_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise RuntimeError("Checker response exceeded 64 MiB")
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--package-only", action="store_true",
                        help="check package files and size without loading the model")
    parser.add_argument("--debug-9b", action="store_true",
                        help="diagnose a Qwen3.5-9B candidate; never a formal score")
    parser.add_argument("--timeout", type=int, default=2100)
    args = parser.parse_args()

    expected = Path("/app/debug-submission" if args.debug_9b else "/app/submission").resolve()
    if args.submission.resolve() != expected:
        raise SystemExit(f"this Checker mode only accepts {expected}")

    request = {
        "protocol": 1,
        "action": "package-only" if args.package_only else "full-check",
    }
    if args.debug_9b:
        request["mode"] = "debug-9b"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(args.timeout)
        connection.connect(str(SOCKET_PATH))
        connection.sendall(json.dumps(request).encode("utf-8") + b"\n")
        payload = receive_line(connection)

    if not payload:
        raise SystemExit("public Checker returned an empty response")
    response = json.loads(payload)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if not response.get("valid", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
