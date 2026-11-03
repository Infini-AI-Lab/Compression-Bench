"""What a run's containers are made of: pure functions, no Docker, no Harbor.

`scripts/agent/harbor_env.py` (run under the Harbor interpreter) builds the
environment from these; the driver and the tests import them directly.

The agent environment is the task's own `environment/` (main + public-checker)
composed by Harbor, plus one per-run overlay that:

* mounts the host's agent CLI binary read-only at `/opt/compression_bench/bin/<cli>`
  (the base image is not rebuilt per CLI version; the binary's SHA-256 is
  recorded);
* mounts the run's staged credentials file read-only into the agent's home;
* sets the CLI's no-telemetry environment on main;
* gives Harbor's egress sidecar a gost config with a 600 s read timeout, so a
  slow model reply is not cut off.

Egress is Harbor's own allowlist (its egress-control sidecar): the agent can
reach its model provider and nothing else. The public Checker runs with
`network_mode: none`, and the hidden Verifier with no network at all.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .backend import CLI_DIR
from .credentials import IN_CONTAINER
from .utils import REPO, sha256_file, until_space

PROVIDER_HOSTS: dict[str, tuple[str, ...]] = {
    "claude": ("api.anthropic.com", "statsig.anthropic.com", "sentry.io"),
    "codex": ("chatgpt.com", "auth.openai.com", "api.openai.com"),
}

#: Where the provider is probed from inside main during the boot proof.
PROVIDER_PROBE_URL = {
    "claude": "https://api.anthropic.com/",
    "codex": "https://api.openai.com/",
}

AGENT_ENV = {
    "claude": {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1",
               "IS_SANDBOX": "1"},
    "codex": {"CODEX_HOME": "/app/work/.agent-home/.codex"},
}

#: Where an API-key provider route's env file (`--agent-env`) is bound inside main;
#: the launch command sources it, so no key is ever on a command line.
AGENT_ENV_FILE = "/run/compression-bench-agent.env"
#: The env-file key that routes Claude Code's model to another provider.
BASE_URL_KEY = "ANTHROPIC_BASE_URL"

DEBUG_OVERLAY = REPO / "compression_bench" / "scripts" / "agent" / "debug-model-compose.yaml"
GOST_CONFIG = REPO / "compression_bench" / "scripts" / "agent" / "egress-gost.yaml"
EGRESS_SERVICE = "harbor-docker-egress-control-sidecar"


def read_env_file(path: Path | str) -> dict[str, str]:
    """KEY=VALUE lines of an agent env file (`#` comments, optional `export`)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def provider_base_url(agent_env: Path | str | None) -> str:
    """The base URL an agent env file routes the model to, or ''."""
    if agent_env is None:
        return ""
    return read_env_file(agent_env).get(BASE_URL_KEY, "")


def allowed_hosts(agent_cli: str, extra: tuple[str, ...] = (), provider_url: str = "") -> list[str]:
    """The hosts the agent may reach: its CLI's provider, or, when a provider route's
    env file sends the model elsewhere (`provider_url`), that provider's host alone.
    `extra` hosts are appended."""
    hosts = [urlparse(provider_url).hostname] if provider_url else list(PROVIDER_HOSTS[agent_cli])
    return hosts + [h for h in extra if h and h not in hosts]


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


#: Where the Codex npm package keeps its native binary: its per-platform optional
#: dependency, `node_modules/@openai/codex-linux-x64/vendor/<triple>/bin/codex`.
CODEX_NATIVE_GLOB = "node_modules/@openai/codex-linux-*/vendor/*linux*/bin/codex"


def resolve_cli_binary(agent_cli: str, *, which=shutil.which) -> Path:
    """The host's native CLI executable for `agent_cli`, fully resolved.

    Claude Code ships a single native binary. Codex's `codex` on PATH is a
    Node launcher; the native binary lives in its package's `vendor/` tree.
    The container has no Node, so only a native ELF is acceptable.
    """
    found = which(agent_cli) or ""
    if agent_cli == "claude" and not found:
        candidate = Path("~/.local/bin/claude").expanduser()
        found = str(candidate) if candidate.exists() else ""
    if not found:
        raise FileNotFoundError(f"no `{agent_cli}` on this host's PATH")
    path = Path(found).resolve()
    if _is_elf(path):
        return path
    if agent_cli == "codex":
        # npm global layout: <prefix>/bin/codex -> ../lib/node_modules/@openai/codex/bin/codex.js
        packages = [path.parent.parent / "lib" / "node_modules" / "@openai" / "codex"]
        package = path.parent
        for _ in range(4):
            packages.append(package)
            package = package.parent
        for package in packages:
            for binary in sorted(package.glob(CODEX_NATIVE_GLOB)):
                if _is_elf(binary) and "x86_64" in str(binary):
                    return binary.resolve()
    raise FileNotFoundError(f"`{agent_cli}` at {path} is not a native executable "
                            f"and no native binary was found beside it")


def model_link_mounts(model_dir: Path, *, container_dir: str = "/input/model",
                      services=("main", "public-checker")) -> dict | None:
    """An overlay that makes a symlinked checkpoint (a Hugging Face cache
    snapshot: relative links into `../../blobs/`) resolve inside containers.

    Each distinct directory the links point into is mounted read-only at the
    path the same relative link reaches from `container_dir`. None when the
    checkpoint has no symlinks."""
    model_dir = Path(model_dir)
    mounts: dict[str, str] = {}
    for path in sorted(model_dir.iterdir()) if model_dir.is_dir() else ():
        if not path.is_symlink():
            continue
        link = os.readlink(path)
        host_target = (path.parent / link).resolve().parent
        inside = os.path.normpath(os.path.join(container_dir, os.path.dirname(link)))
        if inside == os.path.normpath(container_dir) or not inside.startswith("/"):
            continue
        mounts[inside] = str(host_target)
    if not mounts:
        return None
    volumes = [{"type": "bind", "source": source, "target": target, "read_only": True}
               for target, source in sorted(mounts.items())]
    return {"services": {name: {"volumes": volumes} for name in services}}


def compose_variables(*, run_root: Path, model_dir: Path, gpu: int,
                      debug_model_dir: Path | None = None) -> dict[str, str]:
    """The COMPRESSION_BENCH_* variables the task's compose files interpolate."""
    variables = {
        "COMPRESSION_BENCH_RUN_DIR": str(Path(run_root).resolve()),
        "COMPRESSION_BENCH_MODEL_DIR": str(Path(model_dir).resolve()),
        "COMPRESSION_BENCH_GPU_DEVICE": str(int(gpu)),
        "COMPRESSION_BENCH_HOST_UID": str(os.getuid()),
        "COMPRESSION_BENCH_HOST_GID": str(os.getgid()),
    }
    if debug_model_dir is not None:
        variables["COMPRESSION_BENCH_DEBUG_MODEL_DIR"] = str(Path(debug_model_dir).resolve())
    return variables


def agent_overlay(*, agent_cli: str, cli_binary: Path,
                  credentials: Path | None, gost_config: Path = GOST_CONFIG,
                  agent_env: Path | None = None) -> dict:
    """The per-run compose overlay (a dict; JSON is valid compose YAML)."""
    volumes = [{"type": "bind", "source": str(Path(cli_binary).resolve()),
                "target": f"{CLI_DIR}/{agent_cli}", "read_only": True}]
    if credentials is not None:
        volumes.append({"type": "bind", "source": str(Path(credentials).resolve()),
                        "target": IN_CONTAINER[agent_cli], "read_only": True})
    if agent_env is not None:
        volumes.append({"type": "bind", "source": str(Path(agent_env).resolve()),
                        "target": AGENT_ENV_FILE, "read_only": True})
    services = {
        "main": {"environment": dict(AGENT_ENV[agent_cli]), "volumes": volumes},
        EGRESS_SERVICE: {"volumes": [{
            "type": "bind", "source": str(Path(gost_config).resolve()),
            "target": "/opt/egress-sidecar/gost.yaml", "read_only": True}]},
    }
    return {"services": services}


def write_overlay(path: Path, overlay: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    until_space(lambda: path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8"),
                f"writing {path.name}")
    return path


def verifier_mounts(run_root: Path) -> list[dict]:
    """The Verifier's one writable mount: the run's `verifier/` at
    /logs/verifier (passed to Harbor as `mounts`, so Harbor creates, chmods and
    finally chowns it back to the host user)."""
    return [{"type": "bind",
             "source": str((Path(run_root) / "verifier").resolve()),
             "target": "/logs/verifier"}]


def cli_fingerprint(cli_binary: Path) -> dict:
    return {"path": str(cli_binary), "sha256": sha256_file(cli_binary),
            "bytes": Path(cli_binary).stat().st_size}
