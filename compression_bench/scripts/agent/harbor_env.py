#!/usr/bin/env python3
"""Compose, start and stop a run's containers through Harbor.

Runs under COMPRESSION_BENCH_HARBOR_PY (an interpreter with harbor==0.21.0), as a
subprocess of the lifecycle. Harbor is the composition and start engine, as in
ServerBench; after `start` the harness talks to the main container with plain
`docker exec` by its exact name.

    harbor_env.py start  --task-dir T --run-root R --project P --gpu N --model-dir M
                         --agent-cli claude|codex [--credentials F] [--agent-env F]
                         [--debug-model-dir D]
    harbor_env.py stop   --run-root R --project P [--remove-images]
    harbor_env.py verify --task-dir T --run-root R --project P --gpu N --model-dir M [--timeout S]

`start` builds the task's `environment/` (Harbor builds main from its
Dockerfile; Compose builds the public Checker), routes main through Harbor's
egress-control sidecar with the provider allowlist, adds the per-run overlay
(`lifecycle/containers.py`), and records everything needed to stop it again in
`<run>/compose/compose.json`, because Harbor's own overlay files are temporary.

`verify` starts a SEPARATE environment from the task's `tests/` (no network),
mounts `<run>/verifier` at /logs/verifier, runs `bash /tests/test.sh` with the
Verifier timeout, and removes it. `test.sh` refuses to grade while any main
container still holds the submission lock.

Every subcommand prints one JSON object on stdout (the last line).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))

from lifecycle import containers  # noqa: E402
from lifecycle.utils import read_contract, utc_now, write_json_atomically  # noqa: E402


def _harbor():
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    return DockerEnvironment, EnvironmentConfig, NetworkMode, NetworkPolicy, TrialPaths


def _sanitize(project: str) -> str:
    """The project name exactly as Harbor (and Compose) will use it, so label
    filters match what is really running."""
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name
    return _sanitize_docker_compose_project_name(project)


def _project_containers(project: str) -> dict[str, str]:
    project = _sanitize(project)
    got = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={project}",
         "--format", '{{.Names}}\t{{.Label "com.docker.compose.service"}}\t{{.Image}}'],
        capture_output=True, text=True, timeout=60, check=False)
    found = {}
    for line in got.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            found[parts[1]] = parts[0]
    return found


def _container_image(container: str) -> str:
    got = subprocess.run(["docker", "inspect", "-f", "{{.Image}}", container],
                         capture_output=True, text=True, timeout=60, check=False)
    return got.stdout.strip()


def _persist_compose(env, run_root: Path, project: str, variables: dict) -> dict:
    """Copy every compose file Harbor used into the run, and record how to
    replay `docker compose` against them (for `stop`)."""
    folder = run_root / "compose"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for index, path in enumerate(env._docker_compose_paths):
        source = Path(path).resolve()
        target = folder / f"{index:02d}-{source.name}"
        shutil.copyfile(source, target)
        files.append({"original": str(source), "copy": str(target)})
    infra = env._compose_env_vars(include_os_env=False)
    record = {"kind": "compression_bench_compose", "schema_version": 1,
              "project": project,
              "project_directory": str(env.environment_dir.resolve()),
              "files": files, "env": {**infra, **variables},
              "written_utc": utc_now()}
    write_json_atomically(folder / "compose.json", record)
    return record


def _compose(record: dict, *command: str, timeout: float = 900.0) -> subprocess.CompletedProcess:
    argv = ["docker", "compose", "--project-name", record["project"],
            "--project-directory", record["project_directory"]]
    for entry in record["files"]:
        argv += ["-f", entry["copy"]]
    argv += list(command)
    env = dict(os.environ)
    env.update(record["env"])
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=env, check=False)


def _remove_by_label(project: str) -> dict:
    """Last resort: remove the project's containers, networks and volumes by
    their Compose label (never by name pattern)."""
    project = _sanitize(project)
    removed = {"containers": [], "networks": [], "volumes": []}
    for kind, list_cmd, rm_cmd in (
            ("containers", ["docker", "ps", "-aq"], ["docker", "rm", "-f"]),
            ("networks", ["docker", "network", "ls", "-q"], ["docker", "network", "rm"]),
            ("volumes", ["docker", "volume", "ls", "-q"], ["docker", "volume", "rm", "-f"])):
        got = subprocess.run(list_cmd + ["--filter", f"label=com.docker.compose.project={project}"],
                             capture_output=True, text=True, timeout=60, check=False)
        ids = got.stdout.split()
        if ids:
            subprocess.run(rm_cmd + ids, capture_output=True, text=True, timeout=300,
                           check=False)
            removed[kind] = ids
    return removed


# --- start --------------------------------------------------------------------------

async def _start(args) -> dict:
    DockerEnvironment, EnvironmentConfig, NetworkMode, NetworkPolicy, TrialPaths = _harbor()
    task_dir = Path(args.task_dir).resolve()
    run_root = Path(args.run_root).resolve()
    contract = read_contract(task_dir)
    project = _sanitize(args.project)
    variables = containers.compose_variables(
        run_root=run_root, model_dir=Path(args.model_dir), gpu=args.gpu,
        debug_model_dir=Path(args.debug_model_dir) if args.debug_model_dir else None)
    os.environ.update(variables)

    cli_binary = containers.resolve_cli_binary(args.agent_cli)
    credentials = Path(args.credentials) if args.credentials else None
    agent_env = Path(args.agent_env) if args.agent_env else None
    overlay = containers.write_overlay(
        run_root / "compose" / "agent-overlay.json",
        containers.agent_overlay(agent_cli=args.agent_cli, cli_binary=cli_binary,
                                 credentials=credentials, agent_env=agent_env))
    extra = [overlay]
    links = containers.model_link_mounts(Path(args.model_dir))
    if links:
        extra.append(containers.write_overlay(run_root / "compose" / "model-links.json", links))
    if args.debug_model_dir:
        extra.append(containers.DEBUG_OVERLAY)
        debug_links = containers.model_link_mounts(
            Path(args.debug_model_dir), container_dir="/input/debug-model")
        if debug_links:
            extra.append(containers.write_overlay(
                run_root / "compose" / "debug-model-links.json", debug_links))
    hosts = containers.allowed_hosts(args.agent_cli, provider_url=containers.provider_base_url(agent_env))

    trial_dir = run_root / "harbor"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = DockerEnvironment(
        environment_dir=task_dir / "environment",
        environment_name=contract["task_id"],
        session_id=project,
        trial_paths=TrialPaths(trial_dir),
        task_env_config=EnvironmentConfig(build_timeout_sec=float(args.build_timeout)),
        keep_containers=True,
        network_policy=NetworkPolicy(network_mode=NetworkMode.ALLOWLIST,
                                     allowed_hosts=hosts),
        extra_docker_compose=[str(p) for p in extra],
        mounts=[],
    )
    started = time.monotonic()
    try:
        await asyncio.wait_for(env.start(force_build=False), timeout=float(args.build_timeout))
    except BaseException:
        try:
            _persist_compose(env, run_root, project, variables)
        except Exception:                                   # noqa: BLE001
            pass
        raise
    record = _persist_compose(env, run_root, project, variables)
    services = _project_containers(project)
    main = services.get("main", "")
    if not main:
        raise RuntimeError(f"Harbor started project {project} but no main container is listed")
    result = {
        "kind": "compression_bench_container", "schema_version": 1, "status": "started",
        "project": project, "task_id": contract["task_id"],
        "main_container": main, "services": services,
        "main_image_id": _container_image(main),
        "allowed_hosts": hosts, "agent_cli": args.agent_cli,
        "cli": containers.cli_fingerprint(cli_binary),
        "credentials_mounted": credentials is not None,
        "agent_env_mounted": agent_env is not None,
        "debug_model": bool(args.debug_model_dir),
        "compose": {"files": [f["copy"] for f in record["files"]]},
        "compose_variables": variables, "start_seconds": round(time.monotonic() - started, 1),
        "started_utc": utc_now(), "harbor_version": _harbor_version(),
    }
    write_json_atomically(run_root / "harbor_environment.json", result)
    write_json_atomically(run_root / "container.json",
                          {"main_container": main, "project": project,
                           "services": services, "written_utc": utc_now()})
    return result


def _harbor_version() -> str:
    from importlib.metadata import version
    return version("harbor")


# --- stop ---------------------------------------------------------------------------

def _stop(args) -> dict:
    run_root = Path(args.run_root).resolve()
    record_path = run_root / "compose" / "compose.json"
    project = _sanitize(args.project)
    result = {"kind": "compression_bench_container_stop", "project": project, "utc": utc_now()}
    if record_path.exists():
        record = json.loads(record_path.read_text())
        project = record["project"]
        result["project"] = project
        command = ["down", "--remove-orphans", "--volumes", "--timeout", "30"]
        if args.remove_images:
            command += ["--rmi", "local"]
        got = _compose(record, *command)
        result["compose_down"] = {"returncode": got.returncode,
                                  "tail": (got.stdout + got.stderr)[-1500:]}
    left = _project_containers(project)
    if left:
        result["removed_by_label"] = _remove_by_label(project)
    result["remaining"] = list(_project_containers(project).values())
    result["status"] = "stopped" if not result["remaining"] else "incomplete"
    return result


# --- verify -------------------------------------------------------------------------

async def _verify(args) -> dict:
    DockerEnvironment, EnvironmentConfig, NetworkMode, NetworkPolicy, TrialPaths = _harbor()
    task_dir = Path(args.task_dir).resolve()
    run_root = Path(args.run_root).resolve()
    contract = read_contract(task_dir)
    timeout = int(args.timeout or contract["budget"]["verifier_timeout_s"])
    variables = containers.compose_variables(
        run_root=run_root, model_dir=Path(args.model_dir), gpu=args.gpu)
    os.environ.update(variables)
    verifier_dir = run_root / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    project = _sanitize(f"{args.project}-verifier")
    trial_dir = run_root / "harbor-verifier"
    trial_dir.mkdir(parents=True, exist_ok=True)
    links = containers.model_link_mounts(Path(args.model_dir), services=("main",))
    extra = ([str(containers.write_overlay(run_root / "compose" / "verifier-model-links.json",
                                           links))] if links else [])
    env = DockerEnvironment(
        environment_dir=task_dir / "tests",
        environment_name=f"{contract['task_id']}-verifier",
        session_id=project,
        trial_paths=TrialPaths(trial_dir),
        task_env_config=EnvironmentConfig(build_timeout_sec=float(args.build_timeout)),
        keep_containers=False,
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        mounts=containers.verifier_mounts(run_root),
        extra_docker_compose=extra,
    )
    started = time.monotonic()
    result = {"kind": "compression_bench_verify", "schema_version": 1, "project": project,
              "task_id": contract["task_id"], "timeout_s": timeout,
              "started_utc": utc_now(), "harbor_version": _harbor_version()}
    try:
        await asyncio.wait_for(env.start(force_build=False), timeout=float(args.build_timeout))
        result["build_and_start_seconds"] = round(time.monotonic() - started, 1)
        try:
            got = await env.exec("bash /tests/test.sh", timeout_sec=timeout)
            result["returncode"] = got.return_code
            output = (got.stdout or "") + (got.stderr or "")
        except RuntimeError as exc:          # Harbor raises on its own timeout
            result["returncode"] = None
            result["error"] = str(exc)[:2000]
            output = str(exc)
        (verifier_dir / "test-stdout.txt").write_text(output, encoding="utf-8")
        result["output_tail"] = output[-3000:]
    finally:
        try:
            await env.stop(delete=True)
        except Exception as exc:                            # noqa: BLE001
            result["stop_error"] = str(exc)[:500]
        if _project_containers(project):
            result["removed_by_label"] = _remove_by_label(project)
    result["seconds"] = round(time.monotonic() - started, 1)
    result["finished_utc"] = utc_now()
    reward = verifier_dir / "reward.txt"
    result["reward_txt"] = reward.read_text().strip() if reward.exists() else None
    write_json_atomically(verifier_dir / "harbor-verify.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--task-dir", required=True)
        p.add_argument("--run-root", required=True)
        p.add_argument("--project", required=True)
        p.add_argument("--gpu", type=int, required=True)
        p.add_argument("--model-dir", required=True)
        p.add_argument("--build-timeout", type=float, default=3600.0)
    start = sub.choices["start"]
    start.add_argument("--agent-cli", choices=sorted(containers.PROVIDER_HOSTS), default="claude")
    start.add_argument("--credentials")
    start.add_argument("--agent-env", help="an API-key provider route's env file, bound read-only")
    start.add_argument("--debug-model-dir")
    sub.choices["verify"].add_argument("--timeout", type=int)
    stop = sub.add_parser("stop")
    stop.add_argument("--run-root", required=True)
    stop.add_argument("--project", required=True)
    stop.add_argument("--remove-images", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            result = asyncio.run(_start(args))
        elif args.command == "verify":
            result = asyncio.run(_verify(args))
        else:
            result = _stop(args)
    except BaseException as exc:                            # noqa: BLE001
        print(json.dumps({"status": "error", "command": args.command,
                          "error": f"{type(exc).__name__}: {exc}"[-4000:]}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
