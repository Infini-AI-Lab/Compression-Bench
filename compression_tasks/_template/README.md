# `_template/` — what every task is rendered from

| path | used for |
|---|---|
| `instruction/instruction.md.in` | the specification, in ServerBench's fact-only shape: goal and reward, hard gates, measurements, workspace and container inventory, the checkpoint's structure, the Checker, the verdict, the submission contract; `{{KEY}}` scalars and `@@IF FLAG@@ … @@END@@` blocks |
| `instruction/framework.md.in` | what every instruction ends with: versions, checkpoints on disk and the budget, then the `## Continuation`, `## Idle restart` and `## Cold restart` session sections the harness sends on a later launch |
| `docker/environment/` | the agent container's compose file and the agent tools (`check_submission.py`, `model_inventory.py`, `submission_guard.py`) |
| `docker/tests/` | the Verifier's compose file and `test.sh` (the single-copy lock) |
| `docker/MODE.md`, `docker/assets-*.md` | prose copied into each task |
| `starters/inference.<starter>.py` | the editable `/app/inference.py` the agent starts from (`contract.yaml`'s `starter`) |

Dockerfiles and `task.toml` are generated in code (`_manage/render.py`) from the
contract and `compression_bench/environments/images.json`. The evaluator is not here:
it is `compression_bench/harness/final_validate/`, packaged byte-for-byte into each task.
