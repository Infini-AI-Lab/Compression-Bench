"""The compression-bench evolve driver: one run, from workspace to verdict.

`compression_bench/scripts/evolve` hands its arguments to `run.main`. Each stage is one
module here; see `compression_bench/docs/INTERFACES.md` for the seams with the task
tooling and the evaluator. Stdlib only: yaml is imported inside functions.
"""
