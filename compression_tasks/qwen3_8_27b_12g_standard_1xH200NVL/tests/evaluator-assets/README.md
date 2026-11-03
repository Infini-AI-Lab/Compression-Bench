# Hidden evaluation inputs

`ppl_inputs.json` stores the fixed perplexity token blocks (hidden text continued by the BF16 checkpoint). `ppl_config.json` defines the
tokenizer, split, token counts and BF16/Reference PPL baselines.
`decode_inputs.json` stores the fixed decode prompts. `decode_config.json`
defines the request lengths, warmup, measurement rounds and scoring inputs.
`task_policy.json` is this task's evaluation policy, rendered from `contract.yaml`.

These inputs belong only in the Verifier image. They are never delivered to
the agent's container or the public Checker.
