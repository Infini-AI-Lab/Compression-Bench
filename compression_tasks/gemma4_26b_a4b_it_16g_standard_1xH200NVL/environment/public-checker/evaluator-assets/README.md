# Public evaluation inputs

`ppl_inputs.json` stores the fixed perplexity token blocks (hidden text continued by the BF16 checkpoint). `ppl_config.json` defines the
tokenizer, split, token counts and BF16/Reference PPL baselines.
The evaluator checks the protocol, split, token counts and token-ID range before measuring PPL.

`decode_inputs.json` stores the fixed decode prompts. `decode_config.json`
defines the request lengths, warmup, measurement rounds and scoring inputs.

`task_policy.json` is this task's evaluation policy (capacity, payload rule,
quality rule, PPL protocol), rendered from the task's `contract.yaml`.
`evaluation_profiles.json` selects `ppl_assets`, `ppl_split` and `decode_assets`
for each supported Checker profile. The optional `debug-9b` profile has its own
tokenizer-specific inputs and returns diagnostic measurements only.

Public and hidden evaluation use separate inputs. Hidden inputs exist only in
the Verifier image.
