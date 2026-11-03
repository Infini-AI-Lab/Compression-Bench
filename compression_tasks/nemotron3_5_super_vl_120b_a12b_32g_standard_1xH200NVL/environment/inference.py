#!/usr/bin/env python3
"""Editable submitted inference starter; both generation and PPL use this engine.

The task computes PPL/reward. Submitted probabilities and code require separate
review before a reported score is considered accepted.
"""
import argparse
import os

for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VLLM_NO_USAGE_STATS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')


INFERENCE_API_VERSION = 1


def load(model_dir, *, max_model_len=2049, max_num_seqs=8,
         gpu_memory_utilization=0.80):
    return Engine(model_dir, max_model_len=max_model_len,
                  max_num_seqs=max_num_seqs,
                  gpu_memory_utilization=gpu_memory_utilization)


def end_of_sequence_ids(model_dir):
    """The end-of-sequence IDs of the checkpoint's config.json and generation_config.json."""
    import json
    from pathlib import Path
    found = []
    for name in ('config.json', 'generation_config.json'):
        path = Path(model_dir) / name
        if not path.is_file():
            continue
        cfg = json.loads(path.read_text())
        for value in (cfg.get('eos_token_id'), (cfg.get('text_config') or {}).get('eos_token_id')):
            for token in (value if isinstance(value, list) else [value]):
                if isinstance(token, int) and token not in found:
                    found.append(token)
    return found


class Engine:
    def __init__(self, model_dir, **kwargs):
        import json
        from pathlib import Path
        from vllm import LLM
        self.stop_token_ids = end_of_sequence_ids(model_dir)
        config = json.loads((Path(model_dir) / 'config.json').read_text())
        # vLLM's Nemotron-H Omni class tokenizes its media markers when it is built, so the
        # tokenizer is loaded; it reads `vit_hidden_size`, which Transformers' config renames.
        self.model = LLM(model=str(model_dir), skip_tokenizer_init=False,
                         hf_overrides={'vit_hidden_size': config['vit_hidden_size']},
                         trust_remote_code=False, dtype='auto', tensor_parallel_size=1,
                         language_model_only=True, enforce_eager=False,
                         enable_prefix_caching=False, speculative_config=None,
                         generation_config='vllm', disable_log_stats=True, **kwargs)

    def generate(self, token_ids, *, max_new_tokens, ignore_eos=False):
        """Batch of prompts -> batch of NEW token IDs (do not include prompts)."""
        from vllm import SamplingParams
        outputs = self.model.generate(
            [{'prompt_token_ids': ids} for ids in token_ids],
            SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=ignore_eos,
                           stop_token_ids=[] if ignore_eos else self.stop_token_ids,
                           detokenize=False), use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def token_logprobs(self, token_ids):
        """Return aligned natural-log probabilities from this submitted model."""
        from vllm import SamplingParams
        outputs = self.model.generate(
            [{'prompt_token_ids': ids} for ids in token_ids],
            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0,
                           detokenize=False, ignore_eos=True), use_tqdm=False)
        result = []
        for ids, output in zip(token_ids, outputs, strict=True):
            if list(output.prompt_token_ids) != ids or output.prompt_logprobs is None:
                raise ValueError('missing or mismatched prompt logprobs')
            if len(output.prompt_logprobs) != len(ids):
                raise ValueError('incomplete prompt logprobs')
            result.append([None] + [float(output.prompt_logprobs[i][ids[i]].logprob)
                                    for i in range(1, len(ids))])
        return result

    def stream_generate(self, token_ids, *, max_new_tokens, ignore_eos=False):
        """Single prompt -> incremental NEW token IDs, never timestamps or scores."""
        import uuid
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        engine = self.model.llm_engine
        if engine.has_unfinished_requests():
            raise RuntimeError('stream_generate requires an idle, single-request engine')
        request_id = engine.add_request(
            uuid.uuid4().hex, {'prompt_token_ids': token_ids},
            SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                           ignore_eos=ignore_eos, detokenize=False,
                           output_kind=RequestOutputKind.DELTA))
        try:
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if len(output.outputs) != 1:
                        raise ValueError('expected one streamed completion')
                    ids = list(output.outputs[0].token_ids)
                    if ids:
                        if len(ids) != 1:
                            raise ValueError('expected exactly one token per decode step; speculation is prohibited')
                        yield ids
        finally:
            if engine.has_unfinished_requests():
                engine.abort_request([request_id])


def main():
    # Optional CLI for local tests; the Checker calls load() and engine methods.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--tokenizer-dir')
    parser.add_argument('--prompt', default='Explain model compression in one sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.model_dir,
                                             local_files_only=True, trust_remote_code=False)
    ids = tokenizer(args.prompt, add_special_tokens=False).input_ids
    engine = load(args.model_dir, max_model_len=max(2048, len(ids) + args.max_new_tokens),
                  max_num_seqs=1)
    result = engine.generate([ids], max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
