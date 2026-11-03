"""What an agent's tokens cost, in one rule.

* Closed models (Claude, GPT) keep their own CLI's price: Claude Code's
  `modelUsage[model].costUSD` at Anthropic's list rates (`costBasis: "list"`).
* Open-weight models reached through a provider route (e.g. GLM via Fireworks)
  are priced at OpenRouter's published rates, from the snapshot
  `openrouter_prices.json` (refreshed by `compression_bench/scripts/openrouter_prices.py`).
  Claude Code does not know their price (`costBasis: "unknown"`) and would fall
  back to Opus-tier rates, which overstate an open model's cost many times over.

A launch the harness ends mid-turn (the deadline, a cancel) writes no `result`,
so the CLI never prices what it used after its last one. That tail is priced here
(`tail_cost`): its input and cache tokens exactly, from each message's own usage;
its output estimated from the streamed thinking-token deltas and the visible text
(`report.usage_totals`), since a message's usage carries only its first output
tokens. A closed model's tail is priced at Anthropic's list rates (`CLAUDE_LIST`,
which reproduce the CLI's own `costUSD`), an open model's at OpenRouter's.

A model neither rule can price has no cost (None), never a guess.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PRICES = Path(__file__).with_name("openrouter_prices.json")

#: OpenRouter vendor of each open-weight family a provider route reaches
#: (`scripts/agent/routes/`), by the model name's first word.
OPEN_VENDORS = {"glm": "z-ai", "qwen": "qwen", "deepseek": "deepseek",
                "kimi": "moonshotai", "minimax": "minimax"}


@lru_cache(maxsize=1)
def snapshot() -> dict:
    try:
        return json.loads(PRICES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"models": {}, "aliases": {}}


#: USD per million tokens: input, output, cache read, 5-minute cache write, 1-hour
#: cache write, matching Claude Code's own `costUSD` on its result events; the
#: 5-minute write is the list ratio, 1.25x input.
CLAUDE_LIST = {"claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.50, 4.0),
               "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0, 8.0)}


def is_closed(model: str) -> bool:
    """A model its own CLI prices: Claude (Claude Code) or GPT (Codex)."""
    return model.lower().rsplit("/", 1)[-1].startswith(("claude-", "gpt-"))


def openrouter_id(model: str) -> str | None:
    """`accounts/fireworks/models/glm-5p3-flash` -> `z-ai/glm-5.3-flash`."""
    table = snapshot()
    alias = (table.get("aliases") or {}).get(model)
    if alias:
        return alias
    if model in (table.get("models") or {}):
        return model
    name = re.sub(r"(?<=\d)p(?=\d)", ".", model.rsplit("/", 1)[-1].lower())
    for prefix, vendor in OPEN_VENDORS.items():
        if name.startswith(prefix):
            return f"{vendor}/{name}"
    return None


def open_model_cost(model: str, usage: dict) -> tuple[float | None, str]:
    """(USD, basis) of one open model's tokens at OpenRouter's rates."""
    key = openrouter_id(model)
    rates = (snapshot().get("models") or {}).get(key or "")
    if not rates:
        return None, f"unpriced ({model} not in the OpenRouter snapshot)"
    prompt = float(rates["prompt"])
    read = float(rates.get("input_cache_read") or prompt)
    write = float(rates.get("input_cache_write") or prompt)
    usd = (usage.get("input_tokens", 0) * prompt
           + usage.get("cache_read_input_tokens", 0) * read
           + usage.get("cache_creation_input_tokens", 0) * write
           + usage.get("output_tokens", 0) * float(rates["completion"]))
    return round(usd, 6), f"openrouter {key} {snapshot().get('fetched_utc', '')[:10]}"


def tail_cost(model: str, tail: dict) -> tuple[float | None, str]:
    """(USD, basis) of the usage a launch ended mid-turn left unpriced by the CLI."""
    if is_closed(model):
        rates = CLAUDE_LIST.get(model.rsplit("/", 1)[-1])
        if rates is None:
            return None, f"unpriced tail ({model} has no list rates here)"
        inp, out, read, write_5m, write_1h = rates
        usd = (tail.get("input_tokens", 0) * inp + tail.get("output_tokens", 0) * out
               + tail.get("cache_read_input_tokens", 0) * read
               + tail.get("cache_write_5m_tokens", 0) * write_5m
               + tail.get("cache_write_1h_tokens", 0) * write_1h) / 1e6
        return round(usd, 6), "list"
    return open_model_cost(model, tail)


def run_cost(models: dict[str, dict]) -> dict:
    """The run's cost from its per-model usage (`report.usage_totals()["models"]`)."""
    total, bases, unpriced = 0.0, [], False
    for model, usage in sorted(models.items()):
        tail = usage.get("tail")
        if is_closed(model):
            usd = usage.get("cli_cost_usd") or 0.0
            basis = f"cli {usage.get('cost_basis') or 'list'}" if usage.get("cost_basis") or not tail else ""
        else:
            usd, basis = open_model_cost(model, usage)
        if tail and usd is not None:
            extra, how = tail_cost(model, tail)
            if extra is None:
                usd, basis = None, how
            else:
                usd += extra
                rates = "list rates" if is_closed(model) else how
                basis = " + ".join(filter(None, (basis, f"{tail['messages']} messages without a result "
                                                        f"event ({rates}, output estimated)")))
        if usd is None:
            unpriced = True
        else:
            total += usd
        bases.append(basis)
    return {"cost_usd": None if unpriced or not models else round(total, 6),
            "cost_basis": "; ".join(dict.fromkeys(bases))}
