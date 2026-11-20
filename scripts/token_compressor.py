"""Dynamic Token Compression Estimator & Context Window Optimizer.

Evaluates prompt compaction efficiency, strips redundant conversational framing,
calculates compression ratios, and projects micro-dollar cost savings across LLM models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Simple approximation for BPE token count: ~4 chars per token for English text
CHARS_PER_TOKEN = 4.0


def approximate_token_count(text: str) -> int:
    """Estimate token count based on character length and whitespace tokens."""
    if not text:
        return 0
    words = text.split()
    # Word count + 1 token per 4 characters average
    return max(len(words), int(len(text) / CHARS_PER_TOKEN))


# Common repetitive markdown comments and conversational preambles
MARKDOWN_COMMENTS_RE = re.compile(r"<!--[\s\S]*?-->")
EXCESS_WHITESPACE_RE = re.compile(r"\n{3,}")
CONVERSATIONAL_FILLERS: Tuple[str, ...] = (
    "As an AI language model, ",
    "Sure, I can help you with that. ",
    "Here is the information you requested: ",
    "Please let me know if you need any further assistance. ",
    "I hope this helps! ",
)


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    compression_ratio: float
    cost_savings_nanodollars: int  # in 1e-9 USD (nano-dollars)


class PromptTokenCompressor:
    """Optimizes prompts by removing redundant tokens and estimating tariff reductions."""

    def __init__(self, nano_dollars_per_input_token: int = 3000):
        # Default: $3.00 per 1M input tokens = 3,000 nano-dollars per token
        self.nano_dollars_per_input_token = nano_dollars_per_input_token

    def compress(self, text: str) -> CompressionResult:
        """Apply compaction rules and compute token and cost delta."""
        if not text:
            return CompressionResult(
                original_text="",
                compressed_text="",
                original_tokens=0,
                compressed_tokens=0,
                saved_tokens=0,
                compression_ratio=0.0,
                cost_savings_nanodollars=0,
            )

        orig_tokens = approximate_token_count(text)

        # 1. Strip HTML/Markdown comments
        cleaned = MARKDOWN_COMMENTS_RE.sub("", text)

        # 2. Strip conversational fillers
        for filler in CONVERSATIONAL_FILLERS:
            cleaned = cleaned.replace(filler, "")

        # 3. Collapse excessive empty newlines (3+ newlines to 2)
        cleaned = EXCESS_WHITESPACE_RE.sub("\n\n", cleaned)

        # 4. Strip trailing and leading whitespace
        cleaned = cleaned.strip()

        comp_tokens = approximate_token_count(cleaned)
        saved_tokens = max(0, orig_tokens - comp_tokens)

        ratio = 0.0
        if orig_tokens > 0:
            ratio = round(saved_tokens / orig_tokens, 4)

        savings_nanodollars = saved_tokens * self.nano_dollars_per_input_token

        return CompressionResult(
            original_text=text,
            compressed_text=cleaned,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            saved_tokens=saved_tokens,
            compression_ratio=ratio,
            cost_savings_nanodollars=savings_nanodollars,
        )

    def prune_conversation_turns(
        self,
        system_prompt: str,
        turns: List[Dict[str, str]],
        max_token_budget: int,
    ) -> List[Dict[str, str]]:
        """Prune multi-turn conversation history to fit within a strict token budget.

        Preserves system prompt and retains recent turns in reverse chronological order.
        """
        if max_token_budget <= 0:
            raise ValueError("max_token_budget must be positive")

        system_tokens = approximate_token_count(system_prompt)
        remaining_budget = max_token_budget - system_tokens
        if remaining_budget <= 0:
            return []

        selected_turns: List[Dict[str, str]] = []
        # Traverse turns from newest to oldest
        for turn in reversed(turns):
            content = turn.get("content", "")
            turn_tokens = approximate_token_count(content)
            if turn_tokens <= remaining_budget:
                selected_turns.append(turn)
                remaining_budget -= turn_tokens
            else:
                break

        selected_turns.reverse()
        return selected_turns
