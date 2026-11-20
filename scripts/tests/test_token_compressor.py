import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from token_compressor import (
    PromptTokenCompressor,
    approximate_token_count,
)


def test_approximate_token_count():
    assert approximate_token_count("") == 0
    text = "The quick brown fox jumps over the lazy dog."
    assert approximate_token_count(text) >= 9


def test_prompt_token_compressor_compress():
    compressor = PromptTokenCompressor(nano_dollars_per_input_token=3000)

    raw_prompt = (
        "As an AI language model, Sure, I can help you with that. \n\n\n\n"
        "<!-- This is an internal developer instruction comment -->\n"
        "Here is the code to compute factorial:\n"
        "def factorial(n):\n"
        "    return 1 if n <= 1 else n * factorial(n - 1)\n\n\n\n"
        "Please let me know if you need any further assistance. I hope this helps! "
    )

    result = compressor.compress(raw_prompt)

    assert "As an AI language model," not in result.compressed_text
    assert "<!-- This is an internal" not in result.compressed_text
    assert "def factorial(n):" in result.compressed_text
    assert "\n\n\n\n" not in result.compressed_text

    assert result.saved_tokens > 0
    assert result.compression_ratio > 0.0
    assert result.cost_savings_nanodollars > 0


def test_prune_conversation_turns():
    compressor = PromptTokenCompressor()
    system_prompt = "You are a senior systems architect."

    turns = [
        {"role": "user", "content": "Initial question from an hour ago that takes many tokens."},
        {"role": "assistant", "content": "Old answer from an hour ago that also consumes significant tokens."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    # Constrain budget to fit only system prompt + last 2 turns
    pruned = compressor.prune_conversation_turns(system_prompt, turns, max_token_budget=25)

    assert len(pruned) == 2
    assert pruned[0]["content"] == "What is the capital of France?"
    assert pruned[1]["content"] == "The capital of France is Paris."


def test_prune_conversation_turns_invalid_budget():
    compressor = PromptTokenCompressor()
    with pytest.raises(ValueError, match="max_token_budget must be positive"):
        compressor.prune_conversation_turns("sys", [], max_token_budget=0)
