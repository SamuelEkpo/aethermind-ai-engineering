"""
Pre-tokenization: splits raw text into word-like chunks before BPE
merges are applied within each chunk. Leading whitespace stays attached
to the following word (GPT-2 style) so whitespace is itself tokenized
predictably rather than discarded.
"""

import re

# Contractions, runs of letters, runs of digits, runs of "other" symbols
# (operators and punctuation — important for source code), and whitespace.
_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[str]:
    """Split text into pre-tokens using a GPT-2-style regex."""
    return _PATTERN.findall(text)
