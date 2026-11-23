"""
evaluation — model and system evaluation harnesses, kept separate from
training code so benchmarks can't accidentally leak into or depend on the
training loop.

Planned contents (see roadmap Phase 5-9):
    - language_modeling.py   validation loss / perplexity
    - code_eval.py            compilation + execution-based code gen tests
    - reporting.py             benchmark result aggregation/versioning
"""

__all__: list[str] = []
