---
applyTo: "**"
excludeAgent: "cloud-agent"
---

# Code Review Checks — RegRex

## Security / correctness (must catch)

- **Gradient flow in `forward_onehot()` path**: Any `detach()`, `.data`, `torch.no_grad()`,
  or non-differentiable operation in `forward_onehot()` breaks enhancer design. Flag immediately.
- **Sequence encoding hardcoded**: Values `0, 1, 2, 3` used directly for nucleotide indices
  instead of importing from `regrex.utils.sequence_encoding` (use `NUC_TO_IDX["A"]`
  etc., or `N_INDEX`, `PADDING_INDEX`). Also flag hardcoded `-1` for padding or `4` for N.
- **One-hot reimplementation**: Any manual one-hot encoding logic instead of using
  `indices_to_onehot()` from `regrex.utils.sequence_encoding`.
- **Silent shape broadcasting**: Tensors reshaped or broadcast without an assertion or
  comment explaining why the reshape is correct. Especially dangerous for
  `(B, output_bins, num_tracks)` output tensors.
- **Config values hardcoded in Python**: Sequence lengths, bin sizes, track counts, or
  learning rates written as literals instead of coming from Hydra config or model properties.
- **Checkpoint compatibility**: Changes to model `forward()` signatures, state dict keys,
  or Lightning module `__init__` params without updating checkpoint versioning.
- **Backwards compatibility not flagged**: Any breaking change must be explicitly noted.
  If it fixes a bug, that’s fine — but state it. Don’t silently preserve broken behavior.- **Graph-breaking ops in forward/training paths**: `.item()`, `.numpy()`, `.cpu()` in
  `forward()`, `forward_onehot()`, or `training_step()` silently breaks autograd or
  tanks performance. These belong in logging/callbacks only.

## Code quality (flag these)

- **`print()` instead of logger**: All output must use
  `from regrex.utils.logger import setup_logger; logger = setup_logger(__name__)`.
  No `print()` anywhere in production code.
- **Raising without `logger.error()`**: Before raising an exception, call `logger.error()`
  with context. This includes source location in the logs and makes debugging easier.
- **Broad exception handling**: `except Exception` or bare `except:` without re-raising.
  Exceptions must be specific and include context (expected vs actual values).
- **Missing tests for new logic**: Every new function or method needs at least one test.
  Bug fixes need a regression test.
- **Tests longer than needed**: Tests should test behavior, not implementation.
  No unnecessary setup, assertions, or comments that restate the obvious.
- **Missing type hints on public functions**: All public function signatures should have
  parameter and return type annotations.
- **Missing docstring on public functions/classes**: Google-style docstrings required.
- **Comments that restate the code**: Comments explain *why*, not *what*. Flag noise.
- **Magic numbers**: Any numeric literal that should be a named constant or config value.
- **Using `torch.testing.assert_equal` instead of `assert_close`**: Floating point tensor
  comparisons should use `torch.testing.assert_close()`.
- **W&B metric keys with wrong separators**: Must use double underscores (`train__loss`,
  `eval__corr__{track}`), not single underscores or dots.
- **Editing `.ipynb` files directly**: Tutorials are percent-format `.py` scripts.
  Notebooks are generated artifacts.
- **`import *` or relative imports crossing package boundaries**: Use explicit imports.
- **Redundant function parameters**: A function that receives `cfg` should not also
  receive `cfg.something` as a separate argument. Pass explicit values *or* the config
  object — not both.
- **Reimplemented / duplicated component**: A new model, loss, metric, data util, or
  helper that duplicates something already in the codebase instead of reusing or extending
  it. A variant differing only in config should be a config preset (cf. a `DilatedCNN`
  preset), not a new class.
- **Disconnected code**: New code not wired into existing structure (`BaseModel`,
  `model_zoo.yaml`, Hydra config composition, the project logger) — isolated fragments
  bolted on rather than integrated, or dead/duplicate code left behind.
- **Out-of-scope / mixed-concern diff**: A PR spanning unrelated concerns (e.g. data +
  loss + model) that should be separate PRs, or edits unrelated to the stated purpose.
  Flag for splitting.

## Style (suggest, don't block)

- Line length over 120 characters (ruff enforces, but flag if pre-commit might miss)
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Import ordering: standard library → third-party → first-party (`regrex`)
  (isort via ruff handles this, but flag obvious violations)
- Prefer list comprehensions over `map()`/`filter()` for simple transforms
- Avoid nested ternaries — use explicit `if`/`else` blocks
- Docstring parameters should match function signature exactly
- Prefer `pathlib.Path` over string manipulation for file paths
