---
applyTo: "**/*.py"
---

# Python conventions — RegRex

## Naming

- `snake_case` for functions, methods, variables, module names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for module-level constants
- Private helpers: single leading underscore (`_helper_fn`)
- Test classes: `TestFeatureName`, test functions: `test_specific_behavior`

## Type hints

- Required on all public function signatures (parameters and return type)
- Required on non-trivial internal logic where types are not obvious
- Use `torch.Tensor` (not `Tensor`) for tensor types
- Use `tuple[int, ...]` not `Tuple[int, ...]` (Python 3.12+)
- Use `X | None` not `Optional[X]` (Python 3.12+)
- Avoid `Any` — prefer specific types or `object`

## Docstrings

Google-style format:

```python
def predict(self, sequences: torch.Tensor, track_indices: list[int] | None = None) -> torch.Tensor:
    """Run prediction on integer-encoded sequences.

    Args:
        sequences: Integer-encoded input of shape (B, L) where values are
            in {0, 1, 2, 3, 4, -1} for A/C/G/T/N/padding.
        track_indices: Optional subset of output tracks to return.
            If None, returns all tracks.

    Returns:
        Predictions of shape (B, output_bins, num_tracks) or
        (B, output_bins, len(track_indices)) if track_indices provided.

    Raises:
        ValueError: If sequences contain values outside valid range.
    """
```

- Document tensor shapes in Args with dimension semantics (B, L, C)
- Keep descriptions concise — one sentence if possible
- Args/Returns only when non-obvious from type hints
- Comments explain *why*, not *what* — remove comments that restate the code

## Test patterns

```python
import pytest
import torch
from regrex.utils.sequence_encoding import (
    NUC_TO_IDX, N_INDEX, PADDING_INDEX, NUM_BASES,
    indices_to_onehot,
)

class TestMyFunction:
    @pytest.fixture()
    def sample_input(self) -> torch.Tensor:
        return torch.randint(0, 4, (2, 128))

    def test_output_shape(self, sample_input: torch.Tensor) -> None:
        result = my_function(sample_input)
        assert result.shape == (2, 128, 4)

    def test_padding_produces_uniform(self) -> None:
        x = torch.full((1, 10), PADDING_INDEX)
        result = indices_to_onehot(x)
        torch.testing.assert_close(result, torch.full((1, 10, 4), 0.25))
```

- **Keep tests short.** Test behavior, not implementation. No unnecessary setup or assertions.
- Use synthetic data — never depend on external files for unit tests
- Use `torch.testing.assert_close()` for floating-point tensor comparisons
- Minimal mocking — test real behavior where possible
- Mark expensive tests: `@pytest.mark.slow`
- Mark tests needing external resources: `@pytest.mark.integration`

## Ruff rules enforced (not auto-fixable ones to watch for)

- **B006**: Mutable default arguments — use `None` + assignment in body
- **B007**: Unused loop variable — prefix with `_`
- **UP**: Use modern Python syntax (3.10+ patterns, `|` union types, etc.)
- **C4**: Unnecessary list/dict comprehension wrapping (use generator directly)
- **I001**: Import sorting — stdlib, third-party, first-party (`regrex`)

## Patterns specific to this codebase

### Sequence encoding — always use constants

```python
# Correct
from regrex.utils.sequence_encoding import NUC_TO_IDX, PADDING_INDEX
if base_idx == NUC_TO_IDX["A"]: ...

# Wrong — never hardcode
if base_idx == 0: ...
```

### Logger — never print

```python
from regrex.utils.logger import setup_logger
logger = setup_logger(__name__)

# Correct
logger.info("Training started with %d tracks", num_tracks)

# Before raising, always log for readable tracebacks
logger.error("Expected %d tracks, got %d", expected, actual)
raise ValueError(f"Expected {expected} tracks, got {actual}")
```

### Config — no magic numbers

```python
# Correct — values from config or model properties
output = model(x)  # shape determined by model.output_bins, model.num_tracks

# Wrong — hardcoded shape assumptions
assert output.shape == (batch_size, 896, 5313)
```

### BaseModel compliance

All models must implement:

- `forward(x: torch.Tensor) -> torch.Tensor` — integer `(B, L)` input
- `forward_onehot(x: torch.Tensor) -> torch.Tensor` — one-hot `(B, L, 4)` input, must preserve gradients
- Properties: `target_names`, `input_length`, `bin_size`, `output_bins`
- Output: always `(B, output_bins, num_tracks)`
