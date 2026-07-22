# Copilot conventions for RegRex

`AGENTS.md` (repo root, read by Copilot too) holds the general guidance; this is its
review-focused companion — terse rules that shape code suggestions and PR review. Copilot's
PR review reads only the first ~4k characters here, so the highest-priority checks come first.
Keep the scope/reuse/review guidance below aligned with `AGENTS.md` when either changes.

## Review checklist — flag these first

When reviewing a diff (and self-review before requesting review), actively check:

- **Scope** — change spans unrelated concerns (e.g. data + loss + model) that belong in separate PRs; deferred work not stated.
- **Duplication** — reimplements an existing model, loss, metric, or helper instead of reusing or extending it; a config-only variant added as a new class instead of a config preset.
- **Disjointedness** — new code not wired into existing structure (`BaseModel`, `model_zoo.yaml`, Hydra config, project logger); isolated one-off fragments.
- **Quality** — convention drift, missing tests, leftover `print`/debug code, comments restating the code.

## Tensor shape contracts

Every tensor-manipulating function must document expected dimensions:

- `B` = batch, `L` = sequence length, `C` = channels/tracks
- Model input: `(B, L)` integer-encoded or `(B, L, 4)` one-hot
- Model output: always `(B, output_bins, num_tracks)` — never transpose
- Assert shapes on non-trivial operations; do not silently broadcast or reshape

## Sequence encoding — do not hardcode

```python
# ALWAYS import from the canonical module:
from regrex.utils.sequence_encoding import (
    NUC_TO_IDX,       # {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    N_INDEX,          # 4
    PADDING_INDEX,    # -1
    NUM_BASES,        # 4
    indices_to_onehot,  # canonical one-hot conversion — never reimplement
)
```

Never write `0, 1, 2, 3` for nucleotide indices in code. Use `NUC_TO_IDX["A"]` etc.

## Logging — never print

```python
from regrex.utils.logger import setup_logger
logger = setup_logger(__name__)
```

- `logger.info()` for progress and config
- `logger.warning()` for recoverable issues
- `logger.debug()` for temporary diagnostics (remove before merge)
- Never use `print()` — it bypasses structured logging
- Always call `logger.error()` before raising — includes source location in logs

## W&B metric key format

Use double underscores to separate groups: `train__loss`, `eval__corr__{track_name}`.
Never use single underscores or dots as group separators.

## Config is the single source of truth

- No magic numbers in code. All hyperparameters come from Hydra config or named constants.
- Model shapes (input_length, bin_size, output_bins, num_tracks) are in model config.
- Do not hardcode these values in Python files.

## Common mistakes to avoid

1. **Reimplementing one-hot encoding** — use `indices_to_onehot()` exclusively
2. **Detaching in `forward_onehot()` path** — this path must preserve gradients for design
3. **Using `print()` instead of logger** — always use `setup_logger(__name__)`
4. **Broad `except Exception`** — never swallow errors; re-raise or handle specifically
5. **Hardcoding sequence encoding values** — always use constants from `sequence_encoding`
6. **Missing tests** — every new function or bug fix requires tests
7. **Editing `.ipynb` files** — tutorials are percent-format `.py` scripts; notebooks are generated
8. **Silent reshaping** — explain why a reshape is correct; assert shape invariants
9. **Comments restating the code** — comments explain *why*, not *what*; remove noise
10. **Raising without `logger.error()`** — always log before raising for readable tracebacks

## Test patterns

- Keep tests short and focused — test behavior, not implementation
- Use `torch.testing.assert_close()` for tensor comparisons
- Use synthetic data; avoid excessive mocking
- Mark slow tests: `@pytest.mark.slow`
- Mark integration tests: `@pytest.mark.integration`

<!-- ~4k-char limit: content below is read by Copilot chat/agent/CLI but NOT by PR code review -->

## BaseModel interface requirements

All models must implement:

- `forward(x)` — integer input `(B, L)`, training path
- `forward_onehot(x)` — one-hot input `(B, L, 4)`, differentiable design path
- Properties: `target_names`, `input_length`, `bin_size`, `output_bins`

## Error handling

- Always call `logger.error(msg)` before raising — it includes source location in logs
- Raise with context: expected vs actual values
- No silent fallbacks or degradation without documentation
- Validate at system boundaries (data loading, config parsing), not deep in logic

```python
logger.error("Expected shape (B, %d, %d), got %s", expected_bins, expected_tracks, tensor.shape)
raise ValueError(f"Expected shape (B, {expected_bins}, {expected_tracks}), got {tensor.shape}")
```

## Hydra configuration patterns

Configs compose from subdirectories in `conf/`:

```yaml
# conf/config_train.yaml
defaults:
  - model: enformer
  - data: single_species
  - loss: poisson
  - optimizer: adam
  - scheduler: linear_warmup_cosine
  - trainer: default
  - callbacks: default
```

Override on command line:

```bash
uv run pytest                              # run tests, they should pass before pushing any code
uv run python regrex/train.py    # run training, defaults to conf/config_train.yaml
uv run python regrex/train.py model=dilated_cnn data=single_species_toy trainer.max_epochs=5
uv run python regrex/generate.py # sequence generation (Ledidi), defaults to conf/config_generate.yaml
uv run python regrex/train.py --config-name config_finetune     # run fine-tuning (Enformer)
uv run python regrex/train.py --config-name config_finetune_borzoi  # run fine-tuning (Borzoi)
uv run python regrex/train.py --config-name config_finetune_alphagenome  # run fine-tuning (AlphaGenome)
uv run python regrex/evaluate.py # model benchmarking, defaults to conf/config_evaluate.yaml
uv run python regrex/inference.py # embeddings/predictions, defaults to conf/config_inference.yaml
# Hydra overrides can be appended to any command above, e.g.:
#   uv run python regrex/train.py data=single_species_toy trainer.max_epochs=5
#   uv run python regrex/inference.py task=contributions contributions.backend=tangermeme
```

## Model zoo usage

```python
from regrex.models import load_model, list_models
model = load_model("enformer-pretrained")  # downloads and caches checkpoint
```

Models are registered in `regrex/models/model_zoo.yaml`.

## Pre-commit hooks

The repo enforces via pre-commit:

- ruff check + format
- trailing whitespace, end-of-file-fixer
- YAML validation (with unsafe tags for Hydra)
- Large file check (2000KB limit)
- uv.lock currency
- Markdown linting
- Notebook cleaning (nb-clean)

Run all: `uv run pre-commit run --all-files`

## Documentation authoring

- Tutorials: edit the `.py` file (percent format), not the `.ipynb`
- API docs: add `:::` directive to `docs/api/*.md` for new public modules
- Build locally: `ENABLE_GIT_COMMITTERS=false uv run mkdocs serve`
