# Project context: RegRex

## What this is

Modular codebase for training, evaluating, and running inference on sequence-to-function models (e.g., predicting regulatory activity from DNA sequence). Supports generative modeling of regulatory sequences (enhancer design) and training new generation models. Designed for experimentation: swap components (encoders, losses, data loaders) without rewriting the pipeline.

Scientific goals include:

* Documenting cis-regulatory elements specific to different cell types and conditions
* Designing synthetic enhancers that have cell type specific activity
* General framework for interpreting s2f models, focusing on sequence features that drive predictions
* Multiple model architectures: CNNs (DilatedCNN), transformer-based models (Enformer, Borzoi, AlphaGenome), and generative models (Ledidi, Diffusion, etc.)

## Key technical goals

* Modularity: components (models, data, training loops, evaluation) are interchangeable.
* Reproducibility: every experiment must be fully reproducible from config.
* Benchmarking: easy comparison of different model architectures and training strategies.

## Stack

* Python >=3.12
* PyTorch (>=2.8) + PyTorch Lightning (>=2.5)
* Hydra for config management
* W&B for experiment tracking
* pytest for testing (with pytest-cov, pytest-xdist)
* uv for environment and dependency management
* ruff (v0.15) + pylint for linting; mypy for type checking
* MkDocs (Material theme) for documentation
* pre-commit for CI hooks (ruff, trailing whitespace, YAML, markdown, nb-clean)

Key ML dependencies: `enformer-pytorch`, `borzoi-pytorch`, `alphagenome-pytorch`, `ledidi`, `tangermeme`, `alphagenome`, `tfmindi`

## Directory structure

```text
regrex/
  models/           # Model architectures (Enformer, Borzoi, DilatedCNN) and Lightning wrappers
    base.py         # BaseModel ABC: defines forward(), forward_onehot(), predict_on_design()
    model_zoo.py    # Registry/factory for loading pretrained models by name
  data/             # Data loading, preprocessing, sequence providers, zarr datasets
  training/         # Callbacks, schedulers, metrics, batch utilities
  interpretation/   # Feature attribution methods, in silico mutagenesis, etc.
  generating/       # Sequence generation and optimization code (Ledidi, etc.)
  evaluation/       # Model benchmarking: manifest-driven eval, metrics, predictors, reporting
  losses/           # Loss functions (Poisson, etc.)
  hub/              # Model hub integration (HuggingFace, model cards)
  utils/            # Logger, sequence encoding, versioning, visualization
    sequence_encoding.py  # Single source of truth for encoding constants
    logger.py             # Project logger — always use setup_logger(__name__)
conf/               # Experiment configs (YAML, Hydra-structured)
  model/            # Model architecture configs
  data/             # Dataset configs
  loss/             # Loss configs
  optimizer/        # Optimizer configs
  scheduler/        # LR scheduler configs
  trainer/          # Trainer configs
  callbacks/        # Callback configs
  generator/        # Generation configs (Ledidi parameters)
  eval/             # Evaluation configs (manifests, metric selection)
  metrics/          # Metric-set configs (per-model metric definitions)
  preprocess/       # Data preprocessing configs
  logger/           # Logger configs (W&B, etc.)
  hydra/            # Hydra runtime configs (run/sweep output dirs)
tests/              # Mirrors regrex/ structure; pytest with fixtures
scripts/            # Utility scripts (zarr config generation, checkpoint inspection, etc.)
data/               # Helper target files, enformer target metadata
docs/               # MkDocs documentation site
beaker/             # Beaker deployment configs and Docker image
tutorials/          # Percent-format .py scripts converted to Jupyter notebooks
```

## Conventions specific to this project

### Sequence encoding

* Sequences are integer-encoded tensors of shape `(B, L)` with dtype int32/int64.
* Encoding: A=0, C=1, G=2, T=3, N=4, padding=-1.
* One-hot encoding is applied on the fly via `indices_to_onehot()`.
* One-hot N = `[0, 0, 0, 0]`, one-hot padding = `[0.25, 0.25, 0.25, 0.25]`.
* Always import constants from `regrex.utils.sequence_encoding` — never hardcode these values.

### Model interface

* All models implement `regrex.models.base.BaseModel`.
* Two forward paths: `forward(x)` for integer input (training), `forward_onehot(x)` for differentiable one-hot input (design/optimization).
* Output shape is always `(B, output_bins, num_tracks)`.
* Required properties: `target_names`, `input_length`, `bin_size`, `output_bins`.

### Configuration

* Configs are the single source of truth for experiment parameters. No magic numbers in code.
* Model-specific input/output shapes (sequence length, bin count, number of tracks) are defined in each model's config and documented in its class docstring.
* Hydra overrides on the command line; compose configs from `conf/` subdirectories.

### Logging and tracking

* Use the project logger (`from regrex.utils.logger import setup_logger; logger = setup_logger(__name__)`), never `print`.
* Before raising exceptions, call `logger.error(msg)` — this produces readable log output with the location of the error.
* W&B metric keys use double underscores to separate groups: `train__loss`, `eval__corr__{track_name}`.

### Testing

* Every function addition or modification must include unit tests.
* Tests should be concise — verify inputs/outputs, not internal mechanics. If a refactor doesn't change the API, tests shouldn't break.
* Use synthetic data when possible. Avoid excessive mocking.
* Bug fixes require a regression test that fails before the fix and passes after.
* Use `torch.testing.assert_close()` for tensor comparisons.
* Markers: `@pytest.mark.slow`, `@pytest.mark.integration`.

### Documentation

* Google-style docstrings for all public functions and classes. Keep them concise — one sentence summary, then Args/Returns only if non-obvious.
* Comments should explain *why*, not *what*. Remove comments that restate the code.
* Tutorials are written as percent-formatted Python scripts that are converted to Jupyter notebooks. Code changes should be made in Python scripts — never edit `.ipynb` directly.
* API docs are auto-generated by mkdocstrings.

### Code style

* Formatting: ruff (line-length=120, target py310).
* Lint rules: E, W, F, I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade).
* `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
* Type hints on public functions and non-trivial internal logic.
* Default to pre-defined functions instead of hand-rolling code. For example, scipy implements pearson and spearman correlation, we don't need to reinvent the wheel.
* **Function interfaces**: Pass what the function needs, not more. Don't pass a full config *and* values extracted from it — pick one. Prefer explicit parameters for testability; accept a config sub-object only when the function genuinely configures many things.

## Gotchas

* **Checkpoint compatibility**: Version checkpoints and include model architecture info in metadata. Breaking changes are acceptable (especially when fixing bugs — don't preserve broken behavior) but must be documented in release notes with migration instructions.
* Do not treat test coverage failure as failures when running partial tests. If you are working on a specific module, run tests for that module only and ignore coverage failures for the rest of the codebase.
* The `forward_onehot()` path must preserve gradients for design/optimization — never detach or use non-differentiable operations in that path.
* `indices_to_onehot()` is the canonical one-hot conversion — do not reimplement or use alternative approaches.
* Zarr datasets use async I/O — be careful with thread safety when writing custom data loaders.
* **Backwards compatibility**: Flag it explicitly in PRs. If something was broken before, fixing it is more important than keeping backwards compat. If compatible by design, state that briefly.
* **Graph-breaking ops in training/forward paths**: `.item()`, `.numpy()`, `.cpu()` in forward or training-step code silently breaks autograd or kills performance. Keep these to logging/callbacks only.
* **Config backwards compat**: New config parameters must have defaults so existing YAML files continue to work without modification.

## How to run

```bash
# Environment setup
uv sync                                    # install all dependencies
uv sync --group dev                        # include dev tools
uv run pre-commit install                  # set up pre-commit hooks

# Testing
uv run pytest                              # run all tests (strict markers, coverage enforced)
uv run pytest -n 16 --dist=loadscope       # parallel testing
uv run pytest tests/models/                # run subset of tests

# Linting/formatting
uv run pre-commit run --all-files          # run all pre-commit hooks
uv run ruff check .                        # lint only
uv run ruff format .                       # format only

# Training and inference
uv run python regrex/train.py    # defaults to conf/config_train.yaml
uv run python regrex/train.py --config-name config_finetune  # fine-tune (also config_finetune_borzoi, _alphagenome, _dilated_cnn)
uv run python regrex/evaluate.py # benchmarking; defaults to conf/config_evaluate.yaml
uv run python regrex/inference.py # embeddings/predictions; defaults to conf/config_inference.yaml
uv run python regrex/generate.py # sequence generation (Ledidi); defaults to conf/config_generate.yaml

# Hydra overrides:
#   uv run python regrex/train.py data=single_species_toy trainer.max_epochs=5

# Documentation
uv sync --group docs
ENABLE_GIT_COMMITTERS=false uv run mkdocs build   # build docs locally
ENABLE_GIT_COMMITTERS=false uv run mkdocs serve   # serve docs locally
```

## Scope and design

* **Assess scope before non-trivial work.** Name what the change touches. If it spans unrelated concerns — data *and* loss *and* model — split it: these layers are reviewed independently and usually belong in separate PRs.
* **Design before build for large or cross-cutting changes.** For new interfaces, architecture changes, or anything touching many modules, write a short design note (problem, proposed approach, alternatives) and get alignment *before* implementing. Flag it `needs-design` and open an issue or draft PR to discuss — see [CONTRIBUTING.md](CONTRIBUTING.md).
* **Right-size PRs.** Prefer small, focused, independently-reviewable changes. State the scope up front in the PR description and flag anything deferred. If you spot adjacent work, open an issue — do not fold it in.

## Reuse over reinvention

* **Search before writing.** Look for an existing model, loss, metric, data util, or helper that already does the job; extend or parameterize it instead of reimplementing.
* **Wire into existing structure.** New code connects to established patterns — `BaseModel`, the model registry (`model_zoo.yaml`), Hydra config composition, the project logger. Leave behind no disconnected duplicates or one-off fragments.
* **Config variant, not new class.** If a variant differs only in configuration, add a config preset — do not add a new class (e.g. CRESTED-style models are `DilatedCNN` presets, not a separate model).
* **Prefer vetted library implementations** over hand-rolled versions for standard tasks (metrics, validation, config parsing). Add a dependency only when it earns its place; state why in one line.

## Review checklist

Apply to your own diff before requesting review, and when reviewing others' (human, Copilot, or agent):

* **Quality** — follows the conventions in this file; clear, tested, no leftover debug code.
* **Duplication** — does this reimplement an existing model, loss, metric, or helper? Reuse or extend instead.
* **Disjointedness** — is the new code wired into existing structure, or bolted on as an isolated fragment?
* **Scope** — appropriately sized, or should it be split? Are deferred items flagged?

These checks are the PR checklist in [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) — fill it out, request a Copilot review, and disclose any AI/agent tools used. They are also enforced in automated review via [.github/instructions/code-review.instructions.md](.github/instructions/code-review.instructions.md) (the itemized code-review checks) and condensed in [.github/copilot-instructions.md](.github/copilot-instructions.md).

> Keep these four surfaces aligned when the guidance changes: this file, the PR template, `code-review.instructions.md`, and `copilot-instructions.md`. They need not be identical — Copilot's PR review reads only the first ~4k characters of `copilot-instructions.md`.

## Behavioral guidelines for agents

* **Correctness over cleverness.** Explicit over implicit. Fail loudly over silent fallbacks.
* **Efficiency matters** — write efficient code, but never sacrifice correctness for performance. Numerical approximations are fine if by design and documented.
* **Do only what is asked.** If you notice an adjacent issue, mention it in one line but do not fix it unless asked. Never expand scope without approval.
* **Be concise.** Do not add comments that restate the code, tests longer than needed, or docs with filler text. Every line should earn its place.
* **Reproducibility.** Set random seeds where relevant. Log configuration and hyperparameters. Avoid hidden global state.
* **Never swallow exceptions** or use broad `except Exception` without re-raising. Use `logger.error()` before raising — it provides readable context with source location.
* **Tensor shape contracts.** All tensor-manipulating functions must document expected rank and dimension semantics (B=batch, T/L=sequence, C=channels). Assert non-trivial shape invariants. Do not silently reshape or broadcast.
* **No magic numbers.** All constants come from config or from named constants in the codebase.
* **Use the project logger**, never `print`. `info` for progress, `warning` for recoverable issues, `debug` for temporary diagnostics (remove after debugging).
* **Testing is mandatory.** Every new function needs tests. Every bug fix needs a regression test. Keep tests short and focused.
* **Small changes.** Prefer small, focused modifications. Do not refactor code unrelated to the task at hand.
* **Respect existing patterns.** Follow the conventions already present in the codebase rather than introducing new ones.
* **Conventional Commits.** Format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`. Subject under 72 chars, informative but not an essay. Add body only when the *why* isn't obvious from the diff.
* **Disclose agent assistance.** When opening or updating a PR, note in the description which AI/agent tools were used (the PR template has a slot) so reviewers can calibrate scrutiny, and complete the scope/reuse/review checklist.

## Documentation policy

* New users should be able to execute code and run inference in minutes.
* Focus on clear, concise explanations. No filler text.
* Do not write summaries in language, focus on visualizations.
* Only create new markdown files if necessary, otherwise add to existing files.
