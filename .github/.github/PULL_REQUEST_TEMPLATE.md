## What changed?

<!-- Briefly describe the changes in this PR. -->

## Why?

<!-- Link the issue if there is one, or explain the motivation. -->

## How was this tested?

<!-- Describe how you verified the change works: test commands, manual checks, model outputs, etc. -->

## Anything reviewers should watch for?

<!-- Edge cases, performance concerns, known limitations, or "nothing special". -->

## AI / agent assistance

<!-- List any AI/coding-agent tools used (e.g. GitHub Copilot, Claude Code) and for what, or "none". -->

## Checklist

- [ ] Scope is right-sized — unrelated concerns (data / loss / model) are split into separate PRs, or deferred work is noted above
- [ ] Reuse — searched for existing components and extended/parameterized rather than duplicating (a config-only variant is a preset, not a new class)
- [ ] New code is wired into existing structure (`BaseModel`, `model_zoo.yaml`, Hydra config, project logger), not bolted on
- [ ] Self-reviewed my own diff
- [ ] Requested a GitHub Copilot review
- [ ] Tests added/updated for new or changed functionality
- [ ] Docstrings added/updated (Google style)
- [ ] Disclosed any AI/agent assistance above (or "none")
- [ ] CI passes
