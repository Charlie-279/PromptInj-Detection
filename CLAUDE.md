# Project Rules for Claude Code

## Project Context

A Python eval suite that tests whether an LLM can detect prompt injection attempts.
Given a labeled set of prompts (clean vs. injected), the tool scores how accurately
the target model flags the right ones.

- **Stack**: Python 3.12, Anthropic SDK
- **Run**: `python eval.py --verbose`
- **Test corpus**: `prompts.json` (labeled prompt/injection examples)
- **API key**: set via `export ANTHROPIC_API_KEY=sk-ant-...` — never hardcode in files

## Commit Rules

After every logical change:

1. Stage specific files: `git add <files>`
2. Commit with a clear message: `git commit -m "description of change"`
3. Push to `main`: `git push origin main`

- Never force push
- Never commit API keys or `.env` files
- Use descriptive commit messages that explain the "why"
- If a pre-commit hook fails, fix and create a NEW commit (never amend)

## Core Principles

- **Simplicity first**: this is a small tool — keep it that way
- **Minimal impact**: only change what's necessary for the task
- **Verify before done**: run `python eval.py` to confirm nothing is broken after changes
