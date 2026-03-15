"""
Prompt Injection Detector — Multi-Configuration Comparison
-----------------------------------------------------------
Two comparison modes:

1. Prompt-mode comparison (default):
   Runs the same model through strong / weak / none detector prompts.
   Shows how accuracy degrades as the system prompt gets weaker.

2. Model comparison (--models):
   Runs multiple models through the same strong detector prompt.
   Shows how detection accuracy varies across models.

Usage:
    # Prompt-mode comparison
    python compare.py --output comparison.json

    # Model comparison
    python compare.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6 --output model_comparison.json
"""

import argparse
import json
import os
import sys

import anthropic

from eval import (
    SYSTEM_PROMPTS,
    PROMPT_MODE_LABELS,
    run_eval,
    compute_metrics,
    compute_per_category,
)


def print_comparison(title: str, keys: list[str], labels: dict[str, str],
                     all_results: dict, all_metrics: dict, all_per_cat: dict):
    sep = "─" * 72
    col = 28

    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")

    # Overall metrics
    print(f"{sep}")
    print("  OVERALL METRICS")
    print(f"{sep}")
    header = f"  {'Metric':<12}" + "".join(f"  {labels[k]:<{col}}" for k in keys)
    print(header)
    dash_col = "─" * col
    print(f"  {'─'*12}" + "".join(f"  {dash_col}" for _ in keys))

    for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1 Score")]:
        row = f"  {label:<12}"
        for k in keys:
            val = (f"{all_metrics[k][metric]:.1%}  ({all_metrics[k]['correct']}/{all_metrics[k]['total']})"
                   if metric == "accuracy" else f"{all_metrics[k][metric]:.1%}")
            row += f"  {val:<{col}}"
        print(row)

    print()
    for label, key in [("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("TN", "tn")]:
        row = f"  {label:<12}" + "".join(f"  {all_metrics[k][key]:<{col}}" for k in keys)
        print(row)

    # Per-category
    all_cats = sorted(set(cat for pc in all_per_cat.values() for cat in pc))
    print(f"\n{sep}")
    print("  PER-CATEGORY ACCURACY")
    print(f"{sep}")
    print(f"  {'Category':<28}" + "".join(f"  {labels[k]:<22}" for k in keys))
    print(f"  {'─'*28}" + "".join("  " + "─"*22 for _ in keys))

    for cat in all_cats:
        row = f"  {cat:<28}"
        for k in keys:
            if cat in all_per_cat[k]:
                cm = all_per_cat[k][cat]
                bar_len = int(cm["accuracy"] * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                cell = f"{bar} {cm['accuracy']:.0%} ({cm['correct']}/{cm['total']})"
            else:
                cell = "n/a"
            row += f"  {cell:<22}"
        print(row)

    # Divergent predictions
    all_ids = [r.id for r in all_results[keys[0]]]
    diverged = []
    for pid in all_ids:
        preds = {}
        for k in keys:
            match = next(r for r in all_results[k] if r.id == pid)
            preds[k] = (match.predicted_label, match.correct, match.true_label, match.text)
        if len(set(p[0] for p in preds.values())) > 1:
            diverged.append((pid, preds))

    if diverged:
        print(f"\n{sep}")
        print(f"  DIVERGENT PREDICTIONS ({len(diverged)} prompts where configs disagreed)")
        print(f"{sep}")
        for pid, preds in diverged:
            true_label = preds[keys[0]][2].upper()
            text = preds[keys[0]][3]
            print(f"\n  [{true_label}] {pid}")
            print(f"  text : {text[:90]}{'…' if len(text) > 90 else ''}")
            for k in keys:
                pred, correct, _, _ = preds[k]
                status = "✓" if correct else "✗"
                print(f"    {status} {labels[k]}: {pred.upper()}")
    else:
        print(f"\n  All configurations agreed on every prompt.")

    print(f"\n{'═' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare detector configurations or models side by side")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Two or more models to compare (uses strong prompt for all)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Model to use for prompt-mode comparison (ignored if --models is set)")
    parser.add_argument("--prompts", default="prompts.json",
                        help="Path to prompts JSON file")
    parser.add_argument("--output", default=None,
                        help="Optional path to write comparison JSON")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.prompts):
        print(f"Error: prompts file '{args.prompts}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.prompts) as f:
        prompts = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    all_results, all_metrics, all_per_cat = {}, {}, {}

    if args.models:
        # ── Model comparison mode ──────────────────────────────────────────
        keys = args.models
        labels = {m: m for m in keys}
        system_prompt = SYSTEM_PROMPTS["strong"]

        for model in keys:
            print(f"\n{'─'*60}")
            print(f"  Running: {model}")
            print(f"{'─'*60}")
            results = run_eval(client, prompts, model, system_prompt)
            all_results[model] = results
            all_metrics[model] = compute_metrics(results)
            all_per_cat[model] = compute_per_category(results)

        title = "PROMPT INJECTION DETECTOR — MODEL COMPARISON  (strong prompt)"
        output_meta = {"mode": "model_comparison", "prompt_mode": "strong", "models": keys}

    else:
        # ── Prompt-mode comparison (default) ─────────────────────────────
        keys = ["strong", "weak", "none"]
        labels = PROMPT_MODE_LABELS

        for mode in keys:
            print(f"\n{'─'*60}")
            print(f"  Running: {PROMPT_MODE_LABELS[mode]}")
            print(f"{'─'*60}")
            results = run_eval(client, prompts, args.model, SYSTEM_PROMPTS[mode])
            all_results[mode] = results
            all_metrics[mode] = compute_metrics(results)
            all_per_cat[mode] = compute_per_category(results)

        title = f"PROMPT INJECTION DETECTOR — CONFIGURATION COMPARISON  (model: {args.model})"
        output_meta = {"mode": "prompt_comparison", "model": args.model}

    print_comparison(title, keys, labels, all_results, all_metrics, all_per_cat)

    if args.output:
        output_data = {
            **output_meta,
            "configurations": {
                k: {
                    "label": labels[k],
                    "metrics": all_metrics[k],
                    "per_category": all_per_cat[k],
                }
                for k in keys
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Comparison saved to {args.output}\n")


if __name__ == "__main__":
    main()
