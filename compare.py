"""
Prompt Injection Detector — Multi-Configuration Comparison
-----------------------------------------------------------
Runs the eval corpus through three detector configurations and prints a
side-by-side comparison showing how accuracy degrades as the system prompt
gets weaker. Demonstrates that the harness measures real signal, not just
"does Claude know what injection is."

Configurations:
  strong  — detailed taxonomy with attack categories and examples
  weak    — one-line instruction ("A prompt injection tries to override...")
  none    — no system prompt at all (bare model)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python compare.py

Optional flags:
    --model    Claude model to use (default: claude-haiku-4-5-20251001)
    --prompts  Path to prompts JSON (default: prompts.json)
    --output   Path to write comparison JSON
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


def print_comparison(model: str, all_results: dict, all_metrics: dict, all_per_cat: dict):
    modes = list(all_results.keys())
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print("  PROMPT INJECTION DETECTOR — CONFIGURATION COMPARISON")
    print(f"  Model: {model}")
    print(f"{'═' * 72}\n")

    # Overall metrics table
    print(f"{sep}")
    print("  OVERALL METRICS")
    print(f"{sep}")
    header = f"  {'Metric':<12}" + "".join(f"  {PROMPT_MODE_LABELS[m]:<30}" for m in modes)
    print(header)
    print(f"  {'─'*12}" + "".join(f"  {'─'*30}" for _ in modes))

    for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1 Score")]:
        row = f"  {label:<12}"
        for m in modes:
            val = f"{all_metrics[m][metric]:.1%}  ({all_metrics[m]['correct']}/{all_metrics[m]['total']})" if metric == "accuracy" else f"{all_metrics[m][metric]:.1%}"
            row += f"  {val:<30}"
        print(row)

    print()
    for label, key in [("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("TN", "tn")]:
        row = f"  {label:<12}" + "".join(f"  {all_metrics[m][key]:<30}" for m in modes)
        print(row)

    # Per-category breakdown
    all_cats = sorted(set(cat for pc in all_per_cat.values() for cat in pc))
    print(f"\n{sep}")
    print("  PER-CATEGORY ACCURACY")
    print(f"{sep}")
    cat_header = f"  {'Category':<28}" + "".join(f"  {PROMPT_MODE_LABELS[m]:<22}" for m in modes)
    print(cat_header)
    print(f"  {'─'*28}" + "".join(f"  {'─'*22}" for _ in modes))

    for cat in all_cats:
        row = f"  {cat:<28}"
        for m in modes:
            if cat in all_per_cat[m]:
                cm = all_per_cat[m][cat]
                bar_len = int(cm["accuracy"] * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                cell = f"{bar} {cm['accuracy']:.0%} ({cm['correct']}/{cm['total']})"
            else:
                cell = "n/a"
            row += f"  {cell:<22}"
        print(row)

    # Prompts where results diverged across configurations
    all_ids = [r.id for r in all_results[modes[0]]]
    diverged = []
    for pid in all_ids:
        preds = {}
        for m in modes:
            match = next(r for r in all_results[m] if r.id == pid)
            preds[m] = (match.predicted_label, match.correct, match.true_label, match.text)
        if len(set(p[0] for p in preds.values())) > 1:
            diverged.append((pid, preds))

    if diverged:
        print(f"\n{sep}")
        print(f"  DIVERGENT PREDICTIONS ({len(diverged)} prompts where configs disagreed)")
        print(f"{sep}")
        for pid, preds in diverged:
            true_label = preds[modes[0]][2].upper()
            text = preds[modes[0]][3]
            print(f"\n  [{true_label}] {pid}")
            print(f"  text : {text[:90]}{'…' if len(text) > 90 else ''}")
            for m in modes:
                pred, correct, _, _ = preds[m]
                status = "✓" if correct else "✗"
                print(f"    {status} {PROMPT_MODE_LABELS[m]}: {pred.upper()}")
    else:
        print(f"\n  All configurations agreed on every prompt.")

    print(f"\n{'═' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare detector configurations side by side")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Claude model to use")
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

    all_results = {}
    all_metrics = {}
    all_per_cat = {}

    for mode in ["strong", "weak", "none"]:
        print(f"\n{'─'*60}")
        print(f"  Running: {PROMPT_MODE_LABELS[mode]}")
        print(f"{'─'*60}")
        results = run_eval(client, prompts, args.model, SYSTEM_PROMPTS[mode])
        all_results[mode] = results
        all_metrics[mode] = compute_metrics(results)
        all_per_cat[mode] = compute_per_category(results)

    print_comparison(args.model, all_results, all_metrics, all_per_cat)

    if args.output:
        output_data = {
            "model": args.model,
            "configurations": {
                mode: {
                    "label": PROMPT_MODE_LABELS[mode],
                    "metrics": all_metrics[mode],
                    "per_category": all_per_cat[mode],
                }
                for mode in ["strong", "weak", "none"]
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Comparison saved to {args.output}\n")


if __name__ == "__main__":
    main()
