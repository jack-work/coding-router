"""Summarize a live Pier arm wave without treating missing work as failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_arm(path: Path) -> dict:
    rows = []
    for result_path in path.glob("*/result.json"):
        try:
            row = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not row.get("finished_at"):
            continue
        verifier = (row.get("verifier_result") or {}).get("rewards") or {}
        agent = row.get("agent_result") or {}
        steps = row.get("n_agent_steps")
        rows.append({
            "task": str(row.get("task_name") or row.get("trial_name")),
            "graded": (
                float(verifier["f2p_passed"]) / float(verifier["f2p_total"])
                if verifier.get("f2p_total") else None
            ),
            "binary": float(verifier["reward"]) if verifier.get("reward") is not None else None,
            "cost_usd": float(agent.get("cost_usd") or 0.0),
            "input_tokens": int(agent.get("n_input_tokens") or 0),
            "cache_tokens": int(agent.get("n_cache_tokens") or 0),
            "output_tokens": int(agent.get("n_output_tokens") or 0),
            "steps": int(steps) if isinstance(steps, int) else None,
        })
    valid = [r for r in rows if r["graded"] is not None and (r["steps"] or 0) > 0]
    return {
        "arm": path.name,
        "finished": len(rows),
        "valid": len(valid),
        "zero_step": sum((r["steps"] or 0) == 0 for r in rows),
        "missing_grade": sum(r["graded"] is None for r in rows),
        "graded": sum(r["graded"] for r in valid) / len(valid) if valid else None,
        "binary": sum(r["binary"] for r in valid if r["binary"] is not None) / len(valid)
        if valid and all(r["binary"] is not None for r in valid) else None,
        "cost_usd": sum(r["cost_usd"] for r in valid),
        "cost_per_task": sum(r["cost_usd"] for r in valid) / len(valid) if valid else None,
        "input_tokens": sum(r["input_tokens"] for r in valid),
        "cache_tokens": sum(r["cache_tokens"] for r in valid),
        "output_tokens": sum(r["output_tokens"] for r in valid),
        "steps": sum(r["steps"] or 0 for r in valid) / len(valid) if valid else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = [summarize_arm(p) for p in sorted(args.output.iterdir()) if p.is_dir()]
    payload = {"output": str(args.output), "arms": rows}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
