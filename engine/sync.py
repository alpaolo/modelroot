"""
ModelRoot sync orchestrator — single entry point for the engine pipeline.

Runs independent scraper/enricher scripts in dependency order.
Each script owns its data source and write target; sync only sequences and reports.

Usage:
  ./modelroot-env/bin/python3 engine/sync.py
  ./modelroot-env/bin/python3 engine/sync.py --only datasets,tech_domains
  ./modelroot-env/bin/python3 engine/sync.py --dry-run
  ./modelroot-env/bin/python3 engine/sync.py --continue-on-error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPERS_DIR = Path(__file__).resolve().parent / "scrapers"


@dataclass(frozen=True)
class SyncStep:
    key: str
    script_name: str
    reads: str
    writes: str


SYNC_STEPS: tuple[SyncStep, ...] = (
    SyncStep("catalog", "mass_scraper2.py", "HF API", "Neo4j grafo"),
    SyncStep("model_links", "enrich_model_links.py", "HF + Neo4j", "Neo4j grafo"),
    SyncStep("licences", "mass_licences_other.py", "HF + Neo4j", "Neo4j grafo"),
    SyncStep("papers", "papers_enrich.py", "HF + Neo4j", "Neo4j grafo"),
    SyncStep("datasets", "enrich_dataset_urls.py", "Neo4j", "JSON dataset_urls.json"),
    SyncStep(
        "tech_domains",
        "enrich_tech_domains.py",
        "JSON curato",
        "Neo4j grafo + detail",
    ),
)

OPTIONAL_SYNC_STEPS: tuple[SyncStep, ...] = (
    SyncStep(
        "leaderboard",
        "enrich_open_llm_leaderboard.py",
        "HF dataset OLL",
        "Neo4j + JSON open_llm_leaderboard.json",
    ),
)


@dataclass
class StepResult:
    step: SyncStep
    status: str  # OK | FAIL | SKIP
    elapsed_seconds: float
    message: str = ""


def discover_steps(selected_keys: list[str] | None) -> list[SyncStep]:
    available_steps: list[SyncStep] = list(SYNC_STEPS)
    for optional_step in OPTIONAL_SYNC_STEPS:
        if (SCRAPERS_DIR / optional_step.script_name).exists():
            available_steps.append(optional_step)

    if not selected_keys:
        return available_steps

    steps_by_key = {step.key: step for step in available_steps}
    unknown_keys = [key for key in selected_keys if key not in steps_by_key]
    if unknown_keys:
        known = ", ".join(steps_by_key.keys())
        raise SystemExit(f"Unknown --only step(s): {', '.join(unknown_keys)}. Known: {known}")

    return [steps_by_key[key] for key in selected_keys]


def run_step(step: SyncStep, dry_run: bool) -> StepResult:
    script_path = SCRAPERS_DIR / step.script_name
    if not script_path.exists():
        return StepResult(step, "SKIP", 0.0, f"missing {step.script_name}")

    command = [sys.executable, str(script_path)]
    if dry_run:
        print(f"[DRY-RUN] {' '.join(command)}")
        print(f"          reads: {step.reads} → writes: {step.writes}")
        return StepResult(step, "OK", 0.0, "dry-run")

    print(f"\n{'=' * 72}")
    print(f"SYNC  {step.key}  ({step.script_name})")
    print(f"      {step.reads} → {step.writes}")
    print(f"{'=' * 72}\n")

    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed_seconds = time.monotonic() - started_at

    if completed.returncode == 0:
        return StepResult(step, "OK", elapsed_seconds)

    return StepResult(
        step,
        "FAIL",
        elapsed_seconds,
        f"exit code {completed.returncode}",
    )


def print_report(results: list[StepResult]) -> None:
    print(f"\n{'=' * 72}")
    print("SYNC REPORT")
    print(f"{'=' * 72}")
    for result in results:
        elapsed_label = f"{result.elapsed_seconds:.1f}s"
        suffix = f" — {result.message}" if result.message else ""
        print(f"[{result.status:4}] {result.step.key:14} {result.step.script_name:32} {elapsed_label}{suffix}")

    ok_count = sum(1 for result in results if result.status == "OK")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    skip_count = sum(1 for result in results if result.status == "SKIP")
    print(f"\nTotal: {ok_count} OK, {fail_count} FAIL, {skip_count} SKIP")


def parse_args() -> argparse.Namespace:
    step_keys = [step.key for step in SYNC_STEPS]
    optional_keys = [step.key for step in OPTIONAL_SYNC_STEPS]
    parser = argparse.ArgumentParser(
        description="Run ModelRoot engine pipeline (independent scripts, single command).",
    )
    parser.add_argument(
        "--only",
        metavar="STEPS",
        help=f"Comma-separated steps to run. Default: all. Keys: {', '.join(step_keys + optional_keys)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline after a failed step (default: stop on first FAIL).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_keys = [key.strip() for key in args.only.split(",")] if args.only else None
    steps = discover_steps(selected_keys)
    results: list[StepResult] = []

    print(f"ModelRoot sync — {len(steps)} step(s), cwd={PROJECT_ROOT}")

    for step in steps:
        result = run_step(step, dry_run=args.dry_run)
        results.append(result)
        if result.status == "FAIL" and not args.continue_on_error:
            print_report(results)
            return 1

    print_report(results)
    if any(result.status == "FAIL" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
