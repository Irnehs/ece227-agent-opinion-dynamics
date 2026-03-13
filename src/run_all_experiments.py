#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import yaml


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_DIR = ROOT_DIR / "experiments"
DEFAULT_STATE_FILE = ROOT_DIR / ".run_all_state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {"version": 1, "updated_at": now_iso(), "items": {}}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "items" not in data:
        data["items"] = {}
    return data


def save_state(state_path: pathlib.Path, state: dict) -> None:
    state["updated_at"] = now_iso()
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)


def get_experiment_name(yaml_path: pathlib.Path) -> str:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    name = data.get("name")
    if not name:
        raise ValueError(f"Missing 'name' in {yaml_path}")
    return str(name)


def run_step(command: list[str], cwd: pathlib.Path) -> int:
    print(f"\n[RUN] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd))
    return result.returncode


def ensure_item(state: dict, key: str, experiment_name: str) -> dict:
    item = state["items"].setdefault(
        key,
        {
            "experiment_name": experiment_name,
            "status": {
                "main": "pending",
                "analysis": "pending",
                "visualization": "pending",
            },
            "last_error": None,
            "updated_at": now_iso(),
        },
    )
    item["experiment_name"] = experiment_name
    return item


def should_skip(status: str, force: bool) -> bool:
    return (status == "completed") and (not force)


def run_all(
    experiments_dir: pathlib.Path,
    state_path: pathlib.Path,
    force: bool,
    continue_on_error: bool,
) -> int:
    yaml_files = sorted(experiments_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No experiment YAML files found in: {experiments_dir}")
        return 1

    state = load_state(state_path)
    save_state(state_path, state)

    print(f"Discovered {len(yaml_files)} experiment configs")
    print(f"State file: {state_path}")

    for yaml_path in yaml_files:
        rel_key = str(yaml_path.relative_to(ROOT_DIR))
        config_stem = yaml_path.stem
        try:
            experiment_name = get_experiment_name(yaml_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {yaml_path}: {e}")
            if not continue_on_error:
                return 1
            continue

        item = ensure_item(state, rel_key, experiment_name)
        item["updated_at"] = now_iso()
        save_state(state_path, state)

        print("\n" + "=" * 72)
        print(f"Config file: {yaml_path.name}")
        print(f"Config key : {config_stem}")
        print(f"Experiment : {experiment_name}")
        print("=" * 72)

        try:
            if should_skip(item["status"]["main"], force):
                print("[SKIP] main already completed")
            else:
                rc = run_step([sys.executable, "main.py", config_stem], cwd=SRC_DIR)
                if rc != 0:
                    item["status"]["main"] = "failed"
                    item["last_error"] = f"main failed with exit code {rc}"
                    item["updated_at"] = now_iso()
                    save_state(state_path, state)
                    print(f"[ERROR] {item['last_error']}")
                    if not continue_on_error:
                        return rc
                    continue
                item["status"]["main"] = "completed"
                item["last_error"] = None
                item["updated_at"] = now_iso()
                save_state(state_path, state)

            if should_skip(item["status"]["analysis"], force):
                print("[SKIP] analysis already completed")
            else:
                rc = run_step(
                    [sys.executable, "analysis.py", "-e", experiment_name], cwd=SRC_DIR
                )
                if rc != 0:
                    item["status"]["analysis"] = "failed"
                    item["last_error"] = f"analysis failed with exit code {rc}"
                    item["updated_at"] = now_iso()
                    save_state(state_path, state)
                    print(f"[ERROR] {item['last_error']}")
                    if not continue_on_error:
                        return rc
                    continue
                item["status"]["analysis"] = "completed"
                item["last_error"] = None
                item["updated_at"] = now_iso()
                save_state(state_path, state)

            if should_skip(item["status"]["visualization"], force):
                print("[SKIP] visualization already completed")
            else:
                rc = run_step(
                    [sys.executable, "visualization.py", "-e", experiment_name],
                    cwd=SRC_DIR,
                )
                if rc != 0:
                    item["status"]["visualization"] = "failed"
                    item["last_error"] = f"visualization failed with exit code {rc}"
                    item["updated_at"] = now_iso()
                    save_state(state_path, state)
                    print(f"[ERROR] {item['last_error']}")
                    if not continue_on_error:
                        return rc
                    continue
                item["status"]["visualization"] = "completed"
                item["last_error"] = None
                item["updated_at"] = now_iso()
                save_state(state_path, state)

        except KeyboardInterrupt:
            item["updated_at"] = now_iso()
            save_state(state_path, state)
            print("\nInterrupted. Progress saved. Re-run the same command to resume.")
            return 130

    print("\nAll experiment configs processed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all experiment YAMLs with resume support: main -> analysis -> visualization"
    )
    parser.add_argument(
        "--experiments-dir",
        type=pathlib.Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Directory that contains experiment YAML files",
    )
    parser.add_argument(
        "--state-file",
        type=pathlib.Path,
        default=DEFAULT_STATE_FILE,
        help="Path to JSON state file for resume support",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if state says completed",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next experiment when one fails",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir.resolve()
    state_file = args.state_file.resolve()

    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    exit_code = run_all(
        experiments_dir=experiments_dir,
        state_path=state_file,
        force=args.force,
        continue_on_error=args.continue_on_error,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
