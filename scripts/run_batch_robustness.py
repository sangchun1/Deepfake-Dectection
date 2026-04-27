# scripts/run_batch_robustness.py
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch robustness evaluation for merged / by_generator / logo / group_holdout splits."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["merged", "by_generator", "logo", "group_holdout", "all"],
    )
    parser.add_argument("--base_data_config", type=str, default="configs/data/default.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model/fusion.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train/fusion_resnet.yaml")
    parser.add_argument("--robustness_config", type=str, default="configs/train/robustness.yaml")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--generated_config_dir", type=str, default="configs/_generated/openfake_robustness_batch")
    parser.add_argument("--output_root", type=str, default="outputs/fusion/resnet")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--checkpoint_name", type=str, default="best.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--corruptions", type=str, default=None)
    parser.add_argument("--severities", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def resolve_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["merged", "by_generator", "logo", "group_holdout"]
    return [mode]


def discover_experiment_dirs(
    splits_root: Path,
    mode: str,
    selected_names: Sequence[str] | None = None,
) -> List[Path]:
    mode_dir = splits_root / mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {mode_dir}")

    selected = set(selected_names) if selected_names else None

    if mode == "merged":
        exp_name = mode_dir.name
        if selected is not None and exp_name not in selected:
            raise RuntimeError(
                f"No experiment directories found under: {mode_dir} (selected_names={sorted(selected)})"
            )
        return [mode_dir]

    exp_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir()])
    if selected is not None:
        exp_dirs = [p for p in exp_dirs if p.name in selected]
    if not exp_dirs:
        raise RuntimeError(f"No experiment directories found under: {mode_dir}")
    return exp_dirs


def discover_group_holdout_test_csvs(exp_dir: Path) -> List[Path]:
    tests_dir = exp_dir / "tests"
    if not tests_dir.exists():
        raise FileNotFoundError(f"group_holdout tests directory not found: {tests_dir}")
    csv_paths = sorted(p for p in tests_dir.glob("*.csv") if p.is_file())
    if not csv_paths:
        raise RuntimeError(f"No per-generator test csv files found under: {tests_dir}")
    return csv_paths


def build_generated_data_config(
    base_cfg: Dict,
    exp_dir: Path,
    generated_config_path: Path,
    root_dir_override: str | None,
) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    if "paths" not in cfg:
        raise ValueError("Base data config must contain a 'paths' section.")

    if root_dir_override is not None:
        cfg["paths"]["root_dir"] = root_dir_override

    cfg["paths"]["train_csv"] = exp_dir.joinpath("train.csv").as_posix()
    cfg["paths"]["val_csv"] = exp_dir.joinpath("val.csv").as_posix()
    cfg["paths"]["test_csv"] = exp_dir.joinpath("test.csv").as_posix()

    save_yaml(cfg, generated_config_path)
    return cfg


def build_split_override_data_config(
    generated_data_cfg: Dict,
    split_csv_path: Path,
    split_name: str,
    save_path: Path,
) -> Dict:
    cfg = copy.deepcopy(generated_data_cfg)
    cfg["paths"][f"{split_name}_csv"] = split_csv_path.as_posix()
    save_yaml(cfg, save_path)
    return cfg


def make_robustness_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    robustness_config: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    split: str,
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    corruptions: str | None = None,
    severities: str | None = None,
) -> List[str]:
    cmd = [
        python_executable,
        "-u",
        "scripts/evaluate_robustness.py",
        "--data_config", data_config.as_posix(),
        "--model_config", model_config.as_posix(),
        "--train_config", train_config.as_posix(),
        "--robustness_config", robustness_config.as_posix(),
        "--split", split,
        "--output_dir", output_dir.as_posix(),
    ]

    if checkpoint_path is not None:
        cmd.extend(["--checkpoint", checkpoint_path.as_posix()])
    if device is not None:
        cmd.extend(["--device", device])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    if corruptions:
        cmd.extend(["--corruptions", corruptions])
    if severities:
        cmd.extend(["--severities", severities])

    return cmd


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print("=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=cwd, check=True)


def robustness_done(output_dir: Path) -> bool:
    return (
        (output_dir / "robustness_results.json").exists()
        and (output_dir / "robustness_summary.json").exists()
        and (output_dir / "robustness_records.csv").exists()
    )


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    base_data_config_path = resolve_path(args.base_data_config)
    model_config_path = resolve_path(args.model_config)
    train_config_path = resolve_path(args.train_config)
    robustness_config_path = resolve_path(args.robustness_config)
    splits_root = resolve_path(args.splits_root)
    generated_config_dir = resolve_path(args.generated_config_dir)
    output_root = resolve_path(args.output_root)

    base_data_cfg = load_yaml(base_data_config_path)
    selected_names = set(args.names) if args.names else None
    modes = resolve_modes(args.mode)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "resolved_modes": modes,
        "base_data_config": base_data_config_path.as_posix(),
        "model_config": model_config_path.as_posix(),
        "train_config": train_config_path.as_posix(),
        "robustness_config": robustness_config_path.as_posix(),
        "splits_root": splits_root.as_posix(),
        "generated_config_dir": generated_config_dir.as_posix(),
        "output_root": output_root.as_posix(),
        "root_dir_override": args.root_dir,
        "split": args.split,
        "checkpoint_name": args.checkpoint_name,
        "selected_names": sorted(selected_names) if selected_names else None,
        "results": [],
    }

    for mode in modes:
        exp_dirs = discover_experiment_dirs(
            splits_root=splits_root,
            mode=mode,
            selected_names=selected_names,
        )

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            generated_data_config_path = generated_config_dir / mode / "data" / f"{exp_name}.yaml"
            generated_data_cfg = build_generated_data_config(
                base_cfg=base_data_cfg,
                exp_dir=exp_dir,
                generated_config_path=generated_data_config_path,
                root_dir_override=args.root_dir,
            )

            checkpoint_path = output_root / mode / exp_name / args.checkpoint_name
            result = {
                "mode": mode,
                "exp_name": exp_name,
                "split_dir": exp_dir.as_posix(),
                "generated_data_config": generated_data_config_path.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "robustness_status": "not_run",
            }

            try:
                if mode != "group_holdout":
                    robustness_output_dir = output_root / mode / exp_name / f"robustness_{args.split}"
                    if args.skip_existing and robustness_done(robustness_output_dir):
                        print(f"Skip robustness because outputs already exist: {robustness_output_dir}")
                        result["robustness_status"] = "skipped_existing"
                    else:
                        cmd = make_robustness_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            robustness_config=robustness_config_path,
                            output_dir=robustness_output_dir,
                            checkpoint_path=checkpoint_path,
                            split=args.split,
                            device=args.device,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            corruptions=args.corruptions,
                            severities=args.severities,
                        )
                        run_command(cmd, cwd=project_root)
                        result["robustness_status"] = "done"
                        result["robustness_output_dir"] = robustness_output_dir.as_posix()

                else:
                    # pooled
                    pooled_output_dir = output_root / mode / exp_name / f"robustness_{args.split}_pooled"
                    if not (args.skip_existing and robustness_done(pooled_output_dir)):
                        pooled_cmd = make_robustness_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            robustness_config=robustness_config_path,
                            output_dir=pooled_output_dir,
                            checkpoint_path=checkpoint_path,
                            split=args.split,
                            device=args.device,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            corruptions=args.corruptions,
                            severities=args.severities,
                        )
                        run_command(pooled_cmd, cwd=project_root)

                    per_generator = {}
                    for test_csv in discover_group_holdout_test_csvs(exp_dir):
                        generator_name = test_csv.stem
                        per_gen_cfg_path = (
                            generated_config_dir / mode / "data" / exp_name / f"{generator_name}.yaml"
                        )
                        build_split_override_data_config(
                            generated_data_cfg=generated_data_cfg,
                            split_csv_path=test_csv,
                            split_name=args.split,
                            save_path=per_gen_cfg_path,
                        )

                        per_gen_output_dir = (
                            output_root / mode / exp_name / "per_generator" / generator_name / f"robustness_{args.split}"
                        )

                        if args.skip_existing and robustness_done(per_gen_output_dir):
                            per_generator[generator_name] = {
                                "status": "skipped_existing",
                                "output_dir": per_gen_output_dir.as_posix(),
                            }
                            continue

                        per_gen_cmd = make_robustness_command(
                            python_executable=args.python,
                            data_config=per_gen_cfg_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            robustness_config=robustness_config_path,
                            output_dir=per_gen_output_dir,
                            checkpoint_path=checkpoint_path,
                            split=args.split,
                            device=args.device,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            corruptions=args.corruptions,
                            severities=args.severities,
                        )
                        run_command(per_gen_cmd, cwd=project_root)
                        per_generator[generator_name] = {
                            "status": "done",
                            "output_dir": per_gen_output_dir.as_posix(),
                        }

                    result["robustness_status"] = "done"
                    result["pooled_output_dir"] = pooled_output_dir.as_posix()
                    result["per_generator"] = per_generator

            except Exception as e:
                result["robustness_status"] = "failed"
                result["error"] = str(e)
                summary["results"].append(result)
                save_json(summary, output_root / "batch_robustness_summary.json")
                if args.stop_on_error:
                    raise
                continue

            summary["results"].append(result)
            save_json(summary, output_root / "batch_robustness_summary.json")

    print("\n" + "=" * 100)
    print("Batch robustness finished")
    print("=" * 100)
    print(f"Saved summary to: {output_root / 'batch_robustness_summary.json'}")


if __name__ == "__main__":
    main()