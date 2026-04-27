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


def normalize_eval_suffix(suffix: str | None) -> str:
    if suffix is None:
        return ""
    suffix = suffix.strip()
    if not suffix:
        return ""
    if suffix.startswith("_"):
        return suffix
    return f"_{suffix}"


def build_eval_json_name(split: str, suffix: str = "", pooled: bool = False) -> str:
    suffix = normalize_eval_suffix(suffix)
    base = f"eval_{split}"
    if pooled:
        base += "_pooled"
    return f"{base}{suffix}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate existing checkpoints saved by run_batch_experiments.py "
            "for merged / by_generator / logo / group_holdout splits. "
            "Temperature scaling is applied by default using temperature.json "
            "stored next to each checkpoint directory."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["merged", "by_generator", "logo", "group_holdout", "all"],
        help="Which experiment family to evaluate.",
    )
    parser.add_argument(
        "--base_data_config",
        type=str,
        default="configs/data/default.yaml",
        help="Base data config YAML.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
        help="Model config path.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_resnet.yaml",
        help="Base train config path.",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default="data/splits",
        help="Root directory containing merged/by_generator/logo/group_holdout split folders.",
    )
    parser.add_argument(
        "--generated_config_dir",
        type=str,
        default="configs/_generated/eval_batch",
        help="Where generated per-experiment configs will be stored.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/spatial",
        help="Base output directory that contains mode/exp_name/best.pth.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=(
            "Optional override for cfg.data.paths.root_dir. "
            "Useful when raw data is stored on a different disk on the server."
        ),
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Optional subset of experiment folder names to evaluate. "
            "Examples: merged sd-3.5 flux.1-dev midjourney-6"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best.pth",
        help="Checkpoint filename to use under output_root/mode/exp_name/.",
    )
    parser.add_argument(
        "--temperature_json_name",
        type=str,
        default="temperature.json",
        help=(
            "Temperature JSON filename expected under output_root/mode/exp_name/. "
            "By default, temperature scaling is always applied using this file."
        ),
    )
    parser.add_argument(
        "--eval_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix appended to evaluation JSON filenames. "
            "If omitted, '_ts' is used automatically because temperature scaling "
            "is enabled by default."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Optional device override passed to evaluate.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size override passed to evaluate.py.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional num_workers override passed to evaluate.py.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for subprocess calls.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip evaluation if the target eval json already exists.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any experiment fails.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional explicit path to save the batch evaluation summary JSON.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
                f"No experiment directories found under: {mode_dir} "
                f"(selected_names={sorted(selected)})"
            )
        return [mode_dir]

    exp_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir()])

    if selected is not None:
        exp_dirs = [p for p in exp_dirs if p.name in selected]

    if not exp_dirs:
        raise RuntimeError(f"No experiment directories found under: {mode_dir}")

    return exp_dirs


def build_generated_data_config(
    base_cfg: Dict,
    exp_dir: Path,
    generated_config_path: Path,
    root_dir_override: str | None,
    mode: str,
    exp_name: str,
) -> Dict:
    cfg = copy.deepcopy(base_cfg)

    if "paths" not in cfg:
        raise ValueError("Base data config must contain a 'paths' section.")

    if root_dir_override is not None:
        cfg["paths"]["root_dir"] = root_dir_override

    cfg["paths"]["train_csv"] = exp_dir.joinpath("train.csv").as_posix()
    cfg["paths"]["val_csv"] = exp_dir.joinpath("val.csv").as_posix()
    cfg["paths"]["test_csv"] = exp_dir.joinpath("test.csv").as_posix()

    base_name = str(cfg.get("name", "dataset"))
    cfg["name"] = f"{base_name}_{mode}_{exp_name}"

    save_yaml(cfg, generated_config_path)
    return cfg


def build_generated_train_config(
    base_cfg: Dict,
    generated_train_config_path: Path,
    output_dir: Path,
) -> Dict:
    cfg = copy.deepcopy(base_cfg)

    if "experiment" not in cfg:
        raise ValueError("Base train config must contain an 'experiment' section.")

    cfg["experiment"]["output_dir"] = output_dir.as_posix()
    save_yaml(cfg, generated_train_config_path)
    return cfg


def discover_group_holdout_test_csvs(exp_dir: Path) -> List[Path]:
    tests_dir = exp_dir / "tests"
    if not tests_dir.exists():
        raise FileNotFoundError(f"group_holdout tests directory not found: {tests_dir}")

    csv_paths = sorted(p for p in tests_dir.glob("*.csv") if p.is_file())
    if not csv_paths:
        raise RuntimeError(f"No per-generator test csv files found under: {tests_dir}")
    return csv_paths


def get_group_holdout_eval_paths(
    output_dir: Path,
    split: str,
    test_csv_paths: Sequence[Path],
    eval_suffix: str = "",
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {
        "pooled": output_dir / build_eval_json_name(split, suffix=eval_suffix, pooled=True),
    }
    per_generator_root = output_dir / "per_generator"
    for test_csv in test_csv_paths:
        generator_name = test_csv.stem
        paths[generator_name] = (
            per_generator_root / generator_name / build_eval_json_name(split, suffix=eval_suffix)
        )
    return paths


def make_eval_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    checkpoint_path: Path,
    temperature_json_path: Path,
    split: str,
    output_json: Path,
    split_csv_override: Path | None,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
) -> List[str]:
    cmd = [
        python_executable,
        "scripts/evaluate.py",
        "--data_config",
        data_config.as_posix(),
        "--model_config",
        model_config.as_posix(),
        "--train_config",
        train_config.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--temperature_json",
        temperature_json_path.as_posix(),
        "--split",
        split,
        "--output_json",
        output_json.as_posix(),
    ]
    if split_csv_override is not None:
        cmd.extend(["--split_csv_override", split_csv_override.as_posix()])
    if device is not None:
        cmd.extend(["--device", device])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    return cmd


def run_command(cmd: List[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print("-" * 100)
    print(f"RUN: {printable}")
    print("-" * 100)
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    base_data_config_path = resolve_path(args.base_data_config)
    model_config_path = resolve_path(args.model_config)
    train_config_path = resolve_path(args.train_config)
    splits_root = resolve_path(args.splits_root)
    generated_config_dir = resolve_path(args.generated_config_dir)
    output_root = resolve_path(args.output_root)

    eval_suffix = normalize_eval_suffix(args.eval_suffix)
    if not eval_suffix:
        eval_suffix = "_ts"

    base_data_cfg = load_yaml(base_data_config_path)
    base_train_cfg = load_yaml(train_config_path)

    selected_names = set(args.names) if args.names else None
    modes = resolve_modes(args.mode)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "resolved_modes": modes,
        "base_data_config": base_data_config_path.as_posix(),
        "model_config": model_config_path.as_posix(),
        "train_config": train_config_path.as_posix(),
        "splits_root": splits_root.as_posix(),
        "generated_config_dir": generated_config_dir.as_posix(),
        "output_root": output_root.as_posix(),
        "root_dir_override": args.root_dir,
        "split": args.split,
        "checkpoint_name": args.checkpoint_name,
        "temperature_json_name": args.temperature_json_name,
        "eval_suffix": eval_suffix,
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
            generated_train_config_path = generated_config_dir / mode / "train" / f"{exp_name}.yaml"
            output_dir = output_root / mode / exp_name
            checkpoint_path = output_dir / args.checkpoint_name
            temperature_json_path = output_dir / args.temperature_json_name
            eval_json_path = output_dir / build_eval_json_name(args.split, suffix=eval_suffix)

            group_holdout_test_csvs: List[Path] = []
            group_holdout_eval_paths: Dict[str, Path] = {}
            if mode == "group_holdout" and args.split == "test":
                group_holdout_test_csvs = discover_group_holdout_test_csvs(exp_dir)
                group_holdout_eval_paths = get_group_holdout_eval_paths(
                    output_dir=output_dir,
                    split=args.split,
                    test_csv_paths=group_holdout_test_csvs,
                    eval_suffix=eval_suffix,
                )

            build_generated_data_config(
                base_cfg=base_data_cfg,
                exp_dir=exp_dir,
                generated_config_path=generated_data_config_path,
                root_dir_override=args.root_dir,
                mode=mode,
                exp_name=exp_name,
            )
            build_generated_train_config(
                base_cfg=base_train_cfg,
                generated_train_config_path=generated_train_config_path,
                output_dir=output_dir,
            )

            result = {
                "mode": mode,
                "experiment_name": exp_name,
                "split_dir": exp_dir.as_posix(),
                "generated_data_config": generated_data_config_path.as_posix(),
                "generated_train_config": generated_train_config_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "temperature_json_path": temperature_json_path.as_posix(),
                "eval_json_path": eval_json_path.as_posix(),
                "eval_suffix": eval_suffix,
                "group_holdout_test_csvs": [p.as_posix() for p in group_holdout_test_csvs],
                "group_holdout_eval_paths": {k: v.as_posix() for k, v in group_holdout_eval_paths.items()},
                "eval_status": "not_run",
                "error": None,
            }

            try:
                print("\n" + "#" * 100)
                print(f"[{mode}] {exp_name}")
                print("#" * 100)

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint for evaluation not found: {checkpoint_path}"
                    )
                if not temperature_json_path.exists():
                    raise FileNotFoundError(
                        f"Temperature JSON for evaluation not found: {temperature_json_path}"
                    )

                if mode == "group_holdout" and args.split == "test":
                    pooled_eval_json_path = group_holdout_eval_paths["pooled"]
                    per_generator_eval_paths = {
                        k: v for k, v in group_holdout_eval_paths.items() if k != "pooled"
                    }

                    pooled_exists = pooled_eval_json_path.exists()
                    per_generator_all_exist = all(
                        p.exists() for p in per_generator_eval_paths.values()
                    ) if per_generator_eval_paths else False

                    if args.skip_existing and pooled_exists and per_generator_all_exist:
                        print(
                            "Skip group_holdout temperature-scaled evaluation because pooled and "
                            "per-generator eval json files already exist."
                        )
                        result["eval_status"] = "skipped_existing"
                    else:
                        if not (args.skip_existing and pooled_exists):
                            pooled_eval_cmd = make_eval_command(
                                python_executable=args.python,
                                data_config=generated_data_config_path,
                                model_config=model_config_path,
                                train_config=generated_train_config_path,
                                checkpoint_path=checkpoint_path,
                                temperature_json_path=temperature_json_path,
                                split=args.split,
                                output_json=pooled_eval_json_path,
                                split_csv_override=None,
                                device=args.device,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                            )
                            run_command(pooled_eval_cmd, cwd=project_root)
                        else:
                            print(
                                "Skip pooled temperature-scaled evaluation because eval json already exists: "
                                f"{pooled_eval_json_path}"
                            )

                        for test_csv in group_holdout_test_csvs:
                            generator_name = test_csv.stem
                            generator_eval_json_path = per_generator_eval_paths[generator_name]
                            if args.skip_existing and generator_eval_json_path.exists():
                                print(
                                    "Skip per-generator temperature-scaled evaluation because eval json already exists: "
                                    f"{generator_eval_json_path}"
                                )
                                continue

                            eval_cmd = make_eval_command(
                                python_executable=args.python,
                                data_config=generated_data_config_path,
                                model_config=model_config_path,
                                train_config=generated_train_config_path,
                                checkpoint_path=checkpoint_path,
                                temperature_json_path=temperature_json_path,
                                split=args.split,
                                output_json=generator_eval_json_path,
                                split_csv_override=test_csv,
                                device=args.device,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                            )
                            run_command(eval_cmd, cwd=project_root)

                        result["eval_status"] = "done"
                else:
                    if args.skip_existing and eval_json_path.exists():
                        print(
                            "Skip temperature-scaled evaluation because eval json already exists: "
                            f"{eval_json_path}"
                        )
                        result["eval_status"] = "skipped_existing"
                    else:
                        eval_cmd = make_eval_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=generated_train_config_path,
                            checkpoint_path=checkpoint_path,
                            temperature_json_path=temperature_json_path,
                            split=args.split,
                            output_json=eval_json_path,
                            split_csv_override=None,
                            device=args.device,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                        )
                        run_command(eval_cmd, cwd=project_root)
                        result["eval_status"] = "done"
            except Exception as exc:  # noqa: BLE001
                result["eval_status"] = "failed"
                result["error"] = repr(exc)
                print(f"ERROR: {exc}")
                if args.stop_on_error:
                    summary["results"].append(result)
                    raise
            finally:
                summary["results"].append(result)

    if args.summary_json is not None:
        summary_path = resolve_path(args.summary_json)
    else:
        summary_name = f"batch_evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}{eval_suffix}.json"
        summary_path = output_root / "batch_eval_summaries" / summary_name
    save_json(summary, summary_path)

    print("\n" + "=" * 100)
    print("Batch temperature-scaled evaluation finished")
    print("=" * 100)
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
