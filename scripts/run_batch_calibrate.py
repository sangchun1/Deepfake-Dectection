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


def save_yaml(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-fit temperature scaling files (temperature.json) for checkpoints "
            "saved by run_batch_experiments.py over merged / by_generator / logo / group_holdout."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["merged", "by_generator", "logo", "group_holdout", "all"],
        help="Which experiment family to calibrate.",
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
        default="configs/_generated/openfake_calibration_batch",
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
        help="Optional override for cfg.data.paths.root_dir.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional subset of experiment folder names to calibrate.",
    )
    parser.add_argument(
        "--fit_split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to use for fitting temperature. Usually val.",
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
        help="Temperature JSON filename to save under output_root/mode/exp_name/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Optional device override passed to calibrate_temperature.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size override passed to calibrate_temperature.py.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional num_workers override passed to calibrate_temperature.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional reporting threshold passed to calibrate_temperature.py.",
    )
    parser.add_argument(
        "--temperature_init",
        type=float,
        default=1.0,
        help="Initial temperature value before optimization.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Max LBFGS iterations for temperature fitting.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="LBFGS learning rate for temperature fitting.",
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
        help="Skip calibration if the target temperature json already exists.",
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
        help="Optional explicit path to save the batch calibration summary JSON.",
    )
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


def make_calibrate_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    checkpoint_path: Path,
    fit_split: str,
    output_json: Path,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
    threshold: float | None,
    temperature_init: float,
    max_iter: int,
    lr: float,
) -> List[str]:
    cmd = [
        python_executable,
        "scripts/calibrate_temperature.py",
        "--data_config",
        data_config.as_posix(),
        "--model_config",
        model_config.as_posix(),
        "--train_config",
        train_config.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--split",
        fit_split,
        "--output_json",
        output_json.as_posix(),
        "--temperature_init",
        str(temperature_init),
        "--max_iter",
        str(max_iter),
        "--lr",
        str(lr),
    ]
    if device is not None:
        cmd.extend(["--device", device])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
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
        "fit_split": args.fit_split,
        "checkpoint_name": args.checkpoint_name,
        "temperature_json_name": args.temperature_json_name,
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
                "fit_split": args.fit_split,
                "status": "not_run",
                "error": None,
            }

            try:
                print("\n" + "#" * 100)
                print(f"[{mode}] {exp_name}")
                print("#" * 100)

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint for calibration not found: {checkpoint_path}"
                    )

                if args.skip_existing and temperature_json_path.exists():
                    print("Skip calibration because temperature json already exists.")
                    result["status"] = "skipped_existing"
                else:
                    cmd = make_calibrate_command(
                        python_executable=args.python,
                        data_config=generated_data_config_path,
                        model_config=model_config_path,
                        train_config=generated_train_config_path,
                        checkpoint_path=checkpoint_path,
                        fit_split=args.fit_split,
                        output_json=temperature_json_path,
                        device=args.device,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        threshold=args.threshold,
                        temperature_init=args.temperature_init,
                        max_iter=args.max_iter,
                        lr=args.lr,
                    )
                    run_command(cmd, cwd=project_root)
                    result["status"] = "success"
            except Exception as exc:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = str(exc)
                print(f"ERROR: {exc}")
                if args.stop_on_error:
                    summary["results"].append(result)
                    summary_path = (
                        resolve_path(args.summary_json)
                        if args.summary_json is not None
                        else output_root
                        / "batch_eval_summaries"
                        / f"batch_calibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    save_json(summary, summary_path)
                    raise
            finally:
                summary["results"].append(result)

    summary_path = (
        resolve_path(args.summary_json)
        if args.summary_json is not None
        else output_root
        / "batch_eval_summaries"
        / f"batch_calibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_json(summary, summary_path)

    print("\n" + "=" * 100)
    print("Batch temperature calibration finished")
    print("=" * 100)
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
