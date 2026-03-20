from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    filepath: str
    label: int


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CIFAKE train/val/test split CSV files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/cifake",
        help="Root directory of the raw CIFAKE dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits/cifake",
        help="Directory where split CSV files will be saved.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio from the training pool.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio when the dataset has no predefined test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_label_from_path(path: Path) -> int:
    """
    경로에 포함된 폴더명으로 라벨 추론.
    real -> 0
    fake -> 1
    """
    parts_lower = [part.lower() for part in path.parts]

    if "real" in parts_lower:
        return 0
    if "fake" in parts_lower:
        return 1

    raise ValueError(f"Could not infer label from path: {path}")


def infer_predefined_split(path: Path) -> Optional[str]:
    """
    CIFAKE가 train/test 폴더를 이미 가지고 있으면 이를 활용.
    """
    parts_lower = [part.lower() for part in path.parts]

    if "train" in parts_lower:
        return "train"
    if "test" in parts_lower:
        return "test"

    return None


def collect_samples(input_dir: Path, project_root: Path) -> List[Sample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and is_image_file(path)
    )

    if not image_paths:
        raise RuntimeError(f"No image files found under: {input_dir}")

    samples: List[Sample] = []
    for path in image_paths:
        label = infer_label_from_path(path)
        rel_path = path.relative_to(project_root).as_posix()
        samples.append(Sample(filepath=rel_path, label=label))

    return samples


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": [sample.filepath for sample in samples],
            "label": [sample.label for sample in samples],
        }
    )


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, split_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        split_df.reset_index(drop=True),
    )


def split_with_predefined_test(
    df: pd.DataFrame,
    project_root: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    파일 경로에 train/test 폴더가 이미 존재하는 경우:
    - test 폴더는 그대로 test
    - train 폴더만 다시 train/val로 분리
    """
    rel_paths = df["filepath"].apply(lambda x: Path(project_root / x))
    split_tags = rel_paths.apply(infer_predefined_split)

    if split_tags.isnull().any():
        raise ValueError(
            "Some files do not belong to a predefined 'train' or 'test' folder."
        )

    train_pool_df = df[split_tags == "train"].reset_index(drop=True)
    test_df = df[split_tags == "test"].reset_index(drop=True)

    if len(train_pool_df) == 0:
        raise RuntimeError("No training files found in predefined train split.")
    if len(test_df) == 0:
        raise RuntimeError("No test files found in predefined test split.")

    train_df, val_df = stratified_split(
        train_pool_df,
        test_size=val_ratio,
        seed=seed,
    )
    return train_df, val_df, test_df


def split_without_predefined_test(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 데이터를 train/val/test로 직접 stratified split.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    train_val_df, test_df = stratified_split(
        df,
        test_size=test_ratio,
        seed=seed,
    )

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)

    train_df, val_df = stratified_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        seed=seed,
    )
    return train_df, val_df, test_df


def has_predefined_test_split(df: pd.DataFrame, project_root: Path) -> bool:
    rel_paths = df["filepath"].apply(lambda x: Path(project_root / x))
    split_tags = rel_paths.apply(infer_predefined_split)
    return split_tags.notnull().all() and ("test" in set(split_tags.tolist()))


def save_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def print_distribution(name: str, df: pd.DataFrame) -> None:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    print(f"[{name}] total={len(df)} label_counts={label_counts}")


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    samples = collect_samples(input_dir=input_dir, project_root=project_root)
    df = samples_to_dataframe(samples)

    if has_predefined_test_split(df, project_root):
        print("Detected predefined train/test folders. Creating train/val from train only.")
        train_df, val_df, test_df = split_with_predefined_test(
            df=df,
            project_root=project_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        print("No predefined test folder detected. Creating train/val/test from all samples.")
        train_df, val_df, test_df = split_without_predefined_test(
            df=df,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    save_split_csv(train_df, output_dir / "train.csv")
    save_split_csv(val_df, output_dir / "val.csv")
    save_split_csv(test_df, output_dir / "test.csv")

    print_distribution("train", train_df)
    print_distribution("val", val_df)
    print_distribution("test", test_df)

    print(f"Saved train split to: {(output_dir / 'train.csv').as_posix()}")
    print(f"Saved val split to:   {(output_dir / 'val.csv').as_posix()}")
    print(f"Saved test split to:  {(output_dir / 'test.csv').as_posix()}")


if __name__ == "__main__":
    main()