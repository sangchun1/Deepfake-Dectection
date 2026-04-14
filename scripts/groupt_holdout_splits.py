from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_LABEL_NAMES = {
    "real",
    "authentic",
    "original",
    "natural",
    "pristine",
    "genuine",
}
FAKE_LABEL_NAMES = {
    "fake",
    "synthetic",
    "generated",
    "manipulated",
    "deepfake",
    "ai",
}

DEFAULT_GENERATORS = [
    "sd-3.5",
    "flux.1-dev",
    "flux-1.1-pro",
    "midjourney-6",
    "dalle-3",
    "gpt-image-1",
    "ideogram-3.0",
    "hidream-i1-full",
    "grok-2-image-1212",
    "imagen-4.0",
    "sdxl-epic-realism",
    "flux-mvc5000",
]

DEFAULT_GROUP_SPLITS: Dict[str, Dict[str, List[str]]] = {
    "split_a": {
        "train_generators": [
            "sd-3.5",
            "flux.1-dev",
            "flux-mvc5000",
            "midjourney-6",
            "dalle-3",
            "imagen-4.0",
        ],
        "test_generators": [
            "flux-1.1-pro",
            "sdxl-epic-realism",
            "gpt-image-1",
            "ideogram-3.0",
            "hidream-i1-full",
            "grok-2-image-1212",
        ],
    },
    "split_b": {
        "train_generators": [
            "sdxl-epic-realism",
            "flux-1.1-pro",
            "gpt-image-1",
            "ideogram-3.0",
            "hidream-i1-full",
            "imagen-4.0",
        ],
        "test_generators": [
            "sd-3.5",
            "flux.1-dev",
            "flux-mvc5000",
            "midjourney-6",
            "dalle-3",
            "grok-2-image-1212",
        ],
    },
    "split_c": {
        "train_generators": [
            "sd-3.5",
            "flux-1.1-pro",
            "gpt-image-1",
            "midjourney-6",
            "hidream-i1-full",
            "grok-2-image-1212",
        ],
        "test_generators": [
            "sdxl-epic-realism",
            "flux.1-dev",
            "flux-mvc5000",
            "dalle-3",
            "ideogram-3.0",
            "imagen-4.0",
        ],
    },
}


@dataclass(frozen=True)
class Sample:
    filepath: str
    label: int
    generator: Optional[str]


@dataclass(frozen=True)
class SplitCounts:
    train_per_seen_generator: int = 5598
    val_per_seen_generator: int = 800
    test_per_unseen_generator: int = 7997


class SplitCreationError(RuntimeError):
    """Raised when group-holdout splits cannot be created safely."""


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create OpenFake group-holdout split CSVs where seen generators are used "
            "for train/val and unseen generators are evaluated separately."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/openfake",
        help="Root directory of the raw OpenFake dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits/openfake/group_holdout",
        help="Directory where group-holdout split CSV files will be saved.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="data/splits/openfake/summary.json",
        help="Optional summary.json used to load the canonical 12 generators.",
    )
    parser.add_argument(
        "--split_spec_json",
        type=str,
        default="",
        help=(
            "Optional JSON file that overrides the default split_a/split_b/split_c "
            "generator assignments."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--train_per_seen_generator",
        type=int,
        default=SplitCounts.train_per_seen_generator,
        help="Number of fake samples to use for training from each seen generator.",
    )
    parser.add_argument(
        "--val_per_seen_generator",
        type=int,
        default=SplitCounts.val_per_seen_generator,
        help="Number of fake samples to use for validation from each seen generator.",
    )
    parser.add_argument(
        "--test_per_unseen_generator",
        type=int,
        default=SplitCounts.test_per_unseen_generator,
        help=(
            "Maximum number of fake samples to use for each unseen generator test set. "
            "Use -1 to keep all available samples."
        ),
    )
    parser.add_argument(
        "--strict_generators",
        action="store_true",
        help="Raise an error when a fake image path does not map cleanly to one generator.",
    )
    parser.add_argument(
        "--include_reverse",
        action="store_true",
        help="Also create split_a_reverse / split_b_reverse / split_c_reverse.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_label_from_path(path: Path) -> Optional[int]:
    parts_lower = {part.lower() for part in path.parts}
    if parts_lower & REAL_LABEL_NAMES:
        return 0
    if parts_lower & FAKE_LABEL_NAMES:
        return 1
    return None


def load_generators(summary_json_path: Path) -> List[str]:
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        generators = payload.get("generators")
        if isinstance(generators, list) and generators:
            return [str(generator) for generator in generators]
    return DEFAULT_GENERATORS.copy()


def load_group_splits(split_spec_json: Optional[Path], include_reverse: bool) -> Dict[str, Dict[str, List[str]]]:
    if split_spec_json and split_spec_json.exists():
        with split_spec_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        split_specs = {
            str(name): {
                "train_generators": list(spec["train_generators"]),
                "test_generators": list(spec["test_generators"]),
            }
            for name, spec in payload.items()
        }
    else:
        split_specs = {
            name: {
                "train_generators": spec["train_generators"][:],
                "test_generators": spec["test_generators"][:],
            }
            for name, spec in DEFAULT_GROUP_SPLITS.items()
        }

    if include_reverse:
        reverse_specs: Dict[str, Dict[str, List[str]]] = {}
        for name, spec in split_specs.items():
            reverse_specs[f"{name}_reverse"] = {
                "train_generators": spec["test_generators"][:],
                "test_generators": spec["train_generators"][:],
            }
        split_specs.update(reverse_specs)
    return split_specs


def infer_generator_from_path(path: Path, known_generators: Sequence[str]) -> Optional[str]:
    parts_lower = {part.lower() for part in path.parts}
    exact_matches = [generator for generator in known_generators if generator.lower() in parts_lower]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise SplitCreationError(
            f"Ambiguous generator inference for path '{path.as_posix()}': {exact_matches}"
        )
    return None


def collect_samples(
    input_dir: Path,
    project_root: Path,
    known_generators: Sequence[str],
    strict_generators: bool = False,
) -> List[Sample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and is_image_file(path)
    )
    if not image_paths:
        raise SplitCreationError(f"No image files found under: {input_dir}")

    samples: List[Sample] = []
    skipped_label_paths: List[str] = []
    skipped_generator_paths: List[str] = []

    for path in image_paths:
        label = infer_label_from_path(path)
        if label is None:
            skipped_label_paths.append(path.as_posix())
            continue

        generator: Optional[str] = None
        if label == 1:
            generator = infer_generator_from_path(path, known_generators)
            if generator is None:
                if strict_generators:
                    raise SplitCreationError(
                        f"Could not infer generator from fake image path: {path.as_posix()}"
                    )
                skipped_generator_paths.append(path.as_posix())
                continue

        rel_path = path.relative_to(project_root).as_posix()
        samples.append(Sample(filepath=rel_path, label=label, generator=generator))

    if not samples:
        raise SplitCreationError("No valid samples were collected.")

    if skipped_label_paths:
        print(
            "Skipped "
            f"{len(skipped_label_paths)} files because no real/fake label could be inferred."
        )
    if skipped_generator_paths:
        print(
            "Skipped "
            f"{len(skipped_generator_paths)} fake files because the generator name could not be inferred."
        )

    return samples


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "filepath": [sample.filepath for sample in samples],
            "label": [sample.label for sample in samples],
            "generator": [sample.generator for sample in samples],
        }
    )
    if df["filepath"].duplicated().any():
        dup_count = int(df["filepath"].duplicated().sum())
        print(f"Warning: found {dup_count} duplicated filepaths. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["filepath"], keep="first").reset_index(drop=True)
    return df


def validate_split_specs(
    split_specs: Dict[str, Dict[str, List[str]]],
    allowed_generators: Sequence[str],
) -> None:
    allowed_set = set(allowed_generators)
    for split_name, spec in split_specs.items():
        train_generators = spec["train_generators"]
        test_generators = spec["test_generators"]
        train_set = set(train_generators)
        test_set = set(test_generators)

        unknown = (train_set | test_set) - allowed_set
        if unknown:
            raise SplitCreationError(
                f"{split_name} contains unknown generators: {sorted(unknown)}"
            )
        if train_set & test_set:
            raise SplitCreationError(
                f"{split_name} has overlapping train/test generators: {sorted(train_set & test_set)}"
            )
        if len(train_generators) != 6 or len(test_generators) != 6:
            raise SplitCreationError(
                f"{split_name} must have exactly 6 train generators and 6 test generators."
            )
        if train_set | test_set != allowed_set:
            missing = sorted(allowed_set - (train_set | test_set))
            raise SplitCreationError(
                f"{split_name} does not cover all generators. Missing: {missing}"
            )


def save_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["filepath", "label"]].to_csv(path, index=False, encoding="utf-8")


def sample_n(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n < 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df) < n:
        raise SplitCreationError(f"Requested {n} samples, but only {len(df)} are available.")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def select_seen_fake_splits(
    fake_df: pd.DataFrame,
    seen_generators: Sequence[str],
    counts: SplitCounts,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    per_generator_counts: Dict[str, Dict[str, int]] = {}

    required_per_generator = counts.train_per_seen_generator + counts.val_per_seen_generator
    for offset, generator in enumerate(seen_generators):
        generator_df = fake_df[fake_df["generator"] == generator].reset_index(drop=True)
        sampled = sample_n(generator_df, required_per_generator, seed + offset)
        train_df = sampled.iloc[: counts.train_per_seen_generator].reset_index(drop=True)
        val_df = sampled.iloc[counts.train_per_seen_generator :].reset_index(drop=True)
        train_parts.append(train_df)
        val_parts.append(val_df)
        per_generator_counts[generator] = {
            "available": int(len(generator_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
        }

    train_fake_df = pd.concat(train_parts, ignore_index=True)
    val_fake_df = pd.concat(val_parts, ignore_index=True)
    return train_fake_df, val_fake_df, per_generator_counts


def select_unseen_fake_tests(
    fake_df: pd.DataFrame,
    unseen_generators: Sequence[str],
    counts: SplitCounts,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    per_generator_tests: Dict[str, pd.DataFrame] = {}
    per_generator_counts: Dict[str, Dict[str, int]] = {}

    for offset, generator in enumerate(unseen_generators):
        generator_df = fake_df[fake_df["generator"] == generator].reset_index(drop=True)
        requested = None if counts.test_per_unseen_generator < 0 else counts.test_per_unseen_generator
        effective = len(generator_df) if requested is None else min(requested, len(generator_df))
        sampled = sample_n(generator_df, effective, seed + 1000 + offset)
        per_generator_tests[generator] = sampled
        per_generator_counts[generator] = {
            "available": int(len(generator_df)),
            "requested_test": int(requested) if requested is not None else -1,
            "used_test": int(len(sampled)),
        }

    return per_generator_tests, per_generator_counts


def allocate_real_splits(
    real_df: pd.DataFrame,
    train_size: int,
    val_size: int,
    per_generator_test_sizes: Dict[str, int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    total_needed = train_size + val_size + sum(per_generator_test_sizes.values())
    if len(real_df) < total_needed:
        raise SplitCreationError(
            "Not enough real images for the requested group-holdout split. "
            f"Need {total_needed}, but only {len(real_df)} are available."
        )

    shuffled = real_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cursor = 0

    train_real_df = shuffled.iloc[cursor : cursor + train_size].reset_index(drop=True)
    cursor += train_size
    val_real_df = shuffled.iloc[cursor : cursor + val_size].reset_index(drop=True)
    cursor += val_size

    per_generator_real_tests: Dict[str, pd.DataFrame] = {}
    for generator, size in per_generator_test_sizes.items():
        per_generator_real_tests[generator] = shuffled.iloc[cursor : cursor + size].reset_index(drop=True)
        cursor += size

    return train_real_df, val_real_df, per_generator_real_tests


def build_binary_csv(fake_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([fake_df, real_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return combined[["filepath", "label"]]


def build_split(
    all_df: pd.DataFrame,
    split_name: str,
    train_generators: Sequence[str],
    test_generators: Sequence[str],
    counts: SplitCounts,
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    fake_df = all_df[all_df["label"] == 1].reset_index(drop=True)
    real_df = all_df[all_df["label"] == 0].reset_index(drop=True)

    train_fake_df, val_fake_df, seen_counts = select_seen_fake_splits(
        fake_df=fake_df,
        seen_generators=train_generators,
        counts=counts,
        seed=seed,
    )
    per_generator_fake_tests, unseen_counts = select_unseen_fake_tests(
        fake_df=fake_df,
        unseen_generators=test_generators,
        counts=counts,
        seed=seed,
    )

    per_generator_test_sizes = {
        generator: int(len(df)) for generator, df in per_generator_fake_tests.items()
    }
    train_real_df, val_real_df, per_generator_real_tests = allocate_real_splits(
        real_df=real_df,
        train_size=len(train_fake_df),
        val_size=len(val_fake_df),
        per_generator_test_sizes=per_generator_test_sizes,
        seed=seed,
    )

    split_root = output_dir / split_name
    tests_root = split_root / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)

    train_csv = build_binary_csv(train_fake_df, train_real_df)
    val_csv = build_binary_csv(val_fake_df, val_real_df)
    save_split_csv(train_csv, split_root / "train.csv")
    save_split_csv(val_csv, split_root / "val.csv")

    pooled_test_parts: List[pd.DataFrame] = []
    test_metadata: Dict[str, Dict[str, int]] = {}
    for generator in test_generators:
        generator_test_csv = build_binary_csv(
            fake_df=per_generator_fake_tests[generator],
            real_df=per_generator_real_tests[generator],
        )
        save_split_csv(generator_test_csv, tests_root / f"{generator}.csv")
        pooled_test_parts.append(generator_test_csv)
        test_metadata[generator] = {
            "fake": int(len(per_generator_fake_tests[generator])),
            "real": int(len(per_generator_real_tests[generator])),
            "csv_total": int(len(generator_test_csv)),
        }

    pooled_test_csv = pd.concat(pooled_test_parts, ignore_index=True)
    pooled_test_csv = pooled_test_csv.sample(frac=1.0, random_state=0).reset_index(drop=True)
    save_split_csv(pooled_test_csv, split_root / "test.csv")

    metadata = {
        "split_name": split_name,
        "seed": seed,
        "train_generators": list(train_generators),
        "test_generators": list(test_generators),
        "requested_counts": {
            "train_per_seen_generator": counts.train_per_seen_generator,
            "val_per_seen_generator": counts.val_per_seen_generator,
            "test_per_unseen_generator": counts.test_per_unseen_generator,
        },
        "seen_fake_counts": seen_counts,
        "unseen_fake_counts": unseen_counts,
        "csv_counts": {
            "train": int(len(train_csv)),
            "val": int(len(val_csv)),
            "test_pooled": int(len(pooled_test_csv)),
        },
        "real_counts": {
            "train": int(len(train_real_df)),
            "val": int(len(val_real_df)),
            "test_total": int(sum(len(df) for df in per_generator_real_tests.values())),
        },
        "per_generator_test_csv_counts": test_metadata,
        "files": {
            "train_csv": (split_root / "train.csv").as_posix(),
            "val_csv": (split_root / "val.csv").as_posix(),
            "test_csv": (split_root / "test.csv").as_posix(),
            "tests_dir": tests_root.as_posix(),
        },
        "note": (
            "train/val use only seen generators. tests/*.csv contains one unseen-generator "
            "evaluation CSV per generator. test.csv is the pooled union of all unseen tests."
        ),
    }

    with (split_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"[{split_name}] saved train={len(train_csv)} val={len(val_csv)} "
        f"test_pooled={len(pooled_test_csv)}"
    )
    for generator in test_generators:
        print(
            f"  - test/{generator}.csv: "
            f"fake={len(per_generator_fake_tests[generator])} "
            f"real={len(per_generator_real_tests[generator])}"
        )

    return metadata


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    summary_json_path = (project_root / args.summary_json).resolve()
    split_spec_json = (project_root / args.split_spec_json).resolve() if args.split_spec_json else None

    counts = SplitCounts(
        train_per_seen_generator=args.train_per_seen_generator,
        val_per_seen_generator=args.val_per_seen_generator,
        test_per_unseen_generator=args.test_per_unseen_generator,
    )

    generators = load_generators(summary_json_path)
    split_specs = load_group_splits(split_spec_json, include_reverse=args.include_reverse)
    validate_split_specs(split_specs, generators)

    samples = collect_samples(
        input_dir=input_dir,
        project_root=project_root,
        known_generators=generators,
        strict_generators=args.strict_generators,
    )
    all_df = samples_to_dataframe(samples)

    fake_df = all_df[all_df["label"] == 1].reset_index(drop=True)
    real_df = all_df[all_df["label"] == 0].reset_index(drop=True)
    print(f"Collected {len(all_df)} samples: real={len(real_df)} fake={len(fake_df)}")
    print(f"Generators: {generators}")

    generator_availability = (
        fake_df["generator"].value_counts().sort_index().to_dict() if not fake_df.empty else {}
    )
    print(f"Fake availability by generator: {generator_availability}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: Dict[str, object] = {
        "input_root": input_dir.as_posix(),
        "output_root": output_dir.as_posix(),
        "seed": args.seed,
        "generators": generators,
        "counts": {
            "train_per_seen_generator": counts.train_per_seen_generator,
            "val_per_seen_generator": counts.val_per_seen_generator,
            "test_per_unseen_generator": counts.test_per_unseen_generator,
        },
        "splits": {},
    }

    for offset, (split_name, spec) in enumerate(split_specs.items()):
        metadata = build_split(
            all_df=all_df,
            split_name=split_name,
            train_generators=spec["train_generators"],
            test_generators=spec["test_generators"],
            counts=counts,
            output_dir=output_dir,
            seed=args.seed + offset,
        )
        all_metadata["splits"][split_name] = metadata

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved group-holdout summary to: {(output_dir / 'summary.json').as_posix()}")


if __name__ == "__main__":
    main()
