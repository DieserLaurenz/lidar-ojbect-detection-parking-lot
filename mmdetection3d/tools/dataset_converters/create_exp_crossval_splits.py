"""Create paired, experiment-held-out cross-validation splits.

Every fold holds out one complete experiment per target class:

* fold 1: experiments 1 (car), 4 (bicycle), 7 (person)
* fold 2: experiments 2 (car), 5 (bicycle), 8 (person)
* fold 3: experiments 3 (car), 6 (bicycle), 9 (person)

Only timestamps that are valid in merged, os0, and os1 are used. This keeps
the physical frames identical across views. Validation is the final fraction
of each remaining training experiment; an optional guard removes adjacent
frames from training. Test experiments are never used for model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


VIEWS = ("merged", "os0", "os1")
FOLDS = {
    1: ("1", "4", "7"),
    2: ("2", "5", "8"),
    3: ("3", "6", "9"),
}


def logical_key(sample: dict) -> tuple[str, str]:
    sample_id = str(sample.get("sample_id", ""))
    if len(sample_id) < 7:
        raise ValueError(f"Invalid sample_id: {sample_id!r}")
    return sample_id[0], sample_id[6:]


def class_counts(samples: list[dict]) -> dict[str, int]:
    counts = Counter()
    for sample in samples:
        for instance in sample.get("instances", []):
            label = instance.get("bbox_label_3d")
            if label is not None:
                counts[str(label)] += 1
    return dict(sorted(counts.items()))


def relabel_sample_indices(samples: list[dict]) -> list[dict]:
    result = []
    for index, sample in enumerate(samples):
        item = copy.deepcopy(sample)
        item["sample_idx"] = index
        result.append(item)
    return result


def load_views(root: Path) -> tuple[dict, dict[str, dict]]:
    metainfo = {}
    samples_by_view = {}
    for view in VIEWS:
        path = root / f"exp_kitti_infos_{view}.pkl"
        with path.open("rb") as handle:
            data = pickle.load(handle)
        metainfo[view] = data["metainfo"]
        keyed = {}
        for sample in data["data_list"]:
            key = logical_key(sample)
            if key in keyed:
                raise ValueError(f"Duplicate logical frame in {view}: {key}")
            keyed[key] = sample
        samples_by_view[view] = keyed
    return metainfo, samples_by_view


def build_fold_keys(
    common_keys: set[tuple[str, str]],
    fold: int,
    val_ratio: float,
    guard_frames: int,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str]]]:
    test_experiments = set(FOLDS[fold])
    grouped = defaultdict(list)
    for key in common_keys:
        grouped[key[0]].append(key)
    for experiment in grouped:
        grouped[experiment].sort(key=lambda key: int(key[1]))

    split_keys = {"train": [], "val": [], "test": []}
    guard_keys = []
    for experiment, keys in sorted(grouped.items()):
        if experiment in test_experiments:
            split_keys["test"].extend(keys)
            continue

        val_count = max(1, int(round(len(keys) * val_ratio)))
        val_start = len(keys) - val_count
        guard_start = max(0, val_start - guard_frames)
        split_keys["train"].extend(keys[:guard_start])
        guard_keys.extend(keys[guard_start:val_start])
        split_keys["val"].extend(keys[val_start:])

    for split in split_keys:
        split_keys[split].sort(key=lambda key: (int(key[0]), int(key[1])))

    train = set(split_keys["train"])
    val = set(split_keys["val"])
    test = set(split_keys["test"])
    assert not (train & val or train & test or val & test)
    assert {key[0] for key in test} == test_experiments
    assert train | val | test | set(guard_keys) == common_keys
    return split_keys, guard_keys


def write_split(
    root: Path,
    view: str,
    fold: int,
    split: str,
    metainfo: dict,
    keyed_samples: dict,
    keys: list[tuple[str, str]],
) -> tuple[Path, list[dict]]:
    samples = [keyed_samples[key] for key in keys]
    samples.sort(key=lambda sample: str(sample["sample_id"]))
    samples = relabel_sample_indices(samples)
    path = root / f"exp_kitti_infos_{view}_cv{fold}_{split}.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {"metainfo": copy.deepcopy(metainfo), "data_list": samples},
            handle,
        )
    return path, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/exp"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--guard-frames", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 0.5:
        parser.error("--val-ratio must be between 0 and 0.5")
    if args.guard_frames < 0:
        parser.error("--guard-frames must be non-negative")

    metainfo, samples_by_view = load_views(args.root)
    view_keys = {view: set(samples) for view, samples in samples_by_view.items()}
    common_keys = set.intersection(*(view_keys[view] for view in VIEWS))

    manifest = {
        "protocol": "paired experiment-held-out 3-fold cross-validation",
        "views": list(VIEWS),
        "fold_test_experiments": {str(k): list(v) for k, v in FOLDS.items()},
        "val_ratio": args.val_ratio,
        "guard_frames": args.guard_frames,
        "view_full_frame_counts": {
            view: len(samples_by_view[view]) for view in VIEWS
        },
        "common_frame_count": len(common_keys),
        "excluded_noncommon_frames": {
            view: len(view_keys[view] - common_keys) for view in VIEWS
        },
        "folds": {},
    }

    print(f"Common paired frames: {len(common_keys)}")
    for fold in FOLDS:
        split_keys, guard_keys = build_fold_keys(
            common_keys, fold, args.val_ratio, args.guard_frames
        )
        fold_entry = {
            "test_experiments": list(FOLDS[fold]),
            "logical_frame_counts": {
                split: len(keys) for split, keys in split_keys.items()
            },
            "guard_frame_count": len(guard_keys),
            "views": {},
        }
        print(
            f"Fold {fold} test={FOLDS[fold]}: "
            + ", ".join(
                f"{split}={len(keys)}" for split, keys in split_keys.items()
            )
            + f", guard={len(guard_keys)}"
        )
        for view in VIEWS:
            view_entry = {}
            for split, keys in split_keys.items():
                path, samples = write_split(
                    args.root,
                    view,
                    fold,
                    split,
                    metainfo[view],
                    samples_by_view[view],
                    keys,
                )
                view_entry[split] = {
                    "file": path.name,
                    "samples": len(samples),
                    "class_counts": class_counts(samples),
                }
            fold_entry["views"][view] = view_entry
        manifest["folds"][str(fold)] = fold_entry

    manifest_path = args.root / "exp_crossval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
