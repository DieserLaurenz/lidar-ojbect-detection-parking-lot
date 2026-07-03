import argparse
import copy
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def experiment_id(sample):
    sample_id = str(sample.get("sample_id", ""))
    if not sample_id:
        return "unknown"
    return sample_id[0]


def split_group(samples, train_ratio, val_ratio):
    samples = sorted(samples, key=lambda item: str(item.get("sample_id", "")))
    n_samples = len(samples)
    if n_samples < 3:
        return samples, [], []

    train_end = max(1, int(round(n_samples * train_ratio)))
    val_count = max(1, int(round(n_samples * val_ratio)))
    val_end = min(n_samples - 1, train_end + val_count)
    train_end = min(train_end, val_end - 1)

    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def relabel_sample_indices(samples):
    relabeled = []
    for idx, sample in enumerate(samples):
        item = copy.deepcopy(sample)
        item["sample_idx"] = idx
        relabeled.append(item)
    return relabeled


def class_counts(samples):
    counts = Counter()
    for sample in samples:
        for instance in sample.get("instances", []):
            label = instance.get("bbox_label_3d")
            if label is not None:
                counts[label] += 1
    return dict(sorted(counts.items()))


def write_split(root, view, split_name, metainfo, samples):
    out_path = root / f"exp_kitti_infos_{view}_{split_name}.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(
            {"metainfo": copy.deepcopy(metainfo),
             "data_list": relabel_sample_indices(samples)},
            handle,
        )
    return out_path


def split_view(root, view, train_ratio, val_ratio):
    in_path = root / f"exp_kitti_infos_{view}.pkl"
    with in_path.open("rb") as handle:
        data = pickle.load(handle)

    samples = data["data_list"]
    grouped = defaultdict(list)
    for sample in samples:
        grouped[experiment_id(sample)].append(sample)

    splits = {"train": [], "val": [], "test": []}
    for _, group_samples in sorted(grouped.items()):
        train, val, test = split_group(group_samples, train_ratio, val_ratio)
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)

    for split_name in splits:
        splits[split_name] = sorted(
            splits[split_name], key=lambda item: str(item.get("sample_id", "")))
        out_path = write_split(
            root, view, split_name, data["metainfo"], splits[split_name])
        print(
            f"{out_path.name}: samples={len(splits[split_name])} "
            f"labels={class_counts(splits[split_name])}")


def main():
    parser = argparse.ArgumentParser(
        description="Create temporal train/val/test splits for exp KITTI PKLs.")
    parser.add_argument("--root", default="data/exp")
    parser.add_argument(
        "--views", nargs="+", default=["merged", "os0", "os1"],
        choices=["merged", "os0", "os1"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(args.root)
    for view in args.views:
        split_view(root, view, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
