#!/usr/bin/python3

import os
import bisect
import glob
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import rclpy  # noqa: F401
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from pypcd4 import PointCloud
import rosbag2_py


def extract_pc2_from_bag(bag_path, topic):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    raw = []
    timestamps = []
    while reader.has_next():
        tpc, data, t = reader.read_next()
        if tpc != topic:
            continue
        raw.append(data)
        timestamps.append(t)
    return raw, timestamps


def match_timestamps(ts1, ts2, tol_ns=1e10):
    swapped = False
    if len(ts1) > len(ts2):
        ts1, ts2 = ts2, ts1
        swapped = True
    pairs = []
    failed = 0
    for i, t1 in enumerate(ts1):
        idx = bisect.bisect_left(ts2, t1)
        cands = []
        if idx < len(ts2):
            cands.append((ts2[idx], idx))
        if idx > 0:
            cands.append((ts2[idx - 1], idx - 1))
        c_val, c_idx = min(cands, key=lambda x: abs(x[0] - t1))
        if abs(c_val - t1) <= tol_ns:
            pairs.append((c_idx, i) if swapped else (i, c_idx))
        else:
            failed += 1
    if failed:
        print(f"WARN: Failed to match {failed} timestamps")
    return pairs


def convert_and_save(args):
    raw0, raw1, ts, out0_dir, out1_dir = args
    msg0 = deserialize_message(raw0, PointCloud2)
    msg1 = deserialize_message(raw1, PointCloud2)
    for msg, out_dir in ((msg0, out0_dir), (msg1, out1_dir)):
        pc = PointCloud.from_msg(msg)
        arr = np.stack(
            [pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z'], pc.pc_data['intensity']],
            axis=-1,
        )
        PointCloud.from_xyzi_points(arr).save(os.path.join(out_dir, f"{ts}.pcd"))
    return ts


def get_bag_db3(bag_path):
    bag_db_path = glob.glob(os.path.join(bag_path, "*.db3"))
    assert bag_db_path, f"Unable to find db3 in bag, {bag_path}"
    return bag_db_path[0]


def run(bag_os0_path, bag_os1_path, os0_topic, os1_topic, workers):
    os0_bag_path = get_bag_db3(bag_os0_path)
    os1_bag_path = get_bag_db3(bag_os1_path)
    output_dir = os.path.dirname(bag_os0_path)
    out0_dir = os.path.join(output_dir, "os0_pcd")
    out1_dir = os.path.join(output_dir, "os1_pcd")
    os.makedirs(out0_dir, exist_ok=True)
    os.makedirs(out1_dir, exist_ok=True)

    print(f"[{output_dir}] Loading OS0 bag...")
    raw0, ts0 = extract_pc2_from_bag(os0_bag_path, os0_topic)
    print(f"[{output_dir}] OS0: {len(raw0)} frames")

    print(f"[{output_dir}] Loading OS1 bag...")
    raw1, ts1 = extract_pc2_from_bag(os1_bag_path, os1_topic)
    print(f"[{output_dir}] OS1: {len(raw1)} frames")

    pairs = match_timestamps(ts0, ts1)
    print(f"[{output_dir}] Aligned frames: {len(pairs)}")

    tasks = [(raw0[i0], raw1[i1], ts0[i0], out0_dir, out1_dir) for (i0, i1) in pairs]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(convert_and_save, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), unit="msg"):
            pass

    print(f"[{output_dir}] Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export aligned PCDs from two ros2 bags (parallel).")
    parser.add_argument("bag_os0")
    parser.add_argument("bag_os1")
    parser.add_argument("--os0-topic", default="/ouster_os0/points")
    parser.add_argument("--os1-topic", default="/ouster_os1/points")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()
    run(args.bag_os0, args.bag_os1, args.os0_topic, args.os1_topic, args.workers)
