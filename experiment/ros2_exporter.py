#!/usr/bin/python3

import os
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from pypcd4 import PointCloud
from argparse import ArgumentParser
from tqdm import tqdm
import glob
import bisect
import numpy as np
import rosbag2_py


def extract_pc2_from_bag(bag_path, topic):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    msgs = []
    timestamps = []
    while reader.has_next():
        tpc, data, t = reader.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        msgs.append(msg)
        timestamps.append(t)
    return msgs, timestamps


def pc2_to_pcd(msg):
    pc = PointCloud.from_msg(msg)

    arr_xyzi = np.stack(
        [
            pc.pc_data['x'],
            pc.pc_data['y'],
            pc.pc_data['z'],
            pc.pc_data['intensity']
        ],
        axis=-1,
    )
    pc_xyzi = PointCloud.from_xyzi_points(arr_xyzi)

    return pc_xyzi


def match_timestamps(ts1: list, ts2: list, tol_ns: float = 1e10):
    if len(ts1) > len(ts2):
        tmp = ts1
        ts1 = ts2
        ts2 = tmp

    pairs = []
    failed = 0
    for t1 in ts1:
        # Find closest t2
        idx = bisect.bisect_left(ts2, t1)
        candidates = []
        if idx < len(ts2):
            candidates.append(ts2[idx])
        if idx > 0:
            candidates.append(ts2[idx-1])
        # Pick the closest within tolerance
        closest = min(candidates, key=lambda x: abs(x-t1))
        if abs(closest - t1) <= tol_ns:
            pairs.append((ts1.index(t1), ts2.index(closest)))
        else:
            failed += 1
            # print(abs(closest - t1))
    if failed:
        tqdm.write(f"WARN: Failed to match {failed} timestamps")
    return pairs


def get_bag_db3(bag_path):
    bag_db_path = glob.glob(os.path.join(bag_path, "*.db3"))
    assert bag_db_path, f"Unable to find db3 in bag, {bag_path}"
    return bag_db_path[0]


def run(
        bag_os0_path: str,
        bag_os1_path: str,
        os0_topic: str = "/ouster_os0/points",
        os1_topic: str = "/ouster_os1/points"
):
    os0_bag_path = get_bag_db3(bag_os0_path)
    os1_bag_path = get_bag_db3(bag_os1_path)
    output_dir = os.path.dirname(args.bag_os0)

    rclpy.init()

    print("Loading OS0 bag...")
    os0_msgs, os0_timestamps = extract_pc2_from_bag(os0_bag_path, os0_topic)
    print(f"OS0: {len(os0_msgs)} frames")

    print("Loading OS1 bag...")
    os1_msgs, os1_timestamps = extract_pc2_from_bag(os1_bag_path, os1_topic)
    print(f"OS1: {len(os1_msgs)} frames")

    aligned_pairs_idxs = match_timestamps(
        os0_timestamps,
        os1_timestamps,
    )
    print(f"Aligned frames: {len(aligned_pairs_idxs)}")

    for (ts0_idx, ts1_idx) in tqdm(aligned_pairs_idxs, unit="msg"):
        pc0 = pc2_to_pcd(os0_msgs[ts0_idx])
        pc1 = pc2_to_pcd(os1_msgs[ts1_idx])

        ts_both = os0_timestamps[ts0_idx]
        out0_dir = os.path.join(output_dir, "os0_pcd/")
        out1_dir = os.path.join(output_dir, "os1_pcd/")
        os.makedirs(out0_dir, exist_ok=True)
        os.makedirs(out1_dir, exist_ok=True)

        pc0.save(os.path.join(out0_dir, f"{ts_both}.pcd"))
        pc1.save(os.path.join(out1_dir, f"{ts_both}.pcd"))

    print("Done! All aligned PCDs exported.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Export all point clouds from two ros2 bags.")
    parser.add_argument("bag_os0", help="Path to dir of bag os0")
    parser.add_argument("bag_os1", help="Path to dir of bag os1")
    parser.add_argument(
        "--os0-topic", help="ROS2 topic for points in bag os0", default="/ouster_os0/points")
    parser.add_argument(
        "--os1-topic", help="ROS2 topic for points in bag os1", default="/ouster_os1/points")
    args = parser.parse_args()

    run(
        args.bag_os0,
        args.bag_os1,
        os0_topic=args.os0_topic,
        os1_topic=args.os1_topic,
    )
