import numpy as np
from tabulate import tabulate

from mmdet3d.datasets.lumpi import LUMPIDataset

from tools.analysis_tools.dataset_analyzer import (
    DatasetAnalyzer, AnalysisMode, AnchorMode, DatasetAnalyerArgumentParser
)


class ExpAnalyzer(DatasetAnalyzer):
    """
    Analyze quality of Experiment data and calculate needed values for training

    Args:
        labels_path(str):           Path to converted label folder.
        points_path(str):           Path to converted point cloud folder.
        anchor_mode(AnchorMode):    Defines if anchor created per class or
                                    whole dataset. Default PER_CLASS.
        anchor_count(int):          If anchor_mode PER_CLASS is used defines
                                    anchors per class.
                                    If anchor mode PER_DATASET is used it
                                    define the total number of anchors.
                                    (Default: 1)
    """

    def __init__(
            self,
            labels_path: str,
            points_path: str,
            output_dir: str,
            anchor_mode: AnchorMode = AnchorMode.PER_CLASS,
            anchor_count: int = 1,
            modes: list = [AnalysisMode.ALL],
            percentiles: list = [(95, 5), (96, 4), (97, 3), (98, 2), (99, 1)],
    ):
        ExpDataset = LUMPIDataset
        ExpDataset.METAINFO['classes'] = ExpDataset.METAINFO['classes'][:3]

        super().__init__(
            LUMPIDataset,
            labels_path,
            points_path,
            output_dir,
            anchor_mode,
            anchor_count,
            modes,
            percentiles
        )

    def analyze_class_usage(self, low_point_thr: int = 15) -> None:
        """
        Overrides inherrit function to adjust to divide data by sensor
        configuration.

        Args:
            low_point_thr (int):    threshold at which count as bbox with
                                    few points. (Default: 15)
        """
        super().analyze_class_usage()

        data = {}
        sensor_ids = np.array(
            [int(s[1]) for s in self.gt_frames.astype(str)])
        total_samples_merged = len(self.gt_frames[sensor_ids == 2])
        total_samples_os0 = len(self.gt_frames[sensor_ids == 0])
        total_samples_os1 = len(self.gt_frames[sensor_ids == 1])

        for class_id, class_name in enumerate(
                self.dataset.METAINFO['classes']
        ):
            filtered_idxs = self.gt_labels == class_id
            mask_merged = filtered_idxs & (sensor_ids == 2)
            mask_os0 = filtered_idxs & (sensor_ids == 0)
            mask_os1 = filtered_idxs & (sensor_ids == 1)

            total_frames_merged = len(
                np.unique(self.gt_frames[sensor_ids == 2]))
            total_frames_os0 = len(
                np.unique(self.gt_frames[sensor_ids == 0]))
            total_frames_os1 = len(
                np.unique(self.gt_frames[sensor_ids == 1]))

            filtered_frames_merged = self.gt_frames[mask_merged]
            filtered_points_merged = self.gt_num_lidar_pts[mask_merged]
            filtered_frames_os0 = self.gt_frames[mask_os0]
            filtered_frames_os1 = self.gt_frames[mask_os1]
            filtered_points_os0 = self.gt_num_lidar_pts[mask_os0]
            filtered_points_os1 = self.gt_num_lidar_pts[mask_os1]

            num_total_merged = len(filtered_frames_merged)
            num_frames_merged = len(np.unique(filtered_frames_merged))
            freq_merged = (
                num_total_merged /
                total_samples_merged
                if total_samples_merged
                else 0.0
            )
            num_total_os0 = len(filtered_frames_os0)
            num_frames_os0 = len(np.unique(filtered_frames_os0))
            freq_os0 = (
                num_total_os0 /
                total_samples_os0
                if total_samples_os0
                else 0.0
            )
            num_total_os1 = len(filtered_frames_os1)
            num_frames_os1 = len(np.unique(filtered_frames_os1))
            freq_os1 = (
                num_total_os1 /
                total_samples_os1
                if total_samples_os1
                else 0.0
            )
            data[class_name] = {
                "num_total_merged": num_total_merged,
                "num_total_os0": num_total_os0,
                "num_total_os1": num_total_os1,
                "num_frames_merged": num_frames_merged,
                "num_frames_os0": num_frames_os0,
                "num_frames_os1": num_frames_os1,
                "freq_merged": freq_merged,
                "freq_os0": freq_os0,
                "freq_os1": freq_os1,
                "min_points_merged": (
                    filtered_points_merged.min()
                    if filtered_points_merged.size else None
                ),
                "mean_points_merged": (
                    filtered_points_merged.mean()
                    if filtered_points_merged.size else None
                ),
                "max_points_merged": (
                    filtered_points_merged.max()
                    if filtered_points_merged.size else None
                ),
                "count_points_merged": len(filtered_points_merged),
                "min_points_os0": (
                    filtered_points_os0.min()
                    if filtered_points_os0.size else None
                ),
                "mean_points_os0": (
                    filtered_points_os0.mean()
                    if filtered_points_os0.size else None
                ),
                "max_points_os0": (
                    filtered_points_os0.max()
                    if filtered_points_os0.size else None
                ),
                "count_points_os0": len(filtered_points_os0),
                "min_points_os1": (
                    filtered_points_os1.min()
                    if filtered_points_os1.size else None
                ),
                "mean_points_os1": (
                    filtered_points_os1.mean()
                    if filtered_points_os1.size else None
                ),
                "max_points_os1": (
                    filtered_points_os1.max()
                    if filtered_points_os1.size else None
                ),
                "count_points_os1": len(filtered_points_os1),
            }

        sample_table_merged = []
        sample_table_os0 = []
        sample_table_os1 = []
        for cls, val in data.items():
            sample_table_merged.append([
                cls,
                val['num_total_merged'],
                (
                    f"{val['freq_merged']*100:.1f}%"
                    if val['freq_merged'] else "None"
                ),
                val['num_frames_merged'],
                f"{val['num_frames_merged']/total_frames_merged*100:.1f}%",
            ])
            sample_table_os0.append([
                cls,
                val['num_total_os0'],
                f"{val['freq_os0']*100:.1f}%" if val['freq_os0'] else "None",
                val['num_frames_os0'],
                f"{val['num_frames_os0']/total_frames_os0*100:.1f}%",
            ])
            sample_table_os1.append([
                cls,
                val['num_total_os1'],
                f"{val['freq_os1']*100:.1f}%" if val['freq_os1'] else "None",
                val['num_frames_os1'],
                f"{val['num_frames_os1']/total_frames_os1*100:.1f}%",
            ])
        print("Sample Distribution (merged):")
        print(
            f"Total samples: {total_samples_merged}, "
            f"Total frames: {total_frames_merged}"
        )
        print(tabulate(
            sample_table_merged,
            headers=[
              "Class", "Samples", "Freq.", "Frames", "Frame Freq.",
              ],
            tablefmt="github",
        ))
        print("Sample Distribution (Sensor os0):")
        print(
            f"Total samples: {total_samples_os0}, "
            f"Total frames: {total_frames_os0}"
        )
        print(tabulate(
            sample_table_os0,
            headers=[
              "Class", "Samples", "Freq.", "Frames", "Frame Freq.",
              ],
            tablefmt="github",
        ))
        print("Sample Distribution (os1):")
        print(
            f"Total samples: {total_samples_os1} "
            f"Total frames: {total_frames_os1}"
        )
        print(tabulate(
            sample_table_os1,
            headers=[
              "Class", "Samples", "Freq.", "Frames", "Frame Freq.",
              ],
            tablefmt="github",
        ))


if __name__ == "__main__":
    parser = DatasetAnalyerArgumentParser(
        "LUMPI",
        "data/exp/labels",
        "data/exp/points",
        "data/exp/",
    )
    args = parser.parse_args()

    analyzer = ExpAnalyzer(
        args.labels,
        args.points,
        args.output,
        anchor_mode=args.anchor_mode,
        anchor_count=args.anchor_count,
        modes=args.mode,
    )
    analyzer.analyze()
