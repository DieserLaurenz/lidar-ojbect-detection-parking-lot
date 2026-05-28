from mmdet3d.datasets.kitti_dataset import KittiDataset
from tools.dataset_converters.kitti_data_utils import get_kitti_image_info
from tools.dataset_converters.kitti_converter import (
    _calculate_num_points_in_gt
)
from tools.analysis_tools.dataset_analyzer import (
    DatasetAnalyzer, AnalysisMode, AnchorMode, DatasetAnalyerArgumentParser
)

import numpy as np
from joblib import Memory
memory = Memory(location=".joblib_cache", verbose=0)


class KITTIAnalyzer(DatasetAnalyzer):
    """
    Analyze quality of KITTI dataset and calculate needed values for training.

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
        super().__init__(
            KittiDataset,
            labels_path,
            points_path,
            output_dir,
            anchor_mode,
            anchor_count,
            modes,
            percentiles
        )

    def load_labels(self) -> None:
        """
            Overrides inherrit function to adjust to dataset structure.

            Loads labels from given path for evaluation.
            Use joblib cache for multiple runs.
        """
        print("Loading label files")

        @memory.cache
        def _load(labels_path, num_workers=32):
            data = get_kitti_image_info(
                labels_path, num_worker=num_workers, calib=True, velodyne=True)
            _calculate_num_points_in_gt(self.points_path, data, True)

            return data

        data = _load(self.labels_path)
        for element in data:
            annos = element['annos']
            num = len(annos['name'])
            for i in range(num):
                if annos['name'][i] == "DontCare":
                    continue
                bbox3d = (
                    list(annos['location'][i]) +
                    list(annos['dimensions'][i]) +
                    [annos['rotation_y'][i]]
                )
                class_id = list(
                    self.dataset.METAINFO['classes']
                ).index(
                    annos['name'][i]
                )
                self.gt_bboxes.append(bbox3d)
                self.gt_labels.append(class_id)
                self.gt_frames.append(element['image']['image_idx'])
                self.gt_num_lidar_pts.append(annos["num_points_in_gt"][i])
                self.gt_truncated.append(0)
                self.gt_occluded.append(0)

        self.gt_bboxes = np.array(self.gt_bboxes)
        self.gt_labels = np.array(self.gt_labels)
        self.gt_frames = np.array(self.gt_frames)
        self.gt_num_lidar_pts = np.array(self.gt_num_lidar_pts)
        self.gt_truncated = np.array(self.gt_truncated)
        self.gt_occluded = np.array(self.gt_occluded)

        assert len(self.gt_frames) == \
            len(self.gt_bboxes) == \
            len(self.gt_labels) == \
            len(self.gt_num_lidar_pts) == \
            len(self.gt_truncated) == \
            len(self.gt_occluded)
        print(
            f"Loaded {len(np.unique(self.gt_frames))} "
            f"frames with {len(self.gt_labels)} instances in total."
        )


if __name__ == "__main__":
    parser = DatasetAnalyerArgumentParser(
        "KITI",
        "data/kitti/",
        "data/kitti/",
        "data/kitti/",
    )
    args = parser.parse_args()

    analyzer = KITTIAnalyzer(
        args.labels,
        args.points,
        args.output,
        anchor_mode=args.anchor_mode,
        anchor_count=args.anchor_count,
        modes=args.mode,
    )
    analyzer.analyze()
