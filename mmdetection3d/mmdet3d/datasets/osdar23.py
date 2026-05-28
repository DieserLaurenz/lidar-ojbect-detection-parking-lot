import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class OSDaR23Dataset(Det3DDataset):
    """
    This class serves as the API for experiments on the OSDaR23 dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:
            - 'LiDAR': Box in LiDAR coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    # full dataset
    METAINFO = {
        "classes": [
            "person",           # 0
            "crowd",            # 1
            "train",            # 2
            "wagons",           # 3
            "bicycle",          # 4
            "road_vehicle",     # 5
            "animal",           # 6
            "switch",           # 7
            "catenary_pole",    # 8
            "signal_pole",      # 9
            "signal",           # 10
            "signal_bridge",    # 11
            "buffer_stop",      # 12
        ],
    }
    sample_ids = []

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: list = [],
                 modality: dict = dict(use_lidar=True),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 metainfo: dict = [],
                 **kwargs
                 ) -> None:
        # override default meta info
        if metainfo:
            self.METAINFO = metainfo

        super().__init__(
            data_root=data_root,
            metainfo=metainfo,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            print("debug custom ann_info ?")
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros(
                (0, len(self.METAINFO["classes"])), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'])
        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        sample_id = info['sample_id']
        self.sample_ids.append(sample_id)
        info = super().parse_data_info(info)

        if 'axis_align_matrix' in info:
            info['axis_align_matrix'] = np.array(info['axis_align_matrix'])

        if not self.test_mode:
            info['ann_info']['sample_id'] = sample_id
        else:
            info['eval_ann_info']['sample_id'] = sample_id

        return info

    def get_by_sample_id(self, sample_id):
        if sample_id in self.sample_ids:
            idx = self.sample_ids.index(sample_id)
            return self[idx]

        return None
