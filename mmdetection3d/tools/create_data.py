# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from mmengine import print_log
from mmengine.utils import mkdir_or_exist

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters import semantickitti_converter
from tools.dataset_converters import osdar23
from tools.dataset_converters import lumpi
from tools.dataset_converters import exp
from tools.dataset_converters import exp_to_kitti, exp_to_lumpi, exp_to_osdar23
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def experiement_prep(
        root_path: str,
        out_dir: str,
        max_workers: int,
        target_dataset: str = 'lumpi',
        info_prefix="exp",
        only_annotation: bool = False,
) -> None:
    """Prepare data related to own Experiment just for testing.

    Consists of:
        converting pcd data into 4-D bin data: out_dir/points_{target_dataset}
        converting openlabel data into json: out_dir/labels_{target_dataset}
        create annotation files for training, validation and testing

    Args:
        root_path (str):        Path of dataset root.
        out_dir (str):          Output directory of PCD and GT
        max_workers (int):      Max. threads/process for
                                parallizing work
        info_prefix (str):      Prefix for annotation file names
        only_annotation (bool): skip converting format and just
                                dotransformation and annotation
    """
    assert target_dataset in ["lumpi", "kitti", "osdar"], \
        f"Dataset {target_dataset} not supported"

    info_os0_path = osp.join(
        out_dir, f'{info_prefix}_{target_dataset}_infos_os0.pkl')
    info_os1_path = osp.join(
        out_dir, f'{info_prefix}_{target_dataset}_infos_os1.pkl')
    info_merged_path = osp.join(
        out_dir, f'{info_prefix}_{target_dataset}_infos_merged.pkl')

    mkdir_or_exist(osp.join(out_dir, "labels"))
    mkdir_or_exist(osp.join(out_dir, "points"))

    if not only_annotation:
        exp.convert_pcd(root_path, out_dir, max_workers=max_workers)
        exp.convert_labels(root_path, out_dir, max_workers=max_workers)

    if target_dataset == "kitti":
        print("Transforming to KITTI world space")
        mkdir_or_exist(osp.join(out_dir, "points_kitti"))
        mkdir_or_exist(osp.join(out_dir, "labels_kitti"))
        print("Converting pointclouds")
        exp_to_kitti.convert_pointclouds(
            osp.join(out_dir, "points"),
            osp.join(out_dir, "points_kitti"),
            remove_ceiling=False,
            max_workers=max_workers
        )
        print("Converting labels")
        exp_to_kitti.convert_labels(
            osp.join(out_dir, "labels"),
            osp.join(out_dir, "labels_kitti"),
        )
    elif target_dataset == "lumpi":
        print("Transforming to LUMPI world space")
        mkdir_or_exist(osp.join(out_dir, "points_lumpi"))
        mkdir_or_exist(osp.join(out_dir, "labels_lumpi"))
        print("Converting pointclouds")
        exp_to_lumpi.convert_pointclouds(
            osp.join(out_dir, "points"),
            osp.join(out_dir, "points_lumpi"),
            remove_ceiling=True,
            max_workers=max_workers
        )
        print("Converting labels")
        exp_to_lumpi.convert_labels(
            osp.join(out_dir, "labels"),
            osp.join(out_dir, "labels_lumpi"),
        )
    elif target_dataset == "osdar":
        print("Transforming to OSDaR23 world space")
        mkdir_or_exist(osp.join(out_dir, "points_osdar"))
        mkdir_or_exist(osp.join(out_dir, "labels_osdar"))
        print("Converting pointclouds")
        exp_to_osdar23.convert_pointclouds(
            osp.join(out_dir, "points"),
            osp.join(out_dir, "points_osdar"),
            remove_ceiling=True,
            max_workers=max_workers
        )
        print("Converting labels")
        exp_to_osdar23.convert_labels(
            osp.join(out_dir, "labels"),
            osp.join(out_dir, "labels_osdar"),
        )

    exp.create_test_split(
        out_dir,
        info_os0_path,
        info_os1_path,
        info_merged_path,
        target_dataset=target_dataset,
    )

    update_pkl_infos('exp', out_dir=out_dir, pkl_path=info_os0_path)
    update_pkl_infos('exp', out_dir=out_dir, pkl_path=info_os1_path)
    update_pkl_infos('exp', out_dir=out_dir, pkl_path=info_merged_path)


def lumpi_prep(
        root_path: str,
        out_dir: str,
        max_workers: int = 8,
        info_prefix: str = "lumpi",
        only_gt_database: str = False,
        only_annotation: str = False
) -> None:
    """Prepare data related to LUMPI dataset.

    Consists of:
        - converting pcd data into 4-D bin data (out_dir/points)
        - converting openlabel data into json (out_dir/labels)
        - split dataset into training, validation and testing
        - create annotation files for training, validation and testing

    Args:
        root_path (str): Path of dataset root.
        out_dir (str): Output directory of PCD and GT
    """
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')

    mkdir_or_exist(osp.join(out_dir, "labels"))
    mkdir_or_exist(osp.join(out_dir, "points"))

    if not only_annotation and not only_gt_database:
        lumpi.convert_pcd(root_path, out_dir, max_workers=max_workers)
        lumpi.convert_labels(root_path, out_dir, max_workers=max_workers)

    if not only_gt_database:
        lumpi.create_train_val_test_split(
            out_dir, info_train_path, info_val_path, info_test_path)

        update_pkl_infos('lumpi', out_dir=out_dir, pkl_path=info_train_path)
        update_pkl_infos('lumpi', out_dir=out_dir, pkl_path=info_val_path)
        update_pkl_infos('lumpi', out_dir=out_dir, pkl_path=info_test_path)

    if not only_annotation:
        create_groundtruth_database(
            'LUMPIDataset',
            out_dir,
            info_prefix=info_prefix,
            info_path=osp.basename(info_train_path),
            relative_path=True,
            lidar_only=True,
        )


def osdar23_prep(
        root_path: str,
        out_dir: str,
        max_workers: int = 8,
        info_prefix: str = "osdar23",
        only_gt_database: bool = False,
) -> None:
    """Prepare data related to OSDaR23 dataset.

    Consists of:
        - converting pcd data into 4-D bin data (out_dir/points)
        - converting openlabel data into json (out_dir/labels)
        - split dataset into training, validation and testing
        - create annotation files for training, validation and testing

    Args:
        root_path (str): Path of dataset root.
        out_dir (str): Output directory of PCD and GT
    """

    if not only_gt_database:
        info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')

        mkdir_or_exist(osp.join(out_dir, "labels"))
        mkdir_or_exist(osp.join(out_dir, "points"))

        osdar23.convert_pcd(root_path, out_dir, max_workers=max_workers)
        osdar23.convert_labels(root_path, out_dir, max_workers=max_workers)

        osdar23.create_train_val_test_split(
            out_dir,
            info_train_path,
            info_val_path,
            info_test_path,
            mode="measuremen",
        )

        update_pkl_infos('osdar23', out_dir=out_dir, pkl_path=info_train_path)
        update_pkl_infos('osdar23', out_dir=out_dir, pkl_path=info_val_path)
        update_pkl_infos('osdar23', out_dir=out_dir, pkl_path=info_test_path)

    create_groundtruth_database(
        'OSDaR23Dataset',
        out_dir,
        info_prefix=info_prefix,
        info_path=osp.basename(info_train_path),
        relative_path=True,
        lidar_only=True,
    )


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=10,
                    only_gt_database=False,
                    save_senor_data=False,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    from tools.dataset_converters import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 'validation', 'testing',
            'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing', 'testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from tools.dataset_converters.waymo_converter import \
            create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)

    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()

    print_log('Successfully preparing Waymo Open Dataset')


def semantickitti_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(
        info_prefix, out_dir)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--target-dataset',
    choices=['kitti', 'lumpi', 'osdar'],
    default='lumpi',
    help='''Transformation into which dataset world space.
        Only used when dataset is exp!''')
parser.add_argument(
    '--only-annotation',
    action='store_true',
    help='''When using exp: Use KITTI or LUMPI class labels.
        Only used when dataset is exp!''')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'kitti':
        if args.only_gt_database:
            create_groundtruth_database(
                'KittiDataset',
                args.root_path,
                args.extra_tag,
                f'{args.extra_tag}_infos_train.pkl',
                relative_path=False,
                mask_anno_path='instances_train.json',
                with_mask=(args.version == 'mask'))
        else:
            kitti_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                out_dir=args.out_dir,
                with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}-trainval'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
            test_version = f'{args.version}-test'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            only_gt_database=args.only_gt_database,
            save_senor_data=not args.skip_saving_sensor_data,
            skip_cam_instances_infos=args.skip_cam_instances_infos)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'semantickitti':
        semantickitti_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    elif args.dataset == 'osdar23':
        osdar23_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            max_workers=args.workers,
            only_gt_database=args.only_gt_database,
        )
    elif args.dataset == 'lumpi':
        lumpi_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            max_workers=args.workers,
            only_gt_database=args.only_gt_database,
            only_annotation=args.only_annotation,
        )
    elif args.dataset == 'exp':
        experiement_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            max_workers=args.workers,
            target_dataset=args.target_dataset,
            only_annotation=args.only_annotation,
        )
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
