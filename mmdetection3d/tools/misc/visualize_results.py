import argparse

import mmengine
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS, VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--mode',
        help='Choose between test or validation dataloader',
        default='test',
        choices=['test', 'val']
    )
    parser.add_argument(
        '--show-dir',
        help='directory where visualize results will be saved'
    )
    parser.add_argument(
        '--wait',
        help='time before next frame. Toogle Pause with Space',
        default=-1,
        type=float
    )
    parser.add_argument(
        '--score-thr',
        help='filter threshold for predictions to be displayed',
        default=0.3,
        type=float
    )
    args = parser.parse_args()

    assert args.mode in ("test", "val"), "--mode has to be test or val"

    return args


def main():
    args = parse_args()

    if (
            args.result is not None and
            not args.result.endswith(('.pkl', '.pickle'))
    ):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    if "train" in args.result or args.mode == "val":
        print("Loading Validation Dataset")
        dataloader = cfg.val_dataloader.dataset
        dataloader.test_mode = True
    else:
        print("Loading Test Dataset")
        dataloader = cfg.test_dataloader.dataset
        dataloader.test_mode = True

    # build the dataset
    dataset = DATASETS.build(dataloader)
    dataset.full_init()
    datalist = dataset.load_data_list()
    results = mmengine.load(args.result)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    for i, result in enumerate(results):
        assert 'sample_id' in result.metainfo
        sample_id = int(result.metainfo['sample_id'])
        print(
            f"[{i}/{len(results)}] path: "
            f"{dataloader.data_root}/{dataloader.data_prefix['pts']}/"
            f"{sample_id}"
        )
        data_sample = None
        for data in datalist:
            if 'sample_id' not in data:
                data_sample = dataset[sample_id]
                break
            if int(data['sample_id']) == int(sample_id):
                data_sample = dataset[datalist.index(data)]
                break

        if not data_sample:
            print(f"WARN: sample not found {sample_id}")
            continue

        data_sample['data_samples'].pred_instances_3d = result

        visualizer.add_datasample(
            '3d visualizer',
            data_sample['inputs'],
            data_sample=data_sample['data_samples'],
            show=True,
            wait_time=args.wait,
            pred_score_thr=args.score_thr,
            vis_task='lidar_det',
        )


if __name__ == '__main__':
    main()
