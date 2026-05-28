import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mmengine.registry import init_default_scope
from mmengine.dataset import DefaultSampler
from mmengine.dataset import default_collate
from mmengine.structures import InstanceData
from mmdet3d.structures import LiDARInstance3DBoxes

from mmdet3d.evaluation.metrics import OSDaR23Metric
from mmdet3d.datasets import OSDaR23Dataset

data_root = 'tests/data/osdar23'


def _init_evaluate_input(_pred_gen_fn, score: float = 0.9) -> tuple:
    """
        Initialize test data by loading groundtruth and create sample
        predictions.

        Args:
            _preg_gen_fn:  function point to create predcitions
            score (float): default score

        Return:
            tuple of ground truths and predictions
    """
    init_default_scope('mmdet3d')
    val_pipeline = [
        dict(type='LoadPointsFromFile',
             coord_type='LIDAR',
             load_dim=4,
             use_dim=4,
             ),
        dict(
            type='Pack3DDetInputs',
            keys=['points'],
            meta_keys=[
                'box_mode_3d', 'box_type_3d', 'sample_id',
                'num_pts_feats', 'sample_idx', 'lidar_path',
            ],
        ),
    ]
    dataset = OSDaR23Dataset(
        data_root,
        ann_file="osdar23_infos_train.pkl",
        pipeline=val_pipeline,
        data_prefix=dict(pts='points'),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='lidar',
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=DefaultSampler(dataset, shuffle=False),
        collate_fn=default_collate,
        num_workers=1,
        batch_size=1,
    )

    predictions = []
    gt_batches = []
    for data_batch in dataloader:
        gt_batches.append(dict(data_batch))
        predictions += _pred_gen_fn(data_batch, score=score)

    return gt_batches, [predictions]


def _create_perfect_pred(data_batch: dict, score: float = 0.9) -> list:
    """
        Creates sample prefect prediction by copying groundtruth.

        Args:
            data_batch (dict):  batch of groundtruth
            score (float):      score to give predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        pred_instances.bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone()
        pred_instances.labels_3d = torch.tensor(
            eval_ann_info['gt_labels_3d'].copy())
        pred_instances.scores_3d = torch.full(
            eval_ann_info['gt_labels_3d'].shape, score, dtype=float)
        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def _create_perfect_pred_low_score(
        data_batch: dict,
        num_low_score: float = 3,
        score_low: float = 0.2,
        score: float = 0.9,
) -> list:
    """
        Creates sample prefect prediction by copying groundtruth
        but with low score.

        Args:
            data_batch (dict):      batch of groundtruth
            num_low_score (float):  how many preds with low score
            score_low (float):      score to give low predictions
            score (float):          score to give normal predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        pred_instances.bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone()
        pred_instances.labels_3d = torch.tensor(
            eval_ann_info['gt_labels_3d'].copy())

        scores = []
        for i in range(len(eval_ann_info['gt_labels_3d'])):
            if i < num_low_score:
                scores.append(score_low)
            else:
                scores.append(score)
        pred_instances.scores_3d = torch.tensor(scores, dtype=float)

        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def _create_partial_overlap_predictions(
        data_batch: dict,
        shift: float = 0.1,
        yaw_perturb: float = 0.1,
        score: float = 0.9
) -> list:
    """
        Creates sample predictions with partial overlap to
        groundtruth

        Args:
            data_batch (dict):      batch of groundtruth
            shift (float):          shift of center
            yaw_perturb (float):    shift of yaw
            score (float):          score to give predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        pred_instances.bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone()
        pred_instances.bboxes_3d.tensor[:, 0] += shift
        pred_instances.bboxes_3d.tensor[:, 1] += shift
        pred_instances.bboxes_3d.tensor[:, 2] += shift
        pred_instances.bboxes_3d.tensor[:, 6] += yaw_perturb

        pred_instances.labels_3d = torch.tensor(
            eval_ann_info['gt_labels_3d'].copy())
        pred_instances.scores_3d = torch.full(
            eval_ann_info['gt_labels_3d'].shape, score, dtype=float)
        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def _create_partial_class_mismatch(
        data_batch: dict,
        num_mismatch: int = 3,
        score: float = 0.9,
) -> list:
    """
        Creates sample predictions with some class mismatch by
        increasing class id with overflow.

        Args:
            data_batch (dict):      batch of groundtruth
            num_mismatch (int):     how many mismatchs
            score (float):          score to give predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        pred_instances.bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone()

        new_labels = []
        for i, class_id in enumerate(eval_ann_info['gt_labels_3d']):
            if i < num_mismatch:
                # overflow to first class if max clas_id is exceeded
                new_labels.append(
                    (class_id + 1) % len(OSDaR23Dataset.METAINFO['classes'])
                )
            else:
                new_labels.append(class_id)

        pred_instances.labels_3d = torch.tensor(new_labels)
        pred_instances.scores_3d = torch.full(
            eval_ann_info['gt_labels_3d'].shape, score, dtype=float)
        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def _create_missing_preds(
        data_batch: dict,
        num_missing: int = 3,
        score: float = 0.9,
) -> list:
    """
        Creates sample predictions with missing predictions.

        Args:
            data_batch (dict):      batch of groundtruth
            num_missing (int):      how many missing preds
            score (float):          score to give predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        pred_instances.bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone()[
            num_missing:, :]
        pred_instances.labels_3d = torch.Tensor(eval_ann_info['gt_labels_3d'])[
            num_missing:]
        pred_instances.scores_3d = torch.full(
            eval_ann_info['gt_labels_3d'].shape,
            score,
            dtype=float
        )[num_missing:]

        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def _create_duplicate_preds(
        data_batch: dict,
        num_duplicate: int = 3,
        score: float = 0.9,
) -> list:
    """
        Creates sample predictions with some identical predictions.

        Args:
            data_batch (dict):      batch of groundtruth
            num_duplicate (int):    how many duplicated preds
            score (float):          score to give predictions

        Returns:
            list of predictions
    """
    predictions = []
    for data_sample in data_batch['data_samples']:
        eval_ann_info = data_sample.eval_ann_info

        pred_instances = InstanceData()
        bboxes_3d = eval_ann_info['gt_bboxes_3d'].clone().tensor
        pred_instances.bboxes_3d = LiDARInstance3DBoxes(
            torch.cat([
                bboxes_3d,
                bboxes_3d[:num_duplicate, :],
            ], dim=0)
        )
        pred_instances.labels_3d = torch.Tensor(
            np.concatenate([
                eval_ann_info['gt_labels_3d'],
                eval_ann_info['gt_labels_3d'][:num_duplicate],
            ])
        )
        pred_instances.scores_3d = torch.full(
            pred_instances.labels_3d.shape,
            score,
            dtype=float
        )

        predictions.append(
            {'pred_instances_3d': pred_instances,
             'sample_id': 0
             })

    return predictions


def test_osdar23_metric_mAP_perfect():
    """Test metric for perfect prediction"""
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_perfect_pred))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    mAP = 0.9090909059848484
    assert np.isclose(ap_dict['mAP'], mAP)
    assert np.isclose(ap_dict['AP30'], mAP)
    assert np.isclose(ap_dict['AP40'], mAP)
    assert np.isclose(ap_dict['AP50'], mAP)
    assert np.isclose(ap_dict['AP60'], mAP)

    assert np.isclose(ap_dict['pred_gt_ratio'], 1.0)
    assert np.isclose(ap_dict['score_mean'], 0.9)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 0)


def test_osdar23_metric_mAP_partial_overlap():
    """Test metric for partial overlapping predictions"""
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_partial_overlap_predictions))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    assert np.isclose(ap_dict['mAP'], 0.7386363611458333)
    assert np.isclose(ap_dict['AP30'], 0.9090909059848484)
    assert np.isclose(ap_dict['AP40'], 0.9090909059848484)
    assert np.isclose(ap_dict['AP50'], 0.8522727242613637)
    assert np.isclose(ap_dict['AP60'], 0.2840909083522727)

    assert np.isclose(ap_dict['pred_gt_ratio'], 1.0)
    assert np.isclose(ap_dict['score_mean'], 0.9)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 0)


def test_osdar23_metric_mAP_class_mismatch():
    """Test metric for predictions with wrong classes"""
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_partial_class_mismatch))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    assert np.isclose(ap_dict['mAP'], 0.5643939372487374)
    assert np.isclose(ap_dict['AP30'], 0.5643939372487374)
    assert np.isclose(ap_dict['AP40'], 0.5643939372487374)
    assert np.isclose(ap_dict['AP50'], 0.5643939372487374)
    assert np.isclose(ap_dict['AP60'], 0.5643939372487374)

    assert np.isclose(ap_dict['pred_gt_ratio'], 1.0)
    assert np.isclose(ap_dict['score_mean'], 0.9)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 0)


def test_osdar23_metric_mAP_low_score():
    """Test metric for predictions with low score"""
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_perfect_pred_low_score))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    assert np.isclose(ap_dict['mAP'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP30'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP40'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP50'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP60'], 0.6590909059545456)

    assert np.isclose(ap_dict['pred_gt_ratio'], 0.8)
    assert np.isclose(ap_dict['score_mean'], 0.7599999999999999)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 3)


def test_osdar23_metric_missing_preds():
    """Test metric for partial predictions """
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_missing_preds))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    assert np.isclose(ap_dict['mAP'],  0.6590909059545456)
    assert np.isclose(ap_dict['AP30'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP40'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP50'], 0.6590909059545456)
    assert np.isclose(ap_dict['AP60'], 0.6590909059545456)

    assert np.isclose(ap_dict['pred_gt_ratio'], 0.8)
    assert np.isclose(ap_dict['score_mean'], 0.9)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 0)


def test_osdar23_metric_duplicate_preds():
    """Test metric for predictions with duplicates"""
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    metric = OSDaR23Metric()
    metric.dataset_meta = OSDaR23Dataset.METAINFO

    data = zip(*_init_evaluate_input(_create_duplicate_preds))
    for gt_batch, prediction in data:
        metric.process(gt_batch, prediction)
    ap_dict = metric.compute_metrics(metric.results)
    print(ap_dict)

    assert np.isclose(ap_dict['mAP'],  0.8831168797124304)
    assert np.isclose(ap_dict['AP30'], 0.8831168797124304)
    assert np.isclose(ap_dict['AP40'], 0.8831168797124304)
    assert np.isclose(ap_dict['AP50'], 0.8831168797124304)
    assert np.isclose(ap_dict['AP60'], 0.8831168797124304)

    assert np.isclose(ap_dict['pred_gt_ratio'], 1, 2)
    assert np.isclose(ap_dict['score_mean'], 0.8999999999999999)

    assert np.isclose(ap_dict['failed_samples'], 0)
    assert np.isclose(ap_dict['filtered_preds'], 0)
