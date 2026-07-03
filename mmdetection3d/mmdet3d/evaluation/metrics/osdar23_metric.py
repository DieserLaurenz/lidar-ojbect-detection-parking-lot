import traceback
import numpy as np
import torch
import datetime
import time
import json
import random
from tqdm import tqdm
from os import environ as os_environ
from os import path as osp
from prettytable import PrettyTable
from typing import Dict, List, Optional, Sequence, Union
import torch.multiprocessing as mp

from mmcv.ops import diff_iou_rotated_3d
from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric
from mmengine.structures import InstanceData
import mmengine
from mmdet3d.registry import METRICS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


@METRICS.register_module()
class OSDaR23Metric(BaseMetric):
    """OSDaR23 evaluation metric."""

    def __init__(self,
                 metric: Union[str, List[str]] = 'mAP',
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 dump_only: Optional[bool] = False,
                 result_prefix: Optional[str] = None,
                 collect_device: Optional[str] = 'gpu',
                 score_threshold: Optional[float] = 0.2,
                 iou_thresholds: Optional[List[float]] = [0.3, 0.4, 0.5, 0.6],
                 max_workers: Optional[int] = 1,
                 backend_args: Optional[dict] = None,
                 use_kitti: Optional[bool] = False,
                 ) -> None:

        if collect_device != 'gpu' and not dump_only:
            raise AssertionError(
                "Used metric function mmcv.ops.diff_iou_rotated_3d "
                "is not implemented on cpu. Set dump_only=True to skip "
                "mAP computation and only export predictions."
            )

        self.default_prefix = 'osdar23'
        super(OSDaR23Metric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.dump_only = dump_only
        self.pklfile_prefix = pklfile_prefix
        self.result_prefix = result_prefix
        self.score_threshold = score_threshold
        self.iou_thresholds = iou_thresholds
        self.max_workers = max_workers
        self.backend_args = backend_args
        self.use_kitti = use_kitti

        self.results = []
        self.data_infos = []

        allowed_metrics = ['mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("Metric should be one of "
                               f"{allowed_metrics}, but got {metric}."
                               )

        self.classes = []
        self.seen_classes = []

    def convert_groundtruths(self, data_info: dict) -> List[dict]:
        """Convert annotations to OSDaR23 annotations."""

        if self.dump_only:
            return {}

        bboxes_3d = []
        labels_3d = []

        if self.use_kitti:
            labels_3d = data_info['gt_labels_3d']
            bboxes_3d = data_info['gt_bboxes_3d'].tensor.numpy()
        else:
            for instance in data_info['instances']:
                # Skip DontCares
                if instance['bbox_label_3d'] == -1:
                    continue
                labels_3d.append(instance['bbox_label_3d'])
                bboxes_3d.append(instance['bbox_3d'])

        for idx, class_name in enumerate(self.classes):
            if idx in labels_3d and class_name not in self.seen_classes:
                self.seen_classes.append(class_name)

        sample_id = data_info['sample_id'] if 'sample_id' in data_info else -1
        return {
            'bboxes_3d': np.array(bboxes_3d),
            'labels_3d': np.array(labels_3d),
            "num_objs": len(bboxes_3d),
            'sample_id': sample_id,
        }

    def convert_predictions(self, data_sample):
        pred3d = data_sample['pred_instances_3d']
        sample_id = (
            data_sample['sample_id']
            if 'sample_id' in data_sample
            else data_sample['sample_idx']
        )

        good_pred3d_mask = pred3d['scores_3d'] > self.score_threshold
        mean_score = (
            torch.mean(pred3d['scores_3d']).detach().cpu().numpy()
            if pred3d['scores_3d'].numel() != 0
            else torch.tensor(0.0)
        )

        # count only Trues
        num_objs = good_pred3d_mask.sum().detach().cpu().numpy()
        num_filtered = len(good_pred3d_mask) - num_objs

        bboxes_3d = pred3d['bboxes_3d'][good_pred3d_mask].detach(
        ).cpu().numpy()
        labels_3d = pred3d['labels_3d'][good_pred3d_mask].detach(
        ).cpu().numpy()
        scores_3d = pred3d['scores_3d'][good_pred3d_mask].detach(
        ).cpu().numpy()

        # sort preds descending
        sort_idx = np.argsort(-scores_3d)
        bboxes_3d = bboxes_3d[sort_idx]
        labels_3d = labels_3d[sort_idx]
        scores_3d = scores_3d[sort_idx]

        if num_filtered and num_filtered != len(pred3d['scores_3d']):
            assert (
                len(pred3d['scores_3d']) !=
                len(pred3d['scores_3d'][good_pred3d_mask])
            ), (
                f"fitlering Predictions failed!\n"
                f"Before: {pred3d['scores_3d']}\n"
                f"after: {pred3d['scores_3d'][good_pred3d_mask]}\n"
                f"mask: {good_pred3d_mask}"
            )

        return {
            "sample_id": sample_id,
            "mean_score": mean_score,
            "num_filtered": num_filtered,
            "num_objs": num_objs,
            "bboxes_3d": bboxes_3d,
            "labels_3d": labels_3d,
            "scores_3d": scores_3d,
        }

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions."""

        self.classes = self.dataset_meta['classes']
        assert self.classes, "Dataset meta infos should be set!"

        for batch in data_batch['data_samples']:
            data_infos = self.convert_groundtruths(batch.eval_ann_info)
            if not self.dump_only:
                if data_infos['sample_id'] == -1:
                    data_infos['sample_id'] = batch.metainfo['sample_idx']
            self.data_infos.append(data_infos)

        for data_sample in data_samples:
            preds = self.convert_predictions(data_sample)
            self.results.append(preds)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results."""

        logger: MMLogger = MMLogger.get_current_instance()

        self.empty_classes = list(set(self.classes) - set(self.seen_classes))
        if self.empty_classes:
            logger.info(
                f"{self.empty_classes} are defined in dataset metainfo "
                "but not present. They will be excluded from the metric."
            )

        metric_dict = {}

        metric_dict['empty_classes'] = self.empty_classes

        self.dump_results(self.results, logger)
        if self.dump_only:
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.osdar_evaluate(
                self.results,
                self.data_infos,
                filter_by_score=True,
                logger=logger,
            )
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        self.dump_metric(metric_dict, logger=logger)

        self.results = []
        self.data_infos = []

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return metric_dict

    def dump_metric(
            self,
            metric_dict: dict,
            logger: MMLogger = MMLogger.get_current_instance()
    ) -> None:
        """
        Dumps calculated metric to json file.

        Args:
            metric_dict (dict): Dict containing the metric
            logger (MMLogger): Instance of MMLogger to announce location.
                (Default: MMLogger.get_current_instance())
        """
        if self.result_prefix is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            mmengine.mkdir_or_exist(osp.dirname(self.result_prefix))
            out = osp.join(
                osp.dirname(self.result_prefix),
                f"{osp.basename(self.result_prefix)}_{timestamp}.json",
            )

            # makes np arrays dumpable
            class NumpyArrayEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.int64):
                        return int(obj)
                    if isinstance(obj, np.float32):
                        return float(obj)
                    if isinstance(obj, dict):
                        return {
                            key: self.default(value)
                            for key, value in obj.items()
                        }
                    if isinstance(obj, list):
                        return [self.default(item) for item in obj]
                    return super().default(obj)

            with open(out, "w") as f:
                json.dump({"metric": metric_dict},
                          f, cls=NumpyArrayEncoder)
                logger.info(f"Metric stored at {out}")

    def dump_results(
            self,
            results: list,
            logger: MMLogger = MMLogger.get_current_instance()
    ) -> None:
        """
        Dumps predictions made by the network to pkl file.

        Args:
            results (list): List of converted predictions
            logger (MMLogger): Instance of MMLogger to announce location.
                (Default: MMLogger.get_current_instance())
        """
        if self.pklfile_prefix is not None:
            data = []
            for batch in results:
                pred_instances_3d = InstanceData(
                    metainfo={
                        'sample_id': (
                            batch['sample_id']
                            if 'sample_id' in batch else 0
                        ),
                    }
                )
                pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                    batch['bboxes_3d'])
                pred_instances_3d.labels_3d = batch['labels_3d']
                pred_instances_3d.scores_3d = batch['scores_3d']
                data.append(pred_instances_3d)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            mmengine.mkdir_or_exist(osp.dirname(self.pklfile_prefix))
            out = osp.join(
                osp.dirname(self.pklfile_prefix),
                f"{osp.basename(self.pklfile_prefix)}_{timestamp}.pkl",
            )

            mmengine.dump(data, out)
            logger.info(f'Result is saved to {out}')
        else:
            logger.warning(
                "Aborted storing results, because 'pklfile_prefix' was not set"
            )

    @torch.no_grad()
    def compute_mmcv_iou(
            self,
            pred_bboxes: list,
            gt_bboxes: list,
    ) -> np.ndarray:
        """
            Calculates IoU matrix for a set of predictions and
            groundtruth bboxes using the mmcv function
            diff_iou_rotated_3d.

            Note: required are torch.Tensors NOT on cpu!

            ## Arguments

            - pred_bboxes: list or numpy of size (P, 7)
            - pred_bboxes: list or numpy of size (G, 7)

            Each bbox should be defined as [x,y,z,dx,dy,dy,yaw].

            ## Result

            Returns a PxG numpy matrix with IoU values.
        """
        cuda_device = self.select_gpu_with_max_free_memory()
        device = torch.device(cuda_device)

        # (P, 7)
        pred_tensor = torch.tensor(
            pred_bboxes,
            dtype=torch.float32,
            device=device
        )
        # (G, 7)
        gt_tensor = torch.tensor(
            gt_bboxes,
            dtype=torch.float32,
            device=device
        )

        # Boxes arrive in mmdet3d LiDAR convention (z = bottom center),
        # but diff_iou_rotated_3d expects z at the gravity center.
        pred_tensor = pred_tensor.clone()
        gt_tensor = gt_tensor.clone()
        pred_tensor[:, 2] += pred_tensor[:, 5] / 2.0
        gt_tensor[:, 2] += gt_tensor[:, 5] / 2.0

        # Expand for pairwise comparison: (P, G, 7)
        pred_exp = pred_tensor.unsqueeze(1).expand(-1, gt_tensor.shape[0], -1)
        gt_exp = gt_tensor.unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        # (P, G)
        ious = diff_iou_rotated_3d(pred_exp, gt_exp)

        return ious.cpu().numpy()

    def match_gt_and_pred(
            self,
            gt_annos: dict,
            pred_annos: dict,
            iou_matrix: np.ndarray,
            threshold: float = 0.7,
    ) -> list:
        """
        Greedily matches predictions (ordered by descending score) to GT
        at the given IoU threshold.

        Args:
            gt_annos (dict):        Given grountruths to match
            pred_annos (dict):     Given predictions to match
            iou_matrix (ndarray):  Precomputed (P, G) IoU matrix
            threshold (float):     Threshold for IoU to count as match.
                (Default is 0.7)

        Returns:
            List of TP flags (1/0), aligned with the prediction order
        """
        gt_matched = [False] * gt_annos['num_objs']

        tp = []

        for pred_idx in range(pred_annos['num_objs']):
            if gt_annos['num_objs'] > 0:
                ious_for_pred = iou_matrix[pred_idx]
                max_iou = np.max(ious_for_pred)
                best_gt_idx = np.argmax(ious_for_pred)
            else:
                max_iou = 0
                best_gt_idx = -1

            if max_iou >= threshold and not gt_matched[best_gt_idx]:
                tp.append(1)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)

        return tp

    def _compute_iou_with_retry(self, pred_bboxes, gt_bboxes) -> np.ndarray:
        """IoU matrix with CUDA-OOM retry; empty matrix if either side is
        empty."""
        if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            return np.empty((len(pred_bboxes), len(gt_bboxes)))
        MAX_RETRIES = 4
        for i in range(MAX_RETRIES):
            try:
                return self.compute_mmcv_iou(pred_bboxes, gt_bboxes)
            except RuntimeError as e:
                if 'out of memory' in str(e) and i < MAX_RETRIES - 1:
                    backoff_time = random.uniform(1, 5)
                    print(
                        f"[WARN] {i}/{MAX_RETRIES} CUDA Out-of-Memory "
                        "- sleep and retry!"
                    )
                    torch.cuda.empty_cache()
                    time.sleep(backoff_time)
                else:
                    raise

    def calculate_ap11(self, scores: list, tps: list, n_gt: int) -> float:
        """
            Dataset-level AP with 11-point interpolation: predictions of
            one class, pooled over all frames, are ranked by score; the
            precision/recall curve is integrated against the total GT
            count of the split.

            Args:
                scores (list): Confidence of every prediction of the class
                tps (list):    1 if the prediction matched a GT, else 0
                n_gt (int):    Total number of GT instances of the class

            Returns:
                ap (float): Metric value
        """
        if n_gt == 0:
            return 0.0
        if len(scores) == 0:
            return 0.0
        order = np.argsort(-np.asarray(scores))
        tp = np.asarray(tps, dtype=np.float64)[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1.0 - tp)
        recalls = tp_cum / n_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = 0.0
        for level in np.linspace(0.0, 1.0, 11):
            mask = recalls >= level - 1e-9
            ap += np.max(precisions[mask]) if np.any(mask) else 0.0

        return ap / 11.0

    def custom_osdar_eval(
            self,
            gt_annos: dict,
            pred_annos: dict,
            thresholds: List[float],
    ) -> dict:
        """
        Match predictions to GT per class for one frame and return the
        raw match data (scores, TP flags, GT count) for every IoU
        threshold, to be pooled dataset-wide before the AP is computed.

        Args:
            gt_annos (dict):    Given groundtruths of one frame
            pred_annos (dict):  Predictions on that frame
            thresholds (list):  IoU thresholds to match at

        Returns:
            dict: {ap_name: {class_name: [(scores, tps, n_gt)]}}
        """

        ap_dict = {}

        # INFO: Exclude empty classes for metric faireness!
        # Use self.seen_classes instead of self.classes
        for class_name in self.seen_classes:
            class_id = self.classes.index(class_name)

            filtered_gt_idx = gt_annos['labels_3d'] == class_id
            filtered_gt_annos = gt_annos.copy()
            for key in ("bboxes_3d", "labels_3d"):
                filtered_gt_annos[key] = gt_annos[key][filtered_gt_idx]
            filtered_gt_annos['num_objs'] = len(filtered_gt_annos['bboxes_3d'])

            filtered_pred_idx = pred_annos['labels_3d'] == class_id
            filtered_pred_annos = pred_annos.copy()
            for key in ("bboxes_3d", "labels_3d", "scores_3d"):
                filtered_pred_annos[key] = pred_annos[key][filtered_pred_idx]
            filtered_pred_annos['num_objs'] = len(
                filtered_pred_annos['bboxes_3d'])

            # one IoU matrix per class and frame, shared by all thresholds
            iou_matrix = self._compute_iou_with_retry(
                filtered_pred_annos['bboxes_3d'],
                filtered_gt_annos['bboxes_3d'],
            )

            scores = list(filtered_pred_annos['scores_3d'])
            for threshold in thresholds:
                ap_name = f"AP{int(threshold * 100):02d}"
                tps = self.match_gt_and_pred(
                    filtered_gt_annos,
                    filtered_pred_annos,
                    iou_matrix,
                    threshold=threshold,
                )
                ap_dict.setdefault(ap_name, {})[class_name] = [
                    (scores, tps, filtered_gt_annos['num_objs'])
                ]

        return ap_dict

    def print_ap_dict(self, results: dict) -> None:
        """Print the computed per-class AP values as a table"""
        ptable = PrettyTable()
        ap_names = [f"AP{int(ap * 100):02d}" for ap in self.iou_thresholds]
        ptable.field_names = ['Category'] + ap_names
        for category in self.seen_classes:
            row = [category]
            for ap in ap_names:
                value = results.get(f"{ap}_{category}")
                row.append(round(value, 3) if value is not None else -1)
            ptable.add_row(row)

        print(ptable)

    def process_batch_metric(self,
                             gt_annos_batch: dict,
                             pred_annos_batch: dict,
                             filter_by_score: bool,
                             iou_thresholds: list,
                             ) -> dict:
        """
        Wrapper to process single batch

        Args:
            gt_annos_batch (dict):   current batch of groundtruth
            pred_annos_batch (dict): predicition on current batch
            filter_by_score (bool):  where to ignore predictions below
                threshold
            iou_thresholds (float):  list of thresholds to test

        Returns:
            dict with metrics
        """
        batch_time_start = time.time()

        num_gt_annos = gt_annos_batch['num_objs']
        num_pred_annos = pred_annos_batch['num_objs']

        pred_gt_ratio = num_pred_annos / num_gt_annos if num_gt_annos else 0.0

        ap_dict = self.custom_osdar_eval(
            gt_annos_batch,
            pred_annos_batch,
            thresholds=iou_thresholds,
        )

        batch_time = time.time() - batch_time_start
        return {
            "status": "success",
            "ap_dict": ap_dict,
            "pred_gt_ratio": pred_gt_ratio,
            "scores": pred_annos_batch['mean_score'],
            "batch_time": batch_time,
        }

    def merge_results(self, result1: dict, result2: dict) -> dict:
        """Wrapper to merge two result dicts"""
        if not result1 and result2:
            return result2.copy()
        if not result2 and result1:
            return result1.copy()

        merged_result = {}
        for key in result1:
            merged_result[key] = {}
            for class_name in result1[key]:
                # Check if the class exists in both dictionaries
                if class_name in result2[key]:
                    merged_result[key][class_name] = (
                        result1[key][class_name] +
                        result2[key][class_name]
                    )
                else:
                    merged_result[key][class_name] = result1[key][class_name]
        return merged_result

    def calculate_metric_scores(
            self,
            ap_dict: dict,
            pred_gt_ratios: list,
            scores: list,
    ) -> dict:
        """
        Wrapper to calculate final score mAP, score_man and pred_gt_ratio.

        Args:
            ap_dict (dict):         Collected AP metrics
            pred_gt_ratios (list):  List of pred_gt_ratios
            scores: (list):         List of scores

        Returns:
            dict with calculated metrics
        """
        all_ap_values = []
        results = {}

        for ap_key, class_matches in ap_dict.items():
            per_class_aps = {}
            for class_name, frame_matches in class_matches.items():
                # pool matches of all frames, then dataset-level AP
                scores, tps = [], []
                n_gt = 0
                for frame_scores, frame_tps, frame_n_gt in frame_matches:
                    scores.extend(frame_scores)
                    tps.extend(frame_tps)
                    n_gt += frame_n_gt
                ap = self.calculate_ap11(scores, tps, n_gt)
                per_class_aps[class_name] = ap
                results[f"{ap_key}_{class_name}"] = ap
                all_ap_values.append(ap)

            results[ap_key] = np.mean(list(per_class_aps.values()))

        results['mAP'] = np.mean(all_ap_values)
        results['pred_gt_ratio'] = (
            sum(pred_gt_ratios) / max(len(pred_gt_ratios), 1)
        )
        results['score_mean'] = np.array(scores).mean() if scores else 0.0

        return results

    def select_gpu_with_max_free_memory(self) -> str:
        """Selects CUDA gpu with most free RAM"""
        import pynvml
        pynvml.nvmlInit()
        max_free = 0
        best_gpu = 0
        for i in range(int(os_environ.get("WORLD_SIZE", 0))):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            if info.free > max_free:
                max_free = info.free
                best_gpu = i
        return f"cuda:{best_gpu}"

    def osdar_evaluate(self,
                       pred_annos: List[dict],
                       gt_annos: List[dict],
                       filter_by_score=True,
                       logger: MMLogger = MMLogger.get_current_instance(),
                       ) -> Dict[str, float]:
        """
        Wrapper for calculating OSDaR23 mAP metric

        Args:
            pred_annos (list):      List of predictions to each batch
            gt_annos (list):        List of groundtruths to each batch
            filter_by_score (bool): Where to filter predictions below
                threshold
            logger (MMLogger):      MMLogger to give feedback of metric
                process (Default get_current_instance())
        """
        pred_gt_ratio = []
        scores = []
        ap_dict = {}
        failed_samples = 0
        filtered_preds = 0
        total_time_start = time.time()

        if len(gt_annos) != len(pred_annos):
            logger.warning("GT and Pred length does NOT match! "
                           f"gt: {len(gt_annos)}, pred: {len(pred_annos)}"
                           )

        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        def error_callback(error, queue):
            traceback.print_exception(type(error), error, error.__traceback__)
            queue.put({"status": "error"})

        with mp.Pool(processes=self.max_workers) as pool:
            futures = []
            for gt_annos_batch, pred_annos_batch in zip(gt_annos, pred_annos):
                if filter_by_score:
                    filtered_preds += pred_annos_batch['num_filtered']

                futures.append(
                    pool.apply_async(
                        self.process_batch_metric,
                        args=(
                            gt_annos_batch,
                            pred_annos_batch,
                            filter_by_score,
                            self.iou_thresholds,
                        ),
                        callback=lambda result: queue.put(result),
                        error_callback=lambda error: error_callback(
                            error, queue),
                    )
                )

            pool.close()

            for i in tqdm(
                    range(len(futures)),
                    desc="Calculating Metric",
                    unit="sample",
            ):
                result = queue.get()
                if result.get("status") != "success":
                    failed_samples += 1
                    continue

                ap_dict = self.merge_results(ap_dict, result['ap_dict'])
                pred_gt_ratio.append(result['pred_gt_ratio'])
                scores.append(result['scores'])

            # === Calculation Metric Score ===
            results = self.calculate_metric_scores(
                ap_dict,
                pred_gt_ratio,
                scores,
            )
            results['time'] = time.time() - total_time_start
            results['failed_samples'] = failed_samples
            results['filtered_preds'] = filtered_preds
            results['filtered_pred_score'] = self.score_threshold

            if len(pred_annos):
                self.print_ap_dict(results)
                print(f"map: {results['mAP']:0.3f}")
                print(f"pred_gt_ratio: {results['pred_gt_ratio']:0.3f}")
                print(f"score_mean: {results['score_mean']:0.3f}")

            return results
