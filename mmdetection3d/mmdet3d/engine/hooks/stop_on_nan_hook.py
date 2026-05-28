from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.hooks.hook import DATA_BATCH
from typing import Sequence, Optional
import torch


@HOOKS.register_module()
class StopOnNaNHook(Hook):
    """A hook that stops the training if a NaN loss appears."""

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[Sequence] = None,
                         ) -> None:
        """
        Checks training loss and stops if needed

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): idx of current batch
            data_batch (DATA_BATCH, optional): Not used
            outputs (Sequence, Optional): Not used
        """
        failed_metrics = []

        # Check loss
        if outputs is not None:
            for key, val in outputs.items():
                # Check if value is a tensor and contains NaN or Inf
                if (
                        isinstance(val, torch.Tensor) and
                        not torch.isfinite(val).all()
                ):
                    failed_metrics.append(key)

        # Check grad_norm
        try:
            grad_norm = runner.message_hub.get_scalar(
                'train/grad_norm').current()
            if (
                    grad_norm is not None and
                    not torch.isfinite(torch.tensor(grad_norm))
            ):
                failed_metrics.append("grad_norm")
        except KeyError:
            # grad_norm not logged yet — skip
            pass

        if failed_metrics:
            runner.logger.error(
                f'NaN or Inf detected in {failed_metrics} at iter {batch_idx}.'
            )
            raise StopIteration(
                f"Training stopped due to NaN at iter {batch_idx}."
            )
