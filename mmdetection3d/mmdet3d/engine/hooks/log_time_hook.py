import csv
import time
import numpy as np
from typing import Sequence, Optional

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class InferenceTimeHook(Hook):
    """A hook that logs the testing speed of each iteration."""

    priority = 'NORMAL'

    times = []
    data_times = []

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None,
                        ) -> None:
        """
        Store interference time of each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): idx of current batch
            data_batch (DATA_BATCH, optional): Not used
            outputs (Sequence, Optional): Not used
        """

        message_hub = runner.message_hub

        total_time = message_hub.get_scalar('test/time').current()
        data_time = message_hub.get_scalar('test/data_time').current()

        self.times.append((batch_idx, total_time))
        self.data_times.append((batch_idx, data_time))

    def after_test(self, runner) -> None:
        """
        Log average inference speed of entire testing process (per rank).

        Args:
            runner (Runner): The runner of the testing process.
        """
        time_mean = np.mean([t for _, t in self.times])
        data_time_mean = np.mean([t for _, t in self.data_times])
        runner.logger.info('Average test speed of entire testprocess'
                           f'is time: {time_mean:.1f} s/iter'
                           f'data_time: {data_time_mean:.1f} s/iter'
                           )

        rank = runner.rank
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = f"results/{timestamp}_inference_times_rank{rank}.csv"
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "time", "data_time"])
            data = zip(self.times, self.data_times)
            for ((batch_idx, total_time), (batch_idx_2, data_time)) in data:
                if batch_idx != batch_idx_2:
                    runner.logger.warn(
                        f"Batch_IDX does not match. {batch_idx}, {batch_idx_2}"
                    )
                csv.writer(f).writerow([batch_idx, total_time, data_time])
        runner.logger.info(f'CSV stored to {csv_path}')
