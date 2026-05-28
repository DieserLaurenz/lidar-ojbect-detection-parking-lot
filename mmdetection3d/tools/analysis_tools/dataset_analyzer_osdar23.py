from mmdet3d.datasets.osdar23 import OSDaR23Dataset

from tools.analysis_tools.dataset_analyzer import (
    DatasetAnalyzer, AnalysisMode, AnchorMode, DatasetAnalyerArgumentParser
)


class OSDaR23Analyzer(DatasetAnalyzer):
    """
    Analyze quality of OSDaR23 dataset and calculate needed values for training

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
            OSDaR23Dataset,
            labels_path,
            points_path,
            output_dir,
            anchor_mode,
            anchor_count,
            modes,
            percentiles
        )


if __name__ == "__main__":
    parser = DatasetAnalyerArgumentParser(
        "OSDaR23",
        "data/osdar23/labels",
        "data/osdar23/points",
        "data/osdar23/",
    )
    args = parser.parse_args()

    analyzer = OSDaR23Analyzer(
        args.labels,
        args.points,
        args.output,
        anchor_mode=args.anchor_mode,
        anchor_count=args.anchor_count,
        modes=args.mode,
    )
    analyzer.analyze()
