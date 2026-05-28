from mmdet3d.datasets.lumpi import LUMPIDataset

from tools.analysis_tools.dataset_analyzer import (
    DatasetAnalyzer, AnalysisMode, AnchorMode, DatasetAnalyerArgumentParser
)


class LUMPIAnalyzer(DatasetAnalyzer):
    """
    Analyze quality of LUMPI dataset and calculate needed values for training.

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
            LUMPIDataset,
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
        "LUMPI",
        "data/lumpi/labels",
        "data/lumpi/points",
        "data/lumpi/",
    )
    args = parser.parse_args()

    analyzer = LUMPIAnalyzer(
        args.labels,
        args.points,
        args.output,
        anchor_mode=args.anchor_mode,
        anchor_count=args.anchor_count,
        modes=args.mode,
    )
    analyzer.analyze()
