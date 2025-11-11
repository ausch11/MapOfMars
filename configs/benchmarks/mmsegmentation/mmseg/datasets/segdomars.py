# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SegDoMars16kDataset(BaseSegDataset):
    """
    """
    METAINFO = dict(
        classes=('Aeolian Curved (ael)', 'Aeolian Straight (aec)', 'Cliff (cli)', 'Ridge (rid)',
                 'Channel (fsf)', 'Mounds (sfe)', 'Gullies (fsg)', 'Slope Streaks (fse) ',
                 'Mass Wasting (fss)', 'Crater (cra)', 'Crater Field (sfx)', 'Mixed Terrain (mix)',
                 'Rough Terrain (rou)', 'Smooth Terrain (smo)', 'Textured Terrain (tex)'
                 ),
        palette=[[174, 199, 232], [31, 119, 180], [255, 127, 14], [197, 176, 213], [152, 223, 138],
                 [196, 156, 148], [214, 39, 40], [44, 160, 44], [255, 152, 150], [255, 187, 120],
                 [227, 119, 194], [148, 103, 189], [140, 86, 75], [247, 182, 210], [127, 127, 127]
                 ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
