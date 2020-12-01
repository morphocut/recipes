"""Experiment on processing KOSMOS data using MorphoCut."""
import os

import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.segmentation

from morphocut import Call, Pipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.file import Find
from morphocut.image import ExtractROI, ImageProperties, ThresholdConst, RGB2Gray
from morphocut.pandas import JoinMetadata, PandasWriter
from morphocut.str import Format, Parse
from morphocut.stream import TQDM, Enumerate, PrintObjects, StreamBuffer
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.integration.flowcam import FlowCamReader

import_path = "../../morphocut/tests/data/flowcam"
export_path = "/tmp/flowcam"

if __name__ == "__main__":
    print("Processing images under {}...".format(import_path))

    with Pipeline() as p:
        lst_fn = Find(import_path, [".lst"])

        TQDM(lst_fn)

        obj = FlowCamReader(lst_fn)

        img = obj.image
        img_gray = RGB2Gray(img, True)

        mask = obj.mask

        regionprops = ImageProperties(mask, img_gray)

        # TODO: Prefix everything with object_ for ecotaxa
        object_meta = obj.data

        object_id = Format("{lst_name}_{id}", lst_name=obj.lst_name, _kwargs=object_meta)
        object_meta["id"] = object_id
        object_meta = CalculateZooProcessFeatures(regionprops, object_meta)

        EcotaxaWriter(
            os.path.join(export_path, "export.zip"),
            [
                (Format("{object_id}.jpg", object_id=object_id), img),
                (Format("{object_id}_gray.jpg", object_id=object_id), img_gray),
                (Format("{object_id}_mask.jpg", object_id=object_id), mask),
            ],
            object_meta=object_meta,
        )

        TQDM(object_id)

    p.run()
