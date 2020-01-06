from os.path import dirname, join

import rastervision as rv
from benchmark.aux.preprocess import PREPROCESS
from benchmark.constants import TRAIN_IDS, VALID_IDS
from pystac import Catalog


class SplitImages(rv.ExperimentSet):
    def exp_split_images(self,
                         root_uri,
                         train_stac_uri='https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/train_tier_1/catalog.json',
                         split_dir=None):
        
        if not split_dir:
            split_dir = join(root_uri, 'split_images')

        image_ids = TRAIN_IDS + VALID_IDS

        image_uris = [join(dirname(train_stac_uri), area, uid,
                           '{}.tif'.format(uid)) for area, uid in image_ids]

        config = rv.CommandConfig.builder(PREPROCESS) \
                                 .with_root_uri(root_uri) \
                                 .with_config(items=image_uris, split_dir=split_dir) \
                                 .build()
        return config
