from os.path import dirname, join

import rastervision as rv
from benchmark.aux.preprocess import PREPROCESS
from benchmark.constants import TRAIN_IDS, VALID_IDS
from pystac import Catalog


class SplitImages(rv.ExperimentSet):
    def exp_split_images(self, cat_uri, output_dir, root_uri):
        image_ids = TRAIN_IDS + VALID_IDS

        image_uris = [join(dirname(cat_uri), area, uid,
                           '{}.tif'.format(uid)) for area, uid in image_ids]

        config = rv.CommandConfig.builder(PREPROCESS) \
                                 .with_root_uri(root_uri) \
                                 .with_config(items=image_uris, output_dir=output_dir) \
                                 .build()
        return config
