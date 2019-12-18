from os.path import dirname, join

import rastervision as rv
from benchmark.aux.preprocess import PREPROCESS
from pystac import Catalog


class SplitImages(rv.ExperimentSet):
    def exp_split_images(self, output_dir):
        image_ids = [('acc', 'd41d81'),
                     ('mon', '401175'),
                     ('dar', 'a017f9'),
                     ('znz', '75cdfa'),
                     ('acc', 'a42435'),
                     ('mon', 'f15272'),
                     ('dar', '42f235'),
                     ('znz', 'aee7fd')]

        cat_uri = 's3://raster-vision-world-bank-challenge/FINAL/train_tier_1/catalog.json'
        image_uris = [join(dirname(cat_uri), area, iid,
                           '{}.tif'.format(iid)) for area, iid in image_ids]

        config = rv.CommandConfig.builder(PREPROCESS) \
                                 .with_root_uri('s3://raster-vision-world-bank-challenge/benchmark/') \
                                 .with_config(items=image_uris, output_dir=output_dir) \
                                 .build()
        return config
