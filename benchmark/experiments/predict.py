import os
from itertools import islice
from os.path import join

import rastervision as rv
from benchmark.aux.predict_chips import PREDICT_CHIPS
from benchmark.experiments.io import my_read_method, my_write_method
from benchmark.experiments.utils import str_to_bool
from pystac import STAC_IO, Catalog
from rastervision.utils.files import file_exists

STAC_IO.read_text_method = my_read_method
STAC_IO.write_text_method = my_write_method


class BenchmarkExperiment(rv.ExperimentSet):
    def exp_benchmark(self, experiment_id, prediction_package_uri='s3://raster-vision-world-bank-challenge/benchmark/bundle/split_img_v3/predict_package.zip',
                      stac_uri='s3://raster-vision-world-bank-challenge/FINAL/test/catalog.json',
                      test=False):

        test = str_to_bool(test)
        if test:
            experiment_id += '-TEST'

        tmp_dir = join('/opt/data/', experiment_id)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        predictor = rv.Predictor(prediction_package_uri, tmp_dir)
        predictor.load_model()

        cat = Catalog.from_file(stac_uri)
        all_items = cat.get_all_items()
        if test:
            all_items = islice(all_items, 5)

        config = rv.CommandConfig.builder(PREDICT_CHIPS) \
                                 .with_root_uri('s3://raster-vision-world-bank-challenge/benchmark/') \
                                 .with_config(items=all_items, predictor=predictor, stac_uri=stac_uri) \
                                 .build()

        return config