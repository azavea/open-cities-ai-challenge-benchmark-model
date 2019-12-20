import os
from itertools import islice
from os.path import dirname, join

import rastervision as rv
from benchmark.aux.postprocess import POSTPROCESS
from benchmark.constants import CLASSES
from benchmark.io import my_read_method, my_write_method
from benchmark.utils import str_to_bool
from pystac import STAC_IO, Catalog
from rastervision.backend.api import PYTORCH_SEMANTIC_SEGMENTATION
from rastervision.utils.files import file_exists

STAC_IO.read_text_method = my_read_method
STAC_IO.write_text_method = my_write_method


class PredictionExperiment(rv.ExperimentSet):
    def exp_predict(self, experiment_id, stac_uri, root_uri, test=False):
        test = str_to_bool(test)
        if test:
            experiment_id += '-TEST'
            
        cat = Catalog.from_file(stac_uri)
        all_items = cat.get_all_items()
        if test:
            all_items = [next(all_items) for _ in range(5)]

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_predict_chip_size(1024) \
                            .with_classes(CLASSES) \
                            .build()

        backend = rv.BackendConfig.builder(PYTORCH_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .build()

        def make_scene(item, stac_uri):
            raster_uri = join(dirname(stac_uri), item.id,
                              '{}.tif'.format(item.id))
            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(raster_uri) \
                .with_channel_order([0, 1, 2]) \
                .build()

            scene = rv.SceneConfig.builder() \
                .with_id(item.id) \
                .with_raster_source(raster_source) \
                .build()

            return scene

        scenes = [make_scene(i, stac_uri) for i in all_items]
        dataset = rv.DatasetConfig.builder() \
            .with_test_scenes(scenes) \
            .build()

        uris = ['/opt/data/predict/{}/{}.tif'.format(experiment_id, i.id) for i in all_items]

        postprocess_config = {
            'POSTPROCESS': {
                'key': 'postprocess',
                'config': {
                    'uris': uris,
                    'root_uri': root_uri,
                    'experiment_id': experiment_id
                }
            }
        }
        
        experiment = rv.ExperimentConfig.builder() \
            .with_id(experiment_id) \
            .with_train_key('split_img_v3') \
            .with_task(task) \
            .with_backend(backend) \
            .with_dataset(dataset) \
            .with_predict_uri(join('/opt/data/predict/', experiment_id)) \
            .with_root_uri(root_uri) \
            .with_custom_config(postprocess_config) \
            .build()


        return experiment
