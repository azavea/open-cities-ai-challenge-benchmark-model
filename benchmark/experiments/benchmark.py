import json
from functools import reduce
from os.path import basename, dirname, join
from random import sample

import rastervision as rv
from benchmark.constants import CLASSES, TRAIN_IDS, VALID_IDS
from benchmark.io import my_read_method, my_write_method
from benchmark.utils import str_to_bool
from pystac import STAC_IO, Catalog
from rastervision.backend.api import PYTORCH_SEMANTIC_SEGMENTATION
from rastervision.utils.files import file_exists

STAC_IO.read_text_method = my_read_method
STAC_IO.write_text_method = my_write_method


class BenchmarkExperiment(rv.ExperimentSet):
    def exp_benchmark(self, experiment_id, stac_uri, img_dir, root_uri, test=False):
        test = str_to_bool(test)

        chip_opts = {
            'window_method': 'sliding',
            'stride': 300
        }

        config = {
            'batch_size': 8,
            'num_epochs': 30,
            'debug': True,
            'lr': 1e-4,
            'one_cycle': True,
            'sync_interval': 1,
            'model_arch': 'resnet18'
        }

        if test:
            config['batch_size'] = 2
            config['num_epochs'] = 1

            chip_opts = {
                'window_method': 'random_sample',
                'chips_per_scene': 10
            }

            experiment_id += '-TEST'

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(900) \
                            .with_classes(CLASSES) \
                            .with_chip_options(**chip_opts) \
                            .build()

        backend = rv.BackendConfig.builder(PYTORCH_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        stac_dir = dirname(stac_uri)
        cat = Catalog.from_file(stac_uri)

        def make_scenes(item):
            area = item.get_parent().id
            label_uri = join(
                stac_dir, area, '{}-labels'.format(item.id), '{}.geojson'.format(item.id))

            i = 0
            images_remaining = True
            scenes = []
            while images_remaining:
                raster_uri = join(img_dir, area, item.id,
                                  '{}_{}.tif'.format(item.id, i))
                if file_exists(raster_uri):
                    raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                        .with_uri(raster_uri) \
                        .with_channel_order([0, 1, 2]) \
                        .build()

                    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                        .with_vector_source(label_uri) \
                        .with_rasterizer_options(2) \
                        .build()

                    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                        .with_raster_source(label_raster_source) \
                        .build()

                    scene = rv.SceneConfig.builder() \
                        .with_task(task) \
                        .with_id('{}_{}'.format(item.id, i)) \
                        .with_raster_source(raster_source) \
                        .with_label_source(label_source) \
                        .build()

                    scenes.append(scene)
                else:
                    images_remaining = False
                i += 1

            return scenes

        train_ids = TRAIN_IDS
        valid_ids = VALID_IDS
        if test:
            train_ids = sample(train_ids, 2)
            valid_ids = sample(valid_ids, 2)

        train_scenes = reduce(
            lambda a, b: a+b, [make_scenes(cat.get_child(c).get_item(i)) for c, i in train_ids])
        valid_scenes = reduce(
            lambda a, b: a+b, [make_scenes(cat.get_child(c).get_item(i)) for c, i in valid_ids])

        if test:
            train_scenes = sample(train_scenes, 3)
            valid_scenes = sample(valid_scenes, 3)

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(valid_scenes) \
            .build()

        experiment = rv.ExperimentConfig.builder() \
            .with_id(experiment_id) \
            .with_task(task) \
            .with_backend(backend) \
            .with_dataset(dataset) \
            .with_root_uri(root_uri) \
            .build()

        return experiment
