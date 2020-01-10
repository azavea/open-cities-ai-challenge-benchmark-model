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
    def exp_benchmark(self,
                      experiment_id,
                      root_uri,
                      train_stac_uri,
                      test_stac_uri,
                      train_img_dir=None,
                      test_img_dir=None,
                      test=False):

        # Parse 'test' option
        test = str_to_bool(test)

        # Define split image directory: defaults to a directory called 'split_images'
        # within the root directory
        if not train_img_dir:
            train_img_dir = join(root_uri, 'split_images')

        # This example uses scenes from the tier 1 training data for training and
        # evaluation. Load in the tier 1 data using pystac:
        train_stac = Catalog.from_file(train_stac_uri)

        # Use the set of train and validation scene ids that are defined in constants.py
        train_ids = TRAIN_IDS
        valid_ids = VALID_IDS
        if test:
            train_ids = sample(train_ids, 2)
            valid_ids = sample(valid_ids, 2)

        # For this experiment, we are interested in making predictions on two different
        # sets of images:
        #   1. The validation set, which we subset from the tier 1 training data. These
        #      scenes have labels associated with them and will be used to calculate
        #      validation metrics during the eval stage.
        #   2. The test set, which come from the `test` stac. These will not be used for
        #      model evaluation because we don't have the labels for them. Instead, we
        #      will submit the test set predictions to the competition site.
        #
        # We will need to make scenes for each of the 11,481 test chips. It is possible
        # to do this by reading the test STAC directly from s3 but it is not recommended.
        # A better option would be to download the test data from the competition site
        # and uncompress it into the data directory in this repo. That directory will
        # mount into `/opt/data/` within the docker container and then you can supply
        # uri for the test catalog json as the `test_stac_uri` parameter of this experiment
        # runner (i.e. `/opt/data/test/catalog.json`).
        #
        # It may make sense to start with a work flow in which you hold off on making
        # predictions on the test set until you have tried several approaches and validated
        # evaluated them against the validation set. If you would like to do that you can
        # simply remove the `.with_test_scenes()` call from the dataset config builder as well
        # as previous steps that deal with test data.

        # Create a generator of all test images
        test_stac = Catalog.from_file(test_stac_uri)
        all_test_items = test_stac.get_all_items()

        # By default, this script will look for test images in the file tree starting at
        # the location of the test STAC catalog. However, in this example we want to access
        # the catalog json and the images from different locations. As mentioned above, in the
        # interest of time we will get all test ids from a local version of the test STAC but
        # will still need to download the images directly from s3 since the prediction will
        # run remotely. We will use a different location by setting the `test_img_dir`
        # command line parameter to the poblic location of the test data on s3 (i.e.
        # -a test_img_dir s3://drivendata-competition-building-segmentation/test/)
        if not test_img_dir:
            test_img_dir = dirname(test_stac_uri)

        # Configure chip creation
        chip_opts = {
            'window_method': 'sliding',  # use sliding window method of to create chips
            'stride': 300               # slide over 300px to generate each new chip
        }

        # Training configuration: try to improve performance by tuning these
        # hyperparameters
        config = {
            'batch_size': 8,          # 8 chips per batch
            'num_epochs': 6,          # complete 6 epochs
            'debug': True,            # produce example chips to help with debugging
            'lr': 1e-4,               # set learning
            'one_cycle': True,        # use cyclic learning rate scheduler
            'model_arch': 'resnet18'  # model architecture
        }

        # It is generally helpful to try running a 'test' version of the experiment before
        # the entire thing. You can do this by using the 'test' command line parameter (i.e.
        # '-a test True'). This will run a very small version of the experiment that can be
        # run locally.
        if test:
            config['batch_size'] = 2
            config['num_epochs'] = 1

            # In this 'test' scenario, we will generate a small number of chips at random
            # rather than creating a set that exhaustively covers each scene
            chip_opts = {
                'window_method': 'random_sample',
                'chips_per_scene': 10
            }

            # Convert the generator of all items (i.e. chips) within the test set into
            # a list of only five
            all_test_items = [next(all_test_items) for _ in range(5)]

            # Modify the experiment ID so that a future iteration of this same experiment
            # without the 'test' flag will not rely on the same outputs
            experiment_id += '-TEST'

        # Rastervision stores information about images and often their associted labels in
        # SceneConfig objects. We will use this function to created lists of train and
        # validation scenes from pystac items.
        def make_train_scenes(item):
            # We can easily construct the uri using the root directory of the training STAC.
            # Each rv scene will consist of one portion of an image created in the preprocessing
            # stage but it is not necessary to split the labels up in the same way. We can
            # pass a scene's label uri to all of it's child images and rv will subset the
            # labels automatically.
            area = item.get_parent().id
            label_uri = join(dirname(train_stac_uri), area,
                             '{}-labels'.format(item.id), '{}.geojson'.format(item.id))

            # If you preprocessed the imagery using the 'PREPROCESS' aux command, the
            # image splits will have integer suffixes that ascend from 0 until the image
            # has been completely covered. To make sure we get all of the splits we will
            # incrementally create new uri's until one is found to not exist.
            i = 0
            images_remaining = True
            scenes = []
            while images_remaining:
                raster_uri = join(train_img_dir, area, item.id,
                                  '{}_{}.tif'.format(item.id, i))

                if file_exists(raster_uri):
                    # construct a raster source (i.e. the image)
                    raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                        .with_uri(raster_uri) \
                        .with_channel_order([0, 1, 2]) \
                        .build()

                    # construct a label source (i.e. the scene's geojson labels)
                    # The with_rasterizer_options method sets the default pixel value to use
                    # for background pixels in prediction. The value of 0 is reserved for nodata
                    # pixels in rv so we will need to add a postprocessing step at the end
                    # in order for the test set predicitons to match the competition submission
                    # suidelines.
                    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                        .with_vector_source(label_uri) \
                        .with_rasterizer_options(2) \
                        .build()

                    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                        .with_raster_source(label_raster_source) \
                        .build()

                    # Build scene config
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

        # The test image chips are also encoded as SceneConfig objects but
        # lack a label source
        def make_test_scene(item):
            raster_uri = join(test_img_dir, item.id, '{}.tif'.format(item.id))

            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(raster_uri) \
                .with_channel_order([0, 1, 2]) \
                .build()

            scene = rv.SceneConfig.builder() \
                .with_id(item.id) \
                .with_raster_source(raster_source) \
                .build()

            return scene

        # Configure a semantic segmentation task with classes defined in constants.py,
        # use the chipping options defined previously within this script
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_classes(CLASSES) \
                            .with_chip_options(**chip_opts) \
                            .build()

        # This semantic segmentation task uses pytorch as a backend deep learning library.
        # The pytorch backend comes preloaded in rv and installed in this docker container
        # but you can use any backend with additional configuration. Configure the experiment
        # to use the training parameters defined above.
        backend = rv.BackendConfig.builder(PYTORCH_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        # Create train, validation and test scenes
        print('Creating train scenes')
        train_scenes = reduce(
            lambda a, b: a + b, [make_train_scenes(train_stac.get_child(c).get_item(i)) for c, i in train_ids])

        print('Creating validation scenes')
        valid_scenes = reduce(
            lambda a, b: a + b, [make_train_scenes(train_stac.get_child(c).get_item(i)) for c, i in valid_ids])
        
        # Using four different scenes (from four different cities) for the validation
        # set is benefficial becuase it gives us a diverse set of imagery to use when 
        # determining the generalizability of the model. However, it would be overkill
        # to validate the model on the entirety of the four scenes. We will take a random
        # sample of those image splits and validate on those.
        valid_scenes = sample(valid_scenes, 30)

        print('Creating test scenes')
        test_scenes = [make_test_scene(item) for item in all_test_items]

        if test:
            train_scenes = sample(train_scenes, 3)
            valid_scenes = sample(valid_scenes, 3)

        # and use them as inputs to an RV DatasetConfig
        print('Building dataset config')
        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(valid_scenes) \
            .with_test_scenes(test_scenes) \
            .build()

        # As mentioned in a previous comment, 0 is a reserved pixel value in rv and can therefore
        # not be used as the predicted background (or 'no building' value). Those pixels will have
        # a value of 2 in the raw predictions but we need to convert them to 0 to match the
        # submission guidelines.
        #
        # We can do this using a simple aux command called 'POSTPROCESS' that is defined in
        # aux/postprocess.py. This command requires a few parameters that we define in the dict
        # below. It takes the raw prediction tifs as input. These are not created yet but
        # will be written to a specific location below the root uri. Raster Vision will understand
        # that the POSTPROCESS task relies on the output of the predict task and will therefore
        # wait to run the former until the latter has completed.
        postprocess_config = {
            'POSTPROCESS': {
                'key': 'postprocess',
                'config': {
                    'uris': [join(root_uri, 'predict', experiment_id, '{}.tif'.format(scene.id)) for scene in test_scenes],
                    'root_uri': root_uri,
                    'experiment_id': experiment_id
                }
            }
        }

        # Finally build an experiment from all of these constituent parts. Returning an
        # methods that have the prefix 'exp_' (like this one) will run reflexively when this
        # script is run.
        experiment = rv.ExperimentConfig.builder() \
            .with_id(experiment_id) \
            .with_task(task) \
            .with_backend(backend) \
            .with_dataset(dataset) \
            .with_root_uri(root_uri) \
            .with_custom_config(postprocess_config) \
            .build()

        # returning the experiment config will kick off the chain workflow if running locally
        # or submit the sequence of dependent jobs if running on batch
        return experiment
