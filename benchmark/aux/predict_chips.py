import os
from os.path import basename, dirname, join

import numpy as np

import rasterio
import rastervision as rv

PREDICT_CHIPS = 'PREDICT_CHIPS'


def _predict_chip(item, predictor, stac_uri):
    img_uri = join(dirname(stac_uri), item.id, '{}.tif'.format(item.id))
    pred_uri = '/opt/data/preds/{}'.format(basename(img_uri))
    _ = predictor.predict(img_uri, pred_uri)
    with rasterio.open(pred_uri) as src:
        img = src.read()
        img = np.where(img == 2, 0, img)
        profile = src.profile
    os.remove(pred_uri)
    with rasterio.open(pred_uri, 'w', **profile) as dst:
        dst.write(img)


class PredictChipsCommand(rv.AuxCommand):
    command_type = PREDICT_CHIPS
    options = rv.AuxCommandOptions(
        split_on='items',
        inputs=lambda conf: PredictChipsCommand.gather_inputs(conf),
        outputs=lambda conf: PredictChipsCommand.gather_ouputs(conf),
        required_fields=['items', 'predictor', 'stac_uri']
    )

    def run(self):
        for item in self.command_config['items']:
            _predict_chip(
                item, self.command_config['predictor'], self.command_config['stac_uri'])

    @staticmethod
    def gather_inputs(conf):
        return conf['items']

    @staticmethod
    def gather_outputs(conf):
        return ['.phony']
