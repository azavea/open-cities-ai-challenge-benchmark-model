import os
from os.path import basename, dirname, isdir, join

import numpy as np

import rasterio
import rastervision as rv
from rastervision.utils.files import upload_or_copy, download_if_needed

POSTPROCESS = 'POSTPROCESS'


def _postprocess(pred_uri, experiment_id, root_uri):
    tmp_pred_uri = download_if_needed(pred_uri, '/opt/data/predict/')
    tmp_postprocess_uri = tmp_pred_uri.replace('/predict/', '/postprocess/')
    
    os.makedirs(dirname(tmp_postprocess_uri), exist_ok=True)
    out_uri = join(root_uri, 'postprocess', experiment_id, basename(pred_uri))

    with rasterio.open(tmp_pred_uri) as src:
        img = src.read()
        img = np.where(img == 2, 0, img)
        profile = src.profile
    with rasterio.open(tmp_postprocess_uri, 'w', **profile) as dst:
        dst.write(img)

    upload_or_copy(tmp_postprocess_uri, out_uri)
    for t in (tmp_pred_uri, tmp_postprocess_uri):
        os.remove(t)


class PostProcessCommand(rv.AuxCommand):
    command_type = POSTPROCESS
    options = rv.AuxCommandOptions(
        split_on='uris',
        inputs=lambda conf: PostProcessCommand.gather_inputs(conf),
        outputs=lambda conf: PostProcessCommand.gather_outputs(conf),
        required_fields=['uris', 'root_uri', 'experiment_id']
    )

    def run(self):
        root_uri = self.command_config.get('root_uri')
        experiment_id = self.command_config.get('experiment_id')
        postprocess_dir = '/opt/data/postprocess/{}/'.format(experiment_id)
        if not isdir(postprocess_dir):
            os.makedirs(postprocess_dir)

        for uri in self.command_config['uris']:
            _postprocess(uri, experiment_id, root_uri)

    @staticmethod
    def gather_inputs(conf):
        return conf['uris']

    @staticmethod
    def gather_outputs(conf):
        return ['.phony']
