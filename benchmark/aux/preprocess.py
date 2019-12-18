from os.path import basename, join
from subprocess import call

import rasterio
import rastervision as rv
from rasterio.windows import Window

PREPROCESS = 'PREPROCESS'


def split_image(image_uri, output_dir):
    with rasterio.open(image_uri) as src:
        width = src.width
        height = src.height

    win_size = 9000
    wins = []
    for c in list(range(0, width, win_size)):
        if c >= width:
            continue
        for r in list(range(0, height, win_size)):
            if r >= height:
                continue
            wins.append(Window(c, r, win_size, win_size))

    for i, win in enumerate(wins):
        with rasterio.open(image_uri) as src:
            img = src.read(window=win)
            win_transform = src.window_transform(win)
            kwargs = src.meta.copy()

        area = image_uri.split('/')[-3]
        image_id = image_uri.split('/')[-2]
        output_uri = join(output_dir, area, image_id,
                          '{}_{}.tif'.format(image_id, i))
        kwargs.update({
            'height': win_size,
            'width': win_size,
            'transform': win_transform
        })

        tmp_uri = join('/tmp/', basename(output_uri))
        with rasterio.open(tmp_uri, 'w', **kwargs) as dst:
            dst.write(img)
        tmp_cmpr_file = tmp_uri.replace('.tif', '_jpg.tif')
        gdal_command = 'gdal_translate {} {} '.format(tmp_uri, tmp_cmpr_file) +\
            '-co COMPRESS=JPEG -co JPEG_QUALITY=100 -co TILED=YES ' +\
            '-co COPY_SRC_OVERVIEWS=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 --config COMPRESS_OVERVIEW JPEG'
        call(gdal_command, shell=True)
        call('rm {}'.format(tmp_uri), shell=True)
        call('aws s3 mv {} {}'.format(tmp_cmpr_file, output_uri), shell=True)


class PreProcessCommand(rv.AuxCommand):
    command_type = PREPROCESS
    options = rv.AuxCommandOptions(
        split_on='items',
        inputs=lambda conf: PreProcessCommand.gather_inputs(conf),
        outputs=lambda conf: PreProcessCommand.gather_outputs(conf),
        required_fields=['items', 'output_dir'])

    def run(self):
        for image_uri in self.command_config['items']:
            split_image(image_uri, self.command_config['output_dir'])

    @staticmethod
    def gather_inputs(conf):
        return conf['items']

    @staticmethod
    def gather_outputs(conf):
        return ['.phony']
