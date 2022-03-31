# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join, basename
from absl import app, flags
from tqdm import tqdm

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, config as configutil
from third_party.xiuminglib import xiuminglib as xm
import os

scene = '0917'
root_dir = '/data/happily/source/SVL-nerf_data/'

lr = '5e-4'
outroot = root_dir + 'output/train/' + scene + '/' + scene + '_nerf'
ckpt = outroot + "/lr5e-4_batch2048_far3.0_fine/checkpoints/ckpt-17"
# ckpt = outroot + "/lr5e-4/checkpoints/ckpt-20"

# outroot = root_dir + 'output/train/' +scene + '/' + scene + '_nerf'
# ckpt = outroot + "/lr" + lr + "/checkpoints/ckpt-20"
gpus = "0"

logger = logutil.Logger(loggee="nerf_test")

os.environ["CUDA_VISIBLE_DEVICES"] = gpus
debug=False

def main(_):
    if debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_test', basename(ckpt))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=debug)
    ioutil.restore_model(model, ckpt)

    # For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(tqdm(datapipe, desc="Inferring Views", total=n_views)):
        # break
        data_name = batch[0][0].numpy().decode().split('_')
        id = int(data_name[1])
        if id % 4 == int(gpus):
            _, _, _, to_vis = model.call(batch, mode='test')
            # Visualize
            outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
            model.vis_batch(to_vis, outdir, mode='test')
            # Break if debugging
        if debug:
            break

    if gpus == '0':
        import time
        time.sleep(30)
        batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
        outpref = outroot  # proper extension should be added in the function below
        view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test', fps=24)
        logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    main(None)
