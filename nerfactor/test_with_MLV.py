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
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, config as configutil
from third_party.xiuminglib import xiuminglib as xm
from third_party.turbo_colormap import turbo_colormap_data, interpolate_or_clip
import os
from nerfactor.util import vis as visutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil, img as imgutil, \
    math as mathutil, geom as geomutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

root_dir = '/data/happily/SVL-nerf_data/'

flags.DEFINE_string('ckpt', root_dir + 'output/train/pinecone/pinecone_nerfactor/lr5e-3/checkpoints/ckpt-10',
                    "path to checkpoint (prefix only)")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="test")

mlv_checkpoint_dir = root_dir + 'Shared model/mlv_checkpoints/ckpt-1'
scene = '0810-4'
lr = '5e-4'
bg_ckpt = root_dir + 'output/train/0810/' + scene + '/' + scene + '_nerf' + "/lr" + lr + "_viewdirFalse/checkpoints/ckpt-70"


def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_test_mlv', basename(FLAGS.ckpt))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=FLAGS.debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    ioutil.restore_model(model, FLAGS.ckpt)

    model.get_mlv(bg_ckpt=bg_ckpt, mlv_ckpt=mlv_checkpoint_dir)

    # Optionally, edit BRDF
    brdf_z_override = None

    # For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(tqdm(datapipe, desc="Inferring Views", total=n_views)):
        # id = int(batch[0][0].numpy().decode().split('_')[1])
        # if id % 2 == 0:
        relight_olat = batch_i == n_views - 1  # only for the final view
        # Optionally, edit (spatially-varying) albedo
        albedo_override = None
        albedo_scales = None
        # Inference
        env_pose = np.identity(4)
        x_list = np.linspace(-0.5, 0.5, num=10)
        z_list = np.linspace(0.3, 0.7, num=10)
        for x_i, x in enumerate(x_list):
            for z_i, z in enumerate(z_list):
                env_pose[0, -1] = x
                env_pose[2, -1] = z
                _, _, _, to_vis = model.call(batch, mode='test', relight_olat=relight_olat, relight_probes=True,
                                             albedo_scales=albedo_scales, albedo_override=albedo_override,
                                             brdf_z_override=brdf_z_override, with_mlv=True, env_pose=env_pose)
            # Visualize
                mask = dataset.dict_mask[to_vis['id'].numpy()[0].decode()]
                hw = tuple(to_vis.pop('hw').numpy()[0, :])
                mask = mask.reshape(hw)
                outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
                img_tmp = np.zeros(hw + (3,))
                img_tmp[np.where(mask)] =to_vis['pred_rgb'].numpy()
                xm.io.img.write_arr(img_tmp, join(outdir, f'{x_i}_{z_i}.png'), clip=True)

            # model.vis_batch(to_vis, mask, outdir, mode='test', olat_vis=relight_olat)

        print()
        # Break if debugging
        break
        if FLAGS.debug:
            break

    # Compile all visualized batches into a consolidated view (e.g., an HTML or a video)
    batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
    outpref = outroot  # proper extension should be added in the function below
    view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test')
    logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    app.run(main)
