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
from absl import app
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil
from third_party.xiuminglib import xiuminglib as xm
import os
from nerfactor.util import config as configutil, io as ioutil

gpus = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
debug = False
logger = logutil.Logger(loggee="test")

root_dir = '/data/happily/SVL-nerf_data/'
obj_scene = 'pinecone'
obj_ckpt = root_dir + 'output/train/' + obj_scene + '/' + obj_scene + '_nerfactor/lr5e-3/checkpoints/ckpt-10'

mlv_checkpoint_dir = root_dir + 'Shared model/mlv_checkpoints/ckpt-1'
hallucinate = False
bg_scene = '0810-4'
bg_datadir = root_dir + 'real_scene/' + bg_scene
lr = '5e-4'
viewdir = True
ndc = False
ipe = True
bg_ckpt = root_dir + 'output/train/0810/' + bg_scene + '/' + bg_scene + '_nerf/' + f"lr{lr}_viewdir{viewdir}_ndc{ndc}_ipe{ipe}"

shapedir = root_dir + 'output/surf/' + obj_scene + '_' + bg_scene
outroot = root_dir + 'output/insertion/' + obj_scene + '_' + bg_scene
imh = 1008

# obj_pose = '1.0, 0.0, 0.0, -0.25, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0'
# l2w_proj = '0.8333333, 0, 0, -0.25, 0, -0.8333333, 0, 0.25, 0, 0, -0.8333333, 2, 0,0,0,1'
# l2w_proj = np.ndarray([[0.8333333, 0, 0, -0.25],[0, 0.8333333, 0, 0.25],[0, 0, 0.8333333, 2],[0.0, 0.0, 0.0, 1.0]], dtype=np.float)
alpha_thres = 0.8
accu_chunk = 131072
fixed_view = True

def main(_):
    if debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(obj_ckpt)
    config = ioutil.read_config(config_ini)

    config.set('DEFAULT', 'data_root', bg_datadir)
    config.set('DEFAULT', 'shape_root', shapedir)
    config.set('DEFAULT', 'dataset', 'nerfactor_insertion')
    config.set('DEFAULT', 'imh', str(imh))
    config.set('DEFAULT', 'fixed_view', str(fixed_view))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    ckpts = xm.os.sortglob(join(bg_ckpt, 'checkpoints'), 'ckpt-*.index')
    ckpt_ind = [int(os.path.basename(x)[len('ckpt-'):-len('.index')]) for x in ckpts]
    latest_ckpt = ckpts[np.argmax(ckpt_ind)]
    bg_latest_ckpt = latest_ckpt[:-len('.index')]

    config.set('DEFAULT', 'model', 'nerfactor_insertion')
    config.set('DEFAULT', 'bg_ckpt', bg_latest_ckpt)
    config.set('DEFAULT', 'mlv_ckpt', mlv_checkpoint_dir)
    config.set('DEFAULT', 'hallucinate', str(hallucinate))
    config.set('DEFAULT', 'alpha_thres', str(alpha_thres))
    config.set('DEFAULT', 'accu_chunk', str(accu_chunk))

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=debug)
    ioutil.restore_model(model, obj_ckpt)
    # model.set_obj_envmap(obj_pose)

    # For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(tqdm(datapipe, desc="Inferring Views", total=n_views)):
        data_name = batch[0][0].numpy().decode().split('_')
        assert not ((data_name[0] == 'loc') ^ fixed_view), "when using fixed view, we must use loc_ data"
        id = int(data_name[1])
        if id % 4 == int(gpus):
            _, to_vis = model.call(batch, mode='test', fixed_view=fixed_view)
            outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
            model.vis_batch(to_vis, outdir, mode='test')

    import time
    time.sleep(20)
    # Compile all visualized batches into a consolidated view (e.g., an HTML or a video)
    batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
    outpref = outroot  # proper extension should be added in the function below
    view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test')
    logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    app.run(main)
