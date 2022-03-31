"""Test pretrained multiscale lighting volume prediction on our InteriorNet test set."""

import os
from absl import app
import matplotlib.pyplot as plt
import numpy as np
from nerfactor.models.MLV_nerf import MLV
import glob
from third_party.xiuminglib import xiuminglib as xm
from os.path import join, exists
from nerfactor.util import logging as logutil, math as mathutil, io as ioutil, img as imgutil
from shutil import rmtree

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

root_dir = '/data/happily/SVL-nerf_data/'

mlv_checkpoint_dir = root_dir + 'Shared model/mlv_checkpoints/ckpt-1'
output_dir = root_dir + 'output/test_mlv'

scene = '0810-4'
lr = '5e-4'
viewdir = False
ndc = False
ipe = False
hallucinate = False
# ipe = True
bg_ckpt = root_dir + 'output/train/0810/' + scene + '/' + scene + '_nerf/' + f"lr{lr}_viewdir{viewdir}_ndc{ndc}_ipe{ipe}"

def main(argv):
    del argv  # unused
    ckpts = xm.os.sortglob(join(bg_ckpt, 'checkpoints'), 'ckpt-*.index')
    ckpt_ind = [int(os.path.basename(x)[len('ckpt-'):-len('.index')]) for x in ckpts]
    latest_ckpt = ckpts[np.argmax(ckpt_ind)]
    bg_latest_ckpt = latest_ckpt[:-len('.index')]

    model_mlv = MLV(bg_latest_ckpt, theta_res=240, phi_res=120, r_res=128)
    # model_mlv = MLV(bg_ckpt, theta_res = 32, phi_res = 16, r_res = 128)
    ioutil.restore_model_v2(model_mlv, mlv_checkpoint_dir)
    model_mlv.trainable = False
    model_mlv.cube_to_mlv(hallucinate)

    env_pose = np.identity(4)
    if ndc:
        z_list = np.linspace(0.0, 0.7, num=80)
        x_list = np.linspace(-0.3, 0.3, num=5)
    else:
        z_list = np.linspace(0.0, 2.5, num=80)
        x_list = np.linspace(-2.0, 2.0, num=5)

    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)
    frames = []
    # for i_x, x in enumerate(x_list):
    #     env_pose[2, 0] = x
    #     z_list = np.flip(z_list)
    for i_z, z in enumerate(z_list):
        env_pose[2, -1] = z
        env_map = model_mlv(env_pose)
        output_envmap_eval = env_map.numpy()
        # plt.imsave(os.path.join(output_dir, f"lr{lr}_viewdir{viewdir}_ndc{ndc}_ipe{ipe}_{i_x:04d}_{i_z:04d}.png"),
        #            output_envmap_eval[0, :, :, :3])
        plt.imsave(os.path.join(output_dir, f"lr{lr}_viewdir{viewdir}_ndc{ndc}_ipe{ipe}_{i_z:04d}.png"),
                   output_envmap_eval[0, :, :, :3])

    for batch_dir in sorted(glob.glob(output_dir + '/*.png')):
        frames.append(xm.io.img.load(batch_dir))
    xm.vis.video.make_video(frames, fps=15, outpath=output_dir +f"/viewdir{viewdir}_ndc{ndc}_ipe{ipe}_hall{hallucinate}.mp4")


if __name__ == "__main__":
    app.run(main)
