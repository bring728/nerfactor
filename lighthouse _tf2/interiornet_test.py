# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test pretrained multiscale lighting volume prediction on our InteriorNet test set."""

import os
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import geometry.projector as pj
from mlv import MLV
import nets as nets
from nerfactor.util import io


rootdir = '/data/happily/SVL-nerf_data'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
checkpoint_dir =rootdir + "/deprecated data/lighthouse/model/"
data_dir = rootdir + "/deprecated data/lighthouse/testset/"
output_dir = rootdir + "/deprecated data/lighthouse/output/"

flags.DEFINE_string("checkpoint_dir", default=checkpoint_dir, help="Directory for loading checkpoint")
flags.DEFINE_string("data_dir", default=data_dir, help="InteriorNet test dataset directory")
flags.DEFINE_string("output_dir", default=output_dir, help="Output directory to save images")

FLAGS = flags.FLAGS

# Model parameters.
batch_size = 1  # implementation only works for batch size 1 currently.
height = 240  # px
width = 320  # px
env_height = 120  # px
env_width = 240  # px
cube_res = 64  # px
theta_res = 240  # px
phi_res = 120  # px
r_res = 128  # px
scale_factors = [2, 4, 8, 16]
num_planes = 32
depth_clip = 20.0


def main(argv):
    del argv  # unused

    if FLAGS.checkpoint_dir is None:
        raise ValueError("`checkpoint_dir` must be defined")
    if FLAGS.data_dir is None:
        raise ValueError("`data_dir` must be defined")
    if FLAGS.output_dir is None:
        raise ValueError("`output_dir` must be defined")

    # Set up model
    model = MLV()

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    input_files = sorted([f for f in os.listdir(FLAGS.data_dir) if f.endswith(".npz")])
    print("found {:05d} input files".format(len(input_files)))

    for i in range(0, len(input_files)):
        print("running example:", i)

        # Load inputs
        batch = np.load(FLAGS.data_dir + input_files[i])
        ref_image = batch["ref_image"]
        ref_depth = batch["ref_depth"]
        intrinsics = batch["intrinsics"]
        ref_pose = batch["ref_pose"]
        src_images = batch["src_images"]
        src_poses = batch["src_poses"]
        env_pose = batch["env_pose"]

        # We use the true depth bounds for testing
        # Adjust to estimated bounds for your dataset
        min_depth = tf.reduce_min(ref_depth)
        max_depth = tf.reduce_max(ref_depth)
        mpi_planes = pj.inv_depths(min_depth, max_depth, num_planes)
        pred = model.infer_mpi(src_images, ref_image, ref_pose, src_poses, intrinsics, mpi_planes)
        rgba_layers = pred["rgba_layers"]

        lightvols, lightvol_centers, lightvol_side_lengths, cube_rel_shapes, cube_nest_inds = model.predict_lighting_vol(
            rgba_layers, mpi_planes, intrinsics, cube_res, scale_factors, depth_clip=depth_clip)
        lightvols_out = nets.cube_net_multires(lightvols, cube_rel_shapes, cube_nest_inds)

        z_list = np.linspace(1.0, 3.5, num=80)
        for z in z_list:
            # env_pose[0,2, -1] = z
            output_envmap, _ = model.render_envmap(lightvols_out, lightvol_centers, lightvol_side_lengths, cube_rel_shapes,
                                                   cube_nest_inds, ref_pose, env_pose, theta_res, phi_res, r_res)

            output_envmap_eval = output_envmap.numpy()
            # Write environment map image
            plt.imsave(os.path.join(FLAGS.output_dir, f"{z:.2f}_{i:05d}.png"), output_envmap_eval[0, :, :, :3])


if __name__ == "__main__":
    app.run(main)