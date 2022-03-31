"""Test pretrained multiscale lighting volume prediction on our InteriorNet test set."""

import os
from absl import app
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import geometry.projector as pj
from nerfactor.models.MPI import MPI
from nerfactor.models.MLV import MLV
import nets as nets
from nerfactor.util import io as ioutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_dir = '/home/vig-titan2/Data/SVL-nerf_data/deprecated data/lighthouse/testset/'
output_dir = '/home/vig-titan2/Data/SVL-nerf_data/deprecated data/lighthouse/output/'
mlv_checkpoint_dir = '/home/vig-titan2/Data/SVL-nerf_data/Shared model/mlv_checkpoints/ckpt-1'
mpi_checkpoint_dir = '/home/vig-titan2/Data/SVL-nerf_data/Shared model/mpi_checkpoints/ckpt-1'

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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    print("found {:05d} input files".format(len(input_files)))

    model_mpi = MPI()
    # optimizer = tf.keras.optimizers.Adam()
    # ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model_mpi)
    # ckptmanager = tf.train.CheckpointManager(ckpt, ckptdir_mpi, max_to_keep=4)
    ioutil.restore_model_v2(model_mpi, mpi_checkpoint_dir)

    model_mlv = MLV()
    # optimizer = tf.keras.optimizers.Adam()
    # ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model_mlv)
    # ckptmanager = tf.train.CheckpointManager(ckpt, ckptdir_mlv, max_to_keep=4)
    ioutil.restore_model_v2(model_mlv, mlv_checkpoint_dir)

    for i in range(0, len(input_files)):
        print("running example:", i)

        # Load inputs
        batch = np.load(data_dir + input_files[i])
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

        net_input = (ref_image, src_images, ref_pose, src_poses, mpi_planes, intrinsics)
        pred = model_mpi(net_input)
        # for j in range(psv_input.shape[-2]):
        # plt.imsave(os.path.join(FLAGS.output_dir, "aaa__tmp.png".format(j)),
        #            net_tmp[0, :, :, j, :3].numpy())
        # plt.imsave(os.path.join(FLAGS.output_dir, "aaa__ref_{:05d}.png".format(j)),
        #            psv_input[0, :, :, j, :3].numpy())
        # plt.imsave(os.path.join(FLAGS.output_dir, "aaa__psv_{:05d}.png".format(j)),
        #            psv_input[0, :, :, j, 3:].numpy())
        # plt.imsave(os.path.join(FLAGS.output_dir, "aaa__mpi_{:05d}.png".format(j)), pred[0, :, :, j, :].numpy())

        net_input = (pred, mpi_planes, intrinsics, cube_res, scale_factors, depth_clip, ref_pose, env_pose)
        env_map = model_mlv(net_input, mode='mpi')

        # model_mlv.load_weights_from_numpy('mlv_weight.npz')
        # model_mpi.load_weights_from_numpy('mpi_weight.npz')
        # saved_path = ckptmanager.save()
        # print("Checkpointed:\n\t%s", saved_path)

        output_envmap_eval = env_map.numpy()
        plt.imsave(os.path.join(output_dir, "{:05d}.png".format(i)), output_envmap_eval[0, :, :, :3])


if __name__ == "__main__":
    app.run(main)
