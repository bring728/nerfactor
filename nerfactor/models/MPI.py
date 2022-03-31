import tensorflow as tf
import numpy as np
from nerfactor.models.base import Model as BaseModel
from geometry.utils import format_network_input
import os


class MPI(BaseModel):
    def __init__(self):
        super().__init__(None)
        self.net = {}
        self.initializer = tf.initializers.GlorotUniform()
        self.act = tf.nn.relu

        self._init_net()

    def _init_net(self):
        norm = tf.keras.layers.LayerNormalization
        width_list = [8, 16, 32, 64, 128]
        self.net['mpi_net/conv3d'] = self.conv(width_list[0])

        self.net['mpi_net/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_1'] = self.conv(width_list[0])

        self.net['mpi_net/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_2'] = self.conv(width_list[1], strides=2)
        self.net['mpi_net/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_3'] = self.conv(width_list[1])
        self.net['mpi_net/LayerNorm_3'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_4'] = self.conv(width_list[1])

        self.net['mpi_net/LayerNorm_4'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_5'] = self.conv(width_list[2], strides=2)
        self.net['mpi_net/LayerNorm_5'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_6'] = self.conv(width_list[2])
        self.net['mpi_net/LayerNorm_6'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_7'] = self.conv(width_list[2])

        self.net['mpi_net/LayerNorm_7'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_8'] = self.conv(width_list[3], strides=2)
        self.net['mpi_net/LayerNorm_8'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_9'] = self.conv(width_list[3])
        self.net['mpi_net/LayerNorm_9'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_10'] = self.conv(width_list[3])

        self.net['mpi_net/LayerNorm_10'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_11'] = self.conv(width_list[4], strides=2)
        self.net['mpi_net/LayerNorm_11'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_12'] = self.conv(width_list[4])
        self.net['mpi_net/LayerNorm_12'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_13'] = self.conv(width_list[4])

        width_list = [64, 32, 16, 8]
        self.net['mpi_net/up_block0/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block0/conv3d'] = self.conv(width_list[0])
        self.net['mpi_net/up_block0/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block0/conv3d_1'] = self.conv(width_list[0])
        self.net['mpi_net/up_block0/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block0/conv3d_2'] = self.conv(width_list[0])

        self.net['mpi_net/up_block1/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block1/conv3d'] = self.conv(width_list[1])
        self.net['mpi_net/up_block1/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block1/conv3d_1'] = self.conv(width_list[1])
        self.net['mpi_net/up_block1/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block1/conv3d_2'] = self.conv(width_list[1])

        self.net['mpi_net/up_block2/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block2/conv3d'] = self.conv(width_list[2])
        self.net['mpi_net/up_block2/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block2/conv3d_1'] = self.conv(width_list[2])
        self.net['mpi_net/up_block2/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block2/conv3d_2'] = self.conv(width_list[2])

        self.net['mpi_net/up_block3/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/up_block3/conv3d'] = self.conv(width_list[3])

        self.net['mpi_net/LayerNorm_13'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
        self.net['mpi_net/conv3d_14'] = self.conv(5)

    def call(self, batch, mode='train'):
        ref_image, src_images, ref_pose, src_poses, mpi_planes, intrinsics = batch  # all flattened
        net_input = format_network_input(ref_image, src_images, ref_pose, src_poses, mpi_planes, intrinsics)

        skips = []
        net = net_input
        net = self.net['mpi_net/conv3d'](net)
        net = self.net['mpi_net/conv3d_1'](self.act(self.net['mpi_net/LayerNorm'](net)))
        skips.append(net)

        out = self.net['mpi_net/conv3d_2'](self.act(self.net['mpi_net/LayerNorm_1'](net)))
        net = self.net['mpi_net/conv3d_3'](self.act(self.net['mpi_net/LayerNorm_2'](out)))
        net = self.net['mpi_net/conv3d_4'](self.act(self.net['mpi_net/LayerNorm_3'](net)))
        net = out + net
        skips.append(net)

        out = self.net['mpi_net/conv3d_5'](self.act(self.net['mpi_net/LayerNorm_4'](net)))
        net = self.net['mpi_net/conv3d_6'](self.act(self.net['mpi_net/LayerNorm_5'](out)))
        net = self.net['mpi_net/conv3d_7'](self.act(self.net['mpi_net/LayerNorm_6'](net)))
        net = out + net
        skips.append(net)

        out = self.net['mpi_net/conv3d_8'](self.act(self.net['mpi_net/LayerNorm_7'](net)))
        net = self.net['mpi_net/conv3d_9'](self.act(self.net['mpi_net/LayerNorm_8'](out)))
        net = self.net['mpi_net/conv3d_10'](self.act(self.net['mpi_net/LayerNorm_9'](net)))
        net = out + net
        skips.append(net)

        out = self.net['mpi_net/conv3d_11'](self.act(self.net['mpi_net/LayerNorm_10'](net)))
        net = self.net['mpi_net/conv3d_12'](self.act(self.net['mpi_net/LayerNorm_11'](out)))
        net = self.net['mpi_net/conv3d_13'](self.act(self.net['mpi_net/LayerNorm_12'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['mpi_net/up_block0/conv3d'](self.act(self.net['mpi_net/up_block0/LayerNorm'](net)))
        net = self.net['mpi_net/up_block0/conv3d_1'](self.act(self.net['mpi_net/up_block0/LayerNorm_1'](out)))
        net = self.net['mpi_net/up_block0/conv3d_2'](self.act(self.net['mpi_net/up_block0/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['mpi_net/up_block1/conv3d'](self.act(self.net['mpi_net/up_block1/LayerNorm'](net)))
        net = self.net['mpi_net/up_block1/conv3d_1'](self.act(self.net['mpi_net/up_block1/LayerNorm_1'](out)))
        net = self.net['mpi_net/up_block1/conv3d_2'](self.act(self.net['mpi_net/up_block1/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['mpi_net/up_block2/conv3d'](self.act(self.net['mpi_net/up_block2/LayerNorm'](net)))
        net = self.net['mpi_net/up_block2/conv3d_1'](self.act(self.net['mpi_net/up_block2/LayerNorm_1'](out)))
        net = self.net['mpi_net/up_block2/conv3d_2'](self.act(self.net['mpi_net/up_block2/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        net = self.net['mpi_net/up_block3/conv3d'](self.act(self.net['mpi_net/up_block3/LayerNorm'](net)))

        net = self.net['mpi_net/conv3d_14'](self.act(self.net['mpi_net/LayerNorm_13'](net)))

        rgb_bg = tf.reduce_mean(tf.nn.sigmoid(net[Ellipsis, :3]), axis=3, keepdims=True)
        weights = tf.nn.sigmoid(net[Ellipsis, 3:4])
        alpha = tf.nn.sigmoid(net[Ellipsis, -1:])

        ref = net_input[Ellipsis, 0:3]
        rgb = weights * ref + (1.0 - weights) * rgb_bg

        mpi = tf.concat([rgb, alpha], axis=4)

        return mpi

    def load_weights_from_numpy(self, weight_npz):
        weight = dict(np.load(weight_npz))
        for net_name, net in self.net.items():
            if 'conv3d' in net_name:
                bias = weight[net_name + '/bias:0']
                kernel = weight[net_name + '/kernel:0']
                net.set_weights([kernel, bias])
            elif 'Norm' in net_name:
                gamma = weight[net_name + '/gamma:0']
                beta = weight[net_name + '/beta:0']
                gamma = np.tile(gamma, list(net.get_weights()[0].shape[:-1]) + [1])
                beta = np.tile(beta, list(net.get_weights()[0].shape[:-1]) + [1])
                net.set_weights([gamma, beta])

    def conv(self, width, ksize=3, strides=1, d=1):
        """3D conv helper function."""

        return tf.keras.layers.Conv3D(width, ksize, strides=strides, padding='SAME', dilation_rate=(d, d, d),
                                      activation=None, kernel_initializer=self.initializer)

    def tf_repeat(self, tensor, repeats):
        """Nearest neighbor upsampling."""
        # from https://github.com/tensorflow/tensorflow/issues/8246
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tensor

    def up_block(self, net, skip):
        ch = net.get_shape().as_list()[-1]
        net_repeat = self.tf_repeat(net, [1, 2, 2, 2, 1])
        net_repeat.set_shape([None, None, None, None, ch])
        up = net_repeat
        up = tf.cond(tf.equal(tf.shape(up)[1], tf.shape(skip)[1]), lambda: up, lambda: up[:, :-1, Ellipsis])
        up = tf.cond(tf.equal(tf.shape(up)[2], tf.shape(skip)[2]), lambda: up, lambda: up[:, :, :-1, Ellipsis])
        out = tf.concat([up, skip], -1)
        return out
