import tensorflow as tf
import numpy as np
from nerfactor.models.base import Model as BaseModel
from geometry.utils import predict_lighting_vol, render_envmap
import os


class MLV(BaseModel):
    def __init__(self):
        super().__init__(None)
        self.net = {}
        self.initializer = tf.initializers.GlorotUniform()
        self.act = tf.nn.relu

        self._init_net()

    def _init_net(self):
        norm = tf.keras.layers.LayerNormalization

        for i in range(5):
            width_list = [8, 16, 32, 64, 128]
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_1'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_2'] = self.conv(width_list[1], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_3'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_3'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_4'] = self.conv(width_list[1])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_4'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_5'] = self.conv(width_list[2], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_5'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_6'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_6'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_7'] = self.conv(width_list[2])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_7'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_8'] = self.conv(width_list[3], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_8'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_9'] = self.conv(width_list[3])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_9'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_10'] = self.conv(width_list[3])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_10'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_11'] = self.conv(width_list[4], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_11'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_12'] = self.conv(width_list[4])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_12'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_13'] = self.conv(width_list[4])

            width_list = [64, 32, 16, 8]
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d'] = self.conv(width_list[0])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_1'] = self.conv(width_list[0])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_2'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_1'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_2'] = self.conv(width_list[1])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_1'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_1'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_2'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_2'] = self.conv(width_list[2])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/conv3d'] = self.conv(width_list[3])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_13'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_14'] = self.conv(5)

    def call(self, batch, mode='train'):
        pred, mpi_planes, intrinsics, cube_res, scale_factors, depth_clip, ref_pose, env_pose = batch  # all flattened

        inputs, lightvol_centers, lightvol_side_lengths, cube_rel_shapes, cube_nest_inds = predict_lighting_vol(pred,
                                                                                                                mpi_planes,
                                                                                                                intrinsics,
                                                                                                                cube_res,
                                                                                                                scale_factors,
                                                                                                                depth_clip)

        chout = 4

        outvols = []
        i_outvol_next = None
        for i in range(len(inputs)):
            if i == 0:
                i_input = tf.stop_gradient(inputs[0])
            else:
                i_input = tf.concat([tf.stop_gradient(inputs[i]), i_outvol_next], axis=-1)

            i_net_out = self.unet(i_input, i)
            i_outvol_weights = tf.nn.sigmoid(i_net_out[Ellipsis, -1:])
            i_outvol = tf.nn.sigmoid(i_net_out[
                                     Ellipsis, :-1]) * i_outvol_weights + inputs[i] * (1.0 - i_outvol_weights)
            outvols.append(i_outvol)

            if i < len(inputs) - 1:
                # slice and upsample region of volume
                # corresponding to next finer resolution level
                i_outvol_next = i_outvol[:, cube_nest_inds[i][0]:cube_nest_inds[i][0] + cube_rel_shapes[i],
                                cube_nest_inds[i][1]:cube_nest_inds[i][1] + cube_rel_shapes[i],
                                cube_nest_inds[i][2]:cube_nest_inds[i][2] + cube_rel_shapes[i], :]
                i_outvol_next = self.tf_repeat(i_outvol_next, [1, tf.shape(i_input)[1] // tf.shape(i_outvol_next)[1],
                                                               tf.shape(i_input)[2] // tf.shape(i_outvol_next)[2],
                                                               tf.shape(i_input)[3] // tf.shape(i_outvol_next)[3], 1
                                                               ])
                i_outvol_next.set_shape([None, None, None, None, chout])

        theta_res = 240  # px
        phi_res = 120  # px
        r_res = 128  # px

        env_map, _ = render_envmap(outvols, lightvol_centers, lightvol_side_lengths, cube_rel_shapes, cube_nest_inds,
                                   ref_pose, env_pose, theta_res, phi_res, r_res)
        return env_map

    def load_weights_from_numpy(self, weight_npz):
        weight = dict(np.load(weight_npz))
        for net_name, net in self.net.items():
            # print(net_name, net)
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

    def unet(self, net, i):
        skips = []
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d'](net)
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_1'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm'](net)))
        skips.append(net)

        out = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_2'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_1'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_3'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_2'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_4'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_3'](net)))
        net = out + net
        skips.append(net)

        out = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_5'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_4'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_6'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_5'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_7'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_6'](net)))
        net = out + net
        skips.append(net)

        out = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_8'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_7'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_9'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_8'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_10'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_9'](net)))
        net = out + net
        skips.append(net)

        out = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_11'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_10'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_12'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_11'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_13'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_12'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_1'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_1'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_2'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_1'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_1'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_2'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        out = self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm'](net)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_1'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_1'](out)))
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_2'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_2'](net)))
        net = out + net

        skip = skips.pop()
        net = self.up_block(net, skip)
        net = self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/conv3d'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/LayerNorm'](net)))

        net = self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_14'](
            self.act(self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_13'](net)))

        return net
