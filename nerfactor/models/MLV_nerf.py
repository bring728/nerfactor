import tensorflow as tf
import numpy as np
from nerfactor.models.base import Model as BaseModel
from nerfactor.util import logging as logutil, math as mathutil, io as ioutil, img as imgutil, config as configutil
from nerfactor.util.mlv_utils import render_envmap, trilerp_gather
from nerfactor import models


class MLV(BaseModel):
    def __init__(self, bg_ckpt, theta_res=240, phi_res=120, r_res=128, bg_net=None):
        super().__init__(None)
        self.net = {}
        self.initializer = tf.initializers.GlorotUniform()
        self.act = tf.nn.relu

        self._init_net()

        bg_config_ini = configutil.get_config_ini(bg_ckpt)
        bg_config = ioutil.read_config(bg_config_ini)
        self.ndc = bg_config.getboolean('DEFAULT', 'ndc')
        if self.ndc:
            self.bg_near = 0.0
            self.bg_far = 1.0
        else:
            self.bg_near = bg_config.getfloat('DEFAULT', 'near')
            self.bg_far = bg_config.getfloat('DEFAULT', 'far')

        if bg_net is None:
            model_name = bg_config.get('DEFAULT', 'model')
            Model = models.get_model_class(model_name)
            self.bg_net = Model(bg_config, False)
            ioutil.restore_model(self.bg_net, bg_ckpt)
            self.bg_net.trainable = False
        else:
            self.bg_net = bg_net

        self.ref_pose = np.identity(4).astype(np.float32)
        self.theta_res = theta_res  # px
        self.phi_res = phi_res  # px
        self.r_res = r_res  # px
        self.cube = {}
        self._nerf_to_cube()
        self.is_mlv_created = False

    def _init_net(self):
        norm = tf.keras.layers.LayerNormalization

        for i in range(5):
            width_list = [8, 16, 32, 64, 128]
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm'] = norm(axis=[1, 2, 3, 4], epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_1'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_1'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_2'] = self.conv(width_list[1], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_2'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_3'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_3'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_4'] = self.conv(width_list[1])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_4'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_5'] = self.conv(width_list[2], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_5'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_6'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_6'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_7'] = self.conv(width_list[2])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_7'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_8'] = self.conv(width_list[3], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_8'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_9'] = self.conv(width_list[3])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_9'] = norm(axis=[1, 2, 3, 4],
                                                                                          epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_10'] = self.conv(width_list[3])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_10'] = norm(axis=[1, 2, 3, 4],
                                                                                           epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_11'] = self.conv(width_list[4], strides=2)
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_11'] = norm(axis=[1, 2, 3, 4],
                                                                                           epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_12'] = self.conv(width_list[4])
            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_12'] = norm(axis=[1, 2, 3, 4],
                                                                                           epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_13'] = self.conv(width_list[4])

            width_list = [64, 32, 16, 8]
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm'] = norm(axis=[1, 2, 3, 4],
                                                                                                  epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d'] = self.conv(width_list[0])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_1'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_1'] = self.conv(width_list[0])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/LayerNorm_2'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block0/conv3d_2'] = self.conv(width_list[0])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm'] = norm(axis=[1, 2, 3, 4],
                                                                                                  epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_1'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_1'] = self.conv(width_list[1])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/LayerNorm_2'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block1/conv3d_2'] = self.conv(width_list[1])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm'] = norm(axis=[1, 2, 3, 4],
                                                                                                  epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_1'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_1'] = self.conv(width_list[2])
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/LayerNorm_2'] = norm(axis=[1, 2, 3, 4],
                                                                                                    epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block2/conv3d_2'] = self.conv(width_list[2])

            self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/LayerNorm'] = norm(axis=[1, 2, 3, 4],
                                                                                                  epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/up_block3/conv3d'] = self.conv(width_list[3])

            self.net['cube_net_multires/multires_level' + str(i) + '/LayerNorm_13'] = norm(axis=[1, 2, 3, 4],
                                                                                           epsilon=0.0)
            self.net['cube_net_multires/multires_level' + str(i) + '/conv3d_14'] = self.conv(5)

    def _nerf_to_cube(self):
        scale_factors = [2, 4, 8, 16]
        cube_res = 64  # px
        depth_clip = 20.0
        max_depth = tf.minimum(self.bg_net.far - self.bg_net.near, depth_clip)

        cube_side_lengths = [2.0 * max_depth]
        for i in range(len(scale_factors)):
            cube_side_lengths.append(2.0 * max_depth / tf.cast(scale_factors[i], tf.float32))

        # shape of each cube's footprint within the next coarser volume
        cube_rel_shapes = []
        for i in range(len(scale_factors)):
            if i == 0:
                i_rel_shape = cube_res // scale_factors[0]
            else:
                i_rel_shape = (cube_res * scale_factors[i - 1]) // scale_factors[i]
            cube_rel_shapes.append(i_rel_shape)

        cube_centers = [tf.zeros([3])]
        for i in range(len(scale_factors)):
            i_center_depth = (cube_side_lengths[i] / tf.cast((cube_res - 1), tf.float32)) * tf.cast(
                (cube_rel_shapes[i] // 2), tf.float32)
            cube_centers.append(tf.concat([tf.zeros(2), i_center_depth * tf.ones(1)], axis=0))

        cube_nest_inds = []
        for i in range(len(scale_factors)):
            if i == 0:
                i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2, (cube_res - cube_rel_shapes[i]) // 2,
                               cube_res // 2 - cube_rel_shapes[i]]
            else:
                i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2, (cube_res - cube_rel_shapes[i]) // 2,
                               cube_res - cube_rel_shapes[i]]
            cube_nest_inds.append(i_nest_inds)

        cube_list = []
        for i in range(len(cube_centers)):
            i_cube = self.nerf_resample_cube(cube_centers[i], cube_side_lengths[i], cube_res)
            cube_list.append(i_cube)

        self.cube['cube_centers'] = cube_centers
        self.cube['cube_side_lengths'] = cube_side_lengths
        self.cube['cube_rel_shapes'] = cube_rel_shapes
        self.cube['cube_nest_inds'] = cube_nest_inds
        self.cube['cube_list'] = cube_list
        return

    def nerf_resample_cube(self, cube_centers, side_length, cube_res):
        ndc = self.bg_net.ndc
        near = self.bg_net.near
        # create cube coordinates
        x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
        if ndc:
            y_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, cube_res)
        else:
            y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
        z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, cube_res)
        x, y, z = tf.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

        x = x + cube_centers[0, tf.newaxis, tf.newaxis, tf.newaxis]
        y = y + cube_centers[1, tf.newaxis, tf.newaxis, tf.newaxis]
        z = z + cube_centers[2, tf.newaxis, tf.newaxis, tf.newaxis] + near

        ind_valid = tf.where(z > near)
        x_valid = tf.gather_nd(x, ind_valid)
        y_valid = tf.gather_nd(y, ind_valid)
        if ndc:
            z_valid = tf.gather_nd(z, ind_valid) * 2 - 1
        else:
            z_valid = tf.gather_nd(z, ind_valid)
        coords = tf.stack([x_valid, y_valid, z_valid], axis=-1)

        # not using in NDC
        view_dir = tf.linalg.l2_normalize(coords, axis=-1)
        view_dir = tf.reshape(view_dir, (-1, 3))

        # coords = tf.reshape(coords, (-1, 3))
        voxel_size = side_length / cube_res
        voxel_size = tf.expand_dims(voxel_size, axis=0)
        diag_cov = voxel_size / 100.0
        voxel_flat = self.bg_net.mlv_call(coords, view_dir, diag_cov, voxel_size)
        voxel_grid = tf.scatter_nd(ind_valid, voxel_flat, [cube_res, cube_res, cube_res, 4])

        voxel_grid = voxel_grid[tf.newaxis, ...]
        return voxel_grid

    def cube_to_mlv(self, hallucinate):
        cube_rel_shapes = self.cube['cube_rel_shapes']
        cube_nest_inds = self.cube['cube_nest_inds']
        cube_list = self.cube['cube_list']
        if not hallucinate:
            self.mlv = cube_list
            self.is_mlv_created = True
            return

        outvols = []
        i_outvol_next = None
        for i in range(len(cube_list)):
            if i == 0:
                i_input = tf.stop_gradient(cube_list[0])
            else:
                i_input = tf.concat([tf.stop_gradient(cube_list[i]), i_outvol_next], axis=-1)

            i_net_out = self.unet(i_input, i)
            i_outvol_weights = tf.nn.sigmoid(i_net_out[Ellipsis, -1:])
            i_outvol = tf.nn.sigmoid(i_net_out[Ellipsis, :-1]) * i_outvol_weights + cube_list[i] * (
                    1.0 - i_outvol_weights)
            outvols.append(i_outvol)

            if i < len(cube_list) - 1:
                # slice and upsample region of volume corresponding to next finer resolution level
                i_outvol_next = i_outvol[:, cube_nest_inds[i][0]:cube_nest_inds[i][0] + cube_rel_shapes[i],
                                cube_nest_inds[i][1]:cube_nest_inds[i][1] + cube_rel_shapes[i],
                                cube_nest_inds[i][2]:cube_nest_inds[i][2] + cube_rel_shapes[i], :]
                i_outvol_next = self.tf_repeat(i_outvol_next, [1, tf.shape(i_input)[1] // tf.shape(i_outvol_next)[1],
                                                               tf.shape(i_input)[2] // tf.shape(i_outvol_next)[2],
                                                               tf.shape(i_input)[3] // tf.shape(i_outvol_next)[3], 1])
                i_outvol_next.set_shape([None, None, None, None, 4])

        self.mlv = outvols
        self.is_mlv_created = True
        return

    def call(self, env_pose, mode='train'):
        if not self.is_mlv_created:
            print('mlv not created!')
            return None

        env_map, _ = render_envmap(self.mlv, self.cube['cube_centers'], self.cube['cube_side_lengths'],
                                   self.cube['cube_rel_shapes'], self.cube['cube_nest_inds'],
                                   self.ref_pose, env_pose, self.phi_res, self.theta_res, self.r_res)
        return env_map

    def nerfactor_call(self, env_pose):
        if not self.is_mlv_created:
            print('mlv not created!')
            return None

        env_map, _ = render_envmap(self.mlv, self.cube['cube_centers'], self.cube['cube_side_lengths'],
                                   self.cube['cube_rel_shapes'], self.cube['cube_nest_inds'],
                                   self.ref_pose, env_pose, self.phi_res, self.theta_res, self.r_res)
        return env_map

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
