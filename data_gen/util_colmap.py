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

from io import BytesIO
from os.path import basename
import numpy as np

from third_party.xiuminglib import xiuminglib as xm


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, down, pos):
    vec2 = normalize(z)
    vec1_avg = down
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[:, :3, -1:].mean(0)

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    down = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, down, center), hwf], 1)

    return c2w


def render_path_spiral_world(c2w, down, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.0]) * rads)
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.0])) - c)
        render_poses.append(np.concatenate([viewmatrix(z, down, c), hwf], 1))
    return render_poses
