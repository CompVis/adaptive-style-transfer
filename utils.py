# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import division
import math
import scipy.misc
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from ops import *
import random
import copy


def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save

    Returns:

    """
    def batch_to_img(batch):
        return np.reshape(batch,
                          newshape=(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3]))

    inputs = np.concatenate([batch_to_img(input_painting_batch), batch_to_img(input_photo_batch)],
                            axis=0)
    outputs = np.concatenate([batch_to_img(output_painting_batch), batch_to_img(output_photo_batch)],
                             axis=0)

    to_save = np.concatenate([inputs,outputs], axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)

    scipy.misc.imsave(filepath, arr=to_save)


def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr/127.5 - 1.
    # return (arr - np.mean(arr)) / np.std(arr)


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5
