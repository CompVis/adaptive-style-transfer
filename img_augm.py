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

import numpy as np
import scipy.misc
import cv2
from PIL import Image


class Augmentor():
    def __init__(self,
                 crop_size=(256, 256),
                 scale_augm_prb=0.5, scale_augm_range=0.2,
                 rotation_augm_prb=0.5, rotation_augm_range=0.15,
                 hsv_augm_prb=1.0,
                 hue_augm_shift=0.05,
                 saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                 value_augm_shift=0.05, value_augm_scale=0.05,
                 affine_trnsfm_prb=0.5, affine_trnsfm_range=0.05,
                 horizontal_flip_prb=0.5,
                 vertical_flip_prb=0.5):

        self.crop_size = crop_size

        self.scale_augm_prb = scale_augm_prb
        self.scale_augm_range = scale_augm_range

        self.rotation_augm_prb = rotation_augm_prb
        self.rotation_augm_range = rotation_augm_range

        self.hsv_augm_prb = hsv_augm_prb
        self.hue_augm_shift = hue_augm_shift
        self.saturation_augm_scale = saturation_augm_scale
        self.saturation_augm_shift = saturation_augm_shift
        self.value_augm_scale = value_augm_scale
        self.value_augm_shift = value_augm_shift

        self.affine_trnsfm_prb = affine_trnsfm_prb
        self.affine_trnsfm_range = affine_trnsfm_range

        self.horizontal_flip_prb = horizontal_flip_prb
        self.vertical_flip_prb = vertical_flip_prb

    def __call__(self, image, is_inference=False):
        if is_inference:
            return cv2.resize(image, None, fx=self.crop_size[0], fy=self.crop_size[1], interpolation=cv2.INTER_CUBIC)

        # If not inference stage apply the pipeline of augmentations.
        if self.scale_augm_prb > np.random.uniform():
            image = self.scale(image=image,
                               scale_x=1. + np.random.uniform(low=-self.scale_augm_range, high=-self.scale_augm_range),
                               scale_y=1. + np.random.uniform(low=-self.scale_augm_range, high=-self.scale_augm_range)
                               )


        rows, cols, ch = image.shape
        image = np.pad(array=image, pad_width=[[rows // 4, rows // 4], [cols // 4, cols // 4], [0, 0]], mode='reflect')
        if self.rotation_augm_prb > np.random.uniform():
            image = self.rotate(image=image,
                                angle=np.random.uniform(low=-self.rotation_augm_range*90.,
                                                        high=self.rotation_augm_range*90.)
                                )

        if self.affine_trnsfm_prb > np.random.uniform():
            image = self.affine(image=image,
                                rng=self.affine_trnsfm_range
                                )
        image = image[(rows // 4):-(rows // 4), (cols // 4):-(cols // 4), :]

        # Crop out patch of desired size.
        image = self.crop(image=image,
                          crop_size=self.crop_size
                          )

        if self.hsv_augm_prb > np.random.uniform():
            image = self.hsv_transform(image=image,
                                       hue_shift=self.hue_augm_shift,
                                       saturation_shift=self.saturation_augm_shift,
                                       saturation_scale=self.saturation_augm_scale,
                                       value_shift=self.value_augm_shift,
                                       value_scale=self.value_augm_scale)

        if self.horizontal_flip_prb > np.random.uniform():
            image = self.horizontal_flip(image)

        if self.vertical_flip_prb > np.random.uniform():
            image = self.vertical_flip(image)

        return image

    def scale(self, image, scale_x, scale_y):
        """
        Args:
            image:
            scale_x: float positive value. New horizontal scale
            scale_y: float positive value. New vertical scale
        Returns:
        """
        image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        return image

    def rotate(self, image, angle):
        """
        Args:
            image: input image
            angle: angle of rotation in degrees
        Returns:
        """
        rows, cols, ch = image.shape

        rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rot_M, (cols, rows))
        return image

    def crop(self, image, crop_size=(256, 256)):
        rows, cols, chs = image.shape
        x = int(np.random.uniform(low=0, high=max(0, rows - crop_size[0])))
        y = int(np.random.uniform(low=0, high=max(0, cols - crop_size[1])))

        image = image[x:x+crop_size[0], y:y+crop_size[1], :]
        # If the input image was too small to comprise patch of size crop_size,
        # resize obtained patch to desired size.
        if image.shape[0] < crop_size[0] or image.shape[1] < crop_size[1]:
            image = scipy.misc.imresize(arr=image, size=crop_size)
        return image

    def hsv_transform(self, image,
                      hue_shift=0.2,
                      saturation_shift=0.2, saturation_scale=0.2,
                      value_shift=0.2, value_scale=0.2,
                      ):

        image = Image.fromarray(image)
        hsv = np.array(image.convert("HSV"), 'float64')

        # scale the values to fit between 0 and 1
        hsv /= 255.

        # do the scalings & shiftings
        hsv[..., 0] += np.random.uniform(-hue_shift, hue_shift)
        hsv[..., 1] *= np.random.uniform(1. / (1. + saturation_scale), 1. + saturation_scale)
        hsv[..., 1] += np.random.uniform(-saturation_shift, saturation_shift)
        hsv[..., 2] *= np.random.uniform(1. / (1. + value_scale), 1. + value_scale)
        hsv[..., 2] += np.random.uniform(-value_shift, value_shift)

        # cut off invalid values
        hsv.clip(0.01, 0.99, hsv)

        # round to full numbers
        hsv = np.uint8(np.round(hsv * 254.))

        # convert back to rgb image
        return np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))


    def affine(self, image, rng):
        rows, cols, ch = image.shape
        pts1 = np.float32([[0., 0.], [0., 1.], [1., 0.]])
        [x0, y0] = [0. + np.random.uniform(low=-rng, high=rng), 0. + np.random.uniform(low=-rng, high=rng)]
        [x1, y1] = [0. + np.random.uniform(low=-rng, high=rng), 1. + np.random.uniform(low=-rng, high=rng)]
        [x2, y2] = [1. + np.random.uniform(low=-rng, high=rng), 0. + np.random.uniform(low=-rng, high=rng)]
        pts2 = np.float32([[x0, y0], [x1, y1], [x2, y2]])
        affine_M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, affine_M, (cols, rows))

        return image

    def horizontal_flip(self, image):
        return image[:, ::-1, :]

    def vertical_flip(self, image):
        return image[::-1, :, :]

