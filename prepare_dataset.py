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

from __future__ import print_function
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import scipy.misc
import utils
import random


class ArtDataset():
    def __init__(self, path_to_art_dataset):

        self.dataset = [os.path.join(path_to_art_dataset, x) for x in os.listdir(path_to_art_dataset)]
        print("Art dataset contains %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
         also containing artist name, technique name, and period of creation for given artist.
         In case of content images we have only the 'path' column.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch
        Returns:
            dictionary with fields: image
        """

        batch_image = []

        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')

            if max(image.shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image.shape))
            if max(image.shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image.shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            if augmentor:
                batch_image.append(augmentor(image).astype(np.float32))
            else:
                batch_image.append((image).astype(np.float32))
        # Now return a batch in correct form
        batch_image = np.asarray(batch_image)

        return {"image": batch_image}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)


class PlacesDataset():
    categories_names = \
        ['/a/abbey', '/a/arch', '/a/amphitheater', '/a/aqueduct', '/a/arena/rodeo', '/a/athletic_field/outdoor',
         '/b/badlands', '/b/balcony/exterior', '/b/bamboo_forest', '/b/barn', '/b/barndoor', '/b/baseball_field',
         '/b/basilica', '/b/bayou', '/b/beach', '/b/beach_house', '/b/beer_garden', '/b/boardwalk', '/b/boathouse',
         '/b/botanical_garden', '/b/bullring', '/b/butte', '/c/cabin/outdoor', '/c/campsite', '/c/campus',
         '/c/canal/natural', '/c/canal/urban', '/c/canyon', '/c/castle', '/c/church/outdoor', '/c/chalet',
         '/c/cliff', '/c/coast', '/c/corn_field', '/c/corral', '/c/cottage', '/c/courtyard', '/c/crevasse',
         '/d/dam', '/d/desert/vegetation', '/d/desert_road', '/d/doorway/outdoor', '/f/farm', '/f/fairway',
         '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fishpond', '/f/florist_shop/indoor',
         '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road', '/f/formal_garden', '/g/gazebo/exterior',
         '/g/glacier', '/g/golf_course', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto', '/g/gorge',
         '/h/hayfield', '/h/herb_garden', '/h/hot_spring', '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe',
         '/i/ice_shelf', '/i/iceberg', '/i/inn/outdoor', '/i/islet', '/j/japanese_garden', '/k/kasbah',
         '/k/kennel/outdoor', '/l/lagoon', '/l/lake/natural', '/l/lawn', '/l/library/outdoor', '/l/lighthouse',
         '/m/mansion', '/m/marsh', '/m/mausoleum', '/m/moat/water', '/m/mosque/outdoor', '/m/mountain',
         '/m/mountain_path', '/m/mountain_snowy', '/o/oast_house', '/o/ocean', '/o/orchard', '/p/park',
         '/p/pasture', '/p/pavilion', '/p/picnic_area', '/p/pier', '/p/pond', '/r/raft', '/r/railroad_track',
         '/r/rainforest', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge',
         '/r/ruin', '/s/schoolhouse', '/s/sky', '/s/snowfield', '/s/swamp', '/s/swimming_hole',
         '/s/synagogue/outdoor', '/t/temple/asia', '/t/topiary_garden', '/t/tree_farm', '/t/tree_house',
         '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden', '/v/viaduct',
         '/v/village', '/v/vineyard', '/v/volcano', '/w/waterfall', '/w/watering_hole', '/w/wave',
         '/w/wheat_field', '/z/zen_garden', '/a/alcove', '/a/apartment-building/outdoor', '/a/artists_loft',
         '/b/building_facade', '/c/cemetery']
    categories_names = [x[1:] for x in categories_names]

    def __init__(self, path_to_dataset):
        self.dataset = []
        for category_idx, category_name in enumerate(tqdm(self.categories_names)):
            print(category_name, category_idx)
            if os.path.exists(os.path.join(path_to_dataset, category_name)):
                for file_name in tqdm(os.listdir(os.path.join(path_to_dataset, category_name))):
                    self.dataset.append(os.path.join(path_to_dataset, category_name, file_name))
            else:
                print("Category %s can't be found in path %s. Skip it." %
                      (category_name, os.path.join(path_to_dataset, category_name)))

        print("Finished. Constructed Places2 dataset of %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Generate bathes of images with attached labels(place category) in two different formats:
        textual and one-hot-encoded.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch we return
        Returns:
            dictionary with fields: image
        """

        batch_image = []
        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')
            image = scipy.misc.imresize(image, size=2.)
            image_shape = image.shape

            if max(image_shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image_shape))
            if max(image_shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image_shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            batch_image.append(augmentor(image).astype(np.float32))

        return {"image": np.asarray(batch_image)}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)




