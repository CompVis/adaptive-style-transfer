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

# pylint: disable=missing-docstring
from __future__ import division
import os.path
from os.path import join
import time

import tensorflow as tf
import numpy as np
import scipy.stats.mstats as stat
import scipy.spatial.distance as spdis
import sklearn
import sklearn.preprocessing
import scipy.io
import math
from tqdm import tqdm


class MockNet(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_net_with_graph(snapshot_path, gpu_memory_fraction=None,
                        **kwargs):

    graph_path = snapshot_path + '.meta'
    if not os.path.exists(graph_path):
        raise IOError('Graph meta file not found: {}'.format(graph_path))
    if not os.path.exists(snapshot_path):
        raise IOError('Snapshot file not found: {}'.format(snapshot_path))

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    if gpu_memory_fraction is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        new_saver = tf.train.import_meta_graph(graph_path)
        new_saver.restore(sess, snapshot_path)

        net = MockNet(sess=sess, graph=graph)
        for layer_name, tensor_name in kwargs.iteritems():
            net.__dict__[layer_name] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        return net


def extract_features(flipped, net=None, frame_ids=None, layer_names=None,
                     image_getter=None,
                     is_batch_iterator=False,
                     im_shape=(227, 227), batch_size=128, mean=None,
                     verbose=2, should_reshape_vectors=True, input_pl_name='input/x',
                     is_phase_train_pl_name='input/is_phase_train',
                     **params):
    """
    Extract features from the network.

    Args:
        flipped: are frames flipped?
        net: Network object with fields:
             sess - tensorflow session,
             fields corresponding to layers with names as elements of $layer_names;
          if not None it will use this network to extract the features,
          otherwise create new Alexnet net and restore from the snapshot (snapshot_path).
        frame_ids: if None extract from all frames,
          if a list of frames - extract features only for them
        layer_names: list of layer names to use for extraction
        image_getter: image getter object from eval/image_getter.py
        is_batch_iterator: if image_getter is a batch iterator
        im_shape: retrieve from images of this shape from $image_getter
        batch_size: batch size to use for feature extraction
        mean: mean image, must be (h, w, 3) in HxWxC RGB
        verbose: verbosity level from 0 to 100
        input_pl_name: input placeholder name
        is_phase_train_pl_name: name of is_phase_train placeholder,
          If not None the placeholder will be replaced with False (not training),
          If None than not use this placeholder in feed dict.
        should_reshape_vectors: reshape features to the plain 1xD vectors
        params: optional parameters for net if net=None:
                    snapshot_path,
                    number_layers_restore

    """
    if not isinstance(layer_names, list):
        raise TypeError('layer_names must be a list')
    if image_getter is None:
        raise ValueError('image_getter must be provided')

    if net is None:
        if 'number_layers_restore' not in params:
            raise KeyError('number_layers_restore must be in params if net is None')
        raise ValueError('Net must be not None')
    else:
        is_temp_session = False

    tensors_to_get = [net.__getattribute__(name) for name in layer_names]
    if verbose >= 2:
        print 'Tensors to extract:', tensors_to_get
    d = dict()
    if frame_ids is None:
        if hasattr(image_getter, 'total_num_images'):
            frame_ids = np.arange(image_getter.total_num_images())
        else:
            frame_ids = np.arange(len(image_getter.dataset))

    for layer_name in layer_names:
        tensor = net.__getattribute__(layer_name)
        d[layer_name] = np.zeros([len(frame_ids)] + tensor.get_shape().as_list()[1:], dtype=np.float32)

    if is_batch_iterator:
        d = extract_features_from_batch_iterator(net, image_getter, tensors_to_get,
                                                layer_names, d,
                                                input_pl_name=input_pl_name,
                                                is_phase_train_pl_name=is_phase_train_pl_name,
                                                verbose=verbose)
    else:
        num_batches = int(math.ceil(len(frame_ids) / batch_size))
        if verbose >= 2:
            print 'Running {} iterations with batch_size={}'.format(num_batches, batch_size)
        for step, batch_start in tqdm(enumerate(range(0, len(frame_ids), batch_size)),
                                      total=num_batches, disable=(verbose == 0)):
            batch_idxs = frame_ids[batch_start:batch_start + batch_size]
            batch = image_getter.get_batch(batch_idxs, resize_shape=im_shape,
                                                       mean=mean)
            if flipped:
                batch = batch[:, :, ::-1, :]

            feed_dict = {input_pl_name + ':0': batch}
            if is_phase_train_pl_name is not None:
                feed_dict[is_phase_train_pl_name + ':0'] = False

            features = net.sess.run(tensors_to_get, feed_dict=feed_dict)
            pos_begin = batch_size * step
            pos_end = pos_begin + len(batch_idxs)
            for tensor_id, layer_name in enumerate(layer_names):
                d[layer_name][pos_begin:pos_end, ...] = features[tensor_id]

    if should_reshape_vectors:
        for layer_name in layer_names:
            d[layer_name] = d[layer_name].reshape(len(frame_ids), -1)

    if is_temp_session:
        net.sess.close()
    return d


def extract_features_from_batch_iterator(net, batch_iterator, tensors_to_get,
                                         layer_names, out_features_dict, input_pl_name='input/x',
                                         is_phase_train_pl_name='input/is_phase_train',
                                         verbose=2):
    """
    Extract features iterating by batches using batch_iterator.
    Args:
        net:
        batch_iterator:
        tensors_to_get:
        layer_names:
        out_features_dict: dict to store features. Will be modified in-place.
        input_pl_name:
        is_phase_train_pl_name:
        num_batches:
        verbose:

    Returns: out_features_dict
    """
    if verbose >= 2:
        print 'Running {} iterations (from batch_iterator)'.format(len(batch_iterator))
    for step, batch in tqdm(enumerate(batch_iterator), total=len(batch_iterator), disable=(verbose == 0)):

        feed_dict = {input_pl_name + ':0': batch}
        if is_phase_train_pl_name is not None:
            feed_dict[is_phase_train_pl_name + ':0'] = False

        features = net.sess.run(tensors_to_get, feed_dict=feed_dict)
        batch_size = len(batch)
        pos_begin = batch_size * step
        pos_end = pos_begin + batch_size
        for tensor_id, layer_name in enumerate(layer_names):
            out_features_dict[layer_name][pos_begin:pos_end, ...] = features[tensor_id]
    return out_features_dict
