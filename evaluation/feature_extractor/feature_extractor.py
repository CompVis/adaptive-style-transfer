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
import sklearn.preprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from features import extract_features
import image_getter

from nets import nets_factory
from preprocessing import preprocessing_factory

classifier_scope = {
    'inception_v1': 'InceptionV1/Logits',
    'vgg_16': 'vgg_16/fc8'
}


class SlimFeatureExtractor(object):
    def __init__(self, net_name, snapshot_path,
                 feature_norm_method=None,
                 should_restore_classifier=False, gpu_memory_fraction=None, vgg_16_heads=None):
        """
        Args:
            snapshot_path: path or dir with checkpoints
            feature_norm_method:
            should_restore_classifier: if None - do not restore last layer from the snapshot,
                         otherwise must be equal to the number of classes of the snapshot.
                         if vgg_16_heads is not None then the classifiers will be restored anyway.

        """
        self.net_name = net_name
        if net_name != 'vgg_16_multihead' and vgg_16_heads is not None:
            raise ValueError('vgg_16_heads must be not None only for vgg_16_multihead')
        if net_name == 'vgg_16_multihead' and vgg_16_heads is None:
            raise ValueError('vgg_16_heads must be not None for vgg_16_multihead')

        if tf.gfile.IsDirectory(snapshot_path):
            snapshot_path = tf.train.latest_checkpoint(snapshot_path)

        if not isinstance(feature_norm_method, list):
            feature_norm_method = [feature_norm_method]
        accepable_methods = [None, 'signed_sqrt', 'unit_norm']
        for method in feature_norm_method:
            if method not in accepable_methods:
                raise ValueError('unknown norm method: {}. Use one of {}'.format(method, accepable_methods))
        self.feature_norm_method = feature_norm_method
        if vgg_16_heads is not None:
            should_restore_classifier = True

        if should_restore_classifier:
            if vgg_16_heads is None:
                reader = pywrap_tensorflow.NewCheckpointReader(snapshot_path)
                if net_name == 'inception_v1':
                    var_value = reader.get_tensor('InceptionV1/Logits/Conv2d_0c_1x1/weights')
                else:
                    var_value = reader.get_tensor('vgg_16/fc8/weights')
                num_classes = var_value.shape[3]
            else:
                num_classes = vgg_16_heads
        else:
            num_classes = 2 if vgg_16_heads is None else vgg_16_heads

        network_fn = nets_factory.get_network_fn(net_name, num_classes=num_classes, is_training=False)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(net_name, is_training=False)

        eval_image_size = network_fn.default_image_size
        self.img_resize_shape = (eval_image_size, eval_image_size)  # (224, 224) for VGG

        with tf.Graph().as_default() as graph:
            self.graph = graph
            with tf.variable_scope('input'):
                input_pl = tf.placeholder(tf.float32, shape=[None,
                                                             eval_image_size,
                                                             eval_image_size, 3], name='x')
                # not used
                is_phase_train_pl = tf.placeholder(tf.bool, shape=tuple(), name='is_phase_train')

            function_to_map = lambda x: image_preprocessing_fn(x, eval_image_size, eval_image_size)
            images = tf.map_fn(function_to_map, input_pl)

            logits, self.end_points = network_fn(images)
            self.__dict__.update(self.end_points)
            if net_name == 'inception_v1':
                for tensor_name in ['Branch_0/Conv2d_0a_1x1',
                                    'Branch_1/Conv2d_0a_1x1', 'Branch_1/Conv2d_0b_3x3',
                                    'Branch_2/Conv2d_0a_1x1', 'Branch_2/Conv2d_0b_3x3',
                                    'Branch_3/MaxPool_0a_3x3', 'Branch_3/Conv2d_0b_1x1']:
                    full_tensor_name = 'InceptionV1/InceptionV1/Mixed_4d/' + tensor_name
                    if 'MaxPool' in tensor_name:
                        full_tensor_name += '/MaxPool:0'
                    else:
                        full_tensor_name += '/Relu:0'
                    short_name = 'Mixed_4d/' + tensor_name
                    self.__dict__[short_name] = tf.get_default_graph().get_tensor_by_name(full_tensor_name)
                self.MaxPool_0a_7x7 = tf.get_default_graph().get_tensor_by_name("InceptionV1/Logits/MaxPool_0a_7x7/AvgPool:0")
            elif net_name in ['vgg_16', 'vgg_16_multihead']:
                for layer_name in ['fc6', 'fc7'] + \
                        ['conv{0}/conv{0}_{1}'.format(i, j) for i in xrange(3, 6) for j in xrange(1, 4)]:
                    self.__dict__['vgg_16/{}_prerelu'.format(layer_name)] = \
                        tf.get_default_graph().get_tensor_by_name("vgg_16/{}/BiasAdd:0".format(layer_name))
            config = tf.ConfigProto(gpu_options=
                                    tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
            self.sess = tf.Session(config=config)

            if should_restore_classifier:
                variables_to_restore = slim.get_model_variables()
            else:
                variables_to_restore = [var for var in slim.get_model_variables()
                                        if not var.op.name.startswith(classifier_scope[net_name])]

            init_fn = slim.assign_from_checkpoint_fn(snapshot_path, variables_to_restore)
            init_fn(self.sess)

    def extract(self, image_paths, layer_names, flipped=False, batch_size=64,
                should_reshape_vectors=True, verbose=2, spatial_pool=None):
        """
        Extract features from the image
        """
        try:
            image_paths.__getattribute__('__len__')
        except AttributeError:
            raise TypeError('image_paths must be a container of paths')
        if len(self.feature_norm_method) > 1:
            raise NotImplementedError()
        if spatial_pool not in [None, 'max', 'sum']:
            raise ValueError('Unknown spatial pool: {}'.format(spatial_pool))
        if spatial_pool is not None:
            should_reshape_vectors = False
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        if len(layer_names) > 1 and not should_reshape_vectors:
            raise ValueError('Cannot stack features from several layers without reshaping')

        getter = image_getter.ImageGetterFromPaths(image_paths,
                                                   im_shape=self.img_resize_shape,
                                                   rgb_batch=True)

        feature_dict = extract_features(flipped, self,
                                          layer_names=layer_names,
                                          image_getter=getter,
                                          im_shape=self.img_resize_shape,
                                          mean=None,
                                          batch_size=batch_size,
                                          verbose=verbose,
                                          should_reshape_vectors=should_reshape_vectors)

        # feed to the net_stream augmented images anf pool features after
        features = np.hstack(feature_dict.values())
        if spatial_pool is not None and len(features.shape) != 4:
            raise ValueError('Cannot do a spatial pool on features with shape: {}'.format(
                features.shape))
        if spatial_pool == 'max':
            features = np.max(features, axis=(1, 2))
        elif spatial_pool == 'sum':
            features = np.sum(features, axis=(1, 2))

        # print 'features.shape={}'.format(features.shape)
        if 'unit_norm' in self.feature_norm_method:
            if not should_reshape_vectors:
                raise ValueError('Cannot do unit_norm without reshaping the vectors')
            sklearn.preprocessing.normalize(features, norm='l2', axis=1, copy=False)
        assert len(features) == len(image_paths)
        return features

