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
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import multiprocessing

from module import *
from utils import *
import prepare_dataset
import img_augm


class Artgan(object):
    def __init__(self, sess, args):
        self.model_name = args.model_name
        self.root_dir = './models'
        self.checkpoint_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint')
        self.checkpoint_long_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint_long')
        self.sample_dir = os.path.join(self.root_dir, self.model_name, 'sample')
        self.inference_dir = os.path.join(self.root_dir, self.model_name, 'inference')
        self.logs_dir = os.path.join(self.root_dir, self.model_name, 'logs')

        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.image_size

        self.loss = sce_criterion

        self.initial_step = 0

        OPTIONS = namedtuple('OPTIONS',
                             'batch_size image_size \
                              total_steps save_freq lr\
                              gf_dim df_dim \
                              is_training \
                              path_to_content_dataset \
                              path_to_art_dataset \
                              discr_loss_weight transformer_loss_weight feature_loss_weight')
        self.options = OPTIONS._make((args.batch_size, args.image_size,
                                      args.total_steps, args.save_freq, args.lr,
                                      args.ngf, args.ndf,
                                      args.phase == 'train',
                                      args.path_to_content_dataset,
                                      args.path_to_art_dataset,
                                      args.discr_loss_weight, args.transformer_loss_weight, args.feature_loss_weight
                                      ))

        # Create all the folders for saving the model
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(os.path.join(self.root_dir, self.model_name)):
            os.makedirs(os.path.join(self.root_dir, self.model_name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_long_dir):
            os.makedirs(self.checkpoint_long_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.inference_dir):
            os.makedirs(self.inference_dir)

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.saver_long = tf.train.Saver(max_to_keep=None)

    def _build_model(self):
        if self.options.is_training:
            # ==================== Define placeholders. ===================== #
            with tf.name_scope('placeholder'):
                self.input_painting = tf.placeholder(dtype=tf.float32,
                                                     shape=[self.batch_size, None, None, 3],
                                                     name='painting')
                self.input_photo = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size, None, None, 3],
                                                  name='photo')
                self.lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

            # ===================== Wire the graph. ========================= #
            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                options=self.options,
                                                reuse=False)

            # Decode obtained features
            self.output_photo = decoder(features=self.input_photo_features,
                                        options=self.options,
                                        reuse=False)

            # Get features of output images. Need them to compute feature loss.
            self.output_photo_features = encoder(image=self.output_photo,
                                                 options=self.options,
                                                 reuse=True)

            # Add discriminators.
            # Note that each of the predictions contain multiple predictions
            # at different scale.
            self.input_painting_discr_predictions = discriminator(image=self.input_painting,
                                                                  options=self.options,
                                                                  reuse=False)
            self.input_photo_discr_predictions = discriminator(image=self.input_photo,
                                                               options=self.options,
                                                               reuse=True)
            self.output_photo_discr_predictions = discriminator(image=self.output_photo,
                                                                options=self.options,
                                                                reuse=True)

            # ===================== Final losses that we optimize. ===================== #

            # Discriminator.
            # Have to predict ones only for original paintings, otherwise predict zero.
            scale_weight = {"scale_0": 1.,
                            "scale_1": 1.,
                            "scale_3": 1.,
                            "scale_5": 1.,
                            "scale_6": 1.}
            self.input_painting_discr_loss = {key: self.loss(pred, tf.ones_like(pred)) * scale_weight[key]
                                              for key, pred in zip(self.input_painting_discr_predictions.keys(),
                                                                   self.input_painting_discr_predictions.values())}
            self.input_photo_discr_loss = {key: self.loss(pred, tf.zeros_like(pred)) * scale_weight[key]
                                           for key, pred in zip(self.input_photo_discr_predictions.keys(),
                                                                self.input_photo_discr_predictions.values())}
            self.output_photo_discr_loss = {key: self.loss(pred, tf.zeros_like(pred)) * scale_weight[key]
                                            for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                 self.output_photo_discr_predictions.values())}

            self.discr_loss = tf.add_n(list(self.input_painting_discr_loss.values())) + \
                              tf.add_n(list(self.input_photo_discr_loss.values())) + \
                              tf.add_n(list(self.output_photo_discr_loss.values()))

            # Compute discriminator accuracies.
            self.input_painting_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                                         dtype=tf.float32)) * scale_weight[key]
                                             for key, pred in zip(self.input_painting_discr_predictions.keys(),
                                                                  self.input_painting_discr_predictions.values())}
            self.input_photo_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                                      dtype=tf.float32)) * scale_weight[key]
                                          for key, pred in zip(self.input_photo_discr_predictions.keys(),
                                                               self.input_photo_discr_predictions.values())}
            self.output_photo_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                                       dtype=tf.float32)) * scale_weight[key]
                                           for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                self.output_photo_discr_predictions.values())}
            self.discr_acc = (tf.add_n(list(self.input_painting_discr_acc.values())) + \
                              tf.add_n(list(self.input_photo_discr_acc.values())) + \
                              tf.add_n(list(self.output_photo_discr_acc.values()))) / float(len(scale_weight.keys())*3)


            # Generator.
            # Predicts ones for both output images.
            self.output_photo_gener_loss = {key: self.loss(pred, tf.ones_like(pred)) * scale_weight[key]
                                            for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                 self.output_photo_discr_predictions.values())}

            self.gener_loss = tf.add_n(list(self.output_photo_gener_loss.values()))

            # Compute generator accuracies.
            self.output_photo_gener_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                                       dtype=tf.float32)) * scale_weight[key]
                                           for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                self.output_photo_discr_predictions.values())}

            self.gener_acc = tf.add_n(list(self.output_photo_gener_acc.values())) / float(len(scale_weight.keys()))


            # Image loss.
            self.img_loss_photo = mse_criterion(transformer_block(self.output_photo),
                                                transformer_block(self.input_photo))
            self.img_loss = self.img_loss_photo

            # Features loss.
            self.feature_loss_photo = abs_criterion(self.output_photo_features, self.input_photo_features)
            self.feature_loss = self.feature_loss_photo

            # ================== Define optimization steps. =============== #
            t_vars = tf.trainable_variables()
            self.discr_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.encoder_vars = [var for var in t_vars if 'encoder' in var.name]
            self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]

            # Discriminator and generator steps.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.d_optim_step = tf.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.discr_loss_weight * self.discr_loss,
                    var_list=[self.discr_vars])
                self.g_optim_step = tf.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.discr_loss_weight * self.gener_loss +
                         self.options.transformer_loss_weight * self.img_loss +
                         self.options.feature_loss_weight * self.feature_loss,
                    var_list=[self.encoder_vars + self.decoder_vars])

            # ============= Write statistics to tensorboard. ================ #

            # Discriminator loss summary.
            s_d1 = [tf.summary.scalar("discriminator/input_painting_discr_loss/"+key, val)
                    for key, val in zip(self.input_painting_discr_loss.keys(), self.input_painting_discr_loss.values())]
            s_d2 = [tf.summary.scalar("discriminator/input_photo_discr_loss/"+key, val)
                    for key, val in zip(self.input_photo_discr_loss.keys(), self.input_photo_discr_loss.values())]
            s_d3 = [tf.summary.scalar("discriminator/output_photo_discr_loss/" + key, val)
                    for key, val in zip(self.output_photo_discr_loss.keys(), self.output_photo_discr_loss.values())]
            s_d = tf.summary.scalar("discriminator/discr_loss", self.discr_loss)
            self.summary_discriminator_loss = tf.summary.merge(s_d1+s_d2+s_d3+[s_d])

            # Discriminator acc summary.
            s_d1_acc = [tf.summary.scalar("discriminator/input_painting_discr_acc/"+key, val)
                    for key, val in zip(self.input_painting_discr_acc.keys(), self.input_painting_discr_acc.values())]
            s_d2_acc = [tf.summary.scalar("discriminator/input_photo_discr_acc/"+key, val)
                    for key, val in zip(self.input_photo_discr_acc.keys(), self.input_photo_discr_acc.values())]
            s_d3_acc = [tf.summary.scalar("discriminator/output_photo_discr_acc/" + key, val)
                    for key, val in zip(self.output_photo_discr_acc.keys(), self.output_photo_discr_acc.values())]
            s_d_acc = tf.summary.scalar("discriminator/discr_acc", self.discr_acc)
            s_d_acc_g = tf.summary.scalar("discriminator/discr_acc", self.gener_acc)
            self.summary_discriminator_acc = tf.summary.merge(s_d1_acc+s_d2_acc+s_d3_acc+[s_d_acc])

            # Image loss summary.
            s_i1 = tf.summary.scalar("image_loss/photo", self.img_loss_photo)
            s_i = tf.summary.scalar("image_loss/loss", self.img_loss)
            self.summary_image_loss = tf.summary.merge([s_i1 + s_i])

            # Feature loss summary.
            s_f1 = tf.summary.scalar("feature_loss/photo", self.feature_loss_photo)
            s_f = tf.summary.scalar("feature_loss/loss", self.feature_loss)
            self.summary_feature_loss = tf.summary.merge([s_f1 + s_f])

            self.summary_merged_all = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        else:
            # ==================== Define placeholders. ===================== #
            with tf.name_scope('placeholder'):
                self.input_photo = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size, None, None, 3],
                                                  name='photo')

            # ===================== Wire the graph. ========================= #
            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                options=self.options,
                                                reuse=False)

            # Decode obtained features.
            self.output_photo = decoder(features=self.input_photo_features,
                                        options=self.options,
                                        reuse=False)

    def train(self, args, ckpt_nmbr=None):
        # Initialize augmentor.
        augmentor = img_augm.Augmentor(crop_size=[self.options.image_size, self.options.image_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.05,
                                       saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                                       value_augm_shift=0.05, value_augm_scale=0.05, )
        content_dataset_places = prepare_dataset.PlacesDataset(path_to_dataset=self.options.path_to_content_dataset)
        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset)


        # Initialize queue workers for both datasets.
        q_art = multiprocessing.Queue(maxsize=10)
        q_content = multiprocessing.Queue(maxsize=10)
        jobs = []
        for i in range(5):
            p = multiprocessing.Process(target=content_dataset_places.initialize_batch_worker,
                                        args=(q_content, augmentor, self.batch_size, i))
            p.start()
            jobs.append(p)

            p = multiprocessing.Process(target=art_dataset.initialize_batch_worker,
                                        args=(q_art, augmentor, self.batch_size, i))
            p.start()
            jobs.append(p)
        print("Processes are started.")
        time.sleep(3)

        # Now initialize the graph
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start training.")

        if self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Initial discriminator success rate.
        win_rate = args.discr_success_rate
        discr_success = args.discr_success_rate
        alpha = 0.05

        for step in tqdm(range(self.initial_step, self.options.total_steps+1),
                         initial=self.initial_step,
                         total=self.options.total_steps):
            # Get batch from the queue with batches q, if the last is non-empty.
            while q_art.empty() or q_content.empty():
                pass
            batch_art = q_art.get()
            batch_content = q_content.get()

            if discr_success >= win_rate:
                # Train generator
                _, summary_all, gener_acc_ = self.sess.run(
                    [self.g_optim_step, self.summary_merged_all, self.gener_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })
                discr_success = discr_success * (1. - alpha) + alpha * (1. - gener_acc_)
            else:
                # Train discriminator.
                _, summary_all, discr_acc_ = self.sess.run(
                    [self.d_optim_step, self.summary_merged_all, self.discr_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })

                discr_success = discr_success * (1. - alpha) + alpha * discr_acc_
            self.writer.add_summary(summary_all, step * self.batch_size)

            if step % self.options.save_freq == 0 and step > self.initial_step:
                self.save(step)

            # And additionally save all checkpoints each 15000 steps.
            if step % 15000 == 0 and step > self.initial_step:
                self.save(step, is_long=True)

            if step % 500 == 0:
                output_paintings_, output_photos_= self.sess.run(
                    [self.input_painting, self.output_photo],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })

                save_batch(input_painting_batch=batch_art['image'],
                           input_photo_batch=batch_content['image'],
                           output_painting_batch=denormalize_arr_of_imgs(output_paintings_),
                           output_photo_batch=denormalize_arr_of_imgs(output_photos_),
                           filepath='%s/step_%d.jpg' % (self.sample_dir, step))
        print("Training is finished. Terminate jobs.")
        for p in jobs:
            p.join()
            p.terminate()

        print("Done.")

    # Don't use this function yet.
    def inference_video(self, args, path_to_folder, to_save_dir=None, resize_to_original=True,
                        use_time_smooth_randomness=True, ckpt_nmbr=None):
        """
        Run inference on the video frames. Original aspect ratio will be preserved.
        Args:
            args:
            path_to_folder: path to the folder with frames from the video
            to_save_dir:
            resize_to_original:
            use_time_smooth_randomness: change the random vector
            which is added to the bottleneck features linearly over tim

        Returns:

        """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d' % (self.initial_step, self.image_size))

        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        image_paths = sorted(os.listdir(path_to_folder))
        num_images = len(image_paths)
        for img_idx, img_name in enumerate(tqdm(image_paths)):

            img_path = os.path.join(path_to_folder, img_name)
            img = scipy.misc.imread(img_path, mode='RGB')
            img_shape = img.shape[:2]
            # Prepare image for feeding into network.
            scale_mult = self.image_size / np.min(img_shape)
            new_shape = (np.array(img_shape, dtype=float) * scale_mult).astype(int)

            img = scipy.misc.imresize(img, size=new_shape)

            img = np.expand_dims(img, axis=0)

            if use_time_smooth_randomness and img_idx == 0:
                features_delta = self.sess.run(self.labels_to_concatenate_to_features,
                               feed_dict={
                                   self.input_photo: normalize_arr_of_imgs(img),
                               })
                features_delta_start = features_delta + np.random.random(size=features_delta.shape) * 0.5 - 0.25
                features_delta_start = features_delta_start.clip(0, 1000)
                print('features_delta_start.shape=', features_delta_start.shape)
                features_delta_end = features_delta + np.random.random(size=features_delta.shape) * 0.5 - 0.25
                features_delta_end = features_delta_end.clip(0, 1000)
                step = (features_delta_end - features_delta_start) / (num_images - 1)

            feed_dict = {
                self.input_painting: normalize_arr_of_imgs(img),
                self.input_photo: normalize_arr_of_imgs(img),
                self.lr: self.options.lr
            }
            if use_time_smooth_randomness:
                pass

            img = self.sess.run(self.output_photo, feed_dict=feed_dict)

            img = img[0]
            img = denormalize_arr_of_imgs(img)
            if resize_to_original:
                img = scipy.misc.imresize(img, size=img_shape)
            else:
                pass

            scipy.misc.imsave(os.path.join(to_save_dir, img_name[:-4] + "_stylized.jpg"), img)

        print("Inference is finished.")

    def inference(self, args, path_to_folder, to_save_dir=None, resize_to_original=True,
                  ckpt_nmbr=None):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d' % (self.initial_step, self.image_size))

        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for img_idx, img_path in enumerate(tqdm(names)):
            img = scipy.misc.imread(img_path, mode='RGB')
            img_shape = img.shape[:2]

            # Resize the smallest side of the image to the self.image_size
            alpha = float(self.image_size) / float(min(img_shape))
            img = scipy.misc.imresize(img, size=alpha)
            img = np.expand_dims(img, axis=0)

            img = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img),
                })

            img = img[0]
            img = denormalize_arr_of_imgs(img)
            if resize_to_original:
                img = scipy.misc.imresize(img, size=img_shape)
            else:
                pass
            img_name = os.path.basename(img_path)
            scipy.misc.imsave(os.path.join(to_save_dir, img_name[:-4] + "_stylized.jpg"), img)

        print("Inference is finished.")

    def save(self, step, is_long=False):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if is_long:
            self.saver_long.save(self.sess,
                                 os.path.join(self.checkpoint_long_dir, self.model_name+'_%d.ckpt' % step),
                                 global_step=step)
        else:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, self.model_name + '_%d.ckpt' % step),
                            global_step=step)

    def load(self, checkpoint_dir, ckpt_nmbr=None):
        if ckpt_nmbr:
            if len([x for x in os.listdir(checkpoint_dir) if ("ckpt-" + str(ckpt_nmbr)) in x]) > 0:
                print(" [*] Reading checkpoint %d from folder %s." % (ckpt_nmbr, checkpoint_dir))
                ckpt_name = [x for x in os.listdir(checkpoint_dir) if ("ckpt-" + str(ckpt_nmbr)) in x][0]
                ckpt_name = '.'.join(ckpt_name.split('.')[:-1])
                self.initial_step = ckpt_nmbr
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
        else:
            print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
