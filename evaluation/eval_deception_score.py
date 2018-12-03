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

import argparse
import os
import sys
from pprint import pformat
import glob
import numpy as np
import pandas as pd
import re

from feature_extractor.feature_extractor import SlimFeatureExtractor
from logger import Logger
from check_fc8_labels import get_artist_labels_wikiart


def parse_one_or_list(str_value):
    if str_value is not None:
        if str_value.lower() == 'none':
            str_value = None
        elif ',' in str_value:
            str_value = str_value.split(',')
    return str_value


def parse_list(str_value):
    if ',' in str_value:
        str_value = str_value.split(',')
    else:
        str_value = [str_value]
    return str_value


def parse_none(str_value):
    if str_value is not None:
        if str_value.lower() == 'none' or str_value == "":
            str_value = None
    return str_value


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', '--net', help='network type',
                        choices=['vgg_16', 'vgg_16_multihead'], default='vgg_16')
    parser.add_argument('-log', '--log-path', help='log path', type=str,
                        default='/tmp/res.txt'
                        )
    parser.add_argument('-s', '--snapshot_path', type=str,
                        default='vgg_16.ckpt')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--num_classes', type=int, default=624)
    parser.add_argument('--dataset', type=str, default='wikiart', choices=['wikiart'])
    args = parser.parse_args(argv)
    args = vars(args)
    return args


def create_slim_extractor(cli_params):
    extractor_class = SlimFeatureExtractor
    extractor_ = extractor_class(cli_params['net'], cli_params['snapshot_path'],
                                 should_restore_classifier=True,
                                 gpu_memory_fraction=0.95,
                                 vgg_16_heads=None if cli_params['net'] != 'vgg_16_multihead' else {'artist_id': cli_params['num_classes']})
    return extractor_


classification_layer = {
    'vgg_16': 'vgg_16/fc8',
    'vgg_16_multihead': 'vgg_16/fc8_artist_id'
}


def run(extractor, classification_layer, images_df, batch_size=64, logger=Logger()):
    images_df = images_df.copy()
    if len(images_df) == 0:
        print 'No images found!'
        return -1, 0, 0
    probs = extractor.extract(images_df['image_path'].values, [classification_layer],
                              verbose=1, batch_size=batch_size)
    images_df['predicted_class'] = np.argmax(probs, axis=1).tolist()
    is_correct = images_df['label'] == images_df['predicted_class']
    accuracy = float(is_correct.sum()) / len(images_df)

    logger.log('Num images: {}'.format(len(images_df)))
    logger.log('Correctly classified: {}/{}'.format(is_correct.sum(), len(images_df)))
    logger.log('Accuracy: {:.5f}'.format(accuracy))
    logger.log('\n===')
    return accuracy, is_correct.sum(), len(images_df)


# image filenames must be in format "{content_name}_stylized_{artist_name}.jpg"
# uncomment methods which you want to evaluate and set the paths to the folders with the stylized images
results_dir = {
    'ours': 'path/to/our/stylizations',
    # 'gatys': 'path/to/gatys_stylizations',
    # 'cyclegan': '',
    # 'adain': '',
    # 'johnson': '',
    # 'wct': '',
    # 'real_wiki_test': os.path.expanduser('~/workspace/wikiart/images_square_227x227') # uncomment to test on real images from wikiart test set
}


style_2_image_name = {u'berthe-morisot': u'Morisot-1886-the-lesson-in-the-garden',
		      u'claude-monet': u'monet-1914-water-lilies-37.jpg!HD',
		      u'edvard-munch': u'Munch-the-scream-1893',
		      u'el-greco': u'el-greco-the-resurrection-1595.jpg!HD',
		      u'ernst-ludwig-kirchner': u'Kirchner-1913-street-berlin.jpg!HD',
		      u'jackson-pollock': u'Pollock-number-one-moma-November-31-1950-1950',
		      u'nicholas-roerich': u'nicholas-roerich_mongolia-campaign-of-genghis-khan',
		      u'pablo-picasso': u'weeping-woman-1937',
		      u'paul-cezanne': u'still-life-with-apples-1894.jpg!HD',
		      u'paul-gauguin': u'Gauguin-the-seed-of-the-areoi-1892',
		      u'samuel-peploe': u'peploe-ile-de-brehat-1911-1',
		      u'vincent-van-gogh': u'vincent-van-gogh_road-with-cypresses-1890',
		      u'wassily-kandinsky': u'Kandinsky-improvisation-28-second-version-1912'}


artist_2_label_wikiart = get_artist_labels_wikiart()


def get_images_df(dataset, method, artist_slug):
    images_dir = results_dir[method]
    paths = glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png'))
    # print paths
    assert len(paths) or method.startswith('real')

    if not method.startswith('real'):
        cur_style_paths = [x for x in paths if re.match('.*_stylized_({}|{}).(jpg|png)'.format(artist_slug, style_2_image_name[artist_slug]), os.path.basename(x)) is not None]
    elif method == 'real_wiki_test':
        # use only images from the test set
        split_df = pd.read_hdf(os.path.expanduser('evaluation_data/split.hdf5'))
        split_df['image_id'] = split_df.index
        df = split_df[split_df['split'] == 'test']
        df['artist_id'] = df['image_id'].apply(lambda x: x.split('_', 1)[0])
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(results_dir['real_wiki_test'], x + '.png'))
        cur_style_paths = df.loc[df['artist_id'] == artist_slug, 'image_path'].values

    df = pd.DataFrame(index=[os.path.basename(x).split('_stylized_', 1)[0].rstrip('.') for x in
                             cur_style_paths], data={'image_path': cur_style_paths, 'artist': artist_slug})

    df['label'] = artist_2_label_wikiart[artist_slug]
    return df


def sprint_stats(stats):
    msg = ''
    msg += 'artist\t accuracy\t is_correct\t total\n'
    for key in sorted(stats.keys()):
        msg += key + '\t {:.5f}\t {}\t \t{}\n'.format(*stats[key])
    return msg


if __name__ == '__main__':
    import sys

    args = parse_args(sys.argv[1:])

    if not os.path.exists(os.path.dirname(args['log_path'])):
        os.makedirs(os.path.dirname(args['log_path']))
    logger = Logger(args['log_path'])
    print 'Snapshot: {}'.format(args['snapshot_path'])
    extractor = create_slim_extractor(args)
    classification_layer = classification_layer[args['net']]

    stats = dict()
    assert artist_2_label_wikiart is not None
    for artist in artist_2_label_wikiart.keys():
        print('Method:', args['method'])
        logger.log('Artist: {}'.format(artist))
        images_df = get_images_df(dataset=args['dataset'], method=args['method'], artist_slug=artist)
        acc, num_is_correct, num_total = run(extractor, classification_layer, images_df,
                                             batch_size=args['batch_size'], logger=logger)
        stats[artist] = (acc, num_is_correct, num_total)

    logger.log('{}'.format(pformat(args)))
    print 'Images dir:', results_dir[args['method']]
    logger.log('===\n\n')
    logger.log(args['method'])
    logger.log('{}'.format(sprint_stats(stats)))
