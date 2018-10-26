import pandas as pd
import h5py
import numpy as np

ARTISTS = ['claude-monet',
           'paul-cezanne',
           'el-greco',
           'paul-gauguin',
           'samuel-peploe',
           'vincent-van-gogh',
           'edvard-munch',
           'pablo-picasso',
           'berthe-morisot',
           'ernst-ludwig-kirchner',
           'jackson-pollock',
           'wassily-kandinsky',
           'nicholas-roerich']


def get_artist_labels_wikiart(artists=ARTISTS):
    """
    Get mapping of artist name to class label
    """
    split_df = pd.read_hdf('evaluation_data/split.hdf5')

    labels = dict()

    for artist_id in artists:
        artist_id_in_split = artist_id
        print artist_id
        cur_df = split_df[split_df.index.str.startswith(artist_id_in_split)]
        assert len(cur_df)
        if not np.all(cur_df.index.str.startswith(artist_id_in_split + '_')):
            print cur_df[~cur_df.index.str.startswith(artist_id_in_split + '_')]
            assert False

        print '===='
        labels[artist_id] = cur_df['label'][0]
    return labels


if __name__ == '__main__':

    print get_artist_labels_wikiart(ARTISTS)
