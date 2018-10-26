#!/usr/bin/env python
from __future__ import print_function

import requests
import os

from torchvision.datasets.utils import download_url

API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'

EVALUATION_DATA_URL = 'https://yadi.sk/d/A2CBqSGuJ0M_XA'


def get_real_direct_link(sharing_link):
    pk_request = requests.get(API_ENDPOINT.format(sharing_link))

    return pk_request.json()['href']


def unzip(path, target_dir='.'):
    import zipfile
    with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)


def main():
    root = "."
    link = get_real_direct_link(EVALUATION_DATA_URL)
    filename = 'evaluation_data.zip'
    print('Downloadng data (1Gb). This may take a while...')
    download_url(link, root, filename,  None)
    print('Unzipping...')
    unzip(os.path.join(root, filename), target_dir='.')
    print('Done.')


if __name__ == '__main__':
    main()

