import glob
import os
import re
import shutil

from pathlib2 import Path

"""
Utility to clean up for the camera ready submission.
Find all occurrences of image names in the paper and copy only files used in the paper to a new folder.
"""


def recursive_glob(dir_path, filter):
    import fnmatch
    import os

    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, filter):
            matches.append(os.path.join(root, filename))
    return matches


root_dir = u'..'

unique_exts = ['.jpg']
print unique_exts

image_file_names = []
with open('../index.html', 'r') as f:
    text = f.read()
    for ext in unique_exts:
        pattern = u'(images/.*?\.' + ext[1:] + ')"'
        print pattern
        results = re.findall(pattern, text)
        image_file_names.extend(results)

image_file_names = [os.path.join(root_dir, x) for x in image_file_names]
print len(image_file_names)
print image_file_names


new_image_folder = os.path.join(root_dir, u'images_new')

for path in image_file_names:
    path = Path(path)
    assert root_dir == path.parts[0], path.parts[0]
    assert path.parts[1] == 'images', path.parts[1]
    new_path = Path(new_image_folder).joinpath(*path.parts[2:])
    if not new_path.parent.exists():
        new_path.parent.mkdir(parents=True)
    shutil.copyfile(unicode(path), unicode(new_path))
    print u'copy {} -> {}'.format(path, new_path)
