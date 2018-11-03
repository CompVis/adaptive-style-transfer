### Data description

- **eval_paths_700_val.json** - list of paths to the images from MSCOCO and Plcae365 datasets used for stylization and computation of the deception score.

#### The files which will be automatically downloaded by script:

- **model.ckpt-790000**: checkpoint of the VGG-16 network which was trained from scratch to predict artist of the painting.  
    The network was trained on wikiart dataset using all artists, which had at least 50 artworks (624 artists total).   
    (wikiart/cnn/artist_50/vgg_16_0_lr0.005)

- **split.hdf5**: dataframe, containing image ids and labels of images from wikairt which were used for training/testing the artist clasification model.  
    (wikiart/cnn/artist_50)
