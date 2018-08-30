## A Style-Aware Content Loss for Real-time HD Style Transfer
Artsiom Sanakoyeu*, Dmytro Kotovenko*, Sabine Lang, Björn Ommer  
Heidelberg University  
In ECCV 2018 (Oral)  


![pipeline](https://compvis.github.io/adaptive-style-transfer/images/eccv_pipeline_diagram_new_symbols_v2_4.jpg "Method pipeline")
![example](https://compvis.github.io/adaptive-style-transfer/images/teaser_eccv18_cezanne.jpg "Stylization")

**Website**: https://compvis.github.io/adaptive-style-transfer   
**Paper**: https://arxiv.org/abs/1807.10201

### Source code and pretrained models are coming soon.

### Requirements:
- python 2.7
- tensorflow 1.2.
- PIL, numpy, scipy, os 
- tqdm, argparse

### Inference routine:
#### Simplest van Gogh example.
To launch the inference on van Gogh style first download the pretrained model named 'model_van-gogh'
[here](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/XXVKT5grAquXNqi) 
and sample photographs from [here](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf). 
Extract the model to `./models/` folder  and sample photographs to `./data/` folder.
Finally run the following code:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
                 --model_name=model_van-gogh \
                 --phase=inference \
                 --image_size=1280
```
Stylized photographs are stored in the folder ./models/model_van-gogh/inference_ckpt300000_sz1280/

#### Additional settings.
If you want to run the code on your own data please specify additional parameter `--ii_dir` defining a path to folder 
containing your target images. You can specify multiple folders: enumerate them and separate with comma. Don't use spaces 
inbetween.  

You can also change the resolution of the image you generate by changing parameter
`--image_size`, please specify 

To save generated images in custom folder add parameter `--save_dir`.

If you don't have a GPU big enough for current model you can set parameter `CUDA_VISIBLE_DEVICES=""` to use CPU. 
    
All together it looks like this: 
```
CUDA_VISIBLE_DEVICES="" python main.py \
                 --model_name=model_van-gogh \
                 --phase=inference \
                 --image_size=1280 \
                 --ii_dir ../my_photographs1/,../my_photographs2/ \
                 --save_dir=../save_processed_images_here/
``` 
#### Additional artists.
We have pretrained models for following artists:
Paul Cezanne, El-Greco, Paul Gauguin, Wassily Kandinsky _Василий Кандинский_, Ernst Ludwig Kirchner,
Claude Monet, Berthe Morisot, Edvard Munch, Samuel Peploe, Pablo Picasso, Jackson Pollock, 
Nicholal Roerich _Николай Рерих_, Vincent van Gogh. A few more artists will be added in the future.

You can download the models from the [server](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/XXVKT5grAquXNqi)
and extract them to the folder `./models/`. Now use the name of the folder as the `model_name` parameter, for instance 
for Picasso execute: 
```
CUDA_VISIBLE_DEVICES=0 python main.py \
                 --model_name=model_picasso \
                 --phase=inference \
                 --image_size=1280
```
    
### Training.
To start training you need a content dataset with photographs and style dataset with images representing artistic style.
We have trained our models on [Places2 dataset](http://places2.csail.mit.edu/), in particular we've used the Places365-Standard
high-res train dataset images(105GB). Please specify path to it using the parameter `--ptcd` which stands for 
_path_to_content_dataset_. Note that we don't use all the classes from the dataset but only particular subset of it containing 
around 120 classes.

The dataset representing Vincent van Gogh style(only particular part of it) can be 
downloaded from  [here](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf), extract corresponding folder to the folder
`./data/`.   

Now we can launch the training process:
```
CUDA_VISIBLE_DEVICES=1 python main.py \
                 --model_name=model_van-gogh_new \
                 --batch_size=1 \
                 --phase=train \
                 --image_size=768 \
                 --lr=0.0002 \
                 --dsr=0.8 \
                 --ptcd=/path/to/Places2/data_large \
                 --ptad=./data/vincent-van-gogh_road-with-cypresses-1890/
```                 
    
We also provide style images we've used to train Claude Monet model.  
### Video.
Coming soon.

