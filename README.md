[//]: <links>
[sngans]: https://openreview.net/forum?id=B1QRgziT-
[pcgans]: https://openreview.net/forum?id=ByS1VpgRZ

# GANs with spectral normalization and projection discriminator
*NOTE: The setup and example code in this README are for training GANs on **single GPU**.*
*The models are smaller than the ones used in the [papers](https://github.com/pfnet-research/sngan_projection/#references).*
*Please go to [**link**](https://github.com/pfnet-research/sngan_projection/blob/master/README_paper.md) if you are looking for how to reproduce the results in the papers.* 

<img src="https://github.com/pfnet-research/sngan_projection/blob/master/demo/dog_and_cat_1x1_long.gif" width="128">

Official Chainer implementation for conditional image generation on ILSVRC2012 dataset (ImageNet) with [spectral normalization][sngans] and [projection discrimiantor][pcgans]. 

### Demo movies

Consecutive category morphing movies:
- (5x5 panels 128px images) https://www.youtube.com/watch?v=q3yy5Fxs7Lc  
- (10x10 panels 128px images) https://www.youtube.com/watch?v=83D_3WXpPjQ

### Other materials
- [Generated images](https://drive.google.com/drive/folders/1ZzQctZ-loDf9wHJHX90xNN02-_BCYtB-?usp=sharing)
  - from the model trained on all ImageNet images (1K categories): [[128px]](https://drive.google.com/drive/folders/1Mr-fYW0-9QbwKYlIaiFUtgcN6n9qhY8l?usp=sharing)
  - from the model trained on dog and cat images (143 categories): [[64px]](https://drive.google.com/drive/folders/1RVJCDrSSHaoHKiSP9iCaiiimQoq42rQu?usp=sharing) [[128px]](https://drive.google.com/drive/folders/1yA3xWJqWRvhnhkvJsKF3Xbb-2LO4JrJw?usp=sharing) [[256px]](https://drive.google.com/drive/folders/1ejixDbK5gMtuPltr8kY_dIqMwsqGzt6a?usp=sharing)
- [Pretrained models](https://drive.google.com/drive/folders/1xZoL48uFOCnTxNGdknEYqE5YX0ZyoUej?usp=sharing)
- [Movies](https://drive.google.com/drive/folders/1yhV8_VbOcs2rkiMTstO4RHqp4YRnzg6c?usp=sharing)
- 4 corners category morph.

<img src="https://github.com/pfnet-research/sngan_projection/blob/master/demo/interpolated_images_24.png" width="432"> <img src="https://github.com/pfnet-research/sngan_projection/blob/master/demo/interpolated_images_3x3.png" width="432"> 

### References
- Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. *Spectral Normalization for Generative Adversarial Networks*. ICLR2018. [OpenReview][sngans]
- Takeru Miyato, Masanori Koyama. *cGANs with Projection Discriminator*. ICLR2018. [OpenReview][pcgans]

## Setup

### Install required python libraries:

`pip install -r requirements.txt`
### Download ImageNet dataset:
Please download ILSVRC2012 dataset from http://image-net.org/download-images

### Preprocess dataset:
```
cd datasets
IMAGENET_TRAIN_DIR=/path/to/imagenet/train/ # path to the parent directory of category directories named "n0*******".
PREPROCESSED_DATA_DIR=/path/to/save_dir/
bash preprocess.sh $IMAGENET_TRAIN_DIR $PREPROCESSED_DATA_DIR
# Make the list of image-label pairs for all images (1000 categories, 1281167 images).
python imagenet.py $PREPROCESSED_DATA_DIR
# Make the list of image-label pairs for dog and cat images (143 categories, 180373 images). 
python imagenet_dog_and_cat.py $PREPROCESSED_DATA_DIR
```
### Download inception model: 

`python source/inception/download.py --outfile=datasets/inception_model`

## Training examples

### Spectral normalization + projection discriminator for 64x64 dog and cat images:
```
LOGDIR=/path/to/logdir
CONFIG=configs/sn_projection_dog_and_cat_64.yml
python train.py --config=$CONFIG --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
```
- [pretrained model](https://drive.google.com/drive/folders/1KfhQo84fvWUtYQlRVAWf0nswf6X1nawh?usp=sharing)
- [generated images at 250K iterations](https://drive.google.com/drive/u/1/folders/1RVJCDrSSHaoHKiSP9iCaiiimQoq42rQu)
- Examples of 64x64 generated images:
<img src="https://github.com/pfnet-research/sngan_projection/blob/master/demo/images_dog_and_cat_64.jpg">

### Spectral normalization + projection discriminator for 64x64 all ImageNet images:
```
LOGDIR=/path/to/logdir
CONFIG=configs/sn_projection_64.yml
python train.py --config=$CONFIG --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
```

## Evaluation examples
(If you want to use pretrained models for the image generation, please download the model from [link](https://drive.google.com/drive/folders/1xZoL48uFOCnTxNGdknEYqE5YX0ZyoUej?usp=sharing) and set the `snapshot` argument to the path to the downloaded pretrained model file (.npz).)

### Generate images
```
python evaluations/gen_images.py --config=$CONFIG --snapshot=${LOGDIR}/ResNetGenerator_<iterations>.npz --results_dir=${LOGDIR}/gen_images
```

### Generate category morphing images
Regarding the index-category correspondence, please see [1K ImageNet](https://drive.google.com/drive/u/1/folders/1Mr-fYW0-9QbwKYlIaiFUtgcN6n9qhY8l) or [143 dog and cat ImageNet](https://drive.google.com/drive/u/1/folders/1yA3xWJqWRvhnhkvJsKF3Xbb-2LO4JrJw).
```
python evaluations/gen_interpolated_images.py --n_zs=10 --n_intp=10 --classes $CATEGORY1 $CATEGORY2 --config=$CONFIG --snapshot=${LOGDIR}/ResNetGenerator_<iterations>.npz --results_dir=${LOGDIR}/gen_morphing_images
```

### Calculate inception score (with the original OpenAI implementation)
```
python evaluations/calc_inception_score.py --config=$CONFIG --snapshot=${LOGDIR}/ResNetGenerator_<iterations>.npz --results_dir=${LOGDIR}/inception_score --splits=10 --tf
```

