# The code for reproducing the ImageNet results in the ICLR2018 papers; [spectral normalization][sngans] and [projection discrimiantor][pcgans]

Official Chainer implementation for reproducing the results of conditional image generation on ILSVRC2012 dataset (ImageNet) with [spectral normalization][sngans] and [projection discrimiantor][pcgans].

### References
- Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. *Spectral Normalization for Generative Adversarial Networks*. ICLR2018. [OpenReview][sngans]
- Takeru Miyato, Masanori Koyama. *cGANs with Projection Discriminator*. ICLR2018. [OpenReview][pcgans]

## Setup
### Install OpenMPI and NCCL (required for multi-GPU training with [ChainerMN](https://github.com/chainer/chainermn))
Please see the following installation guide: https://chainermn.readthedocs.io/en/latest/installation/guide.html#requirements
 
 (Note: we provide the single GPU training code [here](https://github.com/pfnet-research/sngan_projection/blob/master/README_paper.md#training), but we have not checked the peformance of the models trained on single GPU. 
 All of the results showed in [the papers](https://github.com/pfnet-research/sngan_projection/#references) are produced by the models trained on 4 GPUs)
### Install required python libraries:

`pip install -r requirements_paper.txt`

Additionaly we recommend to install the latest [cupy](https://github.com/cupy/cupy):
```
pip uninstall cupy
git clone https://github.com/cupy/cupy.git
cd cupy
python setup.py install
```


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
# (optional) Make the list of image-label pairs for dog and cat images (143 categories, 180373 images).
python imagenet_dog_and_cat.py $PREPROCESSED_DATA_DIR
```
### Download inception model: 

`python source/inception/download.py --outfile=datasets/inception_model`

## Training

### Spectral normalization + projection discriminator for 128x128 all ImageNet images:
```
LOGDIR=/path/to/logdir/
CONFIG=configs/sn_projection_dog_and_cat.yml
# multi-GPU
mpiexec -n 4 python train_mn.py --config=configs/sn_projection.yml --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
# single-GPU
python train.py --config=$CONFIG --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
```
- [pretrained models](https://drive.google.com/drive/folders/1m04Db3HbN10Tz5XHqiPpIg8kw23fkbSi)
- [generated images at 450K iterations](https://drive.google.com/drive/folders/1Mr-fYW0-9QbwKYlIaiFUtgcN6n9qhY8l)   (Inception score:29.7, Intra class FID:103.1)
- [generated images at 850K iterations](https://drive.google.com/drive/folders/1-PbUUnrII9vUmcTUwYVYUtiwjiixbXpP)  (Inception score:36.8, Intra class FID:92.4)
- Examples of generated images at 450K iterations:

![image](https://github.com/pfnet-research/sngan_projection/blob/master/demo/various_images.jpg)


### Spectral normalization + concat discriminator for 128x128 all ImageNet images:
```
LOGDIR=/path/to/logdir/
CONFIG=configs/sn_projection_dog_and_cat.yml
# multi-GPU
mpiexec -n 4 python train_mn.py --config=configs/sn_concat --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
# single-GPU
python train.py --config=$CONFIG --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
```
- [pretrained models](https://drive.google.com/drive/folders/1xInDUt8nFkq7VWUeIgnqvu2o2ZcsBB-2) 
- [generated images at 450K iterations](https://drive.google.com/drive/folders/11TGLERZsfuVavfgV-dVtYJsUznq2mVIL) (Inception score:21.1, Intra class FID:141.2)

### (optional) Spectral normalization + projection discriminator for 128x128 dog and cat images:
```
LOGDIR=/path/to/logdir/
CONFIG=configs/sn_projection_dog_and_cat.yml
# multi-GPU
mpiexec -n 4 python train_mn.py --config=configs/sn_projection_dog_and_cat.yml --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
# single-GPU
python train.py --config=$CONFIG  --results_dir=$LOGDIR --data_dir=$PREPROCESSED_DATA_DIR
```
- [pretrained models](https://drive.google.com/drive/folders/1wKMG6ontP8ZKdBYOA8l-z_JQUUpuA7XA) 
- [generated images](https://drive.google.com/drive/folders/1yA3xWJqWRvhnhkvJsKF3Xbb-2LO4JrJw?usp=sharing) (Inception score: 28.2)

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


[sngans]: https://openreview.net/forum?id=B1QRgziT-
[pcgans]: https://openreview.net/forum?id=ByS1VpgRZ
