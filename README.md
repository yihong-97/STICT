# STICT for Video Shadow Detection
Code and Dataset for our CVPR 2022 paper "Video Shadow Detection via Spatio-Temporal Interpolation Consistency Training"

## VIdeo ShAdow Detection dataset (VISAD)
VISAD is consisted of 82 videos and was divided into two parts according to semantic of scenes: the `Driving Scenes (VISAD-DS)` part and the `Moving Object Scenes (VISAD-MOS)` part, denoted as DS and MOS respectively.

It is available at [Google Drive](https://drive.google.com/drive/folders/1IkRtl9Hd_b_JBg2PgMqv1l3HM6XGeV-x?usp=sharing).

|scenes|videos/annotated|frames/annotated|resolution|
| :------: | :------: | :------: | :------: |
|[DS-all](https://drive.google.com/drive/folders/1be2BrxwwBQRUUzdXCWoBvP9XM7VBa1k_?usp=sharing)|47 / 17|7953 / 2881|1280×720|
|[DS-test](https://drive.google.com/file/d/1v-Vj-RccLmou0-5y-5t-SNB37M4oNrw3/view?usp=sharing)|13 / 13|2190 / 2190|1280×720|
|[MOS-all](https://drive.google.com/drive/folders/1XGs8ZhN35DevGi8FJ3fGkoX3wW7eLZt9?usp=sharing)|34 / 16|4613 / 1307|(530-1920)×(360-1080)|
|[MOS-test](https://drive.google.com/file/d/1irfq8u85vditoC4pb7YXeHoLGDo8AfvV/view?usp=sharing)|13 / 13|873 / 873|1920×1080,1600×900|

## Spatio-Temporal Interpolation Consistency Training

### Requirement
* cuda (10.0)
* Python (3.6)
* PyTorch (1.1.0)
* spatial-correlation-sampler (0.0.8) 
* Flownet (2.0)

### Download dataset
Download the following datasets and unzip them into ```./data``` folder
* SBU (it can refer [MTMT](https://github.com/eraserNut/MTMT#useful-links))
* DS
* MOS

### Testing
Our pretrained model is available [here](https://drive.google.com/drive/folders/1Ty4ROTRXf5kg7c1cCSzyhTkjvFQGEehd?usp=sharing)
1. Run ```python test.py```
```
important arguments:
--trained_model trained model path (default:'./DS')
--dataset_path your test set path (default: './data/DS/test/')
--dataset_txt_path your test set list path (default: './data/DS/test/test.txt')
```

### Training
1. Download pretrained models ([ReNet](https://download.pytorch.org/models/resnet50-19c8e357.pth) and [FlowNet](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)) into ```./pretrained_model``` folder
2. Run ```python train.py ```
```
important arguments:
--target_domain (options: 'DS_U', 'MOS_U', 'ViSha') (default: 'DS_U')
--dataset_U_path your video domain dataset path (default: './data/DS/train/')
```
