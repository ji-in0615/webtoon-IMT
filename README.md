# letr-proto-WiMT
WiMT (Webtoon images Machine Translation) is a multi-modal webtoon machine translation model.
You can get three results.

1. detect objects (speech bubbles, cuts, and line texts) in webtoon or cartoon.
2. ocr (line text detection + recognition)
3. English translation with papago API

## Install
### Prerequisition

0. Pytorch version 
```
pytorch==1.0.0 (only, but be updated later)
```

1. Set up
```
pip -r requirements.txt
```
```
cd object_detection/lib/
python setup.py build develop
```
2. Generate data folder
`./data/`

## Train
각 detection 폴더 (./object_detection, ./text_detection, ./text_recognition)에서 학습 가능.
### *[ Object detection ]*
1. Download models ([vgg16](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0) | [resnet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)), add to `./weights/`

2. Generate folder `./object_detection/train/images/`,`./object_detection/train/labels/`

3. Add images and labels to `./object_detection/train/images/`,`./object_detection/train/labels/`

4. Run `python ./object_detection/train.py`


### *[ Text detection ]*

- Run `python ./text_detection/train.py`



### *[ Text recognition ]*
#### Create Dataset
1. Add [fonts](https://noonnu.cc/) to `./text_recognition/train/fonts/`
- Run `python ./text_recognition/create_dataset.py`
#### Train
- Run `python ./text_recognition/train.py`

## Demo
0. Generate folder
`./weight/`
1. Download model

|*model name*|*model link*|
|------|----------|
|Speech-Bubble-Detector|[Go!](https://drive.google.com/file/d/1F10sRXWuICKuSQclaUnQVBo1rlxa6ogR/view)|
|Line-Text-Recognizer|[Go!](https://drive.google.com/file/d/1hhAER4rz6Ucgs0J-VzPuIeXbN5ReDOka/view)|
|Line-Text-Detector|[Go!](https://drive.google.com/file/d/1gL0-2IdSqIBN1o3W2AWEtOQRab-t5wx8/view)|

2. `python demo.py --ocr --translation`


### Arguments
- model option

`--object_detector` : folder path to trained speech bubble detector.

`--text_detector` : folder path to trained line text detector.

`--text_recognizer`: folder path to trained line text recognizer.

`--object`: enable object detection.

`--ocr`: enable ocr(text detection + text recognition).

`--papago`: enable translation with papago API.
- augmentation option

`--type`: select background type of toon [white | black | classic]. default : white

`--cls`: probability of speech bubble detection for filtering

`--box_size`: threshold of cut size for filtering

`--large_scale`: whether demo image is large scale.

`--ratio`: ratio of height to width of large scale image
- data option

`--demo_folder`: folder path to demo images
- device option

`--cuda`: use cuda for inference.

## Reference
hanish3464. [WORD-pytorch](https://github.com/hanish3464/WORD-pytorch)

jwyang. [FasterRCNN-pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

Youngmin Baek. [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)

Naver corp. [papago API](https://github.com/naver/naver-openapi-guide/tree/master/ko/papago-apis)

pvaneck. [Hangul Character Recognition](https://github.com/IBM/tensorflow-hangul-recognition)

kuangliu. [Recognition Network](https://github.com/kuangliu/pytorch-cifar)

