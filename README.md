


# Ecomended Development Environment
* GPU: RTX 4090(24GB) 

* CUDA: 11.3

* PyTorch  1.11.0

* Python  3.8(ubuntu20.04)

  

## 1. Dataset
Download datasets and place them in `./data/` folder.  
For example, download [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) and place it:
```
.
└── data
    └── Celeb-DF-v2
        ├── Celeb-real
        │       └── *.mp4
        ├── Celeb-synthesis
        │       └── *.mp4
        ├── Youtube-real
        │       └── *.mp4
        └── List_of_testing_videos.txt
```
For other datasets, please refer to `./data/datasets.md` .


## 2. Pretrained model
We provide weights of EfficientNet-B4 trained on GFADE from FF-raw.  
Download and place it in `./weights/` folder.

# Inference

For example, run the inference on Celeb-DF-v2:
```bash
CUDA_VISIBLE_DEVICES=* python3 src/inference/inference_dataset.py \
-w weights/98_0.9997_val.tar \
-d CDF
```
The result will be displayed.

Using the provided pretrained model, our cross-dataset results are reproduced as follows:

Training Data | CDF | DFD | DFDC | DFDCP | FFIW
:-: | :-: | :-: | :-: | :-: | :-:
FF-raw | 94.68% | 99.12% | 80.81% | 85.39% | 85.55% 

# Training
1. Download [FF++](https://github.com/ondyari/FaceForensics) real videos and place them in `./data/` folder:
```
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── train.json
        ├── val.json
        └── test.json
```
2. Download landmark detector (shape_predictor_81_face_landmarks.dat) from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in `./src/preprocess/` folder.  

3. Run the two codes to extractvideo frames, landmarks, and bounding boxes:
```bash
python3 src/preprocess/crop_dlib_ff.py -d Original
CUDA_VISIBLE_DEVICES=* python3 src/preprocess/crop_retina_ff.py -d Original
```

4.  You can download code for landmark augmentation:
```bash
mkdir src/utils/library
git clone https://github.com/AlgoHunt/Face-Xray.git src/utils/library
```
5. Run the training:
```bash
CUDA_VISIBLE_DEVICES=* python3 src/train_gfade.py \
src/configs/gfade/base.json \
-n gfade
```
Top five checkpoints will be saved in `./output/` folder.
