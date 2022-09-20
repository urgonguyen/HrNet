# How to run HRnet on Google Colab

The paper: Deep High-Resolution Representation Learning for Human Pose Estimation (accepted to CVPR2019)

## Installation
1. Install pytorch >= v1.0.0 (On Google Colab, it done)
2. Clone the project
```bash
!git clone https://github.com/HRNet/HRNet-Human-Pose-Estimation.git
```
3. Install dependencies
```bash
!pip install -r requirements.txt
```
Some libraries are available on Colab so you only need to install libraries that are not available (json_tricks, yacs>=0.1.5, tensorboardX>=1.6, Please check on your Colab) \
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.


```bash
pip install $library$
```

4. Make libs
```bash
cd HRNet-Human-Pose-Estimation/lib
```

```bash
!make
```
5. Install COCOAPI \
The current path: /content/HRNet-Human-Pose-Estimation
```python
#create a new folder to save cocoapi, in my case: foldername: "copcoapi"
mkdir cocoapi

cd cocoapi

!git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI/

!make install

!python3 setup.py install --user
```
6. Create new folder \
Working path: /content/HRNet-Human-Pose-Estimation
```bash
mkdir output
mkdir log
mkdir data
```
7. Download pre-trained models
```bash
!gdown --folder https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing/pytorch
```
```bash
$ HRNet-Human-Pose-Estimation
 `-- models
     `-- pytorch
         |-- imagenet
         |   |-- hrnet_w32-36af842e.pth
         |   |-- hrnet_w48-8ef0771d.pth
         |   |-- resnet50-19c8e357.pth
         |   |-- resnet101-5d3b4d8f.pth
         |   `-- resnet152-b121ed2d.pth
         |-- pose_coco
         |   |-- pose_hrnet_w32_256x192.pth
         |   |-- pose_hrnet_w32_384x288.pth
         |   |-- pose_hrnet_w48_256x192.pth
         |   |-- pose_hrnet_w48_384x288.pth
         |   |-- pose_resnet_101_256x192.pth
         |   |-- pose_resnet_101_384x288.pth
         |   |-- pose_resnet_152_256x192.pth
         |   |-- pose_resnet_152_384x288.pth
         |   |-- pose_resnet_50_256x192.pth
         |   `-- pose_resnet_50_384x288.pth
         `-- pose_mpii
             |-- pose_hrnet_w32_256x256.pth
             |-- pose_hrnet_w48_256x256.pth
             |-- pose_resnet_101_256x256.pth
             |-- pose_resnet_152_256x256.pth
             `-- pose_resnet_50_256x256.pth
             

We have a directory tree look like this:

$ HRNet-Human-Pose-Estimation
├── data
├── experiments
├── lib
├── log
├── models
├── output
├── tools 
├── README.md
└── requirements.txt
```
8. Download data \
Working path: /content/HRNet-Human-Pose-Estimation/data
```bash
!gdown --folder https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk
```
For COCO dataset: Working path: /content/HRNet-Human-Pose-Estimation/data/coco
```bash
#download train, val 2017 and annotations 2017
!wget -c http://images.cocodataset.org/zips/train2017.zip
!wget -c http://images.cocodataset.org/zips/val2017.zip
!wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#unzip file
!unzip train2017.zip
!unzip val2017.zip
!unzip annotations_trainval2017.zip
#remove zip file
!rm train2017.zip
!rm val2017.zip
!rm annotations_trainval2017.zip
```
Note: Move the zip file to the folder as COCO dataset tree before !unzip:
```bash
$ HRNet-Human-Pose-Estimation
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
For MPII dataset:
```bash
#download annotations
!gdown --folder https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing/annot
```
```bash
cd mpii
```
Working path: /content/HRNet-Human-Pose-Estimation/data/mpii
```bash
#download images
!wget -c https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
!tar -xvf mpii_human_pose_v1.tar.gz
```
The MPII dataset tree looks like this:
```bash
$ HRNet-Human-Pose-Estimation
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg

```





## Training and Testing
1. Testing on COCO val2017 dataset using model zoo's models
```bash
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    GPUS [0]\
    TEST.USE_GT_BBOX False
```
2. Training on COCO train2017 dataset
```bash
!python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    GPUS [0]
```

## The pose_hrnet.py (opensource shared)

![alt text](https://github.com/urgonguyen/HrNet/blob/main/HRNet-Diagram-summarized.png?raw=true)

ref: https://2d3d.ai/index.php/2020/06/14/human-pose-estimation-hrnet/

## Reference
```bash
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

