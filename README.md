## Description 
This project is based-on [RADAR](https://github.com/elliottwu/sorderender) and [3-Sweep](https://cg.cs.tsinghua.edu.cn/3sweep/software/index.html).
Our semi-automatic system can de-render and re-render the multi-axis(multi-components) revolutionary object with curved axis.

## Setup (Multi-Axis-Derender)

### 1. Docker Setup
* Install container
```
docker run --name multi-axis-derender -v /raid/dgx_user1/multi-axis-derender:/root -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=0 -dt pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
```
* Start container
```
docker start multi-axis-derender
```
* Execute container
```
docker exec -it multi-axis-derender bash
```

### 2. Install dependencies:
```
conda env create -f environment.yml
```
OR manually :
```
conda install -c conda-forge matplotlib opencv scikit-image pyyaml tensorboard trimesh
```


### 3. Install [PyTorch](https://pytorch.org/):
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

### 4. Install [neural_renderer](https://github.com/daniilidis-group/neural_renderer):
```
pip install neural_renderer_pytorch==1.1.3
```

## Downlaod [RADAR](https://github.com/elliottwu/sorderender) Pretrained Models
Download the pretrained models using the scripts provided in `pretrained/`, eg:
```
cd pretrained && sh download_pretrained_met_vase.sh
```
* If can not unzip the file please install unzip & zip  package.
```
apt-get update
apt-get install unzip
```

## Testing

### 1. Add 3-Sweep Extracted Data
Add 3-Sweep extracted data into folder and specific **rootDir**(input) and **outputRootDir**(output) in preprocess script.

### 2. Preprocessing
```
python preprocess.py
```
* If the error occured: **Llibgl.so.1: cannot open shared object file: no such file or directory**
```
apt-get install freeglut3-dev ffmpeg libsm6 libxext6 -y
```


### 3. Predict Render Factors
Check the configuration files in `configs/` and run experiments, eg:
```
python run.py --config configs/test_batch.yml --gpu 0 --num_workers 4
```
* **test_result_dir_root** : output root dir
* **test_batch_data_dir_root** : input root dir

## 4. Render Animations
To render animations of rotating vases and rotating light, check and run this script:
```
python render_animation.py
```
* **rootDir** : test root dir