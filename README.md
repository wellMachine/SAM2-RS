# Unleashing the Potential of Segment Anything Model 2 for Efficient ORSI Salient Object Detection

## Clone Repository
```shell
git clone https://github.com/wellMachine/SAM2-RS.git
cd SAM2-RS/
```

## Requirements
This project uses the SAM2 code in `./sam2/` and does **not** require installing SAM2 as a separate package.
If you already have a working environment for SAM2, you can reuse it. Otherwise, you may create a new conda environment:

```shell
conda create -n sam2-rs python=3.10
conda activate sam2-rs
pip install -r requirements.txt
```

## Project Structure

- `SAM2-RS.py`: model implementation (contains `class SAM2_RS`)
- `SAM2_RS.py`: import bridge so you can do `from SAM2_RS import SAM2_RS`
- `train.py`: training script
- `test.py`: inference script (saves predicted masks)
- `eval.py`: evaluation script (metrics)
- `train.sh`, `test.sh`, `eval.sh`: runnable examples with placeholder paths
- `dataset.py`, `metrics.py`: dataset/metric utilities
- `sam2/`: SAM2-related code used by this project

## Requirements

This project uses the SAM2 code in `./sam2/` and does **not** require installing SAM2 as a separate package.
If you already have a working environment for SAM2, you can reuse it. Otherwise, you may create a new conda environment:

```shell
conda create -n sam2-rs python=3.10
conda activate sam2-rs
pip install -r requirements.txt
```

## Training

Direct download (example): sam2_hiera_large.pt
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

## 📥 预测图下载

预测结果已上传至百度网盘（提取码：`t4dv`），点击下面链接下载：  
链接: https://pan.baidu.com/s/1lYovmYvIstcomPTlPVvbjQ?pwd=t4dv 提取码: t4dv 
