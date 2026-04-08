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

### Download Weights

Download the pretrained weights file `sam2_hiera`. The download link is wrapped below:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

### Run Training

Use the `train.sh` script to start training. Make sure the weights file is placed in the correct path (e.g., `sam2_hiera_large.pt` in the current directory).

Example `train.sh` content:

```bash
#!/bin/bash
# Training script example
python train.py \
--hiera_path "/path/to/sam2_hiera_large.pt" \
--train_image_path "/path/to/train/images/" \
--train_mask_path "/path/to/train/masks/" \
--save_path "/path/to/output/checkpoints/" \
--epoch  \
--lr  \
--batch_size
```

## Test

Use the `test.sh` script to run testing.

### Run Testing

Example `test.sh` content:

```bash
#!/bin/bash
# Testing script example
CUDA_VISIBLE_DEVICES="" \
python test.py \
--checkpoint "/path/to/checkpoints/SAM2_RS-best.pth" \
--test_image_path "/path/to/test/images/" \
--test_gt_path "/path/to/test/masks/" \
--save_path "/path/to/output/predictions/" 
```

## Eval

Use the `eval.sh` script to run testing.

### Run Eval

Example `eval.sh` content:

```bash
#!/bin/bash
# Eval script example
python eval.py \
--pred_path "/path/to/prediction/results/" \
--gt_path "/path/to/ground_truth/masks/" 
```

## 📥 预测图下载

预测结果已上传至百度网盘（提取码：`t4dv`），点击下面链接下载：  
链接: https://pan.baidu.com/s/1lYovmYvIstcomPTlPVvbjQ?pwd=t4dv 提取码: t4dv 
