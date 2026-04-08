import os
import cv2
import argparse
from tqdm import tqdm
from py_sod_metrics import Fmeasure, Smeasure, Emeasure, MAE

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, required=True, help="path to prediction results")
parser.add_argument("--gt_path",   type=str, required=True, help="path to ground truth masks")
args = parser.parse_args()

# 初始化各指标
FM  = Fmeasure()
SM  = Smeasure()
EM  = Emeasure()
MAE = MAE()

pred_root = args.pred_path
gt_root   = args.gt_path
mask_list = sorted(os.listdir(gt_root))

for mask_name in tqdm(mask_list, desc="Eval", ncols=80):
    gt = cv2.imread(os.path.join(gt_root,   mask_name), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(os.path.join(pred_root, mask_name), cv2.IMREAD_GRAYSCALE)

    FM.step(pred=pr, gt=gt)
    SM.step(pred=pr, gt=gt)
    EM.step(pred=pr, gt=gt)
    MAE.step(pred=pr, gt=gt)

# 汇总结果
fm_res = FM.get_results()["fm"]
sm    = SM.get_results()["sm"]
em_res = EM.get_results()["em"]
mae   = MAE.get_results()["mae"]

results1 = {
    "Smeasure": sm,
    "MAE": mae
}

results2 = {
    "maxFm":  fm_res["curve"].max(),
    "meanFm": fm_res["curve"].mean(),
    "adpFm":  fm_res["adp"]
}

results3 = {
    "maxEm":   em_res["curve"].max(),
    "meanEm":  em_res["curve"].mean(),
    "adpEm":   em_res["adp"]
}

# 打印
print("=== Summary ===")
print("results1:", results1)
print("results2:", results2)
print("results3:", results3)
