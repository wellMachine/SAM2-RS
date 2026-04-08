CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/path/to/checkpoints/SAM2_RS-best.pth" \
--test_image_path "/path/to/test/images/" \
--test_gt_path "/path/to/test/masks/" \
--save_path "/path/to/output/predictions/" 