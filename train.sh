CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/path/to/sam2_hiera_large.pt" \
--train_image_path "/path/to/train/images/" \
--train_mask_path "/path/to/train/masks/" \
--save_path "/path/to/output/checkpoints/" \
--epoch 300 \
--lr 0.0005 \
--batch_size 16