
# fold_A and fold_B are expected to contain the following directory structure:
# - train/
#     image1.png
#     image2.png
# - test/
# - val/

python3.7 datasets/combine_A_and_B.py --fold_A ../downloaded_maps/paired_images_A --fold_B ../downloaded_maps/paired_images_B --fold_AB mb_maps/

