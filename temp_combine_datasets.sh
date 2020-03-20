
# fold_A and fold_B are expected to contain the following directory structure (note that images must contain the same name in A/ and B/ to be paired)
#
# A/
#     train/
#         image1.png
#         image2.png
# B/
#     train/
#         image1.png
#         image2.png

#python3.7 datasets/combine_A_and_B.py --fold_A ../downloaded_maps/paired_images_A --fold_B ../downloaded_maps/paired_images_B --fold_AB mb_maps/
python3.7 datasets/combine_A_and_B.py --fold_A mb_maps/test_small_2ndpass/A/ --fold_B mb_maps/test_small_2ndpass/B --fold_AB mb_maps/test_small_2ndpass

