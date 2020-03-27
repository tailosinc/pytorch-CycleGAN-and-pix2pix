
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
#python3.7 datasets/combine_A_and_B.py --fold_A mb_maps/test_small_2ndpass/A/ --fold_B mb_maps/test_small_2ndpass/B --fold_AB mb_maps/test_small_2ndpass
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_crop256_maintainRatio/train_latest/images/A --fold_B results/mb_maps_pix2pix_train_crop256_maintainRatio/train_latest/images/B --fold_AB mb_maps/train_2ndpass
#python3.7 datasets/combine_A_and_B.py --fold_A mb_maps/test_2ndpass/A/ --fold_B mb_maps/test_2ndpass/B/ --fold_AB mb_maps/test_2ndpass/
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_crop256_maintainRatio_L1400/train_latest/images/A/ --fold_B results/mb_maps_pix2pix_train_crop256_maintainRatio_L1400/train_latest/images/B/ --fold_AB mb_maps/train_2ndpass_L1400
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_noRooms_L1100/train_noRooms_latest/images/A/ --fold_B results/mb_maps_pix2pix_train_noRooms_L1100/train_noRooms_latest/images/B/ --fold_AB mb_maps/train_2ndpass_noRooms_L1100
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_noRooms_L1400/train_noRooms_latest/images/A/ --fold_B results/mb_maps_pix2pix_train_noRooms_L1400/train_noRooms_latest/images/B/ --fold_AB mb_maps/train_2ndpass_noRooms_L1400
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_crop256_maintainRatio_L1400/test_small_latest/images/A/ --fold_B results/mb_maps_pix2pix_train_crop256_maintainRatio_L1400/test_small_latest/images/B/ --fold_AB mb_maps/test_small_2ndpass_L1400
#python3.7 datasets/combine_A_and_B.py --fold_A results/mb_maps_pix2pix_train_noRooms_L1400/test_small_latest/images/A/ --fold_B results/mb_maps_pix2pix_train_noRooms_L1400/test_small_latest/images/B/ --fold_AB mb_maps/test_small_2ndpass_noRooms_L1400

