set -ex

# By default, using pix2pix sets norm='batch', dataset_mode='aligned', netG='unet_256', netD='basic'
# And if is_train: pool_size=0, gan_mode='vanilla', lambda_L1=100.0

# NOTE: test results (generated images from inference) will be generated under results/ (specified through --results_dir flag)

# No rooms, 1st pass, L1 = 400
DATAROOT=mb_maps/
NAME=mb_maps_pix2pix_train_noRooms_L1400
PHASE=test_small

# No rooms, 2nd pass, L1 = 400 --> 200
#DATAROOT=mb_maps/
#NAME=mb_maps_pix2pix_train_2ndpass_noRooms_L1400_L1200
#PHASE=test_small_2ndpass_noRooms_L1400

python3.7 test.py --dataroot $DATAROOT --name $NAME --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase $PHASE
