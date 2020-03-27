set -ex

# By default, using pix2pix sets norm='batch', dataset_mode='aligned', netG='unet_256', netD='basic'
# And if is_train: pool_size=0, gan_mode='vanilla', lambda_L1=100.0

# TODO:
# - input_nc and output_nc (number of channels e.g. 3 for RGB and 1 for BW)
# - gan_mode
# - pool_size

# No rooms, 1st pass, L1 = 400
DATAROOT=mb_maps/
NAME=mb_maps_pix2pix_train_noRooms_L1400
L1=400
PHASE=train_noRooms

# No rooms, 2nd pass, L1 = 400 --> 200
#DATAROOT=mb_maps/
#NAME=mb_maps_pix2pix_train_2ndpass_noRooms_L1400_L1200
#L1=200
#PHASE=train_2ndpass_L1400

python3 train.py --dataroot $DATAROOT --name $NAME --model pix2pix --crop_size 256 --preprocess scale_maintain_ratio_and_crop --lambda_L1 $L1 --phase $PHASE
