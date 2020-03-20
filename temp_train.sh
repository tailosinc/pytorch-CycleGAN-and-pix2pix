set -ex

# By default, using pix2pix sets norm='batch', dataset_mode='aligned', netG='unet_256', netD='basic'
# And if is_train: pool_size=0, gan_mode='vanilla', lambda_L1=100.0

# Set load_size to crop_size to avoid cropping

# Set display_id to -1 to turn off visdom

# TODO: remove (works for their example facades dataset)
#python3.7 train.py --dataroot datasets/facades/ --name facades_pix2pix_train --model pix2pix --gpu_ids -1 --load_size 256 --crop_size 256 --direction BtoA

# TODO:
# - different load and crop sizes
#   - training on random crops of the data might be useful since CALM maps can't hold entire map in memory anyway
#   - may require unet_128 vs unet_256
# - different preprocess methods (e.g. not just resize_and_crop
# - input_nc and output_nc
# - gan_mode
# - pool_size

# TODO: visdom?

##### Experiments

## Bad (overfit for images it trained on, fuzzy and artifacts for new properties)
#python3.7 train.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_256x256 --model pix2pix --gpu_ids -1 --load_size 256 --crop_size 256

## Bad (overfit for images it trained on, fuzzy and artifacts for new properties)
#python3.7 train.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_512x512 --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512

##### In progress

## TODO TODO TODO
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 256 --preprocess scale_maintain_ratio_and_crop
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 128 --preprocess scale_maintain_ratio_and_crop --netG unet_128
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 128 --preprocess crop --netG unet_128
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 256 --load_size 1024 --preprocess resize_and_crop
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 256 --load_size 512 --preprocess scale_width_and_crop
#python3.7 train.py --dataroot mb_maps_temp/ --name mb_maps_pix2pix_train_crop256 --model pix2pix --gpu_ids -1 --crop_size 256 --load_size 256 --preprocess scale_width
