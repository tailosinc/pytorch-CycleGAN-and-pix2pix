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

##### Scale to 256x256, 512x512, 1024x1024 and train on entire image --> resulted in overfitting (memorizing output image)
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_256x256 --model pix2pix --gpu_ids -1 --load_size 256 --crop_size 256 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_512x512 --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_1024x1024 --model pix2pix --gpu_ids -1 --load_size 1024 --crop_size 1024 --phase test_small

##### Random crops after scaling width and height to be at least crop_size (128x128 and 256x256) (scaling maintains original image ratio)
#python3.7 train.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio --model pix2pix --gpu_ids -1 --crop_size 256 --preprocess scale_maintain_ratio_and_crop
#python3.7 train.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio --model pix2pix --gpu_ids -1 --crop_size 128 --preprocess scale_maintain_ratio_and_crop --netG unet_128

