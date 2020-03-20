set -ex

# By default, using pix2pix sets norm='batch', dataset_mode='aligned', netG='unet_256', netD='basic'
# And if is_train: pool_size=0, gan_mode='vanilla', lambda_L1=100.0

# Set load_size to crop_size to avoid cropping

# NOTE: test results (generated images from inference) will be generated under results/ (specified through --results_dir flag)

#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_256x256 --model pix2pix --gpu_ids -1 --load_size 256 --crop_size 256 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_512x512 --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512 --phase test_small --netG unet_128
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio --model pix2pix --gpu_ids -1 --crop_size 128 --preprocess scale_maintain_ratio_and_crop --phase test_small --netG unet_128
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small --netG unet_128

#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio --model pix2pix --gpu_ids -1 --crop_size 256 --preprocess scale_maintain_ratio_and_crop --phase test_small
python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small
