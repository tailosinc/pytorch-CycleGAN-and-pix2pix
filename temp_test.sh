set -ex

# By default, using pix2pix sets norm='batch', dataset_mode='aligned', netG='unet_256', netD='basic'
# And if is_train: pool_size=0, gan_mode='vanilla', lambda_L1=100.0

# Set load_size to crop_size to avoid cropping

# NOTE: test results (generated images from inference) will be generated under results/ (specified through --results_dir flag)

##### Scale image to 256x256, 512x512, 1024x1024 and then perform inference
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_256x256 --model pix2pix --gpu_ids -1 --load_size 256 --crop_size 256 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_512x512 --model pix2pix --gpu_ids -1 --load_size 512 --crop_size 512 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_1024x1024 --model pix2pix --gpu_ids -1 --load_size 1024 --crop_size 1024 --phase test_small

##### Scale each dimension to nearest multiple of 256 and then perform inference
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small --netG unet_128
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small

##### 2nd pass testing (small test dataset)
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_2ndpass --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small_2ndpass
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_2ndpass_bigger --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small_2ndpass
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop128_maintainRatio_2ndpass --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small_2ndpass --netG unet_128

##### 2nd pass testing (entire test dataset)
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_2ndpass --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_2ndpass

##### TODO: tuning L1 loss weighting (for 2nd pass)
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_2ndpass_L1400 --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small_2ndpass
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_2ndpass_L1200 --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small_2ndpass

##### TODO: tuning L1 loss weighting (for 1st pass)
python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_L1400 --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small
#python3.7 test.py --dataroot mb_maps/ --name mb_maps_pix2pix_train_crop256_maintainRatio_L1200 --model pix2pix --gpu_ids -1 --preprocess scale_nearest256 --phase test_small
