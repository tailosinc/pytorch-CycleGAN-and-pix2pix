# concatenate_images.py

import os
import shutil
import numpy as np
import cv2

input_dir = 'pix2pix/results/investor_demo_images/non_concatenated_images/'
output_dir = 'pix2pix/results/investor_demo_images/concatenated_images'

print('=== concatenated output files ===')
for root, dirs, files in os.walk(input_dir):
    for f in files:
        if 'real_A' in f:
            # Get the filenames
            image_real_A_path = os.path.join(root, f)
            image_real_B_path = os.path.join(root, f.replace('real_A', 'real_B'))
            image_fake_B_path = os.path.join(root, f.replace('real_A', 'fake_B'))
            # Read them into image objects
            image_real_A = cv2.imread(image_real_A_path, 1)
            image_real_B = cv2.imread(image_real_B_path, 1)
            image_fake_B = cv2.imread(image_fake_B_path, 1)
            # Create a black line for separating the concatenated images
            line_width = 5
            num_color_channels = 3
            if image_real_A.shape[0] > image_real_A.shape[1]:
                concatenation_axis = 1
                vertical_line = np.zeros(image_real_A.shape[0] * line_width * num_color_channels).reshape(-1, line_width, num_color_channels)
            else:
                concatenation_axis = 0
                vertical_line = np.zeros(image_real_A.shape[1] * line_width * num_color_channels).reshape(line_width, -1, num_color_channels)
            # Concatenate the images together (with a line separating each image)
            new_image = np.concatenate([image_real_A, vertical_line, image_fake_B, vertical_line, image_real_B], concatenation_axis)
            # Get the new file name (e.g. image_1.png instead of image_1_real_A.png)
            new_file_name = 'image_'+ f.split('_')[1] + '.png'
            # Write the new image to the output dir
            output_file_path = os.path.join(output_dir, new_file_name)
            print(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_file_path, new_image)
